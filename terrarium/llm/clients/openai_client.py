import os
import json
import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from terrarium.llm.clients.abstract_client import AbstractClient
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from openai.types.responses.response_input_item_param import Message, FunctionCallOutput
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class OpenAIClient(AbstractClient):
    """
    Client for using OpenAI API for LLM agents.
    
    This module provides a client for using OpenAI models via their Responses API.
    Supports tool use, multi-turn conversations, and various OpenAI models.

    Retries:
      Exponential backoff is applied on transient errors (timeouts, connection errors,
      408/409/429/5xx). Configure via env vars:
        - OPENAI_MAX_RETRIES (default: 6)
        - OPENAI_RETRY_INITIAL_S (default: 0.5)
        - OPENAI_RETRY_MAX_S (default: 20)
        - OPENAI_RETRY_MULTIPLIER (default: 2)
        - OPENAI_RETRY_JITTER (default: 0.1)
    """
    
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")
        self.client = OpenAI(api_key=self.api_key, timeout=30.0)

        # Retry/backoff settings (sync; this client is called from async code paths).
        self._max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
        self._backoff_initial_s = float(os.getenv("OPENAI_RETRY_INITIAL_S", "0.5"))
        self._backoff_max_s = float(os.getenv("OPENAI_RETRY_MAX_S", "20"))
        self._backoff_multiplier = float(os.getenv("OPENAI_RETRY_MULTIPLIER", "2"))
        # Jitter is a fraction (0.0–1.0) applied as (1±jitter) to the base delay.
        self._backoff_jitter = float(os.getenv("OPENAI_RETRY_JITTER", "0.1"))

    def _has_temperature_restrictions(self, model_name: str) -> bool:
        """Check if the model has temperature restrictions (only supports default)."""
        restricted_models = ["gpt-4.1-nano", "gpt-5-nano"]
        return any(restricted_model in model_name.lower() for restricted_model in restricted_models)

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model (o1-series, o3-series)."""
        reasoning_indicators = ["o1-", "o3-", "reasoning"]
        return any(indicator in model_name.lower() for indicator in reasoning_indicators)

    @staticmethod
    def get_usage(response: Any, current_usage: dict[str, int]) -> Dict[str, int]:
        # Accumulate usage stats
        if hasattr(response, 'usage') and response.usage:
            current_usage["prompt_tokens"] += getattr(response.usage, 'input_tokens', 0)
            current_usage["completion_tokens"] += getattr(response.usage, 'output_tokens', 0)
            current_usage["total_tokens"] += getattr(response.usage, 'total_tokens', 0)
        return current_usage
    
    @staticmethod
    def init_context(system_prompt, user_prompt) -> List[Message]:
        """Initialize context/prompt/input for OpenAI responses."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # Build context array for responses.create() - convert messages to proper format
        context = []
        for msg in messages:
            context.append(Message(
                type="message",
                role=msg["role"],
                content=msg["content"]
            ))
        return context
    
    @staticmethod
    def _extract_message_content(output) -> str:
        """Extract text content from response output message (Responses API uses 'output_text' type)."""
        content = ""
        if hasattr(output, 'content') and output.content:
            for content_part in output.content:
                if hasattr(content_part, 'type') and content_part.type == 'output_text':
                    if hasattr(content_part, 'text'):
                        content += content_part.text
        return content

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
        # Best-effort: support different OpenAI SDK exception shapes.
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        if not headers:
            return None

        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        code = getattr(exc, "status_code", None)
        if code is not None:
            try:
                return int(code)
            except Exception:
                return None
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None) if resp is not None else None
        if code is None:
            return None
        try:
            return int(code)
        except Exception:
            return None

    def _should_retry(self, exc: Exception) -> Tuple[bool, Optional[float]]:
        """
        Decide whether to retry and optionally return a server-supplied wait time.

        Retries on:
        - 408/409/429/5xx
        - connection/timeouts
        """
        retry_after = self._extract_retry_after_seconds(exc)

        # Import exception types defensively (OpenAI SDK may vary by version).
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
        except Exception:  # pragma: no cover
            class _NoOpenAIError(Exception):
                pass

            APIConnectionError = APITimeoutError = RateLimitError = APIStatusError = _NoOpenAIError  # type: ignore

        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True, retry_after

        status = self._status_code(exc)
        if status is None and isinstance(exc, APIStatusError):
            # Some SDK versions only expose status via APIStatusError.status_code
            status = self._status_code(exc)

        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True, retry_after

        return False, None

    def _sleep_backoff(self, attempt: int, *, retry_after: Optional[float]) -> None:
        if retry_after is not None and retry_after >= 0:
            delay = retry_after
        else:
            base = self._backoff_initial_s * (self._backoff_multiplier ** max(0, attempt))
            delay = min(self._backoff_max_s, base)

        jitter = max(0.0, self._backoff_jitter)
        if jitter > 0:
            delay *= random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)

        time.sleep(max(0.0, delay))

    def _responses_create_with_retries(self, api_params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self.client.responses.create(**api_params)
            except Exception as exc:
                last_exc = exc
                should_retry, retry_after = self._should_retry(exc)
                if not should_retry or attempt >= self._max_retries:
                    raise
                status = self._status_code(exc)
                logger.warning(
                    "OpenAI request failed (attempt %s/%s, status=%s): %s; backing off and retrying",
                    attempt + 1,
                    self._max_retries + 1,
                    status,
                    str(exc),
                )
                self._sleep_backoff(attempt, retry_after=retry_after)
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        raise RuntimeError("OpenAI retries exhausted without exception")  # pragma: no cover

    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> tuple[Any, str]:
        """
        Outputs the response object, updated context, and response string.
        """
        # Build API call parameters
        api_params = {
            "model": params.get("model", None),
            "max_output_tokens": params.get("max_output_tokens", None),
            "tools": params.get("tools", []),
            "temperature": params.get("temperature", None),
            "reasoning_effort": params.get("reasoning_effort", None),
            "verbosity": params.get("verbosity", None),
        }

        # Remove temperature for reasoning/restricted models
        if self._is_reasoning_model(params.get("model", "")) or self._has_temperature_restrictions(params.get("model", "")):
            api_params.pop("temperature", None)
        # Remove GPT-5 specific parameters for non-GPT-5 models
        if "gpt-5" not in params.get("model", "").lower():
            api_params.pop("reasoning_effort", None)
            api_params.pop("verbosity", None)
        # Remove all None values from api_params
        api_params = {k: v for k, v in api_params.items() if v is not None}

        api_params["input"] = input
        context = input
        # Convert tools to responses.create() format (flattened structure)
        tools_for_api = []
        for tool in api_params.get("tools", []):
            if tool.get("type") == "function" and "function" in tool:
                # Convert from chat completions format to responses format
                func_def = tool["function"]
                converted_tool = {
                    "type": "function",
                    "name": func_def["name"],
                    "description": func_def["description"],
                    "parameters": func_def["parameters"]
                }
                tools_for_api.append(converted_tool)
            else:
                # Skip tools that don't match expected format
                print(f"Warning: Skipping tool with unexpected format: {tool}")
                continue

        # Update api_params with converted tools
        if tools_for_api:
            api_params["tools"] = tools_for_api
        else:
            # Remove tools parameter if empty to avoid API errors
            api_params.pop("tools", None)

        response = self._responses_create_with_retries(api_params)

        """
        The valid output types from response.output include:
        - 'message' - Text responses from the model (ResponseOutputMessage)
        - 'function_call' - Tool/function calls (ResponseFunctionToolCall)
        - 'file_search' - File search tool calls (ResponseFileSearchToolCall)
        - 'web_search' - Web search calls (ResponseFunctionWebSearch)
        - 'computer' - Computer tool calls (ResponseComputerToolCall)
        - 'reasoning' - Reasoning items (ResponseReasoningItem)
        """

        # NOTE: Don't automatically add response.output to context here
        # The agent's _process_tool_calls() will handle adding items to context
        # to avoid duplicate IDs in the input array

        # Extract content from the new response for next iteration
        response_str = ""
        for output in response.output:
            if hasattr(output, 'type') and output.type == 'message':
                response_str += self._extract_message_content(output)

        return response, response_str

    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any
    ) -> tuple[int, list[Any], list[str]]:
        """
        Process tool calls from OpenAI response.

        Parses response.output for function_call items, executes them via callback,
        and adds both the function call and result to the context.

        Args:
            response: OpenAI response object with .output attribute
            context: List of Message/FunctionCallOutput objects
            execute_tool_callback: Async function(tool_name: str, args: dict) -> dict

        Returns:
            Tuple of (tools_executed_count, updated_context, tool_names_list)
        """
        tool_calls_executed = 0
        step_tools = []

        for output in response.output:
            if not hasattr(output, 'type'):
                continue
            if output.type == 'function_call':
                # First, add the function_call output itself to context (before executing)
                context.append(output)
                # Extract and execute tool call
                tool_name = getattr(output, 'name', 'unknown')
                try:
                    args = getattr(output, 'arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
                tool_call_id = getattr(output, 'call_id', getattr(output, 'id', f"call_{tool_name}"))
                # Track tool call for trajectory
                step_tools.append(f"{tool_name} -- {json.dumps(args)}")
                # Execute the tool
                result = await execute_tool_callback(tool_name, args)
                # Add tool result to context (after the function_call)
                context.append(FunctionCallOutput(
                    type="function_call_output",
                    call_id=tool_call_id,
                    output=str(result)
                ))
                tool_calls_executed += 1
            elif output.type == 'message':
                # Add message output to context (already a proper response output object)
                context.append(output)
        return tool_calls_executed, context, step_tools
