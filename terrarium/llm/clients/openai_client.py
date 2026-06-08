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

    Retries use exponential backoff for transient errors such as timeouts,
    connection errors, 408/409/429 responses, and 5xx responses.

    Environment variables:

    - ``OPENAI_MAX_RETRIES`` (default: 6)
    - ``OPENAI_RETRY_INITIAL_S`` (default: 0.5)
    - ``OPENAI_RETRY_MAX_S`` (default: 20)
    - ``OPENAI_RETRY_MULTIPLIER`` (default: 2)
    - ``OPENAI_RETRY_JITTER`` (default: 0.1)
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[Any] = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        default_query: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        provider_name: str = "OpenAI",
        capability_model: Optional[str] = None,
    ):
        load_dotenv(override=True)
        self.api_key = api_key or os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"{provider_name} API key not found. Set {api_key_env_var} in .env file"
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": float(timeout),
        }
        if base_url:
            client_kwargs["base_url"] = str(base_url).rstrip("/") + "/"
        if default_query:
            client_kwargs["default_query"] = default_query
        self.client = OpenAI(**client_kwargs)
        self._capability_model = (
            str(capability_model).strip() if capability_model else None
        )

        # Retry/backoff settings (sync; this client is called from async code paths).
        self._max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
        self._backoff_initial_s = float(os.getenv("OPENAI_RETRY_INITIAL_S", "0.5"))
        self._backoff_max_s = float(os.getenv("OPENAI_RETRY_MAX_S", "20"))
        self._backoff_multiplier = float(os.getenv("OPENAI_RETRY_MULTIPLIER", "2"))
        # Jitter is a fraction (0.0–1.0) applied as (1±jitter) to the base delay.
        self._backoff_jitter = float(os.getenv("OPENAI_RETRY_JITTER", "0.1"))

    def _has_temperature_restrictions(self, model_name: str) -> bool:
        """Check if the model has temperature restrictions (only supports default)."""
        restricted_models = [
            "gpt-4.1-nano",
            "gpt-5-nano",
            "gpt-5.4-nano",
        ]
        return any(restricted_model in model_name.lower() for restricted_model in restricted_models)

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if model is a reasoning model (o1-series, o3-series)."""
        model_name_lower = model_name.lower()
        non_reasoning_indicators = ["non-reasoning", "non_reasoning", "nonreasoning"]
        if any(
            indicator in model_name_lower for indicator in non_reasoning_indicators
        ):
            return any(indicator in model_name_lower for indicator in ["o1-", "o3-"])
        reasoning_indicators = ["o1-", "o3-", "reasoning"]
        return any(indicator in model_name_lower for indicator in reasoning_indicators)

    def _resolve_capability_model(self, params: Dict[str, Any]) -> str:
        """Return the model label used for provider capability checks."""
        return str(self._capability_model or params.get("model", "") or "")

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
    def _get_item_field(item: Any, field: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field, default)
        return getattr(item, field, default)

    @classmethod
    def _normalize_structured_value(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            value = vars(value)
        if isinstance(value, dict):
            return {
                key: cls._normalize_structured_value(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_structured_value(item) for item in value]
        return value

    @classmethod
    def _sanitize_response_input_item(cls, item: Any) -> Optional[dict[str, Any]]:
        item_type = cls._get_item_field(item, "type")
        role = cls._get_item_field(item, "role")

        if item_type == "message":
            content = cls._normalize_structured_value(cls._get_item_field(item, "content"))
            if role in {"system", "user", "developer"}:
                sanitized = {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
                status = cls._get_item_field(item, "status")
                if status in {"in_progress", "completed", "incomplete"}:
                    sanitized["status"] = status
                return sanitized

            if role == "assistant":
                item_id = cls._get_item_field(item, "id")
                status = cls._get_item_field(item, "status")
                if not isinstance(item_id, str) or status not in {
                    "in_progress",
                    "completed",
                    "incomplete",
                }:
                    return None
                return {
                    "type": "message",
                    "role": "assistant",
                    "id": item_id,
                    "status": status,
                    "content": content,
                }

            return None

        if item_type == "function_call":
            sanitized = {
                "type": "function_call",
                "call_id": cls._get_item_field(item, "call_id"),
                "name": cls._get_item_field(item, "name"),
                "arguments": cls._get_item_field(item, "arguments"),
            }
            item_id = cls._get_item_field(item, "id")
            if isinstance(item_id, str) and item_id.startswith("fc"):
                sanitized["id"] = item_id
            status = cls._get_item_field(item, "status")
            if status in {"in_progress", "completed", "incomplete"}:
                sanitized["status"] = status
            return sanitized

        if item_type == "function_call_output":
            sanitized = {
                "type": "function_call_output",
                "call_id": cls._get_item_field(item, "call_id"),
                "output": cls._normalize_structured_value(
                    cls._get_item_field(item, "output")
                ),
            }
            status = cls._get_item_field(item, "status")
            if status in {"in_progress", "completed", "incomplete"}:
                sanitized["status"] = status
            return sanitized

        if item_type == "reasoning":
            item_id = cls._get_item_field(item, "id")
            summary = cls._normalize_structured_value(cls._get_item_field(item, "summary"))
            if not isinstance(item_id, str) or summary is None:
                return None
            sanitized = {
                "type": "reasoning",
                "id": item_id,
                "summary": summary,
            }
            content = cls._normalize_structured_value(cls._get_item_field(item, "content"))
            if content is not None:
                sanitized["content"] = content
            encrypted_content = cls._get_item_field(item, "encrypted_content")
            if encrypted_content is not None:
                sanitized["encrypted_content"] = encrypted_content
            status = cls._get_item_field(item, "status")
            if status in {"in_progress", "completed", "incomplete"}:
                sanitized["status"] = status
            return sanitized

        if isinstance(item, dict):
            return cls._normalize_structured_value(item)

        return None

    @classmethod
    def _sanitize_response_input(cls, items: List[Any]) -> List[Any]:
        sanitized_items: List[Any] = []
        for item in items:
            sanitized_item = cls._sanitize_response_input_item(item)
            if sanitized_item is not None:
                sanitized_items.append(sanitized_item)
        return sanitized_items
    
    @staticmethod
    def _extract_message_content(output) -> str:
        """Extract text content from a response output message."""
        content = ""
        if hasattr(output, 'content') and output.content:
            for content_part in output.content:
                if (
                    hasattr(content_part, 'type')
                    and content_part.type in {'output_text', 'text'}
                ):
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
            "tool_choice": params.get("tool_choice", None),
            "parallel_tool_calls": params.get("parallel_tool_calls", None),
        }

        reasoning_effort = params.get("reasoning_effort", None)
        if reasoning_effort is not None:
            api_params["reasoning"] = {"effort": reasoning_effort}

        verbosity = params.get("verbosity", None)
        if verbosity is not None:
            api_params["text"] = {"verbosity": verbosity}

        capability_model = self._resolve_capability_model(params)

        # Remove temperature for reasoning/restricted models
        if self._is_reasoning_model(capability_model) or self._has_temperature_restrictions(capability_model):
            api_params.pop("temperature", None)
        # Remove GPT-5 specific parameters for non-GPT-5 models
        if "gpt-5" not in capability_model.lower():
            api_params.pop("reasoning", None)
            api_params.pop("text", None)
        # Remove all None values from api_params
        api_params = {k: v for k, v in api_params.items() if v is not None}

        sanitized_input = self._sanitize_response_input(input)
        api_params["input"] = sanitized_input
        context = sanitized_input
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
        response_str = getattr(response, "output_text", "") or ""
        if not response_str:
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

        tool_outputs: list[dict[str, Any]] = []
        for output in response.output:
            if not hasattr(output, 'type'):
                continue
            sanitized_output = self._sanitize_response_input_item(output)
            if sanitized_output is not None:
                context.append(sanitized_output)
            if output.type == 'function_call':
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
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": str(result),
                })
                tool_calls_executed += 1
        context.extend(tool_outputs)
        return tool_calls_executed, context, step_tools
