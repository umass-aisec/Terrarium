"""
Google Gemini client implementation.

This module provides a client for using Google's Gemini models via the `google-genai` SDK.
Supports tool use, multi-turn conversations, and various Gemini models.
"""

import os
import json
import logging
from typing import Any, List, Dict, Optional
from llm_server.clients.abstract_client import AbstractClient


class GeminiClient(AbstractClient):
    """
    Client for using Google Gemini API for LLM agents.

    Provides the same interface as other clients but uses Google's Generative AI API.
    Supports Gemini models including gemini-2.0-flash-exp, gemini-1.5-pro, etc.
    """

    def __init__(self):
        """Initialize the Gemini client."""
        # `google-genai` emits a noisy warning if both GOOGLE_API_KEY and GEMINI_API_KEY
        # are present in the environment, even when an explicit api_key is provided.
        logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)

        try:
            from google import genai
            from dotenv import load_dotenv
        except ImportError as e:
            raise ImportError(
                "Gemini client requires 'google-genai' package. "
                "Install with: uv add google-genai (or pip install google-genai)"
            ) from e

        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY in .env file"
            )

        self.client = genai.Client(api_key=api_key)
        self._current_meta_context = {}  # For logging metadata

    def set_meta_context(
        self,
        agent_name: str,
        phase: str,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> None:
        """
        Set metadata context for logging purposes.

        Args:
            agent_name: Name of the current agent
            phase: Current simulation phase
            iteration: Current iteration number
            round_num: Current round number
        """
        self._current_meta_context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round_num": round_num
        }

    @staticmethod
    def get_usage(response: Any, current_usage: dict[str, int]) -> Dict[str, int]:
        """Accumulate usage statistics from response."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            current_usage["prompt_tokens"] += int(getattr(usage, "prompt_token_count", 0) or 0)
            current_usage["completion_tokens"] += int(
                getattr(usage, "candidates_token_count", 0) or 0
            )
            total = int(getattr(usage, "total_token_count", 0) or 0)
            if total:
                current_usage["total_tokens"] += total
            else:
                current_usage["total_tokens"] = (
                    current_usage["prompt_tokens"] + current_usage["completion_tokens"]
                )
        return current_usage

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict]:
        """
        Initialize context for Gemini API.

        Note: Gemini handles system instructions separately via model initialization,
        not as part of the message history. The system prompt should be passed
        in params["system"] when calling generate_response.

        Args:
            system_prompt: System instructions (will be included in first message for reference)
            user_prompt: Initial user message

        Returns:
            List of message dictionaries in Gemini format
        """
        # Gemini doesn't include system prompt in messages - it goes in model config
        # But we'll store it in the first message's metadata for consistency
        messages = [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
                "_system_prompt": system_prompt  # Store for later use
            }
        ]
        return messages

    @staticmethod
    def _extract_message_content(response) -> str:
        """
        Extract text content from Gemini response.

        Gemini responses have structure: response.candidates[0].content.parts[].
        Each part can have either text or a function call.
        """
        content = ""
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
                for part in candidate.content.parts:
                    # Only extract text parts, skip function_call parts
                    if getattr(part, "text", None):
                        content += part.text
        return content

    @staticmethod
    def _to_part(part: Any):
        from google.genai import types

        if isinstance(part, types.Part):
            return part

        if isinstance(part, dict):
            if "text" in part:
                return types.Part(text=str(part.get("text") or ""))

            function_call = part.get("function_call")
            if isinstance(function_call, dict):
                return types.Part(
                    function_call=types.FunctionCall(
                        id=function_call.get("id"),
                        name=function_call.get("name"),
                        args=function_call.get("args") or {},
                    )
                )

            function_response = part.get("function_response")
            if isinstance(function_response, dict):
                return types.Part(
                    function_response=types.FunctionResponse(
                        id=function_response.get("id"),
                        name=function_response.get("name"),
                        response=function_response.get("response") or {},
                    )
                )

        return None

    @classmethod
    def _to_content(cls, msg: Any):
        from google.genai import types

        if isinstance(msg, types.Content):
            return msg

        if isinstance(msg, dict):
            role = msg.get("role")
            raw_parts = msg.get("parts") or []
            parts = []
            for raw_part in raw_parts:
                part = cls._to_part(raw_part)
                if part is not None:
                    parts.append(part)
            return types.Content(role=role, parts=parts)

        return None

    @staticmethod
    def _build_tools(tools: List[Any]) -> tuple[list[Any], list[str]]:
        from google.genai import types

        function_declarations = []
        allowed_function_names: List[str] = []

        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue

            func_def = tool.get("function") or {}
            func_name = func_def.get("name")
            if not isinstance(func_name, str) or not func_name.strip():
                continue

            allowed_function_names.append(func_name)

            raw_parameters = func_def.get("parameters", {})
            parameters = raw_parameters
            if isinstance(raw_parameters, dict):
                parameters = dict(raw_parameters)
                # Gemini can hard-fail with MALFORMED_FUNCTION_CALL when required
                # arguments are omitted. Validate ourselves instead of crashing.
                parameters.pop("required", None)

            function_declarations.append(
                types.FunctionDeclaration(
                    name=func_name,
                    description=str(func_def.get("description") or ""),
                    parameters_json_schema=parameters if isinstance(parameters, dict) else None,
                )
            )

        if not function_declarations:
            return [], []

        return [types.Tool(function_declarations=function_declarations)], allowed_function_names

    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> tuple[Any, str]:
        """
        Generate a response using Google's Gemini API.

        Args:
            input: List of message dictionaries in Gemini format
            params: Generation parameters including:
                - model: Model name (e.g., "gemini-2.0-flash-exp")
                - max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
                - temperature: Sampling temperature
                - tools: List of tool definitions
                - system: System instruction (optional)

        Returns:
            Tuple of (response_object, response_text)
        """
        from google.genai import types

        model_name = params.get("model", "gemini-2.0-flash-exp")
        max_tokens = params.get("max_tokens") or params.get("max_output_tokens", 1000)
        temperature = params.get("temperature", 0.7)

        system_instruction = params.get("system", "")
        if not system_instruction and input and isinstance(input[0], dict):
            system_instruction = input[0].get("_system_prompt", "")

        converted_tools, allowed_function_names = self._build_tools(params.get("tools", []))

        tool_config = None
        if converted_tools and allowed_function_names:
            has_non_blackboard_tool = any(
                name != "post_message" for name in allowed_function_names
            )
            if has_non_blackboard_tool:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=allowed_function_names,
                    )
                )
            else:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO
                    )
                )

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction or None,
            tools=converted_tools or None,
            tool_config=tool_config,
        )

        contents = []
        for msg in input:
            content = self._to_content(msg)
            if content is not None:
                contents.append(content)

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generation_config,
        )

        if (
            getattr(response, "candidates", None)
            and response.candidates[0].finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL
        ):
            # Retry once without tool calling so we can still get a text response.
            generation_config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.NONE
                )
            )
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )

        response_text = self._extract_message_content(response)
        return response, response_text

    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any
    ) -> tuple[int, list[Any], list[str]]:
        """
        Process tool calls from Gemini response.

        Parses response.candidates[0].content.parts for function_call items,
        executes them via callback, and adds both the function call and result to context.

        Args:
            response: Gemini response object with .candidates attribute
            context: List of message dictionaries in Gemini format
            execute_tool_callback: Async function(tool_name: str, args: dict) -> dict

        Returns:
            Tuple of (tools_executed_count, updated_context, tool_names_list)
        """
        from google.genai import types

        tool_calls_executed = 0
        step_tools = []

        if not getattr(response, "candidates", None):
            return tool_calls_executed, context, step_tools

        candidate = response.candidates[0]
        candidate_content = getattr(candidate, "content", None)
        if not candidate_content or not getattr(candidate_content, "parts", None):
            return tool_calls_executed, context, step_tools

        # Append the model response (includes both text and function calls).
        context.append(candidate_content)

        for part in candidate_content.parts:
            function_call = getattr(part, "function_call", None)
            if not function_call:
                continue

            tool_name = function_call.name
            args = function_call.args or {}

            step_tools.append(f"{tool_name} -- {json.dumps(args)}")

            result = await execute_tool_callback(tool_name, args)

            function_response = types.FunctionResponse(
                id=function_call.id,
                name=tool_name,
                response={"result": result},
            )

            function_response_part = types.Part(function_response=function_response)

            context.append(types.Content(role="user", parts=[function_response_part]))

            tool_calls_executed += 1

        return tool_calls_executed, context, step_tools
