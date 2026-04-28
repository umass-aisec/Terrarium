from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from dotenv import load_dotenv

from terrarium.llm.clients.azure_common import (
    build_entra_token_provider,
    normalize_openai_v1_base_url,
)
from terrarium.llm.clients.openai_client import OpenAIClient

_DEFAULT_SCOPE = "https://ai.azure.com/.default"


def _normalize_api_style(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "responses": "responses",
    }
    resolved = aliases.get(normalized or "responses")
    if not resolved:
        raise ValueError(
            "Invalid Foundry api_style. Expected one of: responses, chat_completions"
        )
    return resolved


def _normalize_foundry_base_url_and_query(endpoint: Any) -> Tuple[str, Dict[str, Any]]:
    """Normalize Foundry endpoint styles supported by the OpenAI SDK client."""
    raw = str(endpoint or "").strip()
    parsed = urlsplit(raw)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    path = (parsed.path or "").rstrip("/")

    if path.endswith("/models/chat/completions"):
        base_path = path[: -len("/chat/completions")]
        return (
            urlunsplit((parsed.scheme, parsed.netloc, base_path + "/", "", "")),
            query,
        )
    if path.endswith("/models"):
        return (
            urlunsplit((parsed.scheme, parsed.netloc, path + "/", "", "")),
            query,
        )

    return (
        normalize_openai_v1_base_url(
            urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
            setting_name="llm.foundry.project_endpoint",
        ),
        query,
    )


def _convert_tools_for_chat(
    tools: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    if not tools:
        return []
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                },
            }
        )
    return normalized


class FoundryClient(OpenAIClient):
    """
    Microsoft Foundry Models client via project endpoint + OpenAI SDK.

    Supports:
    - API key auth
    - Microsoft Entra token auth (azure-identity)
    """

    def __init__(
        self,
        *,
        project_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_mode: Optional[str] = None,
        entra_scope: Optional[str] = None,
        timeout: float = 30.0,
        base_model: Optional[str] = None,
        api_style: Optional[str] = None,
    ):
        load_dotenv(override=True)

        resolved_endpoint = project_endpoint or os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT")
        base_url, default_query = _normalize_foundry_base_url_and_query(
            resolved_endpoint
        )

        resolved_mode = (
            auth_mode
            or os.getenv("AI_FOUNDRY_AUTH_MODE")
            or ("api_key" if (api_key or os.getenv("AI_FOUNDRY_API_KEY")) else "entra")
        ).strip().lower()

        if resolved_mode not in {"api_key", "entra"}:
            raise ValueError(
                "Invalid Foundry auth mode. Expected one of: api_key, entra"
            )

        resolved_api_key: Any
        if resolved_mode == "entra":
            scope = (entra_scope or os.getenv("AI_FOUNDRY_ENTRA_SCOPE") or _DEFAULT_SCOPE).strip()
            resolved_api_key = build_entra_token_provider(scope)
        else:
            resolved_api_key = api_key or os.getenv("AI_FOUNDRY_API_KEY")

        self._api_style = _normalize_api_style(
            api_style or os.getenv("AI_FOUNDRY_API_STYLE")
        )

        super().__init__(
            api_key=resolved_api_key,
            api_key_env_var="AI_FOUNDRY_API_KEY",
            base_url=base_url,
            default_query=default_query,
            timeout=timeout,
            provider_name="Microsoft Foundry",
            capability_model=base_model,
        )

    def init_context(self, system_prompt: str, user_prompt: str):
        if self._api_style == "chat_completions":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return super().init_context(system_prompt, user_prompt)

    def get_usage(
        self, response: Any, current_usage: dict[str, int]
    ) -> Dict[str, int]:
        if self._api_style != "chat_completions":
            return super().get_usage(response, current_usage)

        usage = response.get("usage") or {}
        current_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        current_usage["completion_tokens"] += int(
            usage.get("completion_tokens", 0) or 0
        )
        if "total_tokens" in usage:
            current_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        else:
            current_usage["total_tokens"] = (
                current_usage["prompt_tokens"] + current_usage["completion_tokens"]
            )
        return current_usage

    @staticmethod
    def _extract_chat_message_content(message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    @classmethod
    def _sanitize_chat_tool_call(cls, call: Any) -> Optional[Dict[str, Any]]:
        call_dict = cls._normalize_structured_value(call)
        if not isinstance(call_dict, dict):
            return None

        function_block = call_dict.get("function") or {}
        if not isinstance(function_block, dict):
            return None

        tool_name = function_block.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return None

        arguments = function_block.get("arguments")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments if arguments is not None else {})
            except TypeError:
                arguments = "{}"

        sanitized: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        call_id = call_dict.get("id")
        if isinstance(call_id, str) and call_id:
            sanitized["id"] = call_id
        return sanitized

    @classmethod
    def _sanitize_chat_message(cls, message: Any) -> Optional[Dict[str, Any]]:
        message_dict = cls._normalize_structured_value(message)
        if not isinstance(message_dict, dict):
            return None

        role = message_dict.get("role")
        if not isinstance(role, str) or not role:
            return None

        content = cls._normalize_structured_value(message_dict.get("content"))
        sanitized: Dict[str, Any] = {"role": role}

        if role in {"system", "user", "developer"}:
            sanitized["content"] = content if content is not None else ""
            name = message_dict.get("name")
            if isinstance(name, str) and name:
                sanitized["name"] = name
            return sanitized

        if role == "assistant":
            tool_calls = []
            for call in message_dict.get("tool_calls") or []:
                sanitized_call = cls._sanitize_chat_tool_call(call)
                if sanitized_call is not None:
                    tool_calls.append(sanitized_call)

            if content is not None or not tool_calls:
                sanitized["content"] = content if content is not None else ""
            if tool_calls:
                sanitized["tool_calls"] = tool_calls

            name = message_dict.get("name")
            if isinstance(name, str) and name:
                sanitized["name"] = name
            return sanitized

        if role == "tool":
            tool_call_id = message_dict.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                return None
            sanitized["tool_call_id"] = tool_call_id
            sanitized["content"] = content if content is not None else ""
            return sanitized

        return None

    @classmethod
    def _sanitize_chat_messages(cls, messages: List[Any]) -> List[Dict[str, Any]]:
        sanitized_messages: List[Dict[str, Any]] = []
        for message in messages:
            sanitized_message = cls._sanitize_chat_message(message)
            if sanitized_message is not None:
                sanitized_messages.append(sanitized_message)
        return sanitized_messages

    def _chat_completions_create_with_retries(self, api_params: Dict[str, Any]) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self.client.chat.completions.create(**api_params)
            except Exception as exc:
                last_exc = exc
                should_retry, retry_after = self._should_retry(exc)
                if not should_retry or attempt >= self._max_retries:
                    raise
                status = self._status_code(exc)
                self._sleep_backoff(attempt, retry_after=retry_after)
                # Keep logging consistent with the responses path.
                import logging

                logging.getLogger(__name__).warning(
                    "Foundry chat request failed (attempt %s/%s, status=%s): %s; backing off and retrying",
                    attempt + 1,
                    self._max_retries + 1,
                    status,
                    str(exc),
                )
        if last_exc is not None:  # pragma: no cover
            raise last_exc
        raise RuntimeError("Foundry chat retries exhausted without exception")  # pragma: no cover

    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> Tuple[Any, str]:
        if self._api_style != "chat_completions":
            return super().generate_response(input=input, params=params)

        capability_model = self._resolve_capability_model(params)
        sanitized_messages = self._sanitize_chat_messages(input)
        api_params: Dict[str, Any] = {
            "model": params.get("model"),
            "messages": sanitized_messages,
        }

        max_tokens = params.get("max_completion_tokens") or params.get(
            "max_output_tokens"
        ) or params.get("max_tokens")
        if max_tokens is not None:
            if "gpt-5" in capability_model.lower():
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens

        temperature = params.get("temperature")
        if temperature is not None and not (
            self._is_reasoning_model(capability_model)
            or self._has_temperature_restrictions(capability_model)
        ):
            api_params["temperature"] = temperature

        converted_tools = _convert_tools_for_chat(params.get("tools"))
        if converted_tools:
            api_params["tools"] = converted_tools
        if params.get("tool_choice") is not None:
            api_params["tool_choice"] = params.get("tool_choice")
        if params.get("parallel_tool_calls") is not None:
            api_params["parallel_tool_calls"] = params.get("parallel_tool_calls")

        response = self._chat_completions_create_with_retries(api_params)
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump(exclude_none=True)
            except TypeError:
                response_dict = response.model_dump()
        else:
            response_dict = response
        choices = response_dict.get("choices") or []
        first_message = choices[0].get("message") if choices else {"content": ""}
        response_str = self._extract_chat_message_content(first_message)
        return response_dict, response_str

    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any,
    ) -> Tuple[int, list[Any], list[str]]:
        if self._api_style != "chat_completions":
            return await super().process_tool_calls(
                response=response,
                context=context,
                execute_tool_callback=execute_tool_callback,
            )

        choices = response.get("choices") or []
        if not choices:
            return 0, context, []

        message = self._sanitize_chat_message(choices[0].get("message") or {})
        if message is None:
            message = {"role": "assistant", "content": ""}
        context.append(message)
        tool_calls = message.get("tool_calls") or []
        tool_calls_executed = 0
        step_tools: List[str] = []

        for call in tool_calls:
            function_block = call.get("function") or {}
            tool_name = function_block.get("name", "unknown_tool")
            arguments_raw = function_block.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments = {}
            step_tools.append(f"{tool_name} -- {json.dumps(arguments)}")
            result = await execute_tool_callback(tool_name, arguments)
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result),
                }
            )
            tool_calls_executed += 1

        return tool_calls_executed, context, step_tools
