from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from llm_server.clients.abstract_client import AbstractClient


def _convert_tools(tools: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Convert Terrarium tool schema into OpenAI-compatible tool definitions."""
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


class FireworksClient(AbstractClient):
    """Client that talks to Fireworks.ai's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        api_key: Optional[str] = None,
        request_timeout: int = 60,
    ):
        load_dotenv(override=True)
        resolved_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Fireworks API key not found. Set FIREWORKS_API_KEY in .env file"
            )

        self.base_url = str(base_url).rstrip("/")
        self.api_key = resolved_key
        self.request_timeout = int(request_timeout)
        self.session = requests.Session()

    @staticmethod
    def _get_retry_config() -> tuple[int, float, float]:
        max_retries = int(os.getenv("FIREWORKS_MAX_RETRIES", "6"))
        base_sleep_s = float(os.getenv("FIREWORKS_RETRY_BASE_SLEEP_S", "1.0"))
        max_sleep_s = float(os.getenv("FIREWORKS_RETRY_MAX_SLEEP_S", "30.0"))
        return max_retries, base_sleep_s, max_sleep_s

    @staticmethod
    def _get_retry_after_seconds(response: requests.Response) -> Optional[float]:
        retry_after = response.headers.get("Retry-After")
        if not retry_after:
            return None
        try:
            return float(retry_after)
        except ValueError:
            return None

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Initialize chat-style context for Fireworks."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_message_content(message: Dict[str, Any]) -> str:
        if isinstance(message, dict) and "choices" in message:
            choices = message.get("choices") or []
            message = choices[0].get("message") if choices else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    @staticmethod
    def get_usage(
        response: Dict[str, Any], current_usage: Dict[str, int]
    ) -> Dict[str, int]:
        usage = response.get("usage") or {}
        current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        if "total_tokens" in usage:
            current_usage["total_tokens"] += usage.get("total_tokens", 0)
        else:
            current_usage["total_tokens"] = (
                current_usage["prompt_tokens"] + current_usage["completion_tokens"]
            )
        return current_usage

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_response(
        self,
        input: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        payload: Dict[str, Any] = {
            "model": params.get("model"),
            "messages": input,
        }

        max_tokens = params.get("max_completion_tokens") or params.get("max_output_tokens")
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]

        converted_tools = _convert_tools(params.get("tools"))
        if converted_tools:
            payload["tools"] = converted_tools

        max_retries, base_sleep_s, max_sleep_s = self._get_retry_config()
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Optional[str] = None
        response: Optional[requests.Response] = None

        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    data=json.dumps(payload),
                    timeout=self.request_timeout,
                )
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                response = None
            else:
                if response.ok:
                    last_error = None
                    break
                last_error = f"HTTP {response.status_code}: {response.text}"

            status_code = response.status_code if response is not None else None
            can_retry = (status_code in retryable_statuses) or (response is None)
            if attempt >= max_retries or not can_retry:
                break

            retry_after_s = (
                self._get_retry_after_seconds(response) if response is not None else None
            )
            if retry_after_s is not None and retry_after_s > 0:
                sleep_s = min(retry_after_s, max_sleep_s)
            else:
                exp_sleep = base_sleep_s * (2**attempt)
                jitter = random.uniform(0.0, min(1.0, exp_sleep * 0.25))
                sleep_s = min(max_sleep_s, exp_sleep + jitter)
            time.sleep(sleep_s)

        if last_error is not None:
            raise RuntimeError(
                f"Fireworks chat request failed after {max_retries + 1} attempts: {last_error}"
            )

        data = response.json()  # type: ignore[union-attr]
        choices = data.get("choices") or []
        first_message = choices[0]["message"] if choices else {"content": ""}
        response_str = self._extract_message_content(first_message)
        return data, response_str

    async def process_tool_calls(
        self,
        response: Dict[str, Any],
        context: List[Dict[str, Any]],
        execute_tool_callback: Any,
    ) -> Tuple[int, List[Dict[str, Any]], List[str]]:
        choices = response.get("choices") or []
        if not choices:
            return 0, context, []

        message = choices[0].get("message") or {}
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
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )
            tool_calls_executed += 1

        return tool_calls_executed, context, step_tools

