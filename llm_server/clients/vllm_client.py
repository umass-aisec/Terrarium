from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import requests

from llm_server.clients.abstract_client import AbstractClient


def _convert_tools(tools: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Convert Terrarium tool schema into OpenAI-compatible tool definitions."""
    if not tools:
        return []
    normalized = []
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


class VLLMClient(AbstractClient):
    """Client that talks to a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        request_timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.request_timeout = request_timeout
        self.session = requests.Session()

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Initialize chat-style context for vLLM."""
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
    def _stringify_reasoning(reasoning_block: Any) -> str:
        if isinstance(reasoning_block, str):
            return reasoning_block
        if isinstance(reasoning_block, dict):
            return reasoning_block.get("text") or reasoning_block.get("content") or ""
        if isinstance(reasoning_block, list):
            parts: List[str] = []
            for chunk in reasoning_block:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    parts.append(chunk.get("text") or chunk.get("content") or "")
            return "".join(parts)
        return ""

    def extract_reasoning_trace(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        reasoning_payload = (
            message.get("reasoning")
            or message.get("reasoning_content")
            or message.get("thoughts")
        )
        return self._stringify_reasoning(reasoning_payload)

    @staticmethod
    def get_usage(response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
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
            "model": params.get("model", self.model_name),
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
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            data=json.dumps(payload),
            timeout=self.request_timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"vLLM chat request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
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
