from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_MAX_COMPAT_TOOL_NAME_LEN = 64
_SAFE_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class ExternalMCPServerConfig:
    name: str
    transport: Any
    tool_prefix: str = ""
    include_tools: frozenset[str] = frozenset()
    exclude_tools: frozenset[str] = frozenset()
    timeout_seconds: Optional[float] = None
    auth: Optional[str] = None


def _to_string_set(raw: Any) -> frozenset[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return frozenset()
    return frozenset(str(item) for item in raw if str(item).strip())


def _normalize_schema(raw_schema: Any) -> Dict[str, Any]:
    if not isinstance(raw_schema, dict):
        return {"type": "object", "properties": {}}

    schema = dict(raw_schema)
    schema_type = schema.get("type")
    if schema_type is None:
        schema["type"] = "object"
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema


def _extract_content_text(content_blocks: Any) -> str:
    if not isinstance(content_blocks, list):
        return ""

    text_parts: List[str] = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text)
            continue
        if isinstance(block, dict):
            candidate = block.get("text")
            if isinstance(candidate, str) and candidate.strip():
                text_parts.append(candidate)
    return "\n".join(text_parts)


def _normalize_tool_name_for_model_compat(raw_name: str) -> str:
    """
    Normalize a tool name to a conservative subset accepted by Anthropic/Gemini.

    Result format: ^[A-Za-z_][A-Za-z0-9_]{0,63}$
    """
    name = _SAFE_TOOL_NAME_PATTERN.sub("_", str(raw_name or "").strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "mcp_tool"

    if not (name[0].isalpha() or name[0] == "_"):
        name = f"mcp_{name}"

    if len(name) > _MAX_COMPAT_TOOL_NAME_LEN:
        name = name[:_MAX_COMPAT_TOOL_NAME_LEN].rstrip("_")
    if not name:
        name = "mcp_tool"
    return name


def _build_unique_tool_name(
    preferred_name: str,
    existing_names: set[str],
    *,
    stable_seed: str,
) -> str:
    """Return a unique provider-safe tool name with deterministic suffixing."""
    if preferred_name not in existing_names:
        return preferred_name

    digest = hashlib.sha1(stable_seed.encode("utf-8")).hexdigest()[:8]
    base_suffix = f"_{digest}"

    def _candidate(extra_suffix: str = "") -> str:
        suffix = f"{base_suffix}{extra_suffix}"
        max_base_len = _MAX_COMPAT_TOOL_NAME_LEN - len(suffix)
        base = preferred_name[: max(1, max_base_len)].rstrip("_")
        if not base:
            base = "mcp_tool"[:max(1, max_base_len)]
        return f"{base}{suffix}"

    candidate = _candidate()
    if candidate not in existing_names:
        return candidate

    for idx in range(2, 1000):
        candidate = _candidate(f"_{idx}")
        if candidate not in existing_names:
            return candidate

    raise RuntimeError("Unable to allocate unique external MCP tool name.")


class ExternalMCPToolRegistry:
    """
    Lazy external MCP tool discovery/execution registry.

    Converts discovered MCP tools into OpenAI-style function schemas so the
    existing provider clients can expose them directly to models.
    """

    def __init__(self, servers: Sequence[ExternalMCPServerConfig]):
        self._servers = list(servers)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._tool_defs: List[Dict[str, Any]] = []
        self._tool_routes: Dict[str, Tuple[ExternalMCPServerConfig, str]] = {}

    @classmethod
    def from_llm_config(cls, llm_config: Dict[str, Any]) -> "ExternalMCPToolRegistry":
        raw_servers = llm_config.get("external_mcp_servers")
        if not isinstance(raw_servers, list) or not raw_servers:
            return cls([])

        servers: List[ExternalMCPServerConfig] = []
        for idx, raw in enumerate(raw_servers):
            if not isinstance(raw, dict):
                continue

            if raw.get("enabled", True) is False:
                continue

            transport = raw.get("transport")
            if transport is None:
                transport = raw.get("url")
            if transport is None:
                logger.warning(
                    "Skipping external MCP server config at index %s: missing 'transport' or 'url'.",
                    idx,
                )
                continue

            timeout_raw = raw.get("timeout_seconds")
            timeout_seconds: Optional[float] = None
            if timeout_raw is not None:
                try:
                    timeout_seconds = float(timeout_raw)
                except Exception:
                    logger.warning(
                        "Invalid timeout_seconds=%r in external MCP server index %s; ignoring.",
                        timeout_raw,
                        idx,
                    )

            server = ExternalMCPServerConfig(
                name=str(raw.get("name") or f"external_mcp_{idx + 1}"),
                transport=transport,
                tool_prefix=str(raw.get("tool_prefix") or ""),
                include_tools=_to_string_set(raw.get("include_tools")),
                exclude_tools=_to_string_set(raw.get("exclude_tools")),
                timeout_seconds=timeout_seconds,
                auth=(str(raw.get("auth")) if raw.get("auth") is not None else None),
            )
            servers.append(server)

        return cls(servers)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            if not self._servers:
                self._initialized = True
                return

            for server in self._servers:
                await self._discover_server_tools(server)

            self._initialized = True

    async def _discover_server_tools(self, server: ExternalMCPServerConfig) -> None:
        try:
            from fastmcp import Client

            client = Client(
                server.transport,
                timeout=server.timeout_seconds,
                auth=server.auth,
            )
            async with client as session:
                tools = await session.list_tools()
        except Exception as exc:
            logger.warning(
                "Failed to discover tools from external MCP server '%s': %s",
                server.name,
                exc,
            )
            return

        for tool in tools:
            remote_name = str(getattr(tool, "name", "") or "").strip()
            if not remote_name:
                continue

            if server.include_tools and remote_name not in server.include_tools:
                continue
            if remote_name in server.exclude_tools:
                continue

            proposed_local_name = f"{server.tool_prefix}{remote_name}"
            normalized_local_name = _normalize_tool_name_for_model_compat(
                proposed_local_name
            )
            local_name = _build_unique_tool_name(
                normalized_local_name,
                set(self._tool_routes.keys()),
                stable_seed=f"{server.name}:{remote_name}:{proposed_local_name}",
            )

            if local_name != proposed_local_name:
                logger.info(
                    "External MCP tool '%s' from server '%s' exposed as '%s' for model compatibility.",
                    proposed_local_name,
                    server.name,
                    local_name,
                )

            parameters = _normalize_schema(getattr(tool, "inputSchema", None))
            description = (
                str(getattr(tool, "description", "") or "").strip()
                or f"External MCP tool '{remote_name}' from server '{server.name}'."
            )
            self._tool_defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": local_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
            self._tool_routes[local_name] = (server, remote_name)

    async def get_tools(self) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        return list(self._tool_defs)

    async def get_tool_names(self) -> set[str]:
        await self._ensure_initialized()
        return set(self._tool_routes.keys())

    async def call_tool(
        self, local_tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        route = self._tool_routes.get(local_tool_name)
        if route is None:
            return None

        server, remote_name = route
        try:
            from fastmcp import Client

            client = Client(
                server.transport,
                timeout=server.timeout_seconds,
                auth=server.auth,
            )
            async with client as session:
                result = await session.call_tool(
                    remote_name, arguments or {}, raise_on_error=False
                )
        except Exception as exc:
            return {
                "error": (
                    f"External MCP tool '{local_tool_name}' failed on server "
                    f"'{server.name}': {exc}"
                )
            }

        if getattr(result, "is_error", False):
            details = _extract_content_text(getattr(result, "content", None))
            if not details:
                details = str(getattr(result, "data", "")) or "Tool call returned an error."
            return {
                "error": (
                    f"External MCP tool '{local_tool_name}' failed on server "
                    f"'{server.name}': {details}"
                )
            }

        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        if data is not None:
            return {"result": data}

        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        if structured is not None:
            return {"result": structured}

        content_text = _extract_content_text(getattr(result, "content", None))
        if content_text:
            return {"result": content_text}

        return {"result": None}
