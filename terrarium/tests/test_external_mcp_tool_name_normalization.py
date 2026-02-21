import re
import sys
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from terrarium.llm.clients.external_mcp import ExternalMCPToolRegistry


_COMPAT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class _FakeTool:
    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class _FakeToolResult:
    def __init__(self, data: Dict[str, Any]):
        self.is_error = False
        self.data = data
        self.structured_content = None
        self.content = []


class _FakeFastMCPClient:
    tools_by_transport: Dict[str, List[_FakeTool]] = {}
    calls_by_transport: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, transport, timeout=None, auth=None):
        self.transport = transport
        self.timeout = timeout
        self.auth = auth

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        return list(self.tools_by_transport.get(self.transport, []))

    async def call_tool(self, name, arguments, raise_on_error=False):
        self.calls_by_transport.setdefault(self.transport, []).append(
            {"name": name, "arguments": dict(arguments or {})}
        )
        return _FakeToolResult(
            data={"remote_name": name, "arguments": dict(arguments or {})}
        )


class ExternalMCPToolNameNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_names_are_compatible_with_anthropic_and_gemini(self):
        _FakeFastMCPClient.tools_by_transport = {
            "mock://compat": [
                _FakeTool("jira.create-ticket-v2"),
                _FakeTool("123_start.with.dots-and-dashes"),
                _FakeTool("x" * 120),
            ]
        }
        _FakeFastMCPClient.calls_by_transport = {}

        fake_fastmcp = types.ModuleType("fastmcp")
        fake_fastmcp.Client = _FakeFastMCPClient

        llm_config = {
            "external_mcp_servers": [
                {
                    "name": "compat",
                    "url": "mock://compat",
                    "tool_prefix": "dcops.",
                }
            ]
        }

        with patch.dict(sys.modules, {"fastmcp": fake_fastmcp}):
            registry = ExternalMCPToolRegistry.from_llm_config(llm_config)
            tools = await registry.get_tools()

            tool_names = [tool["function"]["name"] for tool in tools]
            self.assertEqual(len(tool_names), 3)
            for name in tool_names:
                self.assertRegex(name, _COMPAT_NAME_RE)
                self.assertNotIn(".", name)
                self.assertNotIn("-", name)
                self.assertLessEqual(len(name), 64)

            called_remote_names = set()
            for local_name in tool_names:
                result = await registry.call_tool(local_name, {"ok": True})
                self.assertIsInstance(result, dict)
                called_remote_names.add(result.get("remote_name"))

            self.assertEqual(
                called_remote_names,
                {
                    "jira.create-ticket-v2",
                    "123_start.with.dots-and-dashes",
                    "x" * 120,
                },
            )

    async def test_colliding_normalized_names_are_disambiguated_and_routable(self):
        _FakeFastMCPClient.tools_by_transport = {
            "mock://collision": [
                _FakeTool("alpha.beta"),
                _FakeTool("alpha-beta"),
            ]
        }
        _FakeFastMCPClient.calls_by_transport = {}

        fake_fastmcp = types.ModuleType("fastmcp")
        fake_fastmcp.Client = _FakeFastMCPClient

        llm_config = {
            "external_mcp_servers": [
                {
                    "name": "collision",
                    "url": "mock://collision",
                }
            ]
        }

        with patch.dict(sys.modules, {"fastmcp": fake_fastmcp}):
            registry = ExternalMCPToolRegistry.from_llm_config(llm_config)
            tools = await registry.get_tools()
            tool_names = [tool["function"]["name"] for tool in tools]

            self.assertEqual(len(tool_names), 2)
            self.assertEqual(len(set(tool_names)), 2)
            self.assertIn("alpha_beta", set(tool_names))
            for name in tool_names:
                self.assertRegex(name, _COMPAT_NAME_RE)

            called_remote_names = set()
            for local_name in tool_names:
                result = await registry.call_tool(local_name, {"x": 1})
                self.assertIsInstance(result, dict)
                called_remote_names.add(result.get("remote_name"))

            self.assertEqual(called_remote_names, {"alpha.beta", "alpha-beta"})


if __name__ == "__main__":
    unittest.main()
