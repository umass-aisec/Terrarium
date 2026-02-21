import json
import unittest
from typing import Any, Dict, List, Set

from terrarium.agents.base import BaseAgent


class _DummyToolsetDiscovery:
    def get_tools_for_environment(self, environment_name: str, phase: str) -> List[Dict[str, Any]]:
        return []

    def get_env_tool_names(self, environment_name: str) -> Set[str]:
        return set()

    def get_blackboard_tool_names(self) -> Set[str]:
        return set()

    def get_tools_for_blackboard(self, phase: str) -> List[Dict[str, Any]]:
        return []


class _DummyProtocol:
    async def blackboard_handle_tool_call(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("External MCP tool should not route through blackboard handler.")

    async def environment_handle_tool_call(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("External MCP tool should not route through environment handler.")


class _DummyClient:
    def __init__(self):
        self.generate_params_history: List[Dict[str, Any]] = []
        self.external_calls: List[Dict[str, Any]] = []
        self.tool_result: Dict[str, Any] | None = None
        self._processed_once = False

    def init_context(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_response(
        self, input: List[Any], params: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str]:
        self.generate_params_history.append(params)
        return {"model": params.get("model", "dummy-model"), "input": input}, "ok"

    async def process_tool_calls(self, response, context, execute_tool_callback):
        if self._processed_once:
            return 0, context, []

        self._processed_once = True
        args = {"payload": "hello"}
        result = await execute_tool_callback("ext_ping", args)
        self.tool_result = result
        return 1, context, [f"ext_ping -- {json.dumps(args)}"]

    def get_usage(self, response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
        return current_usage

    async def get_external_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "ext_ping",
                    "description": "Ping an external MCP server.",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "string"}},
                        "required": ["payload"],
                    },
                },
            }
        ]

    async def get_external_tool_names(self) -> Set[str]:
        return {"ext_ping"}

    async def execute_external_tool(self, tool_name: str, arguments: Dict[str, Any]):
        self.external_calls.append({"tool_name": tool_name, "arguments": arguments})
        return {"status": "success", "result": {"echo": arguments.get("payload")}}


class BaseAgentExternalMCPRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_tools_are_exposed_and_routed(self):
        client = _DummyClient()
        agent = BaseAgent(
            client=client,  # type: ignore[arg-type]
            name="agent_1",
            model_name="dummy-model",
            max_conversation_steps=1,
            environment_name="DummyEnvironment",
        )
        agent.toolset_discovery = _DummyToolsetDiscovery()  # type: ignore[assignment]
        agent.communication_protocol = _DummyProtocol()
        agent.set_meta_context(agent_name="agent_1", phase="execution", iteration=1)

        result = await agent._multi_step_response_generation(
            system_prompt="system",
            user_prompt="user",
        )

        self.assertTrue(client.generate_params_history, "Expected at least one model call.")
        first_tools = client.generate_params_history[0].get("tools", [])
        tool_names = {
            tool.get("function", {}).get("name")
            for tool in first_tools
            if isinstance(tool, dict)
        }
        self.assertIn("ext_ping", tool_names)

        self.assertEqual(len(client.external_calls), 1)
        self.assertEqual(client.external_calls[0]["tool_name"], "ext_ping")
        self.assertEqual(client.external_calls[0]["arguments"], {"payload": "hello"})
        self.assertEqual(
            client.tool_result,
            {"status": "success", "result": {"echo": "hello"}},
        )
        self.assertTrue(result["has_tool_calls"])


if __name__ == "__main__":
    unittest.main()
