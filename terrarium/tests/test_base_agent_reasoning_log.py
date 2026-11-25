import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from terrarium.agents.base import BaseAgent


class _ToolsetDiscovery:
    def get_tools_for_environment(self, environment_name: str, phase: str) -> List[Dict[str, Any]]:
        return []

    def get_tools_for_blackboard(self, phase: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return []


class _TrajectoryLogger:
    def __init__(self):
        self.entries: List[Tuple[str, int, str, Dict[str, Any], Optional[int]]] = []

    def log_trajectory(self, agent_name, iteration, phase, trajectory_dict, round_num=None):
        self.entries.append((agent_name, iteration, phase, trajectory_dict, round_num))


class _ReasoningClient:
    """Returns a fixed (response, response_str) pair and executes no tools."""

    def __init__(self, response: Any, response_str: str):
        self._response = response
        self._response_str = response_str

    def init_context(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_response(self, input: List[Any], params: Dict[str, Any]):
        return self._response, self._response_str

    async def process_tool_calls(self, response, context, execute_tool_callback):
        return 0, context, []

    def get_usage(self, response: Any, current_usage: Dict[str, int]) -> Dict[str, int]:
        return current_usage

    async def get_external_tools(self) -> List[Dict[str, Any]]:
        return []


def _reasoning_response(summary_parts=(), content_parts=()) -> Any:
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=list(summary_parts),
                content=list(content_parts),
            )
        ],
        output_text="",
        status="completed",
    )


class ExtractReasoningTextTests(unittest.TestCase):
    def test_openai_style_summary_parts(self):
        response = _reasoning_response(
            summary_parts=[SimpleNamespace(type="summary_text", text="step one. "),
                           SimpleNamespace(type="summary_text", text="step two.")]
        )
        self.assertEqual(BaseAgent._extract_reasoning_text(response), "step one. step two.")

    def test_gateway_style_content_parts(self):
        # OpenRouter-style: text lives in content parts of type reasoning_text
        response = _reasoning_response(
            content_parts=[SimpleNamespace(type="reasoning_text", text="thinking hard")]
        )
        self.assertEqual(BaseAgent._extract_reasoning_text(response), "thinking hard")

    def test_dict_shaped_items_and_parts(self):
        response = {"output": [{"type": "reasoning",
                                "summary": [{"type": "summary_text", "text": "dict summary"}],
                                "content": [{"type": "reasoning_text", "text": "dict content"}]}]}
        self.assertEqual(
            BaseAgent._extract_reasoning_text(response), "dict summary\ndict content"
        )

    def test_multiple_reasoning_items_are_joined(self):
        response = SimpleNamespace(output=[
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="first")], content=[]),
            SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="hi")]),
            SimpleNamespace(type="reasoning", summary=[], content=[SimpleNamespace(text="second")]),
        ])
        self.assertEqual(BaseAgent._extract_reasoning_text(response), "first\nsecond")

    def test_no_reasoning_items_returns_empty(self):
        response = SimpleNamespace(output=[SimpleNamespace(type="message")], output_text="hi")
        self.assertEqual(BaseAgent._extract_reasoning_text(response), "")
        self.assertEqual(BaseAgent._extract_reasoning_text(SimpleNamespace(output=[])), "")


class ReasoningTrajectoryLogTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, response: Any, response_str: str):
        trajectory_logger = _TrajectoryLogger()
        agent = BaseAgent(
            client=_ReasoningClient(response, response_str),  # type: ignore[arg-type]
            name="agent_1",
            model_name="dummy-model",
            max_conversation_steps=3,
            trajectory_logger=trajectory_logger,
            environment_name="DummyEnvironment",
        )
        agent.toolset_discovery = _ToolsetDiscovery()  # type: ignore[assignment]
        agent.set_meta_context(agent_name="agent_1", phase="planning", iteration=1)
        await agent._multi_step_response_generation(system_prompt="system", user_prompt="user")
        return trajectory_logger.entries

    async def test_reasoning_item_text_lands_in_trajectory_reasoning_field(self):
        response = _reasoning_response(
            content_parts=[SimpleNamespace(type="reasoning_text", text="surplus everywhere")]
        )
        entries = await self._run(response, response_str="")
        step = entries[0][3]["step_1"]
        self.assertEqual(step["reasoning"], "surplus everywhere")
        self.assertIn("reasoning", step["response_output_types"])

    async def test_message_text_is_fallback_when_no_reasoning_items(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(type="message")], output_text="just a message", status="completed"
        )
        entries = await self._run(response, response_str="just a message")
        step = entries[0][3]["step_1"]
        self.assertEqual(step["reasoning"], "just a message")

    async def test_reasoning_preferred_over_output_text_when_both_present(self):
        response = _reasoning_response(
            summary_parts=[SimpleNamespace(type="summary_text", text="the plan")]
        )
        response.output_text = "the message"
        entries = await self._run(response, response_str="the message")
        step = entries[0][3]["step_1"]
        self.assertEqual(step["reasoning"], "the plan")
        self.assertEqual(step["response_output_text"], "the message")


if __name__ == "__main__":
    unittest.main()
