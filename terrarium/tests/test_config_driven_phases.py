"""Tests for the config-driven phase sequence (simulation.phases).

No LLM access: resolve_phases and the toolset fallbacks are pure functions,
and SequentialCommunicationProtocol.agent_turn is driven with a stub agent
that records the kwargs it is called with.
"""

import unittest
from typing import Any, Dict, List, Optional

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.blackboard import Megaboard
from terrarium.tools.discovery import ToolsetDiscovery
from terrarium.tools.prompts import get_phase_tool_instructions
from terrarium.utils import resolve_phases, validate_config


class ResolvePhasesTests(unittest.TestCase):
    def test_default_derives_from_max_planning_rounds(self):
        self.assertEqual(
            resolve_phases({"max_planning_rounds": 3}),
            [
                {"name": "planning", "rounds": 3},
                {"name": "execution", "rounds": 1},
            ],
        )

    def test_default_when_max_planning_rounds_missing_or_falsy(self):
        expected = [
            {"name": "planning", "rounds": 1},
            {"name": "execution", "rounds": 1},
        ]
        for simulation_config in (
            None,
            {},
            {"max_planning_rounds": None},
            {"max_planning_rounds": 0},
        ):
            with self.subTest(simulation_config=simulation_config):
                self.assertEqual(resolve_phases(simulation_config), expected)

    def test_explicit_phases_pass_through_in_order(self):
        phases = [
            {"name": "planning", "rounds": 2},
            {"name": "critique", "rounds": 1},
            {"name": "execution", "rounds": 1},
        ]
        self.assertEqual(resolve_phases({"phases": phases}), phases)

    def test_explicit_phases_normalize_name_and_rounds(self):
        self.assertEqual(
            resolve_phases({"phases": [{"name": " review ", "rounds": "2"}]}),
            [{"name": "review", "rounds": 2}],
        )

    def test_invalid_phase_entries_raise_value_error(self):
        bad_configs = [
            {"phases": "planning"},
            {"phases": {}},
            {"phases": []},
            {"phases": ["planning"]},
            {"phases": [{"rounds": 1}]},
            {"phases": [{"name": "", "rounds": 1}]},
            {"phases": [{"name": "   ", "rounds": 1}]},
            {"phases": [{"name": 7, "rounds": 1}]},
            {"phases": [{"name": "x"}]},
            {"phases": [{"name": "x", "rounds": None}]},
            {"phases": [{"name": "x", "rounds": 0}]},
            {"phases": [{"name": "x", "rounds": -2}]},
            {"phases": [{"name": "x", "rounds": 1.5}]},
            {"phases": [{"name": "x", "rounds": True}]},
            {"phases": [{"name": "x", "rounds": "many"}]},
        ]
        for simulation_config in bad_configs:
            with self.subTest(simulation_config=simulation_config):
                with self.assertRaises(ValueError):
                    resolve_phases(simulation_config)

    def test_non_dict_simulation_config_raises(self):
        with self.assertRaises(ValueError):
            resolve_phases(["planning"])  # type: ignore[arg-type]

    def test_phases_make_max_planning_rounds_optional_in_validate_config(self):
        config = {
            "simulation": {
                "max_iterations": 1,
                "seed": 1,
                "phases": [
                    {"name": "planning", "rounds": 1},
                    {"name": "execution", "rounds": 1},
                ],
            },
            "environment": {"name": "MeetingSchedulingEnvironment"},
            "communication_network": {"topology": "complete", "num_agents": 2},
            "llm": {"provider": "openai"},
        }
        validate_config(config)  # must not raise
        with self.assertRaises(ValueError):
            validate_config(
                {**config, "simulation": {"max_iterations": 1, "seed": 1, "phases": []}}
            )


class BlackboardToolsetFallbackTests(unittest.TestCase):
    def setUp(self):
        self.discovery = ToolsetDiscovery()

    def _names(self, phase: str, **kwargs: Any) -> List[str]:
        return [
            tool["function"]["name"]
            for tool in self.discovery.get_tools_for_blackboard(phase, **kwargs)
        ]

    def test_planning_gets_blackboard_tools(self):
        self.assertIn("post_message", self._names("planning"))

    def test_execution_gets_no_blackboard_tools(self):
        self.assertEqual(self.discovery.get_tools_for_blackboard("execution"), [])

    def test_unknown_phase_gets_the_planning_toolset(self):
        unknown = self.discovery.get_tools_for_blackboard("critique")
        planning = self.discovery.get_tools_for_blackboard("planning")
        self.assertEqual(unknown, planning)
        self.assertIn("post_message", [t["function"]["name"] for t in unknown])

    def test_unknown_phase_respects_retrieval_and_private_channel_flags(self):
        names = self._names("critique", retrieval_enabled=True)
        self.assertIn("post_message", names)
        self.assertIn("recall", names)
        channel_env = self._names(
            "critique", environment_name="IsThisSeatTakenEnvironment"
        )
        self.assertIn("create_channel", channel_env)


class PhaseToolInstructionFallbackTests(unittest.TestCase):
    INSTRUCTION_MAP = {"planning": "PLAN-BLOCK", "execution": "EXEC-BLOCK"}

    def test_known_phases_are_case_insensitive(self):
        self.assertEqual(
            get_phase_tool_instructions(self.INSTRUCTION_MAP, "planning"), "PLAN-BLOCK"
        )
        self.assertEqual(
            get_phase_tool_instructions(self.INSTRUCTION_MAP, "Execution"), "EXEC-BLOCK"
        )

    def test_unknown_phase_falls_back_to_execution_block(self):
        for phase in ("critique", "review", None, ""):
            with self.subTest(phase=phase):
                self.assertEqual(
                    get_phase_tool_instructions(self.INSTRUCTION_MAP, phase),
                    "EXEC-BLOCK",
                )

    def test_missing_execution_key_returns_empty_string(self):
        self.assertEqual(
            get_phase_tool_instructions({"planning": "PLAN-BLOCK"}, "critique"), ""
        )

    def test_empty_or_missing_map_returns_empty_string(self):
        self.assertEqual(get_phase_tool_instructions({}, "planning"), "")
        self.assertEqual(get_phase_tool_instructions(None, "planning"), "")


class _RecordingAgent:
    """Stub agent: records the kwargs generate_response is called with."""

    def __init__(self):
        self.client = object()
        self.model_name = "stub-model"
        self.calls: List[Dict[str, Any]] = []

    async def generate_response(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"response": "ok"}


class _StubEnvironment:
    def __init__(self):
        self.prompts = object()


def _make_protocol() -> SequentialCommunicationProtocol:
    """Build a protocol without running __init__ (avoids logger filesystem setup)."""
    protocol = SequentialCommunicationProtocol.__new__(SequentialCommunicationProtocol)
    protocol.config = {"simulation": {}, "llm": {}}
    protocol.simulation_config = protocol.config["simulation"]
    protocol.tool_logger = None
    protocol.run_timestamp = None
    protocol.blackboard_logger = None
    protocol.compaction_enabled = False
    protocol.compaction_logger = None
    protocol._compaction_caches = {}
    protocol.megaboard = Megaboard()
    protocol.environment = None
    protocol.environment_tools = None
    protocol._environment_tools_name = None
    return protocol


class AgentTurnTests(unittest.IsolatedAsyncioTestCase):
    def _bind(self, protocol: SequentialCommunicationProtocol, environment) -> None:
        protocol.environment = environment
        # Pretend environment tools are already initialized for this env class.
        protocol.environment_tools = object()
        protocol._environment_tools_name = type(environment).__name__

    def _spy(self, protocol: SequentialCommunicationProtocol):
        prefetch_calls: List[Dict[str, Any]] = []
        log_calls: List[Dict[str, Any]] = []
        original_prefetch = protocol._prefetch_blackboard_events
        original_log = protocol._log_blackboard_states

        async def prefetch_spy(agent_name, **kwargs):
            prefetch_calls.append({"agent_name": agent_name, **kwargs})
            return await original_prefetch(agent_name, **kwargs)

        def log_spy(**kwargs):
            log_calls.append(kwargs)
            original_log(**kwargs)

        protocol._prefetch_blackboard_events = prefetch_spy
        protocol._log_blackboard_states = log_spy
        return prefetch_calls, log_calls

    async def test_agent_turn_forwards_phase_and_round_num(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        prefetch_calls, log_calls = self._spy(protocol)
        agent = _RecordingAgent()

        await protocol.agent_turn(
            agent, "agent_0", {"ctx": 1}, environment, 3, phase="review", round_num=2
        )

        self.assertEqual(len(agent.calls), 1)
        call = agent.calls[0]
        self.assertEqual(call["agent_name"], "agent_0")
        self.assertEqual(call["agent_context"], {"ctx": 1})
        self.assertEqual(call["phase"], "review")
        self.assertEqual(call["round_num"], 2)
        self.assertEqual(call["iteration"], 3)
        self.assertIs(call["communication_protocol"], protocol)
        self.assertIs(call["prompts"], environment.prompts)
        # phase is passed through to the blackboard prefetch and state logging
        self.assertEqual(prefetch_calls[0]["phase"], "review")
        self.assertEqual(prefetch_calls[0]["iteration"], 3)
        self.assertEqual(log_calls[0]["phase"], "review")
        self.assertEqual(log_calls[0]["agent_name"], "agent_0")
        self.assertEqual(log_calls[0]["planning_round"], 2)

    async def test_agent_turn_without_round_num_keeps_agent_default(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        prefetch_calls, log_calls = self._spy(protocol)
        agent = _RecordingAgent()

        await protocol.agent_turn(agent, "agent_0", {}, environment, 1, phase="execution")

        call = agent.calls[0]
        self.assertEqual(call["phase"], "execution")
        # round_num is not forwarded, so generate_response keeps its default (0)
        self.assertNotIn("round_num", call)
        self.assertEqual(prefetch_calls[0]["phase"], "execution")
        self.assertIsNone(log_calls[0].get("planning_round"))

    async def test_legacy_wrappers_delegate_to_agent_turn_with_fixed_phase(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        agent = _RecordingAgent()
        seen: List[Any] = []

        async def fake_turn(*args, **kwargs):
            seen.append((args, kwargs))

        protocol.agent_turn = fake_turn  # type: ignore[method-assign]

        await protocol.agent_planning_turn(agent, "agent_0", {"ctx": 2}, environment, 5, 3)
        await protocol.agent_execution_turn(agent, "agent_0", {"ctx": 2}, environment, 5)

        self.assertEqual(len(seen), 2)
        planning_args, planning_kwargs = seen[0]
        self.assertEqual(
            planning_args, (agent, "agent_0", {"ctx": 2}, environment, 5)
        )
        self.assertEqual(planning_kwargs, {"phase": "planning", "round_num": 3})
        execution_args, execution_kwargs = seen[1]
        self.assertEqual(
            execution_args, (agent, "agent_0", {"ctx": 2}, environment, 5)
        )
        self.assertEqual(execution_kwargs, {"phase": "execution", "round_num": None})

    async def test_planning_and_execution_wrappers_end_to_end(self):
        protocol = _make_protocol()
        environment = _StubEnvironment()
        self._bind(protocol, environment)
        agent = _RecordingAgent()

        await protocol.agent_planning_turn(agent, "agent_0", {}, environment, 2, 4)
        await protocol.agent_execution_turn(agent, "agent_0", {}, environment, 2)

        planning_call, execution_call = agent.calls
        self.assertEqual(planning_call["phase"], "planning")
        self.assertEqual(planning_call["round_num"], 4)
        self.assertEqual(planning_call["iteration"], 2)
        self.assertEqual(execution_call["phase"], "execution")
        self.assertNotIn("round_num", execution_call)
        self.assertEqual(execution_call["iteration"], 2)


if __name__ == "__main__":
    unittest.main()
