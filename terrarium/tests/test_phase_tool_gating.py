import unittest
from types import SimpleNamespace

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.blackboard import Megaboard
from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools
from terrarium.tools.environment import instantiate_environment_tools
from terrarium.tools.gates import tool_blocked


def _env_with_blocked(blocked_tools):
    return SimpleNamespace(env_config={"blocked_tools": blocked_tools})


def _tool_names(tools, phase):
    return {tool["function"]["name"] for tool in tools.get_tools(phase)}


class ToolBlockedTests(unittest.TestCase):
    def test_no_config_returns_false(self):
        self.assertFalse(tool_blocked(None, "schedule_patient", "execution"))
        self.assertFalse(tool_blocked(SimpleNamespace(), "schedule_patient", "execution"))
        self.assertFalse(
            tool_blocked(_env_with_blocked({}), "schedule_patient", "execution")
        )

    def test_phase_match(self):
        env = _env_with_blocked({"pre_coordination": ["schedule_patient"]})
        self.assertTrue(tool_blocked(env, "schedule_patient", "pre_coordination"))
        self.assertFalse(tool_blocked(env, "schedule_patient", "execution"))

    def test_wildcard_matches_env_tools(self):
        env = _env_with_blocked({"pre_coordination": ["*"]})
        self.assertTrue(tool_blocked(env, "schedule_patient", "pre_coordination"))
        self.assertTrue(tool_blocked(env, "get_job_queue", "pre_coordination"))

    def test_wildcard_never_matches_blackboard_tools(self):
        env = _env_with_blocked({"pre_coordination": ["*"]})
        self.assertFalse(tool_blocked(env, "post_message", "pre_coordination"))

    def test_explicit_post_message(self):
        env = _env_with_blocked({"pre_coordination": ["post_message"]})
        self.assertTrue(tool_blocked(env, "post_message", "pre_coordination"))
        self.assertFalse(tool_blocked(env, "schedule_patient", "pre_coordination"))

    def test_unknown_phase_returns_false(self):
        env = _env_with_blocked({"pre_coordination": ["*"]})
        self.assertFalse(tool_blocked(env, "schedule_patient", "execution"))
        self.assertFalse(tool_blocked(env, "schedule_patient", None))


class HospitalGetToolsGatingTests(unittest.TestCase):
    def _tools(self, env):
        return HospitalTools(blackboard_manager=None, environment=env)

    def test_default_config_unchanged(self):
        tools = self._tools(SimpleNamespace())
        self.assertEqual(
            _tool_names(tools, "execution"),
            {
                "find_available_slots",
                "get_job_queue",
                "transfer_resources",
                "broadcast_message",
                "schedule_patient",
            },
        )
        self.assertEqual(
            _tool_names(tools, "planning"),
            {
                "find_available_slots",
                "get_job_queue",
                "transfer_resources",
                "broadcast_message",
            },
        )

    def test_wildcard_returns_empty_list(self):
        tools = self._tools(_env_with_blocked({"pre_coordination": ["*"]}))
        self.assertEqual(tools.get_tools("pre_coordination"), [])

    def test_specific_tool_omitted_only_in_blocked_phase(self):
        tools = self._tools(_env_with_blocked({"execution": ["schedule_patient"]}))
        names = _tool_names(tools, "execution")
        self.assertNotIn("schedule_patient", names)
        self.assertEqual(len(names), 4)
        self.assertEqual(len(_tool_names(tools, "planning")), 4)

    def test_read_only_tools_kept_unless_explicitly_listed(self):
        tools = self._tools(_env_with_blocked({"pre_coordination": ["transfer_resources"]}))
        names = _tool_names(tools, "pre_coordination")
        self.assertNotIn("transfer_resources", names)
        self.assertIn("get_job_queue", names)
        self.assertIn("find_available_slots", names)


class EnvironmentToolGatingTests(unittest.TestCase):
    def _tools(self, env):
        return instantiate_environment_tools(
            "HospitalEnvironment", None, environment=env
        )

    def test_blocked_env_tool_returns_blocked_dict(self):
        tools = self._tools(_env_with_blocked({"pre_coordination": ["*"]}))
        result = tools.handle_tool_call(
            "schedule_patient",
            "Agent_1",
            {"patient_id": "P1", "step_index": 0, "start_time": 1, "rationale": "x"},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(
            result,
            {
                "blocked": True,
                "status": "blocked",
                "error": "Tool 'schedule_patient' is blocked in phase 'pre_coordination'.",
            },
        )

    def test_unblocked_phase_executes_normally(self):
        tools = self._tools(_env_with_blocked({"pre_coordination": ["*"]}))
        result = tools.handle_tool_call(
            "get_job_queue", "Agent_1", {}, phase="execution", iteration=0
        )
        self.assertEqual(result, {"result": "Refer to prompt context."})

    def test_no_config_executes_normally(self):
        tools = self._tools(SimpleNamespace())
        result = tools.handle_tool_call(
            "get_job_queue", "Agent_1", {}, phase="execution", iteration=0
        )
        self.assertEqual(result, {"result": "Refer to prompt context."})

    def test_without_environment_passes_through(self):
        tools = instantiate_environment_tools("HospitalEnvironment", None)
        self.assertEqual(type(tools).__name__, "HospitalTools")
        result = tools.handle_tool_call(
            "get_job_queue", "Agent_1", {}, phase="execution", iteration=0
        )
        self.assertEqual(result, {"error": "Environment not initialized for tools."})


class HospitalEnvironment:
    """Stub environment; the class name drives tools discovery by name."""

    def __init__(self, blocked_tools):
        self.env_config = {"blocked_tools": blocked_tools}


class ProtocolEnvironmentToolGatingTests(unittest.IsolatedAsyncioTestCase):
    def _protocol(self, blocked_tools):
        protocol = SequentialCommunicationProtocol.__new__(
            SequentialCommunicationProtocol
        )
        protocol.megaboard = Megaboard()
        protocol.environment = HospitalEnvironment(blocked_tools)
        protocol.environment_tools = None
        protocol._environment_tools_name = None
        return protocol

    async def test_blocked_env_tool_via_protocol_returns_blocked_dict(self):
        protocol = self._protocol({"pre_coordination": ["*"]})
        result = await protocol.environment_handle_tool_call(
            "schedule_patient",
            "Agent_1",
            {"patient_id": "P1", "step_index": 0, "start_time": 1, "rationale": "x"},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(
            result,
            {
                "blocked": True,
                "status": "blocked",
                "error": "Tool 'schedule_patient' is blocked in phase 'pre_coordination'.",
            },
        )

    async def test_unblocked_phase_env_tool_executes_via_protocol(self):
        protocol = self._protocol({"pre_coordination": ["*"]})
        result = await protocol.environment_handle_tool_call(
            "get_job_queue", "Agent_1", {}, phase="execution", iteration=0
        )
        self.assertEqual(result, {"result": "Refer to prompt context."})


class BlackboardToolGatingTests(unittest.IsolatedAsyncioTestCase):
    def _protocol(self, env):
        protocol = SequentialCommunicationProtocol.__new__(
            SequentialCommunicationProtocol
        )
        protocol.megaboard = Megaboard()
        protocol.megaboard.add_blackboard(["Agent_1", "Agent_2"])
        protocol.environment = env
        return protocol

    async def test_blocked_post_message_returns_blocked_dict(self):
        protocol = self._protocol(
            _env_with_blocked({"pre_coordination": ["*", "post_message"]})
        )
        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Agent_1",
            {"message": "hello", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertEqual(
            result,
            {
                "blocked": True,
                "status": "blocked",
                "error": "Tool 'post_message' is blocked in phase 'pre_coordination'.",
            },
        )
        self.assertEqual(protocol.megaboard.get(0, "Agent_1", limit=None), [])

    async def test_unblocked_phase_posts_normally(self):
        protocol = self._protocol(_env_with_blocked({"pre_coordination": ["post_message"]}))
        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Agent_1",
            {"message": "hello", "blackboard_id": 0},
            phase="planning",
            iteration=0,
        )
        self.assertIn("event_id", result)

    async def test_wildcard_alone_does_not_block_post_message(self):
        protocol = self._protocol(_env_with_blocked({"pre_coordination": ["*"]}))
        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Agent_1",
            {"message": "hello", "blackboard_id": 0},
            phase="pre_coordination",
            iteration=0,
        )
        self.assertIn("event_id", result)

    async def test_no_config_posts_normally(self):
        protocol = self._protocol(SimpleNamespace())
        result = await protocol.blackboard_handle_tool_call(
            "post_message",
            "Agent_1",
            {"message": "hello", "blackboard_id": 0},
            phase="planning",
            iteration=0,
        )
        self.assertIn("event_id", result)


if __name__ == "__main__":
    unittest.main()
