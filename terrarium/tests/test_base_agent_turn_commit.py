import unittest
from typing import Any, Dict, List, Set, Tuple

from terrarium.agents.base import BaseAgent

ENV_TOOLS = {"get_job_queue", "schedule_patient", "schedule_full", "legacy_assign"}
BLACKBOARD_TOOLS = {"post_message"}

# Canned environment/blackboard results, shaped like the shipped DCOP tools return them.
RESULTS: Dict[str, Dict[str, Any]] = {
    "get_job_queue": {"result": "Refer to prompt context."},
    "schedule_patient": {"status": "success", "result": "Scheduling P1.", "action_committed": True},
    "schedule_full": {"status": "failed", "reason": "No capacity."},
    "legacy_assign": {"status": "success", "result": {"state_updates": {"assignment": {"a": "t"}}}},
    "post_message": {"status": "success", "action_committed": True},
}


def _tool_def(name: str) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


class _ToolsetDiscovery:
    def get_tools_for_environment(self, environment_name: str, phase: str) -> List[Dict[str, Any]]:
        return [_tool_def(n) for n in sorted(ENV_TOOLS)]

    def get_env_tool_names(self, environment_name: str) -> Set[str]:
        return set(ENV_TOOLS)

    # Optional args match newer ToolsetDiscovery signatures (environment_name, retrieval_enabled).
    def get_blackboard_tool_names(self, *args: Any, **kwargs: Any) -> Set[str]:
        return set(BLACKBOARD_TOOLS)

    def get_tools_for_blackboard(self, phase: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [_tool_def(n) for n in sorted(BLACKBOARD_TOOLS)]


class _Protocol:
    def __init__(self):
        self.calls: List[str] = []

    async def environment_handle_tool_call(self, tool_name, agent_name, arguments, **kwargs):
        self.calls.append(tool_name)
        return dict(RESULTS[tool_name])

    async def blackboard_handle_tool_call(self, tool_name, agent_name, arguments, **kwargs):
        self.calls.append(tool_name)
        return dict(RESULTS[tool_name])


class _ScriptedClient:
    """Issues a fixed list of tool calls per conversation step; records what the model sees."""

    def __init__(self, steps: List[List[str]]):
        self.steps = list(steps)
        self.seen_results: List[Tuple[str, Dict[str, Any]]] = []

    def init_context(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def generate_response(self, input: List[Any], params: Dict[str, Any]):
        return {"model": "dummy-model"}, "ok"

    async def process_tool_calls(self, response, context, execute_tool_callback):
        if not self.steps:
            return 0, context, []
        names = self.steps.pop(0)
        for name in names:
            self.seen_results.append((name, await execute_tool_callback(name, {})))
        return len(names), context, names

    def get_usage(self, response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
        return current_usage

    async def get_external_tools(self) -> List[Dict[str, Any]]:
        return []

    async def get_external_tool_names(self) -> Set[str]:
        return set()


class BaseAgentTurnCommitTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, steps: List[List[str]], max_steps: int = 3):
        client, protocol = _ScriptedClient(steps), _Protocol()
        agent = BaseAgent(
            client=client,  # type: ignore[arg-type]
            name="agent_1",
            model_name="dummy-model",
            max_conversation_steps=max_steps,
            environment_name="DummyEnvironment",
        )
        agent.toolset_discovery = _ToolsetDiscovery()  # type: ignore[assignment]
        agent.communication_protocol = protocol
        agent.set_meta_context(agent_name="agent_1", phase="planning", iteration=1)
        await agent._multi_step_response_generation(system_prompt="system", user_prompt="user")
        return protocol.calls, client.seen_results

    async def test_read_only_tool_does_not_end_turn(self):
        calls, _ = await self._run([["get_job_queue"], ["schedule_patient"], ["get_job_queue"]])
        self.assertEqual(calls, ["get_job_queue", "schedule_patient"])

    async def test_committed_action_ends_turn_and_flag_is_hidden_from_model(self):
        calls, seen = await self._run([["schedule_patient"], ["get_job_queue"]])
        self.assertEqual(calls, ["schedule_patient"])
        self.assertNotIn("action_committed", seen[0][1])
        self.assertEqual(seen[0][1]["status"], "success")

    async def test_failed_action_does_not_end_turn(self):
        calls, _ = await self._run([["schedule_full"], ["schedule_patient"], ["get_job_queue"]])
        self.assertEqual(calls, ["schedule_full", "schedule_patient"])

    async def test_legacy_state_updates_result_ends_turn(self):
        calls, _ = await self._run([["legacy_assign"], ["get_job_queue"]])
        self.assertEqual(calls, ["legacy_assign"])

    async def test_blackboard_tool_never_ends_turn(self):
        calls, _ = await self._run([["post_message"], ["schedule_patient"], ["get_job_queue"]])
        self.assertEqual(calls, ["post_message", "schedule_patient"])

    async def test_step_limit_still_applies(self):
        calls, _ = await self._run([["get_job_queue"]] * 5, max_steps=2)
        self.assertEqual(calls, ["get_job_queue", "get_job_queue"])


class DcopToolsCommitFlagTests(unittest.TestCase):
    """The shipped DCOP tools flag successful actions, and never read-only or failed calls."""

    def test_hospital_read_tools_are_not_committed(self):
        from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools

        tools = HospitalTools(blackboard_manager=None, environment=object())
        for name in ("get_job_queue", "find_available_slots"):
            self.assertNotIn("action_committed", tools.handle_tool_call(name, "a", {}))

    def test_hospital_broadcast_is_committed(self):
        from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools

        tools = HospitalTools(blackboard_manager=None, environment=object())
        result = tools.handle_tool_call("broadcast_message", "a", {"message": "hi"})
        self.assertEqual(result["status"], "success")
        self.assertIs(result["action_committed"], True)

    def test_jira_success_committed_and_failure_not(self):
        from terrarium.environments.dcops.jira_ticket.jira_ticket_tools import JiraTicketTools

        class _Env:
            tasks = {"T1": {"id": "T1"}}
            assignment: Dict[str, Any] = {}
            agent_names = ["a"]

            def joint_reward(self, assignment):
                return 0.0

        tools = JiraTicketTools(blackboard_manager=None, environment=_Env())
        ok = tools.handle_tool_call("assign_task", "a", {"task_id": "T1"})
        again = tools.handle_tool_call("assign_task", "a", {"task_id": "T1"})
        self.assertIs(ok.get("action_committed"), True)
        self.assertEqual(again["status"], "failed")
        self.assertNotIn("action_committed", again)



class SeatToolsCommitFlagTests(unittest.TestCase):
    """IsThisSeatTaken has only action tools: successes commit, failures and retries do not."""

    def _tools(self, status: str):
        from terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_tools import (
            IsThisSeatTakenTools,
        )

        class _Env:
            def _apply_agent_action(self, agent_name, action_name, arguments):
                return {"status": status, "agent": agent_name}

        return IsThisSeatTakenTools(blackboard_manager=None, environment=_Env())

    def test_successful_physical_action_is_committed(self):
        result = self._tools("success").handle_tool_call("stand", "a", {}, phase="execution")
        self.assertIs(result.get("action_committed"), True)

    def test_failed_or_retry_action_is_not_committed(self):
        for status in ("failed", "retry"):
            result = self._tools(status).handle_tool_call("stand", "a", {}, phase="execution")
            self.assertNotIn("action_committed", result)


if __name__ == "__main__":
    unittest.main()
