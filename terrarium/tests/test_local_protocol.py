import unittest

try:
    from experiments.common.local_protocol import LocalCommunicationProtocol
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("experiments"):
        raise unittest.SkipTest("Optional experiments package is not available") from exc
    raise


class _DummyJiraEnvironment:
    tools_environment_name = "JiraTicketEnvironment"

    def __init__(self):
        self.tasks = {"ISSUE-1": {"summary": "Fix the failing test"}}
        self.assignment = {}
        self.agent_names = ["Alex"]

    def joint_reward(self, assignment):
        return float(sum(1 for task_id in assignment.values() if task_id is not None))


class LocalCommunicationProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_jira_tool_call_uses_active_environment_instance(self):
        protocol = LocalCommunicationProtocol(config={})
        protocol.megaboard.log_action_to_blackboards = lambda *args, **kwargs: None

        first_env = _DummyJiraEnvironment()
        protocol.environment = first_env

        first_result = await protocol.environment_handle_tool_call(
            "assign_task",
            "Alex",
            {"task_id": "ISSUE-1"},
            phase="execution",
            iteration=1,
        )

        self.assertEqual(first_result["status"], "success")
        self.assertEqual(first_env.assignment, {"Alex": "ISSUE-1"})
        self.assertIs(protocol._env_tools.environment, first_env)

        second_env = _DummyJiraEnvironment()
        protocol.environment = second_env

        second_result = await protocol.environment_handle_tool_call(
            "assign_task",
            "Alex",
            {"task_id": "ISSUE-1"},
            phase="execution",
            iteration=1,
        )

        self.assertEqual(second_result["status"], "success")
        self.assertEqual(second_env.assignment, {"Alex": "ISSUE-1"})
        self.assertIs(protocol._env_tools.environment, second_env)


if __name__ == "__main__":
    unittest.main()
