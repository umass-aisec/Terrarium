import unittest

from terrarium.environment_tools import (
    get_environment_tools_class,
    instantiate_environment_tools,
)
from terrarium.llm.vllm.runtime import VLLMProviderRuntime


class TerrariumSmokeTests(unittest.TestCase):
    def test_public_namespaces_expose_expected_symbols(self):
        import terrarium.environments as environments
        import terrarium.llm as llm

        self.assertIn("JiraTicketEnvironment", environments.__all__)
        self.assertIn("HospitalEnvironment", environments.__all__)
        self.assertIn("AbstractClient", llm.__all__)
        self.assertIn("VLLMProviderRuntime", llm.__all__)

    def test_environment_tool_discovery_uses_terrarium_namespace(self):
        tools_cls = get_environment_tools_class("JiraTicketEnvironment")
        self.assertEqual(tools_cls.__name__, "JiraTicketTools")

        tools = instantiate_environment_tools(
            "JiraTicketEnvironment", blackboard_manager=None
        )
        self.assertEqual(type(tools).__name__, "JiraTicketTools")
        self.assertIn("assign_task", tools.get_tool_names())

    def test_vllm_runtime_can_initialize_from_minimal_config(self):
        runtime = VLLMProviderRuntime(
            {
                "auto_start_server": False,
                "persistent_server": False,
                "models": [
                    {
                        "checkpoint": "Qwen/Qwen2.5-7B-Instruct",
                        "host": "127.0.0.1",
                        "port": 8020,
                    }
                ],
            }
        )
        self.assertEqual(runtime.describe_default_model(), "Qwen/Qwen2.5-7B-Instruct")
        self.assertIn("logs/vllm/", runtime.describe_log_path())
        runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
