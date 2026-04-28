import os
import unittest
from unittest.mock import patch

from terrarium.tools.environment import (
    get_environment_tools_class,
    instantiate_environment_tools,
)
from terrarium.llm.vllm.runtime import VLLMProviderRuntime
from terrarium.utils import get_generation_params, get_model_name, _resolve_optional_value


class TerrariumSmokeTests(unittest.TestCase):
    def test_public_namespaces_expose_expected_symbols(self):
        import terrarium.environments as environments
        import terrarium.llm as llm
        import terrarium.llm.clients as clients

        self.assertIn("JiraTicketEnvironment", environments.__all__)
        self.assertIn("HospitalEnvironment", environments.__all__)
        self.assertIn("AbstractClient", llm.__all__)
        self.assertIn("VLLMProviderRuntime", llm.__all__)
        self.assertIn("GrokClient", clients.__all__)
        self.assertNotIn("AzureOpenAIClient", clients.__all__)

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

    def test_provider_alias_foundary_maps_to_foundry(self):
        cfg = {"foundry": {"model": "gpt-4.1-mini"}}
        self.assertEqual(get_model_name("foundary", cfg), "gpt-4.1-mini")

    def test_provider_alias_xai_maps_to_grok(self):
        cfg = {"grok": {"model": "grok-4-1-fast-reasoning"}}
        self.assertEqual(get_model_name("xai", cfg), "grok-4-1-fast-reasoning")

    def test_xai_provider_block_works_for_generation_params(self):
        cfg = {
            "provider": "xai",
            "xai": {"params": {"max_tokens": 1500, "temperature": 0.7}},
        }
        self.assertEqual(
            get_generation_params(cfg),
            {"max_tokens": 1500, "temperature": 0.7},
        )

    def test_foundry_endpoint_can_come_from_named_env_var(self):
        with patch.dict(
            "os.environ",
            {
                "AI_FOUNDRY_RBR_EAST_US_2_PROJECT_ENDPOINT": (
                    "https://rbr-east-us-2-resource.services.ai.azure.com/api/projects/rbr-east-us-2"
                )
            },
            clear=False,
        ):
            self.assertEqual(
                _resolve_optional_value(
                    {"project_endpoint_env_var": "AI_FOUNDRY_RBR_EAST_US_2_PROJECT_ENDPOINT"},
                    direct_keys=["project_endpoint", "endpoint", "base_url"],
                    env_var_keys=[
                        "project_endpoint_env_var",
                        "endpoint_env_var",
                        "base_url_env_var",
                    ],
                ),
                "https://rbr-east-us-2-resource.services.ai.azure.com/api/projects/rbr-east-us-2",
            )

    def test_named_env_var_resolver_loads_dotenv_before_lookup(self):
        import terrarium.utils as utils

        def fake_load_dotenv(*, override=False):
            os.environ.setdefault(
                "AI_FOUNDRY_GROK_OPENAI_ENDPOINT",
                "https://collusion.services.ai.azure.com/openai/v1/",
            )
            return True

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(utils, "load_dotenv", side_effect=fake_load_dotenv),
        ):
            self.assertEqual(
                _resolve_optional_value(
                    {"project_endpoint_env_var": "AI_FOUNDRY_GROK_OPENAI_ENDPOINT"},
                    direct_keys=["project_endpoint", "endpoint", "base_url"],
                    env_var_keys=[
                        "project_endpoint_env_var",
                        "endpoint_env_var",
                        "base_url_env_var",
                    ],
                ),
                "https://collusion.services.ai.azure.com/openai/v1/",
            )


if __name__ == "__main__":
    unittest.main()
