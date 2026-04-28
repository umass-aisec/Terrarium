import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _import_module_with_mocks(module_name: str):
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = SimpleNamespace(create=lambda **_kwargs: None)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: None)
            )
            _FakeOpenAI.instances.append(self)

    class _APIConnectionError(Exception):
        pass

    class _APIStatusError(Exception):
        pass

    class _APITimeoutError(Exception):
        pass

    class _RateLimitError(Exception):
        pass

    class _FakeMessage(dict):
        pass

    class _FakeFunctionCallOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.APIConnectionError = _APIConnectionError
    fake_openai.APIStatusError = _APIStatusError
    fake_openai.APITimeoutError = _APITimeoutError
    fake_openai.RateLimitError = _RateLimitError

    fake_oai_types = types.ModuleType("openai.types")
    fake_oai_types_responses = types.ModuleType("openai.types.responses")
    fake_oai_types_response_input = types.ModuleType(
        "openai.types.responses.response_input_item_param"
    )
    fake_oai_types_response_input.Message = _FakeMessage
    fake_oai_types_response_input.FunctionCallOutput = _FakeFunctionCallOutput

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None

    modules = {
        "openai": fake_openai,
        "openai.types": fake_oai_types,
        "openai.types.responses": fake_oai_types_responses,
        "openai.types.responses.response_input_item_param": fake_oai_types_response_input,
        "dotenv": fake_dotenv,
    }

    for stale_module in (
        "terrarium.llm.clients.openai_client",
        "terrarium.llm.clients.azure_common",
        "terrarium.llm.clients.foundry_client",
    ):
        sys.modules.pop(stale_module, None)

    with patch.dict(sys.modules, modules):
        module = importlib.import_module(module_name)

    return module, _FakeOpenAI


class FoundryClientTests(unittest.TestCase):
    @staticmethod
    def _foundry_env():
        return patch.dict(
            os.environ,
            {
                "AI_FOUNDRY_API_KEY": "foundry-key",
                "AI_FOUNDRY_AUTH_MODE": "api_key",
            },
            clear=False,
        )

    def test_entra_is_default_when_api_key_missing(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        scope_calls = []

        with (
            patch.object(
                module,
                "build_entra_token_provider",
                side_effect=lambda scope: scope_calls.append(scope)
                or (lambda: "fake-token"),
            ),
            patch.dict(
                os.environ,
                {},
                clear=True,
            ),
        ):
            FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project"
            )

        self.assertEqual(scope_calls, ["https://ai.azure.com/.default"])
        kwargs = fake_openai_cls.instances[0].kwargs
        self.assertTrue(callable(kwargs["api_key"]))
        self.assertEqual(
            kwargs["base_url"],
            "https://example.services.ai.azure.com/api/projects/demo-project/openai/v1/",
        )

    def test_api_key_mode_uses_foundry_key_env(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        with self._foundry_env():
            FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project/openai/v1/"
            )

        kwargs = fake_openai_cls.instances[0].kwargs
        self.assertEqual(kwargs["api_key"], "foundry-key")
        self.assertEqual(
            kwargs["base_url"],
            "https://example.services.ai.azure.com/api/projects/demo-project/openai/v1/",
        )

    def test_project_endpoint_can_come_from_env(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        with patch.dict(
            os.environ,
            {
                "AI_FOUNDRY_API_KEY": "foundry-key",
                "AI_FOUNDRY_AUTH_MODE": "api_key",
                "AI_FOUNDRY_PROJECT_ENDPOINT": (
                    "https://example.services.ai.azure.com/api/projects/demo-project"
                ),
            },
            clear=False,
        ):
            FoundryClient()

        kwargs = fake_openai_cls.instances[0].kwargs
        self.assertEqual(
            kwargs["base_url"],
            "https://example.services.ai.azure.com/api/projects/demo-project/openai/v1/",
        )

    def test_target_uri_endpoint_uses_models_base_url_and_api_version_query(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        with self._foundry_env():
            FoundryClient(
                project_endpoint=(
                    "https://example.services.ai.azure.com/models/chat/completions"
                    "?api-version=2024-05-01-preview"
                ),
                api_style="chat_completions",
            )

        kwargs = fake_openai_cls.instances[0].kwargs
        self.assertEqual(
            kwargs["base_url"],
            "https://example.services.ai.azure.com/models/",
        )
        self.assertEqual(
            kwargs["default_query"],
            {"api-version": "2024-05-01-preview"},
        )

    def test_base_model_preserves_gpt5_request_options_for_deployment_names(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        captured_calls = []
        fake_response = SimpleNamespace(output=[], output_text="", usage=None)

        with self._foundry_env():
            client = FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project",
                base_model="gpt-5-nano",
            )
            fake_openai_cls.instances[-1].responses = SimpleNamespace(
                create=lambda **kwargs: captured_calls.append(kwargs) or fake_response
            )

            context = client.init_context("system", "user")
            client.generate_response(
                input=context,
                params={
                    "model": "prod-deployment",
                    "max_output_tokens": 128,
                    "temperature": 0.9,
                    "reasoning_effort": "low",
                    "verbosity": "low",
                },
            )

        call = captured_calls[-1]
        self.assertEqual(call["model"], "prod-deployment")
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["text"], {"verbosity": "low"})
        self.assertNotIn("temperature", call)

    def test_chat_api_style_uses_chat_completions_for_chat_only_models(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        captured_calls = []
        fake_response = SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

        with self._foundry_env():
            client = FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project",
                base_model="reasoning-chat-model",
                api_style="chat_completions",
            )
            fake_openai_cls.instances[-1].chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: captured_calls.append(kwargs)
                    or fake_response
                )
            )

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={
                    "model": "reasoning-chat-model",
                    "max_output_tokens": 128,
                    "temperature": 0.9,
                },
            )

        call = captured_calls[-1]
        self.assertEqual(call["model"], "reasoning-chat-model")
        self.assertEqual(call["max_tokens"], 128)
        self.assertNotIn("temperature", call)
        self.assertEqual(response_str, "OK")
        usage = client.get_usage(
            response,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            usage,
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    def test_chat_api_style_uses_max_completion_tokens_for_gpt5_family(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        captured_calls = []
        fake_response = SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {},
            }
        )

        with self._foundry_env():
            client = FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project",
                base_model="gpt-5.4",
                api_style="chat_completions",
            )
            fake_openai_cls.instances[-1].chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: captured_calls.append(kwargs)
                    or fake_response
                )
            )

            client.generate_response(
                input=client.init_context("system", "user"),
                params={"model": "gpt-5.4", "max_output_tokens": 128},
            )

        call = captured_calls[-1]
        self.assertEqual(call["model"], "gpt-5.4")
        self.assertEqual(call["max_completion_tokens"], 128)
        self.assertNotIn("max_tokens", call)

    def test_chat_api_style_keeps_temperature_for_non_reasoning_model_name(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        captured_calls = []
        fake_response = SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {},
            }
        )

        with self._foundry_env():
            client = FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project",
                base_model="grok-4-1-fast-non-reasoning",
                api_style="chat_completions",
            )
            fake_openai_cls.instances[-1].chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: captured_calls.append(kwargs)
                    or fake_response
                )
            )

            client.generate_response(
                input=client.init_context("system", "user"),
                params={
                    "model": "grok-4-1-fast-non-reasoning",
                    "max_output_tokens": 128,
                    "temperature": 0.7,
                },
            )

        call = captured_calls[-1]
        self.assertEqual(call["model"], "grok-4-1-fast-non-reasoning")
        self.assertEqual(call["temperature"], 0.7)

    def test_chat_api_style_sanitizes_followup_messages_for_foundry_validation(self):
        module, fake_openai_cls = _import_module_with_mocks(
            "terrarium.llm.clients.foundry_client"
        )
        FoundryClient = module.FoundryClient

        captured_calls = []
        fake_response = SimpleNamespace(
            model_dump=lambda: {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        )

        with self._foundry_env():
            client = FoundryClient(
                project_endpoint="https://example.services.ai.azure.com/api/projects/demo-project",
                base_model="reasoning-chat-model",
                api_style="chat_completions",
            )
            fake_openai_cls.instances[-1].chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: captured_calls.append(kwargs)
                    or fake_response
                )
            )

            client.generate_response(
                input=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "post_message",
                                    "arguments": {"message": "hi"},
                                },
                                "index": None,
                            }
                        ],
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_123",
                        "name": "post_message",
                        "content": "{\"ok\": true}",
                    },
                ],
                params={"model": "reasoning-chat-model"},
            )

        call = captured_calls[-1]
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "post_message",
                                "arguments": "{\"message\": \"hi\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "{\"ok\": true}",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
