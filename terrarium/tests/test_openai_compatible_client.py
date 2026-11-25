import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeResponsesAPI:
    def __init__(self):
        self.calls = []
        self.next_response = SimpleNamespace(output=[], output_text="", usage=None)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_response


class _FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = _FakeResponsesAPI()
        _FakeOpenAI.instances.append(self)


class _FakeMessage(dict):
    pass


class _FakeFunctionCallOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _import_openai_compatible_client_with_mocks():
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI

    class _APIConnectionError(Exception):
        pass

    class _APIStatusError(Exception):
        pass

    class _APITimeoutError(Exception):
        pass

    class _RateLimitError(Exception):
        pass

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

    for module_name in (
        "terrarium.llm.clients.openai_client",
        "terrarium.llm.clients.openai_compatible_client",
    ):
        sys.modules.pop(module_name, None)

    with patch.dict(
        sys.modules,
        {
            "openai": fake_openai,
            "openai.types": fake_oai_types,
            "openai.types.responses": fake_oai_types_responses,
            "openai.types.responses.response_input_item_param": fake_oai_types_response_input,
            "dotenv": fake_dotenv,
        },
    ):
        module = importlib.import_module(
            "terrarium.llm.clients.openai_compatible_client"
        )
    return module


class OpenAICompatibleClientTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances = []

    def test_uses_env_base_url_and_api_key(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient()

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "compat-key")
        self.assertEqual(kwargs["base_url"], "https://my-gateway.example.com/v1/")
        self.assertEqual(kwargs["timeout"], 30.0)

    def test_custom_env_var_names_are_used(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "MY_PROVIDER_KEY": "custom-key",
                "MY_PROVIDER_URL": "https://custom.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient(
                api_key_env_var="MY_PROVIDER_KEY",
                base_url_env_var="MY_PROVIDER_URL",
            )

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "custom-key")
        self.assertEqual(kwargs["base_url"], "https://custom.example.com/v1/")

    def test_explicit_args_override_env(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "env-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://env.example.com/v1",
            },
            clear=True,
        ):
            OpenAICompatibleClient(
                api_key="explicit-key",
                base_url="https://explicit.example.com/v1",
            )

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "explicit-key")
        self.assertEqual(kwargs["base_url"], "https://explicit.example.com/v1/")

    def test_missing_api_key_raises(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as ctx:
                OpenAICompatibleClient()

        self.assertIn("OPENAI_COMPATIBLE_API_KEY", str(ctx.exception))

    def test_missing_base_url_raises(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {"OPENAI_COMPATIBLE_API_KEY": "compat-key"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as ctx:
                OpenAICompatibleClient()

        self.assertIn("OPENAI_COMPATIBLE_BASE_URL", str(ctx.exception))

    def test_non_reasoning_model_keeps_temperature(self):
        module = _import_openai_compatible_client_with_mocks()
        OpenAICompatibleClient = module.OpenAICompatibleClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "compat-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://my-gateway.example.com/v1",
            },
            clear=True,
        ):
            client = OpenAICompatibleClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            context = client.init_context("system", "user")
            client.generate_response(
                input=context,
                params={
                    "model": "my-local-chat-model",
                    "max_output_tokens": 128,
                    "temperature": 0.7,
                },
            )

        call = sdk_instance.responses.calls[-1]
        self.assertEqual(call["model"], "my-local-chat-model")
        self.assertEqual(call["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
