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


def _import_grok_client_with_mocks():
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
        "terrarium.llm.clients.grok_client",
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
        module = importlib.import_module("terrarium.llm.clients.grok_client")
    return module


class GrokClientTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances = []

    def test_uses_first_party_xai_base_url_and_grok_api_key(self):
        module = _import_grok_client_with_mocks()
        GrokClient = module.GrokClient

        with patch.dict(os.environ, {"GROK_API_KEY": "grok-key"}, clear=True):
            GrokClient()

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "grok-key")
        self.assertEqual(kwargs["base_url"], "https://api.x.ai/v1/")
        self.assertEqual(kwargs["timeout"], 3600.0)

    def test_falls_back_to_official_xai_api_key_env_name(self):
        module = _import_grok_client_with_mocks()
        GrokClient = module.GrokClient

        with patch.dict(os.environ, {"XAI_API_KEY": "xai-key"}, clear=True):
            GrokClient()

        kwargs = _FakeOpenAI.instances[-1].kwargs
        self.assertEqual(kwargs["api_key"], "xai-key")

    def test_grok_reasoning_model_does_not_send_reasoning_effort_param(self):
        module = _import_grok_client_with_mocks()
        GrokClient = module.GrokClient

        with patch.dict(os.environ, {"GROK_API_KEY": "grok-key"}, clear=True):
            client = GrokClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            context = client.init_context("system", "user")
            client.generate_response(
                input=context,
                params={
                    "model": "grok-4-1-fast-reasoning",
                    "max_output_tokens": 128,
                    "temperature": 0.7,
                    "reasoning_effort": "low",
                },
            )

        call = sdk_instance.responses.calls[-1]
        self.assertEqual(call["model"], "grok-4-1-fast-reasoning")
        self.assertNotIn("reasoning", call)
        self.assertNotIn("temperature", call)

    def test_grok_non_reasoning_model_keeps_temperature(self):
        module = _import_grok_client_with_mocks()
        GrokClient = module.GrokClient

        with patch.dict(os.environ, {"GROK_API_KEY": "grok-key"}, clear=True):
            client = GrokClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            context = client.init_context("system", "user")
            client.generate_response(
                input=context,
                params={
                    "model": "grok-4-1-fast-non-reasoning",
                    "max_output_tokens": 128,
                    "temperature": 0.7,
                },
            )

        call = sdk_instance.responses.calls[-1]
        self.assertEqual(call["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
