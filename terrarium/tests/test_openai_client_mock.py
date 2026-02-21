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
        self.next_response = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.next_response, Exception):
            raise self.next_response
        return self.next_response


class _FakeOpenAI:
    instances = []

    def __init__(self, api_key=None, timeout=None):
        self.api_key = api_key
        self.timeout = timeout
        self.responses = _FakeResponsesAPI()
        _FakeOpenAI.instances.append(self)


class _FakeMessage(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _FakeFunctionCallOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _import_openai_client_with_mocks():
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

    module_name = "terrarium.llm.clients.openai_client"
    if module_name in sys.modules:
        del sys.modules[module_name]

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
        module = importlib.import_module(module_name)
    return module


class OpenAIClientMockTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FakeOpenAI.instances = []

    def test_generate_response_translates_tools_and_extracts_text(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="mock-response")],
                )
            ]
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={
                    "model": "gpt-5-nano",
                    "max_output_tokens": 128,
                    "temperature": 0.9,
                    "reasoning_effort": "low",
                    "verbosity": "low",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "assign_task",
                                "description": "Assign a task.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"task_id": {"type": "string"}},
                                },
                            },
                        }
                    ],
                },
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "mock-response")
        self.assertEqual(len(sdk_instance.responses.calls), 1)

        call = sdk_instance.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5-nano")
        self.assertEqual(call["max_output_tokens"], 128)
        self.assertNotIn("temperature", call)
        self.assertEqual(call["reasoning_effort"], "low")
        self.assertEqual(call["verbosity"], "low")
        self.assertEqual(
            call["tools"],
            [
                {
                    "type": "function",
                    "name": "assign_task",
                    "description": "Assign a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                    },
                }
            ],
        )

    async def test_process_tool_calls_executes_callback_and_appends_outputs(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()

        function_call = SimpleNamespace(
            type="function_call",
            name="ext_ping",
            arguments='{"payload":"hello"}',
            call_id="call_1",
        )
        message = SimpleNamespace(type="message", content=[])
        response = SimpleNamespace(output=[function_call, message])

        async def _execute_tool(tool_name, args):
            return {"tool_name": tool_name, "args": args, "status": "ok"}

        context = []
        tools_executed, updated_context, step_tools = await client.process_tool_calls(
            response, context, _execute_tool
        )

        self.assertEqual(tools_executed, 1)
        self.assertEqual(len(updated_context), 3)
        self.assertIn('ext_ping -- {"payload": "hello"}', step_tools)
        self.assertEqual(updated_context[0], function_call)
        self.assertEqual(updated_context[1].type, "function_call_output")
        self.assertEqual(updated_context[1].call_id, "call_1")
        self.assertIn("payload", updated_context[1].output)
        self.assertEqual(updated_context[2], message)


if __name__ == "__main__":
    unittest.main()
