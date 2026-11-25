import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeEventStream:
    """Iterable SSE event stream; raises ``error`` once ``events`` are consumed."""

    def __init__(self, events=(), error=None):
        self._events = list(events)
        self._error = error

    def __iter__(self):
        return self

    def __next__(self):
        if self._events:
            return self._events.pop(0)
        if self._error is not None:
            raise self._error
        raise StopIteration


class _FakeResponsesAPI:
    def __init__(self):
        self.calls = []
        self.next_response = None
        self.queued = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queued.pop(0) if self.queued else self.next_response
        if isinstance(item, Exception):
            raise item
        if kwargs.get("stream"):
            if isinstance(item, list):
                return iter(item)
            if isinstance(item, _FakeEventStream):
                return item
            return iter([SimpleNamespace(type="response.completed", response=item)])
        return item


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
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["text"], {"verbosity": "low"})
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
        self.assertNotIn("reasoning_effort", call)
        self.assertNotIn("verbosity", call)

    def test_generate_response_falls_back_to_response_output_text(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="text", text="ignored-content")],
                )
            ],
            output_text="preferred-output-text",
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={"model": "gpt-5-nano"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "preferred-output-text")

    def test_generate_response_sanitizes_followup_items(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(strip_reasoning_content=False)
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            response, response_str = client.generate_response(
                input=[
                    {"type": "message", "role": "system", "content": "system"},
                    {"type": "message", "role": "user", "content": "user"},
                    SimpleNamespace(
                        type="function_call",
                        id="msg_bad_id",
                        call_id="call_1",
                        name="ext_ping",
                        arguments='{"payload":"hello"}',
                    ),
                    SimpleNamespace(
                        type="function_call_output",
                        id="msg_bad_tool_output",
                        call_id="call_1",
                        output='{"status":"ok"}',
                    ),
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        id="msg_123",
                        status="completed",
                        content=[],
                    ),
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_123",
                        summary=[SimpleNamespace(type="summary_text", text="summary")],
                        content=[SimpleNamespace(type="reasoning_text", text="trace")],
                        status="completed",
                    ),
                ],
                params={"model": "gpt-5-nano"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "")
        call = sdk_instance.responses.calls[0]
        self.assertEqual(len(call["input"]), 6)
        self.assertEqual(call["input"][2]["type"], "function_call")
        self.assertEqual(call["input"][2]["call_id"], "call_1")
        self.assertNotIn("id", call["input"][2])
        self.assertEqual(call["input"][3]["type"], "function_call_output")
        self.assertEqual(call["input"][3]["call_id"], "call_1")
        self.assertNotIn("id", call["input"][3])
        self.assertEqual(call["input"][4]["role"], "assistant")
        self.assertEqual(call["input"][4]["id"], "msg_123")
        self.assertEqual(call["input"][5]["type"], "reasoning")
        self.assertEqual(call["input"][5]["id"], "rs_123")
        self.assertEqual(call["input"][5]["summary"][0]["text"], "summary")

    async def test_process_tool_calls_executes_callback_and_appends_outputs(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()

        function_call = SimpleNamespace(
            type="function_call",
            id="msg_bad_id",
            name="ext_ping",
            arguments='{"payload":"hello"}',
            call_id="call_1",
        )
        message = SimpleNamespace(
            type="message",
            role="assistant",
            id="msg_123",
            status="completed",
            content=[],
        )
        reasoning = SimpleNamespace(
            type="reasoning",
            id="rs_123",
            summary=[SimpleNamespace(type="summary_text", text="summary")],
            content=[],
            status="completed",
        )
        response = SimpleNamespace(output=[reasoning, function_call, message])

        async def _execute_tool(tool_name, args):
            return {"tool_name": tool_name, "args": args, "status": "ok"}

        context = []
        tools_executed, updated_context, step_tools = await client.process_tool_calls(
            response, context, _execute_tool
        )

        self.assertEqual(tools_executed, 1)
        self.assertEqual(len(updated_context), 4)
        self.assertIn('ext_ping -- {"payload": "hello"}', step_tools)
        self.assertEqual(updated_context[0]["type"], "reasoning")
        self.assertEqual(updated_context[1]["type"], "function_call")
        self.assertEqual(updated_context[1]["call_id"], "call_1")
        self.assertNotIn("id", updated_context[1])
        self.assertEqual(updated_context[2]["role"], "assistant")
        self.assertEqual(updated_context[2]["id"], "msg_123")
        self.assertEqual(updated_context[3]["type"], "function_call_output")
        self.assertEqual(updated_context[3]["call_id"], "call_1")
        self.assertIn("payload", updated_context[3]["output"])

    async def test_process_tool_calls_skips_message_outputs_in_followup_context(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()

        response = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", id="msg_123", content=[]),
            ]
        )

        context = [{"role": "system", "content": "system"}]
        tools_executed, updated_context, step_tools = await client.process_tool_calls(
            response, context, None
        )

        self.assertEqual(tools_executed, 0)
        self.assertEqual(updated_context, context)
        self.assertEqual(step_tools, [])

    def test_defaults_stream_strip_and_timeout(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            for key in (
                "OPENAI_MAX_RETRIES",
                "OPENAI_RETRY_INITIAL_S",
                "OPENAI_RETRY_MAX_S",
                "OPENAI_RETRY_MULTIPLIER",
                "OPENAI_RETRY_JITTER",
            ):
                os.environ.pop(key, None)
            client = OpenAIClient()

        self.assertTrue(client.stream)
        self.assertTrue(client.strip_reasoning_content)
        self.assertEqual(_FakeOpenAI.instances[-1].timeout, 180.0)
        self.assertEqual(client._max_retries, 6)
        self.assertEqual(client._backoff_initial_s, 0.5)
        self.assertEqual(client._backoff_max_s, 20.0)
        self.assertEqual(client._backoff_multiplier, 2.0)
        self.assertEqual(client._backoff_jitter, 0.1)

    def test_generate_response_streams_and_returns_completed_response(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="streamed-text")],
                )
            ],
            output_text="streamed-text",
        )
        events = [
            SimpleNamespace(type="response.created", response=SimpleNamespace()),
            SimpleNamespace(type="response.completed", response=fake_response),
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = events

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={"model": "gpt-5-nano"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "streamed-text")
        self.assertEqual(len(sdk_instance.responses.calls), 1)
        self.assertIs(sdk_instance.responses.calls[0]["stream"], True)

    def test_generate_response_stream_disabled_uses_blocking_call(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="blocking-text")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(stream=False)
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={"model": "gpt-5-nano"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "blocking-text")
        self.assertNotIn("stream", sdk_instance.responses.calls[0])

    def _status_error(self, status_code):
        exc = Exception(f"HTTP {status_code}")
        exc.status_code = status_code
        return exc

    def test_stream_reject_falls_back_to_blocking_call(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="fallback-text")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.queued = [
                self._status_error(400),
                fake_response,
            ]

            context = client.init_context("system", "user")
            response, response_str = client.generate_response(
                input=context,
                params={"model": "gpt-5-nano"},
            )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "fallback-text")
        calls = sdk_instance.responses.calls
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["stream"], True)
        self.assertNotIn("stream", calls[1])

    def test_stream_reject_ladder_ends_in_blocking_call_with_retries(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="ladder-text")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.queued = [
                self._status_error(422),   # stream rejected
                self._status_error(429),   # single non-streaming retry fails (retryable)
                fake_response,             # plain blocking call succeeds
            ]

            context = client.init_context("system", "user")
            with patch("time.sleep"):
                response, response_str = client.generate_response(
                    input=context,
                    params={"model": "gpt-5-nano"},
                )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "ladder-text")
        calls = sdk_instance.responses.calls
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[0]["stream"], True)
        self.assertNotIn("stream", calls[1])
        self.assertNotIn("stream", calls[2])

    def test_mid_stream_error_is_retried(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="retried-text")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.queued = [
                _FakeEventStream(error=self._status_error(503)),
                [SimpleNamespace(type="response.completed", response=fake_response)],
            ]

            context = client.init_context("system", "user")
            with patch("time.sleep"):
                response, response_str = client.generate_response(
                    input=context,
                    params={"model": "gpt-5-nano"},
                )

        self.assertIs(response, fake_response)
        self.assertEqual(response_str, "retried-text")
        calls = sdk_instance.responses.calls
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["stream"], True)
        self.assertIs(calls[1]["stream"], True)

    def test_retry_config_env_vars_override_defaults(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MAX_RETRIES": "9",
                "OPENAI_RETRY_INITIAL_S": "1.5",
                "OPENAI_RETRY_MAX_S": "30",
                "OPENAI_RETRY_MULTIPLIER": "3",
                "OPENAI_RETRY_JITTER": "0.25",
            },
        ):
            client = OpenAIClient()

        self.assertEqual(client._max_retries, 9)
        self.assertEqual(client._backoff_initial_s, 1.5)
        self.assertEqual(client._backoff_max_s, 30.0)
        self.assertEqual(client._backoff_multiplier, 3.0)
        self.assertEqual(client._backoff_jitter, 0.25)

    def test_retry_config_constructor_args_beat_env_vars(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MAX_RETRIES": "9",
                "OPENAI_RETRY_JITTER": "0.9",
            },
        ):
            client = OpenAIClient(max_retries=3, retry_jitter=0.05, retry_max_s=99)

        self.assertEqual(client._max_retries, 3)
        self.assertEqual(client._backoff_jitter, 0.05)
        self.assertEqual(client._backoff_max_s, 99.0)

    def test_retry_config_env_prefix(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MAX_RETRIES": "9",
                "CUSTOM_MAX_RETRIES": "4",
                "CUSTOM_RETRY_INITIAL_S": "2.5",
            },
        ):
            client = OpenAIClient(env_prefix="CUSTOM")

        self.assertEqual(client._max_retries, 4)
        self.assertEqual(client._backoff_initial_s, 2.5)
        self.assertEqual(client._backoff_max_s, 20.0)

    def test_generate_response_strips_reasoning_items_by_default(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            client.generate_response(
                input=[
                    {"type": "message", "role": "user", "content": "user"},
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_123",
                        summary=[],
                        status="completed",
                    ),
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        id="msg_123",
                        status="completed",
                        content=[],
                    ),
                ],
                params={"model": "gpt-5-nano"},
            )

        call = sdk_instance.responses.calls[0]
        self.assertEqual(len(call["input"]), 2)
        self.assertNotIn("reasoning", [item["type"] for item in call["input"]])

    def test_generate_response_keeps_reasoning_items_when_strip_disabled(self):
        module = _import_openai_client_with_mocks()
        OpenAIClient = module.OpenAIClient

        fake_response = SimpleNamespace(output=[], output_text="")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(strip_reasoning_content=False)
            sdk_instance = _FakeOpenAI.instances[-1]
            sdk_instance.responses.next_response = fake_response

            client.generate_response(
                input=[
                    {"type": "message", "role": "user", "content": "user"},
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_123",
                        summary=[],
                        status="completed",
                    ),
                ],
                params={"model": "gpt-5-nano"},
            )

        call = sdk_instance.responses.calls[0]
        self.assertEqual(len(call["input"]), 2)
        self.assertEqual(call["input"][1]["type"], "reasoning")


if __name__ == "__main__":
    unittest.main()
