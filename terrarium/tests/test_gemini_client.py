import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from terrarium.llm.clients.gemini_client import GeminiClient


class GeminiClientTests(unittest.TestCase):
    def test_vertex_project_location_does_not_pass_api_key(self):
        from google import genai

        captured = []

        with (
            patch("dotenv.load_dotenv", lambda override=True: None),
            patch.object(
                genai,
                "Client",
                side_effect=lambda **kwargs: captured.append(kwargs)
                or SimpleNamespace(),
            ),
            patch.dict(
                os.environ,
                {
                    "GOOGLE_GENAI_USE_VERTEXAI": "True",
                    "GOOGLE_CLOUD_PROJECT": "test-project",
                    "GOOGLE_CLOUD_LOCATION": "global",
                    "GOOGLE_API_KEY": "developer-api-key",
                },
                clear=True,
            ),
        ):
            GeminiClient()

        kwargs = captured[-1]
        self.assertTrue(kwargs["vertexai"])
        self.assertEqual(kwargs["project"], "test-project")
        self.assertEqual(kwargs["location"], "global")
        self.assertNotIn("api_key", kwargs)
        self.assertEqual(kwargs["http_options"].api_version, "v1")

    def test_non_vertex_uses_api_key(self):
        from google import genai

        captured = []

        with (
            patch("dotenv.load_dotenv", lambda override=True: None),
            patch.object(
                genai,
                "Client",
                side_effect=lambda **kwargs: captured.append(kwargs)
                or SimpleNamespace(),
            ),
            patch.dict(os.environ, {"GOOGLE_API_KEY": "developer-api-key"}, clear=True),
        ):
            GeminiClient()

        self.assertEqual(captured[-1], {"api_key": "developer-api-key"})

    def test_generate_response_accepts_thinking_level(self):
        captured = []
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=None,
                    content=SimpleNamespace(parts=[SimpleNamespace(text="ok")]),
                )
            ]
        )

        client = GeminiClient.__new__(GeminiClient)
        client.client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: captured.append(kwargs) or response
            )
        )

        _response, response_text = client.generate_response(
            input=[{"role": "user", "parts": [{"text": "hello"}]}],
            params={
                "model": "gemini-3-flash-preview",
                "max_tokens": 128,
                "thinking_level": "low",
            },
        )

        self.assertEqual(response_text, "ok")
        config = captured[-1]["config"]
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")


class GeminiClientToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_tool_calls_groups_function_responses_for_one_model_turn(self):
        from google.genai import types

        client = GeminiClient.__new__(GeminiClient)
        context = [{"role": "user", "parts": [{"text": "hello"}]}]
        candidate_content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_1",
                        name="first_tool",
                        args={"value": 1},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_2",
                        name="second_tool",
                        args={"value": 2},
                    )
                ),
            ],
        )
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=candidate_content)]
        )

        async def execute_tool_callback(tool_name, args):
            return {"tool_name": tool_name, "args": dict(args)}

        tools_executed, updated_context, step_tools = await client.process_tool_calls(
            response,
            context,
            execute_tool_callback,
        )

        self.assertEqual(tools_executed, 2)
        self.assertEqual(step_tools, [
            'first_tool -- {"value": 1}',
            'second_tool -- {"value": 2}',
        ])
        self.assertEqual(len(updated_context), 3)
        self.assertIs(updated_context[1], candidate_content)

        function_response_turn = updated_context[2]
        self.assertEqual(function_response_turn.role, "user")
        self.assertEqual(len(function_response_turn.parts), 2)
        self.assertEqual(
            function_response_turn.parts[0].function_response.name,
            "first_tool",
        )
        self.assertEqual(
            function_response_turn.parts[1].function_response.name,
            "second_tool",
        )


if __name__ == "__main__":
    unittest.main()
