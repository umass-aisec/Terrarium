"""
Google Gemini client implementation.

This module provides a client for using Google's Gemini models via their Generative AI API.
Supports tool use, multi-turn conversations, and various Gemini models.
"""

import os
import json
from typing import Any, List, Dict, Optional
from llm_server.clients.abstract_client import AbstractClient


class GeminiClient(AbstractClient):
    """
    Client for using Google Gemini API for LLM agents.

    Provides the same interface as other clients but uses Google's Generative AI API.
    Supports Gemini models including gemini-2.0-flash-exp, gemini-1.5-pro, etc.
    """

    def __init__(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
        except ImportError as e:
            raise ImportError(
                "Gemini client requires 'google-generativeai' package. "
                "Install with: pip install google-generativeai"
            ) from e

        load_dotenv(override=True)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in .env file"
            )

        genai.configure(api_key=self.api_key)
        self.genai = genai
        self._current_meta_context = {}  # For logging metadata

    def set_meta_context(
        self,
        agent_name: str,
        phase: str,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> None:
        """
        Set metadata context for logging purposes.

        Args:
            agent_name: Name of the current agent
            phase: Current simulation phase
            iteration: Current iteration number
            round_num: Current round number
        """
        self._current_meta_context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round_num": round_num
        }

    @staticmethod
    def get_usage(response: Any, current_usage: dict[str, int]) -> Dict[str, int]:
        """Accumulate usage statistics from response."""
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            current_usage["prompt_tokens"] += getattr(usage, 'prompt_token_count', 0)
            current_usage["completion_tokens"] += getattr(usage, 'candidates_token_count', 0)
            current_usage["total_tokens"] = getattr(usage, 'total_token_count', 0)
        return current_usage

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict]:
        """
        Initialize context for Gemini API.

        Note: Gemini handles system instructions separately via model initialization,
        not as part of the message history. The system prompt should be passed
        in params["system"] when calling generate_response.

        Args:
            system_prompt: System instructions (will be included in first message for reference)
            user_prompt: Initial user message

        Returns:
            List of message dictionaries in Gemini format
        """
        # Gemini doesn't include system prompt in messages - it goes in model config
        # But we'll store it in the first message's metadata for consistency
        messages = [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
                "_system_prompt": system_prompt  # Store for later use
            }
        ]
        return messages

    @staticmethod
    def _extract_message_content(response) -> str:
        """
        Extract text content from Gemini response.

        Gemini responses have structure: response.candidates[0].content.parts[]
        Each part can have either .text or .function_call attributes.
        """
        content = ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Only extract text parts, skip function_call parts
                    if hasattr(part, 'text') and part.text:
                        content += part.text
        return content

    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> tuple[Any, str]:
        """
        Generate a response using Google's Gemini API.

        Args:
            input: List of message dictionaries in Gemini format
            params: Generation parameters including:
                - model: Model name (e.g., "gemini-2.0-flash-exp")
                - max_tokens: Maximum tokens to generate (max_output_tokens in Gemini)
                - temperature: Sampling temperature
                - tools: List of tool definitions
                - system: System instruction (optional)

        Returns:
            Tuple of (response_object, response_text)
        """
        # Extract parameters
        model_name = params.get("model", "gemini-2.0-flash-exp")
        max_tokens = params.get("max_tokens") or params.get("max_output_tokens", 1000)
        temperature = params.get("temperature", 0.7)
        tools = params.get("tools", [])

        # Extract system prompt - check params first, then first message metadata
        system_instruction = params.get("system", "")
        if not system_instruction and input and isinstance(input[0], dict):
            system_instruction = input[0].get("_system_prompt", "")

        # Convert tools to Gemini format if provided
        gemini_tools = []
        if tools:
            from google.generativeai.types import FunctionDeclaration, Tool

            function_declarations = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    func_def = tool.get("function", {})
                    function_declarations.append(
                        FunctionDeclaration(
                            name=func_def.get("name"),
                            description=func_def.get("description"),
                            parameters=func_def.get("parameters", {})
                        )
                    )

            if function_declarations:
                gemini_tools = [Tool(function_declarations=function_declarations)]

        # Create generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Initialize model
        model_config = {
            "model_name": model_name,
            "generation_config": generation_config,
        }

        if system_instruction:
            model_config["system_instruction"] = system_instruction

        if gemini_tools:
            model_config["tools"] = gemini_tools

        model = self.genai.GenerativeModel(**model_config)

        # Convert input to Gemini format, handling both dicts and Content objects
        from google.ai.generativelanguage_v1beta.types.content import Content as GeminiContent

        history = []
        for msg in input:
            # Handle Content objects (from process_tool_calls)
            if isinstance(msg, GeminiContent):
                history.append(msg)
            # Handle dict messages (from init_context)
            elif isinstance(msg, dict):
                role = msg.get("role")
                parts = msg.get("parts", [])
                # Filter out metadata (keys starting with _)
                clean_msg = {"role": role, "parts": parts}
                history.append(clean_msg)

        # For multi-turn: use all history except the last message for chat initialization
        # Then send the last message to get the response
        if len(history) > 1:
            chat = model.start_chat(history=history[:-1])
            # Send the last message
            last_msg = history[-1]
            if isinstance(last_msg, GeminiContent):
                response = chat.send_message(last_msg.parts)
            else:
                response = chat.send_message(last_msg["parts"])
        elif history:
            # First turn: no history yet, just send the message
            chat = model.start_chat(history=[])
            last_msg = history[0]
            if isinstance(last_msg, GeminiContent):
                response = chat.send_message(last_msg.parts)
            else:
                response = chat.send_message(last_msg["parts"])
        else:
            # Fallback: empty context
            chat = model.start_chat(history=[])
            response = chat.send_message("")

        # Extract text content
        response_text = self._extract_message_content(response)

        return response, response_text

    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any
    ) -> tuple[int, list[Any], list[str]]:
        """
        Process tool calls from Gemini response.

        Parses response.candidates[0].content.parts for function_call items,
        executes them via callback, and adds both the function call and result to context.

        Args:
            response: Gemini response object with .candidates attribute
            context: List of message dictionaries in Gemini format
            execute_tool_callback: Async function(tool_name: str, args: dict) -> dict

        Returns:
            Tuple of (tools_executed_count, updated_context, tool_names_list)
        """
        from google.ai.generativelanguage_v1beta.types import content

        tool_calls_executed = 0
        step_tools = []

        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return tool_calls_executed, context, step_tools

        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
            return tool_calls_executed, context, step_tools

        # First, append the model's response to context (includes both text and function calls)
        context.append(candidate.content)

        # Process each part looking for function calls
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                tool_name = function_call.name

                # Extract arguments (already a dict in Gemini)
                args = dict(function_call.args) if function_call.args else {}

                # Track tool call for trajectory
                step_tools.append(f"{tool_name} -- {json.dumps(args)}")

                # Execute the tool
                result = await execute_tool_callback(tool_name, args)

                # Create function response using Gemini's proto format
                function_response = content.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                )

                # Create a Part with the function response
                function_response_part = content.Part(
                    function_response=function_response
                )

                # Append function result to context as a user Content
                context.append(content.Content(
                    role="user",
                    parts=[function_response_part]
                ))

                tool_calls_executed += 1

        return tool_calls_executed, context, step_tools