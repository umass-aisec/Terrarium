"""
Anthropic Claude client implementation.

This module provides a client for using Anthropic's Claude models via their Messages API.
Supports tool use, extended thinking, and multi-turn conversations.
"""

import os
import json
from typing import Any, List, Dict, Optional
from llm_server.clients.abstract_client import AbstractClient


class AnthropicClient(AbstractClient):
    """
    Client for using Anthropic Claude API for LLM agents.

    Provides the same interface as other clients but uses Anthropic's Messages API.
    Supports Claude models including claude-3-5-sonnet, claude-opus-4, etc.
    """

    def __init__(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
            from dotenv import load_dotenv
        except ImportError as e:
            raise ImportError(
                "Anthropic client requires 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from e

        load_dotenv(override=True)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env file"
            )

        self.client = Anthropic(api_key=self.api_key)
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
        if hasattr(response, 'usage') and response.usage:
            current_usage["prompt_tokens"] += getattr(response.usage, 'input_tokens', 0)
            current_usage["completion_tokens"] += getattr(response.usage, 'output_tokens', 0)
            current_usage["total_tokens"] = (
                current_usage["prompt_tokens"] + current_usage["completion_tokens"]
            )
        return current_usage

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict]:
        """
        Initialize context for Anthropic Messages API.

        Note: Anthropic handles system instructions separately via the API parameter,
        not as part of the message history. The system prompt should be passed
        in params["system"] when calling generate_response.

        Args:
            system_prompt: System instructions (stored in first message metadata)
            user_prompt: Initial user message

        Returns:
            List of message dictionaries in Anthropic format
        """
        # Store system prompt in first message metadata (like Gemini does)
        # It will be extracted and used as API parameter in generate_response
        messages = [
            {
                "role": "user",
                "content": user_prompt,
                "_system_prompt": system_prompt  # Store for later use
            }
        ]
        return messages

    @staticmethod
    def _extract_message_content(response) -> str:
        """
        Extract text content from Anthropic response.

        Anthropic responses have structure: response.content (list of content blocks)
        Each block can have type 'text', 'tool_use', or 'thinking'.
        """
        content = ""
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                # Extract text blocks (skip tool_use and thinking blocks)
                if hasattr(block, 'type') and block.type == 'text':
                    if hasattr(block, 'text'):
                        content += block.text
        return content

    @staticmethod
    def _extract_thinking_content(response) -> str:
        """
        Extract thinking content from Anthropic response (for extended thinking mode).

        Returns thinking blocks as separate content for logging/debugging.
        """
        thinking = ""
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                # Extract thinking blocks
                if hasattr(block, 'type') and block.type == 'thinking':
                    if hasattr(block, 'thinking'):
                        thinking += block.thinking + "\n"
        return thinking.strip()

    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> tuple[Any, str]:
        """
        Generate a response using Anthropic's Messages API.

        Args:
            input: List of message dictionaries in Anthropic format
            params: Generation parameters including:
                - model: Model name (e.g., "claude-3-5-sonnet-20241022")
                - max_tokens: Maximum tokens to generate
                - max_output_tokens: Alternative parameter name for max_tokens
                - temperature: Sampling temperature
                - tools: List of tool definitions (OpenAI format, will be converted)
                - system: System prompt (optional)
                - thinking: Extended thinking config (optional)
                    Format: {"type": "enabled", "budget_tokens": 1024}

        Returns:
            Tuple of (response_object, response_text)
        """
        # Extract parameters
        model = params.get("model", "claude-3-5-sonnet-20241022")
        max_tokens = params.get("max_tokens") or params.get("max_output_tokens", 4000)
        temperature = params.get("temperature", 0.7)
        tools = params.get("tools", [])
        thinking = params.get("thinking")  # Optional extended thinking config

        # Extract system prompt - check params first, then first message metadata
        system_prompt = params.get("system", "")
        if not system_prompt and input and isinstance(input[0], dict):
            system_prompt = input[0].get("_system_prompt", "")

        # Convert tools from OpenAI format to Anthropic format
        anthropic_tools = []
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    func_def = tool.get("function", {})
                    # Anthropic uses "input_schema" instead of "parameters"
                    anthropic_tools.append({
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                        "input_schema": func_def.get("parameters", {})
                    })

        # Prepare messages - filter out metadata (keys starting with _)
        clean_messages = []
        for msg in input:
            if isinstance(msg, dict):
                clean_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content")
                }
                clean_messages.append(clean_msg)
            else:
                # Already a properly formatted message object
                clean_messages.append(msg)

        # Prepare API call parameters
        api_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": clean_messages,
        }

        # Add optional parameters
        if system_prompt:
            api_params["system"] = system_prompt
        if temperature is not None:
            api_params["temperature"] = temperature
        if anthropic_tools:
            api_params["tools"] = anthropic_tools
        if thinking:
            # Extended thinking support for Claude 4 models
            api_params["thinking"] = thinking

        # Make API call
        response = self.client.messages.create(**api_params)

        # Extract text content from response
        response_text = self._extract_message_content(response)

        return response, response_text

    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any
    ) -> tuple[int, list[Any], list[str]]:
        """
        Process tool calls from Anthropic response.

        Parses response.content for tool_use blocks, executes them via callback,
        and adds both the function call and result to the context.

        Args:
            response: Anthropic response object with .content attribute
            context: List of message dictionaries in Anthropic format
            execute_tool_callback: Async function(tool_name: str, args: dict) -> dict

        Returns:
            Tuple of (tools_executed_count, updated_context, tool_names_list)
        """
        tool_calls_executed = 0
        step_tools = []

        # Check if response has content
        if not hasattr(response, 'content') or not response.content:
            return tool_calls_executed, context, step_tools

        # First, append the assistant's response to context (includes both text and tool_use blocks)
        assistant_message = {
            "role": "assistant",
            "content": response.content
        }
        context.append(assistant_message)

        # Collect all tool results to send back in a single user message
        tool_results = []

        # Process each content block looking for tool_use
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_name = getattr(block, 'name', 'unknown')
                tool_use_id = getattr(block, 'id', 'unknown')

                # Extract arguments (already a dict in Anthropic)
                args = getattr(block, 'input', {})
                if not isinstance(args, dict):
                    args = {}

                # Track tool call for trajectory
                step_tools.append(f"{tool_name} -- {json.dumps(args)}")

                # Execute the tool
                result = await execute_tool_callback(tool_name, args)

                # Create tool_result block for this execution
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(result)
                }
                tool_results.append(tool_result_block)

                tool_calls_executed += 1

        # If any tools were executed, add a user message with all tool results
        if tool_results:
            user_message = {
                "role": "user",
                "content": tool_results
            }
            context.append(user_message)

        return tool_calls_executed, context, step_tools
