"""
Abstract base class for LLM client implementations.

This module defines the interface that all LLM clients must implement,
along with optional helper methods that can be shared across implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any


class AbstractClient(ABC):
    """
    Abstract base class for LLM client implementations.

    All client implementations (OpenAI, vLLM, GPT-OSS, etc.) should inherit from this class
    and implement the generate_response method. This ensures a consistent interface across
    different LLM backends.
    """

    @abstractmethod
    def generate_response(
        self,
        input: list[Any],
        params: dict[str, Any],
    ) -> tuple[Any, str]:
        """
        Generate a response from the LLM.

        This is the core method that all client implementations must provide.
        The method signature is standardized to use flexible input and params
        to accommodate different API styles.

        Args:
            input: List of message/context items representing the conversation history.
                   Format depends on the specific client implementation (e.g., OpenAI Message objects).
            params: Dictionary of generation parameters including:
                   - model: str - Model identifier to use
                   - max_completion_tokens: int - Maximum tokens to generate
                   - max_output_tokens: int - Alternative token limit parameter
                   - temperature: float - Sampling temperature (0.0 = deterministic)
                   - tools: list - Available tools/functions for the model
                   - reasoning_effort: str - Effort level for reasoning models
                   - verbosity: str - Verbosity setting for model output
                   Additional client-specific parameters can be included.

        Returns:
            Tuple containing:
            - response_object: The raw response object from the LLM API
            - response_string: Extracted text content from the response

        Raises:
            Exception: If the LLM request fails or encounters an error
        """
        pass

    @abstractmethod
    async def process_tool_calls(
        self,
        response: Any,
        context: list[Any],
        execute_tool_callback: Any
    ) -> tuple[int, list[Any], list[str]]:
        """
        Process tool calls from the LLM response.

        This method handles provider-specific tool call parsing and execution.
        Each client implementation must parse its own response format, detect
        tool calls, execute them via the callback, and update the context
        appropriately for the next conversation turn.

        Args:
            response: The raw response object from generate_response()
            context: Current conversation context/history (will be updated)
            execute_tool_callback: Async function to execute tools.
                                  Signature: async def(tool_name: str, args: dict) -> dict

        Returns:
            Tuple containing:
            - tool_calls_executed: Number of tool calls that were executed
            - updated_context: Context list with tool calls and results appended
            - tool_names_list: List of executed tool names with args (for logging)
                              Format: ["tool_name -- {args_json}", ...]

        Raises:
            Exception: If tool call parsing or execution fails
        """
        pass

    def configure_external_mcp(self, llm_config: dict[str, Any]) -> None:
        """
        Configure optional external MCP servers for this client instance.

        The default implementation is a no-op when no external server config
        is provided and lazily initializes a shared registry otherwise.
        """
        from terrarium.llm.clients.external_mcp import ExternalMCPToolRegistry

        self._external_mcp_registry = ExternalMCPToolRegistry.from_llm_config(
            llm_config
        )

    async def get_external_tools(self) -> list[dict[str, Any]]:
        """Return external MCP tools in OpenAI-compatible tool schema."""
        registry = getattr(self, "_external_mcp_registry", None)
        if registry is None:
            return []
        return await registry.get_tools()

    async def get_external_tool_names(self) -> set[str]:
        """Return discovered external MCP tool names."""
        registry = getattr(self, "_external_mcp_registry", None)
        if registry is None:
            return set()
        return await registry.get_tool_names()

    async def execute_external_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Execute a discovered external MCP tool.

        Returns None when the tool name is not managed by the external registry.
        """
        registry = getattr(self, "_external_mcp_registry", None)
        if registry is None:
            return None
        return await registry.call_tool(tool_name, arguments or {})

    # Optional methods with default no-op implementations that subclasses can override

    def set_meta_context(
        self,
        agent_name: str,
        phase: str,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> None:
        """
        Set metadata context for logging and tracking purposes.

        This is an optional method used primarily for debugging and logging.
        Subclasses that support detailed logging (like OpenAI client with tool call logging)
        can override this method to track agent context across turns.

        Args:
            agent_name: Name of the current agent making the request
            phase: Current simulation phase (e.g., "planning", "execution")
            iteration: Current iteration number in the simulation
            round_num: Current round number within the iteration
        """
        # Default no-op implementation
        pass
