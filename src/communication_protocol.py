"""
Communication protocol for managing multi-agent interactions and phases.

This module implements the generic communication protocol that can work with
any environment implementing the AbstractEnvironment interface. It handles
phases, iterations, blackboard management, and LLM interactions.
"""

import random
from typing import Any, Dict, List, Optional, Set

# Import our modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent import Agent
from typing import Dict, Any, Optional
from src.logger import BlackboardLogger

class CommunicationProtocol:
    """
    Manages the overall communication protocol.

    This class is environment-agnostic and handles:
    - Phase management (planning, execution)
    - Agent turn ordering and iteration
    NOTE: DCOP Environments only have 1 iteration per simulation (e.g., MeetingScheduling, PersonalAssistant, SmartGrid)
    """

    def __init__(self, config: Dict[str, Any],
                 tool_logger, mcp_client):
        """
        Initialize the communication protocol.

        Args:
            environment: The specific environment implementation
            config: Full configuration dictionary with structured sections
            client: Initialized LLM client for agent interactions
            tool_logger: Tool call logger instance
        """
        self.config = config
        self.simulation_config = config["simulation"]
        # self.blackboard_manager = Megaboard()
        self.tool_logger = tool_logger
        self.blackboard_logger = BlackboardLogger(self.config)
        self.blackboard_logger.clear_blackboard_logs()
        self.mcp_client = mcp_client
        self.environment = None
        self._server_logger_initialized = False  # Track if MCP server logger is initialized
        
    async def _ensure_server_logger_initialized(self, client):
        """
        Ensure the MCP server-side logger is initialized.
        """
        if not self._server_logger_initialized and self.blackboard_logger:
            await client.call_tool("initialize_blackboard_logger", {"config": self.config})
            self._server_logger_initialized = True

    def _extract_environment_state(self) -> Dict[str, Any]:
        """
        Extract serializable state from environment for MCP transmission.
        Delegates to environment-specific implementation.
        """
        assert self.environment is not None, "Environment should be set for CommunicationProtocol"
        # Call environment-specific state extraction
        if hasattr(self.environment, 'get_serializable_state'):
            return self.environment.get_serializable_state()

        return {}

    def _apply_environment_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates back to the environment.
        Delegates to environment-specific implementation.
        """
        if not self.environment:
            return

        # Call environment-specific state application
        if hasattr(self.environment, 'apply_state_updates'):
            self.environment.apply_state_updates(state_updates)

    async def environment_handle_tool_call(self, tool_name, agent_name:str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle environment tool calls by extracting serializable state, calling MCP tools,
        and applying state updates back to the environment.
        """
        # Extract serializable state from environment
        env_state = self._extract_environment_state()

        async with self.mcp_client as client:
            response = (await client.call_tool("handle_environment_tool_call", {
                "tool_name": tool_name,
                "agent_name": agent_name,
                "arguments": arguments,
                "phase": phase,
                "iteration": iteration,
                "env_state": env_state
            })).data

            # Apply state updates back to environment if present
            # Check both top-level and nested in "result"
            state_updates = None
            if "state_updates" in response:
                state_updates = response["state_updates"]
            elif "result" in response and isinstance(response["result"], dict) and "state_updates" in response["result"]:
                state_updates = response["result"]["state_updates"]

            if state_updates:
                self._apply_environment_state_updates(state_updates)

                # Call environment-specific post-tool execution callback if available
                # This allows environments to do custom processing after state updates
                if hasattr(self.environment, 'post_tool_execution_callback'):
                    self.environment.post_tool_execution_callback(state_updates, response)

            return response
    
    async def blackboard_handle_tool_call(self, tool_name, agent_name:str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        async with self.mcp_client as client:
            response = (await client.call_tool("handle_blackboard_tool_call", {"tool_name": tool_name, "agent_name": agent_name, "arguments": arguments, "phase": phase, "iteration": iteration})).data
            return response
        #return self.blackboard_manager.handle_tool_call(tool_name, agent_name, arguments, phase, iteration)
    
    async def get_all_blackboard_ids(self) -> List[str]:
        async with self.mcp_client as client:
            return (await client.call_tool("get_blackboard_string_ids")).data

    async def post_system_message(self, blackboard_id: int, kind: str, payload: Optional[Dict[str, Any]] = None,
                           phase: Optional[str] = None, iteration: Optional[int] = None) -> str:
        async with self.mcp_client as client:
            # Only pass the parameters that the MCP server tool accepts
            return (await client.call_tool("post_system_message", {"blackboard_id": blackboard_id, "kind": kind, "payload": payload})).data
        
    async def agent_planning_turn(self, agent: 'Agent', agent_name: str, agent_context, environment, iteration: int, planning_round: int):
        """Handle a single agent's planning turn."""
        # Get blackboard contexts from blackboard manager
        async with self.mcp_client as client:
            blackboard_contexts = (await client.call_tool("get_agent_blackboard_contexts", {"agent_name": agent_name})).data
            self.environment = environment
            #blackboard_contexts = self.blackboard_manager.get_agent_blackboard_contexts()
            prompts = environment.prompts
            response_data = await agent.generate_agent_response(
                agent_name=agent_name,
                agent_context=agent_context,
                blackboard_context=blackboard_contexts,
                communication_protocol = self,
                prompts=prompts,
                phase="planning",
                iteration=iteration,
                round_num=planning_round
            )

            # Log blackboard states after agent's turn via MCP
            if self.blackboard_logger:
                await self._ensure_server_logger_initialized(client)
                result = await client.call_tool("log_blackboard_states", {
                    "iteration": iteration,
                    "phase": "planning",
                    "agent_name": agent_name,
                    "planning_round": planning_round
                })

    async def generate_blackboard_network_from_factor(self, factor, context):
        async with self.mcp_client as client:
            blackboard_id = (await client.call_tool("add_blackboard", {
                "agents": factor.agent_scope,
                "template": None
            })).data

            # Post initial context as a system message with phase metadata
            await client.call_tool("post_system_message", {
                "blackboard_id": blackboard_id,
                "kind": "context",
                "payload": {"message": context}
            })
        return blackboard_id

    async def agent_execution_turn(self, agent: 'Agent', agent_name: str, agent_context, environment, iteration: int):
        """
        Handle a single agent's execution turn with retry logic.

        Args:
            agent_name: Name of the agent taking the turn
            iteration: Current iteration
        """

        # Get blackboard contexts
        async with self.mcp_client as client:
            blackboard_contexts = (await client.call_tool("get_agent_blackboard_contexts", {"agent_name": agent_name})).data
            self.environment = environment
            prompts = environment.prompts
            await agent.generate_agent_response(
                agent_name=agent_name,
                agent_context=agent_context,
                blackboard_context=blackboard_contexts,
                communication_protocol = self,
                prompts=prompts,
                phase="execution",
                iteration=iteration
            )

            # Log blackboard states after agent's turn via MCP
            if self.blackboard_logger:
                await self._ensure_server_logger_initialized(client)
                await client.call_tool("log_blackboard_states", {
                    "iteration": iteration,
                    "phase": "execution",
                    "agent_name": agent_name
                })