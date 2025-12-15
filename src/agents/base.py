import logging
import time
from typing import Dict, Optional, Any
from llm_server.clients.abstract_client import AbstractClient
from ..toolset_discovery import ToolsetDiscovery

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Agent-dependent methods for LLM agents.
    """

    def __init__(
        self,
        client: AbstractClient,
        name: str,
        model_name: str = "",
        max_conversation_steps: int = 3,
        tool_logger: Optional[Any] = None,
        trajectory_logger: Optional[Any] = None,
        environment_name: str = "",
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an Agent.

        Args:
            client: LLM client instance
            name: Agent name
            model_name: Model name (provider-specific)
            max_conversation_steps: Max conversation turns for multi-step tool execution (default: 3)
            tool_logger: Logger for tool call tracking
            trajectory_logger: Logger for agent reasoning trajectories
            environment_name: Environment name (prefer passing environment.__class__.__name__)
            generation_params: Generation parameters specific to the provider/model (e.g., temperature, top_p)
        """
        self.name = name
        self.model_name = model_name
        self.generation_params = generation_params or {}
        self.max_conversation_steps = max_conversation_steps
        self.tool_logger = tool_logger
        self.trajectory_logger = trajectory_logger
        # Used for tool discovery and error messages; should be the class environment name
        self.environment_name = environment_name
        self.toolset_discovery = ToolsetDiscovery()
        self.client = client

        # Agent context for logging (set via set_meta_context)
        self.current_agent_name = None
        self.current_phase = None
        self.current_iteration = None
        self.current_round = None
        self.communication_protocol = None

    def _build_generation_params(self, tool_set) -> Dict[str, Any]:
        """
        Merge generic generation defaults with provider-specific params.
        """
        max_tokens = self.generation_params.get("max_tokens")
        base_params = {
            "model": self.model_name,
            # Supply all common token limit keys to support different clients
            "max_completion_tokens": max_tokens,
            "max_output_tokens": max_tokens,
            "max_tokens": max_tokens,
            "tools": tool_set,
        }
        # Drop None values
        base_params = {k: v for k, v in base_params.items() if v is not None}

        # Provider/model specific overrides from config
        generation_params_clean = {k: v for k, v in self.generation_params.items() if v is not None}
        base_params.update(generation_params_clean)
        return base_params


    def _log_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any], start_time: float) -> None:
        """Helper to log tool calls."""
        if not (self.tool_logger and self.current_agent_name):
            return
        duration_ms = (time.time() - start_time) * 1000  # s -> ms
        self.tool_logger.log_tool_call(
            agent_name=self.current_agent_name,
            phase=self.current_phase or "unknown",
            tool_name=tool_name,
            parameters=arguments,
            result=result,
            iteration=self.current_iteration,
            round_num=self.current_round,
            duration_ms=duration_ms
        )

    def _log_trajectory(self, trajectory_dict: Dict[str, Any]) -> None:
        """Helper to log agent trajectories."""
        if self.trajectory_logger and self.current_agent_name and trajectory_dict:
            self.trajectory_logger.log_trajectory(
                agent_name=self.current_agent_name,
                iteration=self.current_iteration or 0,
                phase=self.current_phase or "unknown",
                trajectory_dict=trajectory_dict,
                round_num=self.current_round
            )


    def set_meta_context(self, agent_name: str, phase: str, iteration: int, round_num: Optional[int] = None) -> None:
        """
        Set the current agent context for tool call logging.

        Args:
            agent_name: Name of the current agent
            phase: Current phase (planning, execution, etc.)
            iteration: Current iteration number
            round_num: Current round number
        """
        self.current_agent_name = agent_name
        self.current_phase = phase
        self.current_iteration = iteration
        self.current_round = round_num

    async def _execute_tool_call(self, tool_name: str, tool_arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call by delegating to the appropriate tool handler.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        start_time = time.time()
        assert self.communication_protocol is not None, "Communication protocol not set for agent"
        assert self.current_agent_name is not None, "Agent context not set - call set_meta_context first"
        assert self.current_phase is not None, "Agent context not set - call set_meta_context first"

        result: Dict[str, Any] = {"error": "Unknown error"}
        try:
            # NOTE: Depending on the communication protocol implementation, this logic may have to change.
            env_name = self.environment_name or ""

            available_env_tools = {
                tool.get("function", {}).get("name")
                for tool in self.toolset_discovery.get_tools_for_environment(env_name, self.current_phase)
            }
            available_env_tools.discard(None)
            all_env_tools = self.toolset_discovery.get_env_tool_names(env_name)
            blackboard_tool_names = self.toolset_discovery.get_blackboard_tool_names()

            handler = None
            if tool_name in blackboard_tool_names:
                handler = self.communication_protocol.blackboard_handle_tool_call
            elif tool_name in available_env_tools:
                handler = self.communication_protocol.environment_handle_tool_call
            elif tool_name in all_env_tools:
                result = {
                    "error": f"Tool '{tool_name}' is not available during the {self.current_phase} phase."
                }
            else:
                raise ValueError(f"Tool '{tool_name}' is not recognized in environment '{self.environment_name}'.")

            if handler is not None:
                result = await handler(
                    tool_name,
                    self.current_agent_name,
                    tool_arguments,
                    phase=self.current_phase,
                    iteration=self.current_iteration,
                )
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {e}"
            result = {"error": error_msg}
            logger.exception(error_msg)
        finally:
            self._log_tool_call(tool_name, tool_arguments, result, start_time)

        return result



    async def generate_response(
        self,
        *,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, str],
        prompts: Any,
        communication_protocol: Any,
        phase: str,
        iteration: int,
        round_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate a response for a specific agent with full context.

        Returns:
            Dictionary containing agent's response.
        """
        try:
            self.communication_protocol = communication_protocol
            if self.communication_protocol is None:
                raise ValueError("communication_protocol must be set for tool-based execution")

            if prompts is None:
                raise ValueError("prompts must be provided to generate a response")

            if not self.environment_name:
                raise ValueError("environment_name must be set when initializing the agent")

            self.set_meta_context(
                agent_name=agent_name,
                phase=phase,
                iteration=iteration,
                round_num=round_num,
            )

            system_prompt = prompts.get_system_prompt()
            user_prompt = prompts.get_user_prompt(
                agent_name=agent_name,
                agent_context=agent_context,
                blackboard_context=blackboard_context,
            )
            if system_prompt is None or user_prompt is None:
                raise ValueError("Prompts returned empty system/user prompt")

            return await self._multi_step_response_generation(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as e:
            logger.exception("Can't get LLM response: %s", e)
            raise

    async def _multi_step_response_generation(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate a response multi-step tool execution loop.
        Example: Call Tool -> Execute Tool (us) -> Append output to context -> Call Tool -> ...

        Args:
            system_prompt: System prompt for the agent
            user_prompt: User prompt with context and query

        Returns:
            Dict with keys: response, usage, model, has_tool_calls, etc.
        """
        # Get tools for this environment and phase.
        assert self.current_phase is not None, "Agent context not set - call set_meta_context first"
        env_tools = self.toolset_discovery.get_tools_for_environment(
            self.environment_name,
            self.current_phase,
        )
        blackboard_tools = self.toolset_discovery.get_tools_for_blackboard(self.current_phase)
        tool_set = env_tools + blackboard_tools
        if not tool_set:
            raise ValueError(
                "[ERROR] tool_set is empty. Agents are required to have access to tools for tool-based execution."
            )
        params = self._build_generation_params(tool_set=tool_set)
        # Get system and user prompt into Reponses API format
        # TODO: Add functions to abstract client class such as init_context()
        context = self.client.init_context(system_prompt, user_prompt)
        total_tools_executed = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Get a response and update context for next conversation step
        # response object will have tool calls embedded inside and text response
        response, response_str = self.client.generate_response(input=context, params=params)
        total_usage = self.client.get_usage(response, total_usage)
        current_response = response

        trajectory_dict: Dict[str, Any] = {}
        conversation_steps = 1

        for step in range(self.max_conversation_steps):
            # Process tool calls and extract content
            tool_calls_executed, context, step_tools = await self.client.process_tool_calls(
                current_response,
                context,
                self._execute_tool_call,
            )
            total_tools_executed += tool_calls_executed

            if step_tools:
                # For logging purposes
                trajectory_dict[f"step_{step + 1}"] = {"tools": step_tools}

            # Heuristic: No tool calls => we're done.
            if tool_calls_executed == 0:
                break

            # Reached the max number of model calls allowed.
            if step >= self.max_conversation_steps - 1:
                break

            # Continue the conversation using the accumulated context
            try:
                response, response_str = self.client.generate_response(input=context, params=params)
                total_usage = self.client.get_usage(response, total_usage)
                current_response = response
                conversation_steps += 1
            except Exception as e:
                logger.exception("Failed to continue conversation at step %s: %s", step + 1, e)
                self._log_trajectory(trajectory_dict)

                return {
                    "response": (
                        f"[ERROR] Tool execution completed but failed to continue conversation at step {step + 1}: {e}"
                    ),
                    "full_content": response_str,
                    "usage": total_usage,
                    "model": getattr(current_response, "model", None) or self.model_name,
                    "tool_calls": None,
                    "has_tool_calls": total_tools_executed > 0,
                    "manual_tool_execution": True,
                    "tools_executed": total_tools_executed,
                    "conversation_steps": conversation_steps,
                }

        self._log_trajectory(trajectory_dict)

        return {
            "response": response_str,
            "full_content": response_str,
            "usage": total_usage,
            "model": getattr(current_response, "model", None) or self.model_name,
            "tool_calls": None,
            "has_tool_calls": total_tools_executed > 0,
            "manual_tool_execution": True,
            "tools_executed": total_tools_executed,
            "conversation_steps": conversation_steps,
        }
