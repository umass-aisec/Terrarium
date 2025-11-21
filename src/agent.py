import time
from typing import Dict, Optional, Any
from llm_server.clients.abstract_client import AbstractClient
from dotenv import load_dotenv
import traceback
from .toolset_discovery import ToolsetDiscovery

class Agent:
    """
    Agent-dependent methods for LLM agents.
    """

    def __init__(self, client: AbstractClient, name, model_name: str = "", max_conversation_steps: int = 3,
                 tool_logger: Optional[Any] = None, trajectory_logger: Optional[Any] = None, environment_name: str = ""):
        """
        Initialize an Agent.

        Args:
            client: LLM client instance
            name: Agent name
            model_name: OpenAI model name (e.g., "gpt-4o", "o1-preview")
            max_conversation_steps: Max conversation turns for multi-step tool execution (default: 3)
            tool_logger: Logger for tool call tracking
            trajectory_logger: Logger for agent reasoning trajectories
            environment_name: Name of the environment this agent operates in (e.g., "MeetingScheduling")
        """
        load_dotenv()
        self.name = name
        self.model_name = model_name
        self.max_conversation_steps = max_conversation_steps
        self.tool_logger = tool_logger
        self.trajectory_logger = trajectory_logger
        self.environment_name = environment_name  # Store environment name for tool discovery

        # Agent context for logging (set via set_agent_context)
        self.current_agent_name = None
        self.current_phase = None
        self.current_iteration = None
        self.current_round = None
        self.toolset_discovery = ToolsetDiscovery()
        # Attack configurations for message replacement (set via set_attack_config)
        self.attack_configs = {
            "poisoning": {"agent": None, "config": {}},
            "adversarial_agent": {"agent": None, "config": {}},
            "context_overflow": {"agent": None, "config": {}}
        }

        # An already instantiated client
        self.client = client

    
    def set_meta_context(self, agent_name: str, phase: Optional[str] = None, iteration: Optional[int] = None, round_num: Optional[int] = None):
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

    def set_attack_config(self, attack_agent: str, config: dict):
        """
        Set attack agent configuration (supports poisoning and adversarial_agent attacks).

        Args:
            attack_agent: Agent name or list of agent names that will have messages replaced
            config: Attack configuration dictionary with keys: "poisoning" or "adversarial_agent"
        """
        # Detect attack type and store configuration
        for attack_type in ["poisoning", "adversarial_agent", "context_overflow"]:
            if attack_type in config:
                self.attack_configs[attack_type]["agent"] = attack_agent
                self.attack_configs[attack_type]["config"] = config[attack_type]

                agent_desc = f"(ALL): {attack_agent}" if isinstance(attack_agent, list) else f": {attack_agent}"
                print(f"OpenAI client configured with {attack_type} agent{agent_desc}")
                return

        print(f"Warning: No valid attack configuration found in config")


    async def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call by delegating to the appropriate handler.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        start_time = time.time()

        try:
            # Ensure environment and blackboard_manager exist
            if not self.communication_protocol:
                return {"error": "No communication protocol available"}

            # Ensure agent context is set
            if not self.current_agent_name:
                return {"error": "Agent context not set - call set_agent_context first"}

            env_name_normalized = (self.environment_name or "").lower()
            available_env_tools = {
                tool.get("function", {}).get("name")
                for tool in self.toolset_discovery.get_tools_for_environment(env_name_normalized, self.current_phase)
            }
            supported_env_tools = self.toolset_discovery.get_supported_tools_for_environment(env_name_normalized)

            # Check blackboard tools first
            if tool_name in self.toolset_discovery.get_supported_tools_for_blackboard():
                result = await self.communication_protocol.blackboard_handle_tool_call(tool_name, self.current_agent_name, arguments,
                                                            phase=self.current_phase, iteration=self.current_iteration)
            # Then check environment tools (normalize environment name to lowercase)
            elif tool_name in available_env_tools:
                result = await self.communication_protocol.environment_handle_tool_call(tool_name, self.current_agent_name, arguments,
                                                          phase=self.current_phase, iteration=self.current_iteration)
            elif tool_name in supported_env_tools:
                result = {"error": f"Tool '{tool_name}' is not available during the {self.current_phase or 'current'} phase."}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Log the tool call if logger is available
            duration_ms = (time.time() - start_time) * 1000
            if self.tool_logger and self.current_agent_name:
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

            return result

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            error_result = {"error": error_msg}

            # Log the failed tool call if logger is available
            duration_ms = (time.time() - start_time) * 1000
            if self.tool_logger and self.current_agent_name:
                self.tool_logger.log_tool_call(
                    agent_name=self.current_agent_name,
                    phase=self.current_phase or "unknown",
                    tool_name=tool_name,
                    parameters=arguments,
                    result=error_result,
                    iteration=self.current_iteration,
                    round_num=self.current_round,
                    duration_ms=duration_ms
                )

            print(f"ERROR: {error_msg}")
            return error_result



    async def generate_response(self, system_prompt: str, user_prompt: str,
                                   max_tokens: int = 4000, temperature: float = 0.7,
                                   reasoning_effort: str = "low", verbosity: str = "low") -> Dict[str, Any]:
        """
        Generate a response from OpenAI API with optional tool execution.

        Supports two modes:
        1. With tools: Uses responses.create() API with multi-step tool execution
        2. Without tools: Uses chat.completions.create() for simple text generation

        Args:
            system_prompt: System prompt for the agent
            user_prompt: User prompt with context and query
            mcp_tools: Tool definitions (enables multi-step tool execution if provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored for reasoning/restricted models)
            reasoning_effort: Reasoning effort level for GPT-5 models
            verbosity: Response verbosity for GPT-5 models

        Returns:
            Dict with keys: response, thinking, usage, model, has_tool_calls, etc.
        """
        # Get tools for this environment and phase (normalize environment name to lowercase)
        tool_set = self.toolset_discovery.get_tools_for_environment(self.environment_name.lower(), self.current_phase) + self.toolset_discovery.get_tools_for_blackboard(self.current_phase)
        try:
            # Prepare API call parameters
            params = {
                "model": self.model_name,
                "max_completion_tokens": max_tokens,
                "max_output_tokens": 1024,
                "tools": tool_set if tool_set else [],
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "verbosity": verbosity
            }

            # Get system and user prompt into Reponses API format
            context = self.client.init_context(system_prompt, user_prompt)
            # Make API call with or without tools
            if tool_set:
                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                # Get a response and update context for next conversation step
                response, response_str = self.client.generate_response(
                    input=context,
                    params=params
                )
                total_usage = self.client.get_usage(response, total_usage)
                current_response = response
                total_tools_executed = 0

                # Initialize trajectory tracking as dict
                trajectory_dict = {}
                has_reasoning_trace = False

                for step in range(self.max_conversation_steps):
                    # Process tool calls and extract content
                    tool_calls_executed, context, step_tools = await self.client.process_tool_calls(
                        current_response, context, self._execute_tool_call
                    )

                    # Extract text reasoning from current response for trajectory
                    reasoning_trace = self.client.extract_reasoning_trace(current_response)
                    # Fall back to regular content if no explicit reasoning payload
                    step_reasoning = reasoning_trace or self.client._extract_message_content(current_response)
                    if reasoning_trace:
                        has_reasoning_trace = True

                    # Add trajectory step to dict if there was any activity
                    if step_tools or step_reasoning:
                        step_key = f"step_{step + 1}"
                        trajectory_dict[step_key] = {
                            "tools": step_tools,
                            "reasoning": step_reasoning
                        }

                    # If no tool calls were made, we're done
                    total_tools_executed += tool_calls_executed

                    # If this is the last allowed step, don't continue
                    if step >= self.max_conversation_steps - 1:
                        break

                    # Continue the conversation using the accumulated context
                    try:
                        # Get a response and update context for next conversation step
                        response, response_str = self.client.generate_response(
                            input=context,
                            params=params
                        )
                        total_usage = self.client.get_usage(response, total_usage)
                        current_response = response  # Update current_response for next iteration


                    except Exception as e:
                        print(f"ERROR: Failed to continue conversation at step {step + 1}: {e}")

                        # Log trajectory even if conversation failed
                        if self.trajectory_logger and self.current_agent_name and trajectory_dict:
                            self.trajectory_logger.log_trajectory(
                                agent_name=self.current_agent_name,
                                iteration=self.current_iteration or 0,
                                phase=self.current_phase or "unknown",
                                trajectory_dict=trajectory_dict,
                                round_num=self.current_round
                            )

                        return {
                            "response": f"Tool execution completed but failed to continue conversation at step {step + 1}: {e}",
                            "thinking": "",
                            "full_content": response_str,
                            "usage": total_usage,
                            "model": self.model_name,
                            "reasoning_available": has_reasoning_trace,
                            "tool_calls": None,
                            "has_tool_calls": False,
                            "manual_tool_execution": True,
                            "tools_executed": total_tools_executed,
                            "conversation_steps": step + 1
                        }

                # Log trajectory if trajectory logger is available
                if self.trajectory_logger and self.current_agent_name and trajectory_dict:
                    self.trajectory_logger.log_trajectory(
                        agent_name=self.current_agent_name,
                        iteration=self.current_iteration or 0,
                        phase=self.current_phase or "unknown",
                        trajectory_dict=trajectory_dict,
                        round_num=self.current_round
                    )

                # Return final response with accumulated stats
                return {
                    "response": response_str,
                    "thinking": "",
                    "full_content": response_str,
                    "usage": total_usage,
                    "model": getattr(current_response, 'model', None) or self.model_name,
                    "reasoning_available": has_reasoning_trace,
                    "tool_calls": None,
                    "has_tool_calls": total_tools_executed > 0,
                    "manual_tool_execution": True,
                    "tools_executed": total_tools_executed,
                    "conversation_steps": step + 1 if total_tools_executed > 0 else 1
                }
            else:
                raise NotImplementedError("Agents SHOULD have access to tools for tool-based execution, otherwise they can't interact with agents ")

        except Exception as e:
            print(f"Error generating agent response: {e}")
            raise

    def stop_server(self):
        """No-op for compatibility with other client interfaces."""
        pass
    
    async def generate_agent_response(self, agent_name: str, agent_context: Dict[str, Any],
                              blackboard_context: Dict[str, str], prompts: Any = None, communication_protocol: Any = None,
                               phase: Optional[str] = None,
                              iteration: Optional[int] = None, round_num: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a response for a specific agent with full context.

        Args:
            agent_name: Name of the agent
            agent_context: Agent's private context (budget, inventory, utilities)
            blackboard_context: Recent blackboard activity context
            system_prompt: Base system prompt for the agent (optional if environment provided)
            user_prompt: Optional user prompt (if not provided, will be generated from environment)
            mcp_tools: Optional MCP tools for function calling
            environment: Environment instance for generating prompts (if prompts not directly provided)
            phase: Current phase (for backend context setting)
            iteration: Current iteration (for backend context setting)
            round_num: Planning round number if applicable (for backend context setting)

        Returns:
            Dictionary containing agent's response and thinking, or None if error
        """

        try:
            self.set_meta_context(
                agent_name=agent_name,
                phase=phase,
                iteration=iteration,
                round_num=round_num
            )

            # Get prompts from environment if not directly provided
            system_prompt = prompts.get_system_prompt()
            
            user_prompt = prompts.get_user_prompt(
                agent_name=agent_name,
                agent_context=agent_context,
                blackboard_context=blackboard_context
            )
            self.communication_protocol = communication_protocol
            # Validate we have prompts
            if system_prompt is None or user_prompt is None:
                raise ValueError("System prompt and user prompt must be provided either directly or via environment")
            #self.tools = tools
            self.prompts = prompts
            # For OpenAI clients, pass MCP tools if available
            return await self.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                reasoning_effort="low",
                verbosity="low"
            )
        except Exception as e:
            print(f"      Error getting LLM response: {e}")
            print(f"      Traceback: {traceback.format_exc()}")
            return None
