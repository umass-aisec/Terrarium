from typing import Dict, List, Any, Optional, Set

class PersonalAssistantTools:
    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def execute_action(self, agent_name: str, action: Dict[str, Any], log_to_blackboards: bool = True,
                      phase: Optional[str] = None, iteration: Optional[int] = None, env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an agent action (only choose_outfit supported).

        Args:
            agent_name: Name of the agent performing the action
            action: Dictionary containing action type and parameters
            log_to_blackboards: Whether to log the action to blackboards (default: True)
            phase: Current simulation phase (planning/execution)
            iteration: Current iteration number
            env_state: Serializable environment state dictionary

        Returns:
            Dictionary with execution result, status, and state updates
        """
        if not env_state:
            return {"status": "failed", "reason": "Environment state not provided"}

        # Extract state from serializable dict
        outfit_selections = env_state.get("outfit_selections", {})
        agent_names = env_state.get("agent_names", [])
        wardrobe_options = env_state.get("wardrobe_options", {})
        max_possible_score = env_state.get("max_possible_score", 0)

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        action_type = action.get("action")
        if action_type != "choose_outfit":
            return {"status": "failed", "reason": f"Unknown action type: {action_type}"}

        # Check if agent already made a selection
        if agent_name in outfit_selections:
            return {
                "status": "failed",
                "reason": f"Agent {agent_name} has already chosen an outfit"
            }

        # Get outfit number from action parameters
        outfit_number = action.get("outfit_number")
        if outfit_number is None:
            return {
                "status": "retry",
                "reason": "Missing outfit_number parameter",
                "suggestions": ["Specify outfit_number as an integer (1-based index)"]
            }

        try:
            outfit_idx = int(outfit_number) - 1  # Convert to 0-based index
        except (ValueError, TypeError):
            return {
                "status": "retry",
                "reason": "outfit_number must be an integer",
                "suggestions": ["Use a valid integer for outfit_number"]
            }

        # Get agent's wardrobe options from serialized state
        if agent_name not in wardrobe_options:
            return {"status": "failed", "reason": f"No wardrobe found for agent {agent_name}"}

        agent_wardrobe = wardrobe_options[agent_name]
        if outfit_idx < 0 or outfit_idx >= len(agent_wardrobe):
            valid_options = ", ".join(
                f"{idx + 1}: {opt['article']} ({opt['color']})"
                for idx, opt in enumerate(agent_wardrobe)
            )
            return {
                "status": "retry",
                "reason": (
                    f"outfit_number out of range (1-{len(agent_wardrobe)}). "
                    f"Valid outfits: {valid_options or 'None'}"
                ),
                "suggestions": [f"Choose a number between 1 and {len(agent_wardrobe)}"]
            }

        # Make the selection (create updated copy for serialization)
        selected_outfit = agent_wardrobe[outfit_idx]
        updated_selections = outfit_selections.copy()
        updated_selections[agent_name] = selected_outfit

        # Note: Score calculation will be done by environment after applying state updates
        # We just track the selection count here
        result_dict = {
            "agent": agent_name,
            "outfit": {
                "number": outfit_number,
                "article": selected_outfit["article"],
                "color": selected_outfit["color"]
            },
            "total_selections": len(updated_selections),
            "remaining_agents": len(agent_names) - len(updated_selections),
            # State updates to be applied by environment
            "state_updates": {
                "outfit_selections": updated_selections
            }
        }

        execution_result = {
            "status": "success",
            "result": result_dict
        }

        # Log action to blackboards if enabled and action type is in config
        if log_to_blackboards and self.blackboard_manager:
            action_type = action.get("action")
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, execution_result, phase, iteration
            )

        return execution_result

    def get_supported_tools(self) -> Set[str]:
        """Return set of tool names this environment supports."""
        return {"choose_outfit"}

    def handle_tool_call(self, tool_name: str, agent_name: str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None, env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle tool calls by routing to execute_action.

        Args:
            tool_name: Name of the tool to execute
            agent_name: Name of the agent calling the tool
            arguments: Parameters for the tool call
            phase: Current simulation phase (planning/execution)
            iteration: Current iteration number
            env_state: Serializable environment state dictionary

        Returns:
            Dictionary with tool execution result and state updates
        """
        # PersonalAssistant only has one tool that's actually an action
        if tool_name == "choose_outfit":
            outfit_number = arguments.get("outfit_number")

            if outfit_number is None:
                return {"error": "outfit_number is required for choose_outfit"}

            action = {"action": "choose_outfit", "outfit_number": outfit_number}
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True,
                                           phase=phase, iteration=iteration, env_state=env_state)

            # Return the full result including status and state_updates
            return env_result

        # All other tools are not supported by this environment
        else:
            return {"error": f"PersonalAssistant environment does not support tool: {tool_name}"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get environment-specific tools available for a phase.

        Args:
            phase: Current phase ("planning" or "execution")

        Returns:
            List of tool definitions
        """
        if phase == "execution":
            # During execution phase, provide the choose_outfit tool
            return [{
                "type": "function",
                "function": {
                    "name": "choose_outfit",
                    "description": "Choose your final outfit from your wardrobe options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "outfit_number": {
                                "type": "integer",
                                "description": "The number of the outfit to choose (1-based index from your wardrobe)"
                            }
                        },
                        "required": ["outfit_number"]
                    }
                }
            }]
        # During planning phase, no environment-specific tools
        # Agents use blackboard tools for communication
        return []
