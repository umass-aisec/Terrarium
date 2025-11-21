"""
SmartGrid Tools Module

Handles tool execution for the SmartGrid environment, specifically
the schedule_task action for power grid task scheduling.
"""

from typing import Dict, List, Any, Optional, Set


class SmartGridTools:
    """
    Handles tool execution for SmartGrid environment.

    Manages the schedule_task action which allows homes to schedule
    power-consuming tasks at specific time slots.
    """

    def __init__(self, blackboard_manager):
        """
        Initialize SmartGridTools.

        Args:
            blackboard_manager: Blackboard manager instance for logging
        """
        self.blackboard_manager = blackboard_manager

    def execute_action(self, agent_name: str, action: Dict[str, Any], log_to_blackboards: bool = True,
                      phase: Optional[str] = None, iteration: Optional[int] = None, env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an agent action (only schedule_task supported).

        Args:
            agent_name: Name of the home performing the action
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
        task_schedules = env_state.get("task_schedules", {})
        agents_list = env_state.get("agents_list", [])
        homes_data = env_state.get("homes_data", {})
        T = env_state.get("T", 24)

        if agent_name not in agents_list:
            return {"status": "failed", "reason": f"Home {agent_name} not found"}

        action_type = action.get("action")
        if action_type != "schedule_task":
            return {"status": "failed", "reason": f"Unknown action type: {action_type}"}

        # Get task and start time from action parameters
        task_id = action.get("task_id")
        start_time = action.get("start_time")

        if task_id is None:
            return {
                "status": "retry",
                "reason": "Missing task_id parameter",
                "suggestions": ["Specify task_id as a string"]
            }

        if start_time is None:
            return {
                "status": "retry",
                "reason": "Missing start_time parameter",
                "suggestions": ["Specify start_time as an integer (0-based time slot)"]
            }

        # Check if task already scheduled
        # Keys are in format "home_id:task_id"
        task_key = f"{agent_name}:{task_id}"

        if task_key in task_schedules:
            return {
                "status": "failed",
                "reason": f"Task {task_id} for home {agent_name} is already scheduled"
            }

        # Validate the task exists and start time is allowed
        if agent_name not in homes_data:
            return {"status": "failed", "reason": f"Home {agent_name} not found in state"}

        home_tasks = homes_data[agent_name].get("tasks", [])
        task_spec = None
        for task in home_tasks:
            if task["id"] == task_id:
                task_spec = task
                break

        if task_spec is None:
            valid_tasks = ", ".join(task["id"] for task in home_tasks)
            return {
                "status": "failed",
                "reason": (
                    f"Task {task_id} not found for home {agent_name}. "
                    f"Valid tasks: {valid_tasks or 'None'}"
                ),
            }

        try:
            start_time = int(start_time)
        except (ValueError, TypeError):
            return {
                "status": "retry",
                "reason": "start_time must be an integer",
                "suggestions": ["Use a valid integer for start_time"]
            }

        if start_time not in task_spec["allowed_starts"]:
            return {
                "status": "retry",
                "reason": f"start_time {start_time} not allowed for task {task_id}",
                "suggestions": [f"Choose from allowed starts: {task_spec['allowed_starts']}"]
            }

        # Make the scheduling (create updated copy)
        updated_schedules = task_schedules.copy()
        updated_schedules[task_key] = start_time

        # Calculate total tasks
        total_tasks = sum(len(home_data["tasks"]) for home_data in homes_data.values())

        # Note: Main grid draw calculation will be done by environment after applying state updates
        result_dict = {
            "home": agent_name,
            "task": {
                "id": task_id,
                "start_time": start_time,
                "duration": task_spec["duration"],
                "consumption": task_spec["consumption"]
            },
            "total_scheduled": len(updated_schedules),
            "remaining_tasks": total_tasks - len(updated_schedules),
            # State updates to be applied by environment
            "state_updates": {
                "task_schedules": updated_schedules
            }
        }

        execution_result = {
            "status": "success",
            "result": result_dict
        }

        # Log action to blackboards if enabled
        if log_to_blackboards and self.blackboard_manager:
            action_type = action.get("action")
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, execution_result, phase, iteration
            )

        return execution_result

    def get_supported_tools(self) -> Set[str]:
        """
        Return set of tool names this environment supports.

        Returns:
            Set of supported tool names
        """
        return {"schedule_task"}

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
        # SmartGrid only has one tool that's actually an action
        if tool_name == "schedule_task":
            task_id = arguments.get("task_id")
            start_time = arguments.get("start_time")

            if task_id is None:
                return {"error": "task_id is required for schedule_task"}
            if start_time is None:
                return {"error": "start_time is required for schedule_task"}

            action = {"action": "schedule_task", "task_id": task_id, "start_time": start_time}
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True,
                                           phase=phase, iteration=iteration, env_state=env_state)

            # Return the full result including status and state_updates
            return env_result

        # All other tools are not supported by this environment
        else:
            return {"error": f"SmartGrid environment does not support tool: {tool_name}"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get environment-specific tools available for a phase.

        Args:
            phase: Current phase ("planning" or "execution")

        Returns:
            List of tool definitions
        """
        if phase == "execution":
            # During execution phase, provide the schedule_task tool
            return [{
                "type": "function",
                "function": {
                    "name": "schedule_task",
                    "description": "Schedule a power-consuming task at a specific start time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The ID of the task to schedule from your task list"
                            },
                            "start_time": {
                                "type": "integer",
                                "description": "The time slot to start the task (0-based index within allowed windows)"
                            }
                        },
                        "required": ["task_id", "start_time"]
                    }
                }
            }]
        # During planning phase, no environment-specific tools
        # Homes use blackboard tools for communication
        return []
