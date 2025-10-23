from typing import Dict, List, Any, Optional, Set

class SmartGridEnvironmentTools:
    def __init__(self):
        pass

    def get_supported_tools(self) -> Set[str]:
        """Return set of tool names this environment supports."""
        return {"schedule_task"}

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
