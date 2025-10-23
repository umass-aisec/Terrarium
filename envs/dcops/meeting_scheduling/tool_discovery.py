from typing import Dict, List, Any, Optional, Set
from envs.dcops.CoLLAB.MeetingScheduling.data_structure import SLOT_LABELS

class MeetingSchedulingEnvironmentTools:
    def __init__(self):
        pass
    def get_supported_tools(self) -> Set[str]:
        """Return set of tool names this environment supports."""
        return {"schedule_meeting"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get environment-specific tools available for a phase.

        Args:
            phase: Current phase ("planning" or "execution")

        Returns:
            List of tool definitions
        """
        if phase == "execution":
            # During execution phase, provide the schedule_meeting tool
            return [{
                "type": "function",
                "function": {
                    "name": "schedule_meeting",
                    "description": "Schedule a meeting that you own to a specific time slot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meeting_id": {
                                "type": "string",
                                "description": "The ID of the meeting to schedule (must be owned by you)"
                            },
                            "slot": {
                                "type": "integer",
                                "description": "The time slot to schedule the meeting (1-10 for 8:00-17:00)"
                            }
                        },
                        "required": ["meeting_id", "slot"]
                    }
                }
            }]
        # During planning phase, no environment-specific tools
        # Agents use blackboard tools for communication
        return []