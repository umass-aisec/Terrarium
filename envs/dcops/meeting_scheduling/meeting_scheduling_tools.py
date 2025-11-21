from typing import Dict, List, Any, Optional, Set
from envs.dcops.CoLLAB.MeetingScheduling.data_structure import SLOT_LABELS

class MeetingSchedulingTools:
    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def execute_action(self, agent_name: str, action: Dict[str, Any], log_to_blackboards: bool = True,
                      phase: Optional[str] = None, iteration: Optional[int] = None, env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute an agent action (only schedule_meeting supported).

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
        meetings = env_state.get("meetings", {})
        meeting_schedules = env_state.get("meeting_schedules", {})
        agent_names = env_state.get("agent_names", [])

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        action_type = action.get("action")
        if action_type != "schedule_meeting":
            return {"status": "failed", "reason": f"Unknown action type: {action_type}"}

        # Get meeting and slot from action parameters
        meeting_id = action.get("meeting_id")
        slot = action.get("slot")

        if meeting_id is None:
            return {
                "status": "retry",
                "reason": "Missing meeting_id parameter",
                "suggestions": ["Specify meeting_id as a string"]
            }

        if slot is None:
            return {
                "status": "retry",
                "reason": "Missing slot parameter",
                "suggestions": ["Specify slot as an integer (1-10 for 8:00-17:00)"]
            }

        # Check if meeting already scheduled
        if meeting_id in meeting_schedules:
            return {
                "status": "failed",
                "reason": f"Meeting {meeting_id} is already scheduled"
            }

        # Validate the meeting exists and agent owns it
        if meeting_id not in meetings:
            return {"status": "failed", "reason": f"Meeting {meeting_id} not found"}

        meeting = meetings[meeting_id]
        if meeting.get("owner") != agent_name:
            owned_meetings = sorted(
                mid for mid, data in meetings.items() if data.get("owner") == agent_name
            )
            return {
                "status": "failed",
                "reason": (
                    f"Agent {agent_name} does not own meeting {meeting_id}. "
                    f"Valid meetings for {agent_name}: {', '.join(owned_meetings) or 'None'}"
                ),
            }

        try:
            # 0-based indexing
            slot_int = int(slot) - 1
        except (ValueError, TypeError):
            return {
                "status": "retry",
                "reason": "slot must be an integer",
                "suggestions": ["Use a valid integer for slot (1-10)"]
            }

        if slot_int < 0 or slot_int > 9:
            return {
                "status": "retry",
                "reason": f"slot {slot_int} out of range",
                "suggestions": ["Choose a slot between 1 and 10 (8:00-17:00)"]
            }

        # Update meeting schedules (create updated copy for serialization)
        updated_schedules = meeting_schedules.copy()
        updated_schedules[meeting_id] = slot_int

        total_meetings = len(meetings)

        result_dict = {
            "agent": agent_name,
            "meeting": {
                "id": meeting_id,
                "slot": slot_int,
                "time_label": SLOT_LABELS[slot_int],
                "mode": meeting.get("mode"),
                "location": meeting.get("location"),
                "attendees": meeting.get("attendees", [])
            },
            "total_scheduled": len(updated_schedules),
            "remaining_meetings": total_meetings - len(updated_schedules),
            # State updates to be applied by environment
            "state_updates": {
                "meeting_schedules": updated_schedules
            }
        }
        # Note: global_score will be calculated by environment after applying state updates

        execution_result = {
            "status": "success",
            "result": result_dict
        }

        # Log action to blackboards if enabled and actiosn type is in config
        if log_to_blackboards and self.blackboard_manager:
            action_type = action.get("action")
            # TODO: Rename log to post
            self.blackboard_manager.log_action_to_blackboards(agent_name, action, execution_result, phase, iteration)

        return execution_result

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
        # MeetingScheduling only has one tool that's actually an action
        if tool_name == "schedule_meeting":
            meeting_id = arguments.get("meeting_id")
            slot = arguments.get("slot")

            if meeting_id is None:
                return {"error": "meeting_id is required for schedule_meeting"}
            if slot is None:
                return {"error": "slot is required for schedule_meeting"}

            action = {"action": "schedule_meeting", "meeting_id": meeting_id, "slot": slot}
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True,
                                           phase=phase, iteration=iteration, env_state=env_state)

            # Return the full result including status and state_updates
            return env_result

        # All other tools are not supported by this environment
        else:
            return {"error": f"MeetingScheduling environment does not support tool: {tool_name}"}
