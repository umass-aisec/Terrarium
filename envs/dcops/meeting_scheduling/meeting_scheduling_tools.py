from typing import Dict, List, Any, Optional, Set


class MeetingSchedulingTools:
    """
    CoLLAB v2 MeetingScheduling tools.

    The execution action is `attend_meeting`, which sets an agent's attendance
    interval for a specific meeting they participate in.
    """

    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def get_tool_names(self) -> Set[str]:
        return {"attend_meeting"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase == "execution":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "attend_meeting",
                        "description": "Choose your attendance interval for a meeting you participate in.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "meeting_id": {
                                    "type": "string",
                                    "description": "Meeting ID from your meeting list (e.g., m001).",
                                },
                                "interval": {
                                    "type": "string",
                                    "description": "Attendance interval within the meeting window. Use 'skip' or 'join-leave' (e.g., '3-5').",
                                },
                            },
                            "required": ["meeting_id", "interval"],
                        },
                    },
                }
            ]
        return []

    @staticmethod
    def _allowed_intervals(start: int, end: int, meeting_type: str) -> Set[str]:
        allowed: Set[str] = {"skip"}
        for join in range(start, end):
            for leave in range(join + 1, end + 1):
                allowed.add(f"{join}-{leave}")
            if meeting_type == "strict":
                break
        return allowed

    def execute_action(
        self,
        agent_name: str,
        action: Dict[str, Any],
        log_to_blackboards: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not env_state:
            return {"status": "failed", "reason": "Environment state not provided"}

        meetings = env_state.get("meetings", {})
        attendance = env_state.get("attendance", {})
        agent_names = env_state.get("agent_names", [])

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        if action.get("action") != "attend_meeting":
            return {"status": "failed", "reason": f"Unknown action type: {action.get('action')}"}

        meeting_id = action.get("meeting_id")
        interval = action.get("interval")
        if meeting_id is None:
            return {"status": "retry", "reason": "meeting_id is required"}
        if interval is None:
            return {"status": "retry", "reason": "interval is required"}

        if meeting_id not in meetings:
            return {"status": "failed", "reason": f"Meeting {meeting_id} not found"}

        meeting = meetings[meeting_id]
        participants = meeting.get("participants", [])
        if agent_name not in participants:
            return {
                "status": "failed",
                "reason": f"Agent {agent_name} is not a participant of meeting {meeting_id}",
            }

        var_name = f"{agent_name}__{meeting_id}"
        if var_name in attendance:
            return {"status": "failed", "reason": f"Attendance for {meeting_id} already set"}

        try:
            start = int(meeting.get("start", 0))
            end = int(meeting.get("end", 0))
        except Exception:
            return {"status": "failed", "reason": "Invalid meeting window in state"}

        meeting_type = str(meeting.get("meeting_type", "soft"))
        allowed = self._allowed_intervals(start, end, meeting_type)
        if interval not in allowed:
            sample = ", ".join(sorted(list(allowed))[:5])
            return {
                "status": "retry",
                "reason": f"interval '{interval}' not allowed for meeting window [{start},{end})",
                "suggestions": [f"Choose from allowed intervals like: {sample}"],
            }

        updated_attendance = dict(attendance)
        updated_attendance[var_name] = interval

        total_vars = env_state.get("total_variables")
        if total_vars is None:
            try:
                total_vars = sum(len(m.get("participants", [])) for m in meetings.values())
            except Exception:
                total_vars = len(updated_attendance)

        result_dict = {
            "agent": agent_name,
            "meeting": {
                "id": meeting_id,
                "title": meeting.get("title"),
                "meeting_type": meeting_type,
                "window": [start, end],
                "participants": participants,
            },
            "interval": interval,
            "total_assigned": len(updated_attendance),
            "remaining_variables": int(total_vars) - len(updated_attendance),
            "state_updates": {"attendance": updated_attendance},
        }

        execution_result = {"status": "success", "result": result_dict}

        if log_to_blackboards and self.blackboard_manager:
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, execution_result, phase, iteration
            )

        return execution_result

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if tool_name != "attend_meeting":
            return {"error": f"MeetingScheduling environment does not support tool: {tool_name}"}

        meeting_id = arguments.get("meeting_id")
        interval = arguments.get("interval")
        if meeting_id is None:
            return {"error": "meeting_id is required for attend_meeting"}
        if interval is None:
            return {"error": "interval is required for attend_meeting"}

        action = {"action": "attend_meeting", "meeting_id": meeting_id, "interval": interval}
        return self.execute_action(
            agent_name,
            action,
            log_to_blackboards=True,
            phase=phase,
            iteration=iteration,
            env_state=env_state,
        )

