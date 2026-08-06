from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_SOCIAL_ACTIONS = frozenset({"request_move", "complain"})
EXECUTION_PHYSICAL_ACTIONS = frozenset({"move", "settle", "stand"})


class IsThisSeatTakenTools:
    # Agents may open their own private channels (create_channel) during planning.
    supports_private_channels = True

    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tool_names(self) -> Set[str]:
        return set(PLANNING_SOCIAL_ACTIONS) | set(EXECUTION_PHYSICAL_ACTIONS)

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase == "planning":
            return self._planning_tools()
        if phase == "execution":
            return self._execution_tools()
        return []

    def _social_tool_schema(self, name: str, description: str) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Neighboring agent you are addressing.",
                        },
                        "message": {
                            "type": "string",
                            "description": "Optional natural-language line to post in the group chat.",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        }

    def _planning_tools(self) -> List[Dict[str, Any]]:
        return [
            self._social_tool_schema(
                "request_move",
                (
                    "Politely ask a neighboring agent to move. Posts your request to the group chat "
                    "and applies social pressure. Use during planning, not execution."
                ),
            ),
            self._social_tool_schema(
                "complain",
                (
                    "Tell a neighboring agent you need them to move because their presence is "
                    "unacceptable. Posts to the group chat and applies stronger social pressure."
                ),
            ),
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "move",
                    "description": "Move to a specific empty seat. IMPORTANT: seat_id must use the full format like 'seat_2_3' or 'seat_1_1' — never just '2_3' or '1_1'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seat_id": {
                                "type": "string",
                                "description": "Target empty seat identifier.",
                            }
                        },
                        "required": ["seat_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "settle",
                    "description": (
                        "Declare you are satisfied with your current seat. "
                        "The simulation ends when every agent has settled. "
                        "Use when your seat is good enough, especially under time pressure."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stand",
                    "description": "Vacate your current seat. Use only as a last resort when no visible seat is tolerable.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    @staticmethod
    def _default_social_message(
        agent_name: str,
        target_agent: str,
        action_name: str,
    ) -> str:
        if action_name == "complain":
            return (
                f"Hey {target_agent} — I really need you to move. "
                f"I'm not comfortable sitting next to you."
            )
        return (
            f"Hey {target_agent}, would you mind shifting? "
            f"I'd really appreciate the space."
        )

    def _post_social_message(
        self,
        agent_name: str,
        message: str,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> None:
        if self.blackboard_manager is None:
            return

        blackboard_ids = self.blackboard_manager.get_agent_blackboards(agent_name)
        for bb_id in blackboard_ids:
            # request_move/complain are public acts; keep them out of side channels.
            if self.blackboard_manager.is_private_channel(bb_id):
                continue
            try:
                self.blackboard_manager.post(
                    int(bb_id),
                    agent_name,
                    "communication",
                    {"content": message},
                    phase=phase,
                    iteration=iteration,
                )
            except Exception:
                continue

    def execute_action(
        self,
        agent_name: str,
        action: Dict[str, Any],
        log_to_blackboards: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.environment is None:
            return {"status": "failed", "reason": "Environment not initialized for tools"}

        tool_name = action.get("action")
        if tool_name in PLANNING_SOCIAL_ACTIONS:
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": f"{tool_name} is only available during planning",
                }
            result = self.environment.apply_planning_social_action(
                agent_name=agent_name,
                action_name=str(tool_name),
                arguments=action,
            )
            if result.get("status") == "success":
                target_agent = str(action.get("agent_id", ""))
                message = str(action.get("message") or "").strip()
                if not message:
                    message = self._default_social_message(agent_name, target_agent, str(tool_name))
                self._post_social_message(agent_name, message, phase, iteration)
            return result

        if tool_name not in EXECUTION_PHYSICAL_ACTIONS:
            return {"error": f"Unknown action type: {tool_name}"}

        result = self.environment._apply_agent_action(
            agent_name=agent_name,
            action_name=str(tool_name),
            arguments=action,
        )

        if log_to_blackboards and self.blackboard_manager:
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, result, phase, iteration
            )

        return result

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if tool_name not in self.get_tool_names():
            return {
                "error": f"IsThisSeatTaken environment does not support tool: {tool_name}"
            }

        action = {"action": tool_name, **dict(arguments or {})}
        return self.execute_action(
            agent_name,
            action,
            log_to_blackboards=True,
            phase=phase,
            iteration=iteration,
        )
