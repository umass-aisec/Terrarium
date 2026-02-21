from typing import Dict, List, Any, Optional, Set


class JiraTicketTools:
    """Tools for JiraTicket task assignment."""

    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tool_names(self) -> Set[str]:
        return {"assign_task"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase != "execution":
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": "assign_task",
                    "description": "Claim a task id (or use 'skip' to skip).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task id from your task list (or 'skip').",
                            }
                        },
                        "required": ["task_id"],
                    },
                },
            }
        ]

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

        tasks = self.environment.tasks
        assignment = self.environment.assignment
        agent_names = self.environment.agent_names

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        if action.get("action") != "assign_task":
            return {"status": "failed", "reason": f"Unknown action type: {action.get('action')}"}

        if agent_name in assignment:
            return {"status": "failed", "reason": f"Agent {agent_name} already assigned"}

        task_id = action.get("task_id")
        if task_id is None:
            return {"status": "retry", "reason": "task_id is required"}

        task_id = str(task_id)
        if task_id.lower() in {"idle", "none"}:
            return {
                "status": "retry",
                "reason": "Use task_id='skip' to skip (only 'skip' and valid task ids are accepted).",
                "suggestions": ["Use task_id='skip' or choose a task_id from your TASKS list."],
            }
        if task_id.lower() == "skip":
            self.environment.assignment[agent_name] = None
            result = {
                "agent": agent_name,
                "task_id": None,
                "status": "skip",
                "total_assigned": len(self.environment.assignment),
                "remaining_agents": len(agent_names) - len(self.environment.assignment),
                "joint_reward": self.environment.joint_reward(self.environment.assignment),
            }
            execution_result = {"status": "success", "result": result}
        else:
            if task_id not in tasks:
                return {
                    "status": "retry",
                    "reason": f"Unknown task_id {task_id}",
                    "suggestions": ["Choose a task_id from your TASKS list or use 'skip'."],
                }

            self.environment.assignment[agent_name] = task_id
            result = {
                "agent": agent_name,
                "task_id": task_id,
                "task": tasks.get(task_id, {}),
                "total_assigned": len(self.environment.assignment),
                "remaining_agents": len(agent_names) - len(self.environment.assignment),
                "joint_reward": self.environment.joint_reward(self.environment.assignment),
            }
            execution_result = {"status": "success", "result": result}

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
    ) -> Dict[str, Any]:
        if tool_name != "assign_task":
            return {"error": f"JiraTicket environment does not support tool: {tool_name}"}

        task_id = arguments.get("task_id")
        if task_id is None:
            return {"error": "task_id is required for assign_task"}

        action = {"action": "assign_task", "task_id": task_id}
        return self.execute_action(
            agent_name,
            action,
            log_to_blackboards=True,
            phase=phase,
            iteration=iteration,
        )
