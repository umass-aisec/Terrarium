"""
SmartGrid Tools Module (CoLLAB v2)

Handles tool execution for the SmartGrid environment, specifically the
assign_source action for assigning machines to shared renewable sources.
"""

from typing import Dict, List, Any, Optional, Set


class SmartGridTools:
    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tool_names(self) -> Set[str]:
        return {"assign_source"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase == "execution":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "assign_source",
                        "description": "Assign one of your machines to a renewable source you can use.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "machine_id": {
                                    "type": "string",
                                    "description": "ID of a machine you own (e.g., machine_001).",
                                },
                                "source_id": {
                                    "type": "string",
                                    "description": "Renewable source ID from your available sources list.",
                                },
                            },
                            "required": ["machine_id", "source_id"],
                        },
                    },
                }
            ]
        return []

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

        machines: Dict[str, Any] = {}
        for machine_id, machine in self.environment.instance.machines.items():
            var_spec = self.environment.problem.variables.get(machine_id)
            machines[machine_id] = {
                "owner": machine.owner,
                "label": machine.label,
                "start": machine.start,
                "end": machine.end,
                "power": float(self.environment.instance.machine_powers.get(machine_id, 0.0)),
                "allowed_sources": list(var_spec.domain) if var_spec else [],
            }

        assignment = self.environment.assignment
        agent_names = self.environment.agent_names

        if agent_name not in agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        if action.get("action") != "assign_source":
            return {"status": "failed", "reason": f"Unknown action type: {action.get('action')}"}

        machine_id = action.get("machine_id")
        source_id = action.get("source_id")
        if machine_id is None:
            return {"status": "retry", "reason": "machine_id is required"}
        if source_id is None:
            return {"status": "retry", "reason": "source_id is required"}

        if machine_id not in machines:
            return {"status": "failed", "reason": f"Machine {machine_id} not found"}

        machine = machines[machine_id]
        if machine.get("owner") != agent_name:
            owned = sorted(mid for mid, m in machines.items() if m.get("owner") == agent_name)
            return {
                "status": "failed",
                "reason": f"Machine {machine_id} is not owned by {agent_name}. Owned machines: {', '.join(owned) or 'None'}",
            }

        if machine_id in assignment:
            return {"status": "failed", "reason": f"Machine {machine_id} already assigned"}

        allowed_sources = machine.get("allowed_sources", [])
        if source_id not in allowed_sources:
            return {
                "status": "retry",
                "reason": f"source_id {source_id} not allowed for machine {machine_id}",
                "suggestions": [f"Choose from: {allowed_sources}"],
            }

        self.environment.assignment[machine_id] = source_id

        total_vars = len(self.environment.problem.variables)
        total_assigned = len(self.environment.assignment)
        result_dict = {
            "agent": agent_name,
            "machine": {
                "id": machine_id,
                "label": machine.get("label"),
                "window": [machine.get("start"), machine.get("end")],
                "power": machine.get("power"),
            },
            "source_id": source_id,
            "total_assigned": total_assigned,
            "remaining_machines": total_vars - total_assigned,
            "joint_reward": self.environment.joint_reward(self.environment.assignment),
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
    ) -> Dict[str, Any]:
        if tool_name != "assign_source":
            return {"error": f"SmartGrid environment does not support tool: {tool_name}"}

        machine_id = arguments.get("machine_id")
        source_id = arguments.get("source_id")
        if machine_id is None:
            return {"error": "machine_id is required for assign_source"}
        if source_id is None:
            return {"error": "source_id is required for assign_source"}

        action = {"action": "assign_source", "machine_id": machine_id, "source_id": source_id}
        return self.execute_action(
            agent_name,
            action,
            log_to_blackboards=True,
            phase=phase,
            iteration=iteration,
        )
