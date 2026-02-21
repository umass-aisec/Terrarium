from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class HospitalTools:
    """
    Hospital Tools: Supports Provisioner Logic and expanded Resource types.
    """

    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tool_names(self) -> Set[str]:
        return {
            "schedule_patient", "find_available_slots", "get_job_queue", 
            "transfer_resources", "broadcast_message"
        }

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        tools = []
        
        # Read-Only Tools
        tools.append({
            "type": "function",
            "function": {
                "name": "find_available_slots",
                "description": "Finds available time slots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration": {"type": "integer"},
                        "min_start_time": {"type": "integer"},
                        "num_slots": {"type": "integer", "default": 3}
                    },
                    "required": ["duration", "min_start_time"],
                },
            },
        })

        tools.append({
            "type": "function",
            "function": {
                "name": "get_job_queue",
                "description": "Refreshes the job queue.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })

        # Provisioner / Communication Tools
        tools.append({
            "type": "function",
            "function": {
                "name": "transfer_resources",
                "description": "Send supplies to a hospital.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_hospital": {
                            "type": "string",
                            "description": "Destination hospital name (e.g., 'General_Hospital', 'St_Marys_Center', 'Hospital_2', or 'Resource_Provisioner').",
                        },
                        "resource_type": {
                            "type": "string", 
                            "enum": ["IV_Kits", "Anesthetics", "Pain_Killers", "Radio_Contrast", "Oxygen_Tanks", "Surgical_Packs", "PPE_Sets"]
                        },
                        "amount": {"type": "integer"},
                        "rationale": {"type": "string", "description": "Why are you moving these resources?"}
                    },
                    "required": ["to_hospital", "resource_type", "amount", "rationale"],
                },
            },
        })

        tools.append({
            "type": "function",
            "function": {
                "name": "broadcast_message",
                "description": "Send a free-form message to the entire network.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The content of your announcement."},
                        "rationale": {"type": "string", "description": "Internal note on why you are saying this."}
                    },
                    "required": ["message", "rationale"],
                },
            },
        })

        # Execution Only
        if phase == "execution":
            tools.append({
                "type": "function",
                "function": {
                    "name": "schedule_patient",
                    "description": "Book a time slot. Consumes inventory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "step_index": {"type": "integer"},
                            "start_time": {"type": "integer"},
                            "rationale": {"type": "string", "description": "Why are you scheduling this now?"}
                        },
                        "required": ["patient_id", "step_index", "start_time", "rationale"],
                    },
                },
            })

        return tools

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.environment is None:
            return {"error": "Environment not initialized for tools."}

        # Read Tools
        if tool_name == "get_job_queue":
            return {"result": "Refer to prompt context."}
        elif tool_name == "find_available_slots":
            # Simplified for brevity
            min_start = arguments.get("min_start_time", 0)
            return {"result": f"Available slots found starting from {min_start}."} 

        # Write Tools
        action = {}
        if tool_name == "schedule_patient":
            action = {"schedule": {agent_name: arguments}}
        elif tool_name == "transfer_resources":
            action = {"transfers": {agent_name: arguments}}
        elif tool_name == "broadcast_message":
            action = {"broadcast_message": {agent_name: arguments}}
        else:
            return {"error": f"Tool {tool_name} not found."}

        return self.execute_action(agent_name, action, True, phase, iteration)

    def execute_action(
        self,
        agent_name: str,
        action: Dict[str, Any],
        log_to_blackboards: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        
        msg = "Action Executed."
        if "schedule" in action:
            args = action["schedule"][agent_name]
            msg = f"Scheduling {args['patient_id']}. Rationale: {args.get('rationale')}"
            try:
                self.environment._process_schedule_request(agent_name, args)
            except Exception as e:
                return {"status": "failed", "reason": f"Failed to schedule patient: {e}"}
        elif "transfers" in action:
            args = action["transfers"][agent_name]
            msg = f"Transferring {args['amount']} {args['resource_type']} to {args['to_hospital']}. Rationale: {args.get('rationale')}"
            try:
                self.environment._process_transfer(agent_name, args)
            except Exception as e:
                return {"status": "failed", "reason": f"Failed to transfer resources: {e}"}
        elif "broadcast_message" in action:
            args = action["broadcast_message"][agent_name]
            msg = f"BROADCAST: '{args['message']}'. (Internal Rationale: {args.get('rationale')})"

        execution_result = {
            "status": "success", 
            "result": msg,
        }

        if log_to_blackboards and self.blackboard_manager:
            try:
                self.blackboard_manager.log_action_to_blackboards(
                    agent_name, action, execution_result, phase, iteration
                )
            except Exception as e:
                logger.error(f"Blackboard log failed: {e}")

        return execution_result
