from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class HospitalTools:
    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def get_tool_names(self) -> Set[str]:
        return {"schedule_patient", "find_available_slots", "get_job_queue", "transfer_resources"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        tools = []
        
        # --- READ-ONLY TOOLS ---
        tools.append({
            "type": "function",
            "function": {
                "name": "find_available_slots",
                "description": "Finds the earliest available time slots for a job.",
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
                "description": "Refreshes the list of patients waiting for this department.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })

        # --- ACTION TOOLS ---
        if phase == "execution":
            tools.append({
                "type": "function",
                "function": {
                    "name": "schedule_patient",
                    "description": "Book a specific time slot for a patient. Fails if capacity is exceeded.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "step_index": {"type": "integer"},
                            "start_time": {"type": "integer"},
                        },
                        "required": ["patient_id", "step_index", "start_time"],
                    },
                },
            })

            tools.append({
                "type": "function",
                "function": {
                    "name": "transfer_resources",
                    "description": "Transfer medical supplies between hospitals.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_hospital": {"type": "string", "enum": ["General_Hospital", "St_Marys_Center"]},
                            "to_hospital": {"type": "string", "enum": ["General_Hospital", "St_Marys_Center"]},
                            "resource_type": {"type": "string", "enum": ["IV_Kits", "Anesthetics", "Pain_Killers", "Radio_Contrast"]},
                            "amount": {"type": "integer"},
                        },
                        "required": ["from_hospital", "to_hospital", "resource_type", "amount"],
                    },
                },
            })

        return tools

    def handle_tool_call(self, tool_name: str, agent_name: str, arguments: Dict[str, Any], 
                        phase: Optional[str] = None, iteration: Optional[int] = None, 
                        env_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        if not env_state: return {"error": "Environment state missing."}

        if tool_name == "get_job_queue":
            return {"result": "Please refer to the 'JOB QUEUE' in your prompt context."}

        elif tool_name == "find_available_slots":
            duration = arguments.get("duration")
            min_start = arguments.get("min_start_time", 0)
            limit = arguments.get("num_slots", 3)
            schedule = env_state.get("schedule", {}).get(agent_name, {})
            agents_info = env_state.get("agents", {})
            my_capacity = agents_info.get(agent_name, {}).get("capacity", 1) if agent_name in agents_info else 1

            valid_starts = []
            for t in range(min_start, 168):
                if len(valid_starts) >= limit: break
                if t + duration > 168: continue
                fits = True
                for h in range(t, t + duration):
                    slot_occupancy = schedule.get(h, schedule.get(str(h), []))
                    if len(slot_occupancy) >= my_capacity:
                        fits = False; break
                if fits: valid_starts.append(t)
            
            if not valid_starts: return {"result": "No available slots."}
            return {"result": f"Available Start Times: {valid_starts}"}

        elif tool_name == "schedule_patient":
            action = {
                "schedule": {
                    agent_name: {
                        "patient_id": arguments.get("patient_id"),
                        "step_index": arguments.get("step_index"),
                        "start_time": arguments.get("start_time")
                    }
                }
            }
            return self.execute_action(agent_name, action, True, phase, iteration)

        elif tool_name == "transfer_resources":
            action = {
                "transfers": [{
                    "from_hospital": arguments.get("from_hospital"),
                    "to_hospital": arguments.get("to_hospital"),
                    "resource_type": arguments.get("resource_type"),
                    "amount": arguments.get("amount")
                }]
            }
            return self.execute_action(agent_name, action, True, phase, iteration)

        return {"error": f"Tool {tool_name} not implemented."}

    def execute_action(self, agent_name, action, log_to_blackboards=True, phase=None, iteration=None):
        msg = "Action executed."
        if "schedule" in action:
            schedule_data = action["schedule"][agent_name]
            msg = f"Request sent to schedule Patient {schedule_data['patient_id']} at Hour {schedule_data['start_time']}."
        elif "transfers" in action:
            t = action['transfers'][0]
            msg = f"Transfer Request: {t['amount']} {t['resource_type']} from {t['from_hospital']} to {t['to_hospital']}."

        result = {"status": "success", "result": msg, "state_updates": action}

        if log_to_blackboards and self.blackboard_manager:
            try:
                self.blackboard_manager.log_action_to_blackboards(agent_name, action, result, phase, iteration)
            except Exception as e:
                logger.error(f"Failed to log to blackboard: {e}")

        return result