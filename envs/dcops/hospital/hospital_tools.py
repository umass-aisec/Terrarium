from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class HospitalTools:
    """
    Hospital Job Shop Scheduling Tools.
    
    Tools allow agents to:
    1. Check their own schedule availability (find gaps).
    2. Commit a patient to a time slot.
    """

    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def get_tool_names(self) -> Set[str]:
        return {"schedule_patient", "find_available_slots", "get_job_queue"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Returns the JSON schema for tools available in the current phase.
        """
        tools = []
        
        # --- READ-ONLY TOOLS (Available in ALL phases) ---
        
        # Tool 1: Find Available Slots (Helper to avoid LLM math errors)
        tools.append({
            "type": "function",
            "function": {
                "name": "find_available_slots",
                "description": "Finds the earliest available time slots for a job of a given duration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "integer",
                            "description": "The length of the procedure in hours.",
                        },
                        "min_start_time": {
                            "type": "integer",
                            "description": "The earliest hour the job can start (e.g., due to arrival time).",
                        },
                        "num_slots": {
                            "type": "integer",
                            "description": "How many options to return. Default is 3.",
                            "default": 3
                        }
                    },
                    "required": ["duration", "min_start_time"],
                },
            },
        })

        # Tool 2: Get Queue
        tools.append({
            "type": "function",
            "function": {
                "name": "get_job_queue",
                "description": "Refreshes the list of patients waiting for this department.",
                "parameters": {
                    "type": "object",
                    "properties": {}, 
                    "required": [],
                },
            },
        })

        # --- ACTION TOOLS (Execution ONLY) ---
        if phase == "execution":
            tools.append({
                "type": "function",
                "function": {
                    "name": "schedule_patient",
                    "description": "Book a specific time slot for a patient. Fails if capacity is exceeded.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "The ID of the patient.",
                            },
                            "step_index": {
                                "type": "integer",
                                "description": "The step number in the patient's pathway.",
                            },
                            "start_time": {
                                "type": "integer",
                                "description": "The specific hour (0-167) to start the procedure.",
                            },
                        },
                        "required": ["patient_id", "step_index", "start_time"],
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
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        
        if not env_state:
            return {"error": "Environment state missing."}

        # --- READ HANDLERS ---
        
        if tool_name == "get_job_queue":
            # Just return the subset of patients ready for this agent
            # We rely on 'build_agent_context' having populated this in the prompt.
            return {"result": "Please refer to the 'JOB QUEUE' in your prompt context."}

        elif tool_name == "find_available_slots":
            duration = arguments.get("duration")
            min_start = arguments.get("min_start_time", 0)
            limit = arguments.get("num_slots", 3)
            
            # Extract schedule from serializable state
            schedule = env_state.get("schedule", {}).get(agent_name, {})
            
            # Extract capacity
            agents_info = env_state.get("agents", {})
            my_capacity = 1
            if agent_name in agents_info:
                my_capacity = agents_info[agent_name].get("capacity", 1)

            valid_starts = []
            
            # Scan the week (0 to 167 hours)
            for t in range(min_start, 168):
                if len(valid_starts) >= limit: break
                
                # Check if block [t, t+duration] fits
                fits = True
                if t + duration > 168:
                    fits = False
                else:
                    for h in range(t, t + duration):
                        # SAFETY FIX: Handle both Integer (Environment) and String (JSON) keys
                        # This prevents the agent from seeing empty slots when keys are actually integers
                        slot_occupancy = schedule.get(h, schedule.get(str(h), []))
                        
                        if len(slot_occupancy) >= my_capacity:
                            fits = False
                            break
                
                if fits:
                    valid_starts.append(t)
            
            if not valid_starts:
                return {"result": "No available slots found in the remaining week."}
            return {"result": f"Available Start Times: {valid_starts}"}

        # --- WRITE HANDLER ---

        elif tool_name == "schedule_patient":
            # Construct the action dictionary
            action = {
                "schedule": {
                    agent_name: {
                        "patient_id": arguments.get("patient_id"),
                        "step_index": arguments.get("step_index"),
                        "start_time": arguments.get("start_time")
                    }
                }
            }
            
            return self.execute_action(agent_name, action, log_to_blackboards=True, phase=phase, iteration=iteration)

        return {"error": f"Tool {tool_name} not implemented."}

    def execute_action(
        self,
        agent_name: str,
        action: Dict[str, Any],
        log_to_blackboards: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes the action and returns the state updates for the environment to apply.
        """
        
        # We assume the action structure is correct as constructed in handle_tool_call
        schedule_data = action["schedule"][agent_name]
        
        patient_id = schedule_data['patient_id']
        start_time = schedule_data['start_time']
        
        # Construct result message
        result_msg = f"Request sent to schedule Patient {patient_id} at Hour {start_time}."

        # CRITICAL FIX FOR ATTRIBUTE ERROR:
        # We do NOT call self.blackboard_manager.apply_state_updates(...) because Megaboard doesn't have it.
        # Instead, we return 'state_updates'. The Framework will pass this to the Environment.
        execution_result = {
            "status": "success", 
            "result": result_msg,
            "state_updates": action # Pass the whole action dict as the update payload
        }

        # We DO call log_action_to_blackboards, which IS available on Megaboard
        if log_to_blackboards and self.blackboard_manager:
            try:
                self.blackboard_manager.log_action_to_blackboards(
                    agent_name, action, execution_result, phase, iteration
                )
            except Exception as e:
                logger.error(f"Failed to log to blackboard: {e}")

        return execution_result