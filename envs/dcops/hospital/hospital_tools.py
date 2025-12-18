from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class HospitalTools:
    """
    Hospital coordination tools.
    
    Includes:
    1. Sensory tools (available in Planning & Execution): Check costs, capacity, and patient lists.
    2. Action tools (available in Execution only): Transfer patients.
    """

    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def get_tool_names(self) -> Set[str]:
        return {"transfer_patient", "get_transport_cost", "get_hospital_capacity", "get_my_patients"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Returns the JSON schema for tools available in the current phase.
        """
        tools = []
        
        # --- READ-ONLY TOOLS (Available in ALL phases) ---
        
        # Tool 1: Check Cost/Distance
        tools.append({
            "type": "function",
            "function": {
                "name": "get_transport_cost",
                "description": "Calculate the transport cost (distance/price) to move a patient to a target hospital.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_hospital_id": {
                            "type": "string",
                            "description": "The THCIC_ID of the destination hospital.",
                        },
                    },
                    "required": ["target_hospital_id"],
                },
            },
        })

        # Tool 2: Check Capacity
        tools.append({
            "type": "function",
            "function": {
                "name": "get_hospital_capacity",
                "description": "Check the total capacity and current load of a target hospital (or yourself).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hospital_id": {
                            "type": "string",
                            "description": "The THCIC_ID of the hospital to check.",
                        },
                    },
                    "required": ["hospital_id"],
                },
            },
        })
        
        # Tool 3: Inspect Own Patients
        tools.append({
            "type": "function",
            "function": {
                "name": "get_my_patients",
                "description": "Get a list of all patients currently located at your facility with their required specialty.",
                "parameters": {
                    "type": "object",
                    "properties": {}, # No arguments needed
                    "required": [],
                },
            },
        })

        # --- ACTION TOOLS (Execution ONLY) ---
        # if phase == "execution":
        tools.append({
            "type": "function",
            "function": {
                "name": "transfer_patient",
                "description": "Assign a patient currently at your facility to a final destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "The RECORD_ID of the patient.",
                        },
                        "target_hospital_id": {
                            "type": "string",
                            "description": "The THCIC_ID of the receiving hospital.",
                        },
                    },
                    "required": ["patient_id", "target_hospital_id"],
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
        
        if tool_name == "get_transport_cost":
            # Assuming env_state has a 'distance_matrix' or similar
            target = arguments.get("target_hospital_id")
            # Logic to look up distance matrix in env_state...
            # This is a placeholder logic:
            dist_matrix = env_state.get("distance_matrix", {})
            cost = dist_matrix.get(agent_name, {}).get(target, "Unknown")
            return {"result": f"Transport cost from {agent_name} to {target} is {cost}."}

        elif tool_name == "get_hospital_capacity":
            target = arguments.get("hospital_id")
            # Logic to look up agent profile in env_state...
            agents_data = env_state.get("agents_data", {}) # Assuming this exists
            target_data = agents_data.get(target, {})
            cap = target_data.get("capacity", "Unknown")
            load = target_data.get("current_load", "Unknown")
            return {"result": f"Hospital {target}: Load {load}/{cap}"}

        elif tool_name == "get_my_patients":
            patients = env_state.get("patients", {})
            my_list = [
                {"id": pid, "condition": p.get("condition"), "needed_specialty": p.get("specialty")}
                for pid, p in patients.items() 
                if p.get("location") == agent_name
            ]
            return {"result": my_list}

        # --- WRITE HANDLER ---
        
        elif tool_name == "transfer_patient":
            action = {
                "action": "transfer_patient", 
                "patient_id": arguments.get("patient_id"), 
                "target_hospital_id": arguments.get("target_hospital_id")
            }
            return self.execute_action(
                agent_name, action, log_to_blackboards=True, phase=phase, iteration=iteration, env_state=env_state
            )

        return {"error": f"Tool {tool_name} not implemented."}

    # ... execute_action remains the same as your code ...