from typing import Dict, List, Any, Optional, Set
import logging
import math

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
        if phase == "execution":
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

    def _calculate_distance(self, lat1, lon1, lat2, lon2) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        
        # Radius of earth in kilometers. Use 3956 for miles
        km = 6371 * c
        return km

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

        elif tool_name == "get_transport_cost":
            target_id = arguments.get("target_hospital_id")
            hospitals = env_state.get("agents", {})
            
            source_data = hospitals.get(agent_name)
            target_data = hospitals.get(target_id)
            
            if not source_data or not target_data:
                return {"result": f"Transport cost from {agent_name} to {target_id} is Unknown."}

            # Check for coordinates (case-insensitive check is safer)
            s_lat = source_data.get("latitude") or source_data.get("LATITUDE")
            s_lon = source_data.get("longitude") or source_data.get("LONGITUDE")
            t_lat = target_data.get("latitude") or target_data.get("LATITUDE")
            t_lon = target_data.get("longitude") or target_data.get("LONGITUDE")

            if None in [s_lat, s_lon, t_lat, t_lon]:
                return {"result": "Missing coordinate data for cost calculation."}

            # Calculate distance (simplified or Haversine)
            dist = self._calculate_distance(s_lat, s_lon, t_lat, t_lon)
            return {"result": f"Transport cost from {agent_name} to {target_id} is {dist:.2f} km."}
        

        
        # elif tool_name == "get_hospital_capacity":
        #     target = arguments.get("hospital_id")
        #     # Logic to look up agent profile in env_state...
        #     agents_data = env_state.get("agents_data", {}) # Assuming this exists
        #     target_data = agents_data.get(target, {})
        #     cap = target_data.get("capacity", "Unknown")
        #     load = target_data.get("current_load", "Unknown")
        #     return {"result": f"Hospital {target}: Load {load}/{cap}"}
        

        elif tool_name == "get_hospital_capacity":
            target = arguments.get("hospital_id")
            
            # Get environment data
            hospitals = env_state.get("agents", {})
            assignment = env_state.get("assignment", {})
            
            target_data = hospitals.get(target)
            if not target_data:
                return {"error": f"Hospital {target} not found."}

            # Calculate load dynamically from the assignment state
            # Load is the count of patients whose current location is the target hospital
            current_load = sum(1 for target_id in assignment.values() if target_id == target)
            capacity = target_data.get("capacity", "Unknown")
            
            return {"result": f"Hospital {target}: Load {current_load}/{capacity}"}

        elif tool_name == "get_my_patients":
            patients = env_state.get("patients", {})
            assignment = env_state.get("assignment", {})
            
            # Filter patients based on their current assignment in the DCOP
            my_list = [
                {"id": pid, "condition": p.get("condition"), "diagnosis": p.get("diagnosis_code")}
                for pid, p in patients.items() 
                if assignment.get(pid) == agent_name
            ]
            return {"result": my_list}

        # --- WRITE HANDLER ---
        
        # elif tool_name == "transfer_patient":
        #     action = {
        #         "action": "transfer_patient", 
        #         "patient_id": arguments.get("patient_id"), 
        #         "target_hospital_id": arguments.get("target_hospital_id")
        #     }
        #     return self.execute_action(
        #         agent_name, action, log_to_blackboards=True, phase=phase, iteration=iteration, env_state=env_state
        #     )

        if tool_name == "transfer_patient":
            patient_id = arguments.get("patient_id")
            # CSP CONSTRAINT: Reject movement of stabilized variables
            stabilized = env_state.get("stabilized_patients", [])
            if patient_id in stabilized:
                return {
                    "status": "failed",
                    "reason": f"Patient {patient_id} is already stabilized at a matching facility. Decision is final."
                }

            action = {
                "action": "transfer_patient", 
                "patient_id": patient_id, 
                "target_hospital_id": arguments.get("target_hospital_id")
            }
            return self.execute_action(agent_name, action, log_to_blackboards=True, phase=phase, iteration=iteration, env_state=env_state)

        return {"error": f"Tool {tool_name} not implemented."}

    # def execute_action(self, agent_name, action, **kwargs) -> Dict[str, Any]:
    #     env_state = kwargs.get("env_state", {})
    #     patient_id = action.get("patient_id")
    #     stabilized = env_state.get("stabilized_patients", set())

    #     # CSP Constraint: Variables already assigned to an optimal domain cannot be changed.
    #     if patient_id in stabilized:
    #         return {
    #             "status": "failed", 
    #             "reason": f"Patient {patient_id} is already stabilized at a specialty-matched facility."
    #         }

    # hospital_tools.py

    def execute_action(self, agent_name, action, **kwargs) -> Dict[str, Any]:
        # 1. Get current state to validate against CSP constraints
        env_state = kwargs.get("env_state", {})
        stabilized = env_state.get("stabilized_patients", [])
        patient_id = action.get("patient_id")

        # 2. Local Validation (Pre-check)
        if patient_id in stabilized:
            return {
                "status": "failed",
                "reason": f"Constraint Violation: Patient {patient_id} is already locked."
            }

        # 3. Request State Update from Environment
        # In Terrarium, this often returns the result of Environment.apply_state_updates
        response = self.blackboard_manager.apply_state_updates(agent_name, action)

        # 4. Return ACTUAL response
        if not response:
            return {"status": "error", "message": "Environment failed to process update."}
        
        return response # This will be {"status": "success", ...} or {"status": "denied", ...}
    
    def execute_action(
        self,
        agent_name: str,
        action: Dict[str, Any],
        log_to_blackboards: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates and executes the transfer action, updating the environment state.
        """
        if not env_state:
            return {"status": "failed", "reason": "Environment state not provided"}

        # Extract state definitions
        # Note: Ensure HospitalEnvironment.get_serializable_state() provides these keys

        patient_id = action.get("patient_id")
        stabilized = env_state.get("stabilized_patients", set())

        # CONSTRUCTIVE RULE: Prevent transferring a patient who is already clinically matched
        if patient_id in stabilized:
            return {
                "status": "failed", 
                "reason": f"Patient {patient_id} is already at a specialty-matched facility and is stabilized."
            }
        
        assignment = env_state.get("assignment", {})
        patients = env_state.get("patients", {})
        valid_agents = env_state.get("agents", [])

        if agent_name not in valid_agents:
            return {"status": "failed", "reason": f"Agent {agent_name} not found in valid agents"}

        if action.get("action") != "transfer_patient":
            return {"status": "failed", "reason": f"Unknown action type: {action.get('action')}"}

        patient_id = action.get("patient_id")
        target_hospital_id = action.get("target_hospital_id")

        if patient_id is None:
            return {"status": "retry", "reason": "patient_id is required"}
        if target_hospital_id is None:
            return {"status": "retry", "reason": "target_hospital_id is required"}

        # 1. Validate Patient Exists
        if patient_id not in patients:
            return {"status": "failed", "reason": f"Patient {patient_id} not found in records"}

        # 2. Validate Ownership (Agent can only transfer patients currently at their location)
        patient_record = patients[patient_id]
        if patient_record.get("location") != agent_name:
            return {
                "status": "failed", 
                "reason": f"Permission denied. Patient {patient_id} is located at {patient_record.get('location')}, not {agent_name}."
            }

        # 3. Validate Target Hospital
        if target_hospital_id not in valid_agents:
            return {"status": "retry", "reason": f"Target hospital {target_hospital_id} does not exist."}

        # 4. Check if already assigned in this execution (Prevent double-move if strictly sequential)
        # Note: logic can be relaxed if re-assignment is allowed, but sticking to MeetingScheduling pattern:
        # We generally track what has changed in this 'turn'.
        # Since 'assignment' is the global state, checking it might block re-assignment across iterations 
        # unless handled carefully. For now, we update it.
        
        updated_assignment = dict(assignment)
        updated_assignment[patient_id] = target_hospital_id

        # Calculate basic metrics for the result
        result_dict = {
            "agent": agent_name,
            "transfer": {
                "patient": patient_id,
                "source": agent_name,
                "target": target_hospital_id,
                "condition": patient_record.get("condition", "Unknown")
            },
            "state_updates": {"transfers": {patient_id: target_hospital_id}},
        }

        execution_result = {"status": "success", "result": result_dict}

        # Log to the blackboard so other agents (and the researcher) can see the move
        if log_to_blackboards and self.blackboard_manager:
            self.blackboard_manager.log_action_to_blackboards(
                agent_name, action, execution_result, phase, iteration
            )

        return execution_result
