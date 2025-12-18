from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class HospitalTools:
    """
    Hospital coordination tools.

    The execution action is `transfer_patient`, which allows a hospital to 
    assign a final destination (target hospital) for a patient currently in their care.
    """

    def __init__(self, blackboard_manager):
        self.blackboard_manager = blackboard_manager

    def get_tool_names(self) -> Set[str]:
        return {"transfer_patient"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Returns the JSON schema for tools available in the current phase.
        The 'transfer_patient' tool is only available during the 'execution' phase.
        """
        if phase == "execution":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "transfer_patient",
                        "description": "Assign a patient currently at your facility to a target hospital (or keep them).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_id": {
                                    "type": "string",
                                    "description": "The RECORD_ID of the patient (e.g., 'P101').",
                                },
                                "target_hospital_id": {
                                    "type": "string",
                                    "description": "The THCIC_ID of the receiving hospital (e.g., 'H5').",
                                },
                            },
                            "required": ["patient_id", "target_hospital_id"],
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
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates and executes the transfer action, updating the environment state.
        """
        if not env_state:
            return {"status": "failed", "reason": "Environment state not provided"}

        # Extract state definitions
        # Note: Ensure HospitalEnvironment.get_serializable_state() provides these keys
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

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        env_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Entry point for the agent's tool call. Maps the LLM's arguments to the execution logic.
        """
        if tool_name != "transfer_patient":
            return {"error": f"Hospital environment does not support tool: {tool_name}"}

        patient_id = arguments.get("patient_id")
        target_hospital_id = arguments.get("target_hospital_id")

        if patient_id is None:
            return {"error": "patient_id is required for transfer_patient"}
        if target_hospital_id is None:
            return {"error": "target_hospital_id is required for transfer_patient"}

        # Construct the internal action dictionary
        action = {
            "action": "transfer_patient", 
            "patient_id": patient_id, 
            "target_hospital_id": target_hospital_id
        }

        return self.execute_action(
            agent_name,
            action,
            log_to_blackboards=True,
            phase=phase,
            iteration=iteration,
            env_state=env_state,
        )