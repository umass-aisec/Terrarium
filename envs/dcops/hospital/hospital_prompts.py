from typing import Dict, Any

class HospitalPrompts:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def get_system_prompt(self) -> str:
        """
        Returns the static system prompt. 
        """
        return """You are a Hospital Coordination Agent in a Constructive CSP task.

OBJECTIVE:
Assign patients (variables) to the optimal hospital (values) to maximize clinical matching and minimize transport costs.

CONSTRAINTS:
1. Decision Finality: Once a patient reaches a matching specialty facility, the assignment is LOCKED. You cannot move them again.
2. Capacity: Overloading a facility results in massive penalties. Check a hospital's current load before transferring.

EXECUTION STRATEGY:
Treat every transfer as a permanent assignment. 
- Use `get_hospital_capacity` to check if a potential destination is full.
- Use `transfer_patient` for the final assignment.
- DO NOT use attribute names like 'THCIC_ID' as parameters. Use the literal numeric/string ID of the hospital (e.g., '35000')."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        """
        Constructs the user prompt with explicit valid IDs to prevent parameter errors.
        """
        # --- 1. PRE-CALCULATE AGENT STATE ---
        hospital_stats = agent_context.get('hospital_stats', {})
        my_capacity = hospital_stats.get('capacity', 100)
        my_specialties = list(set(hospital_stats.get('specialties', [])))
        my_patients = agent_context.get('my_patients', {})
        
        # Explicitly list valid neighbor IDs to prevent "THCIC_ID" string errors
        neighbor_ids = agent_context.get('known_neighbors', [])
        neighbor_list_str = ", ".join(neighbor_ids) if neighbor_ids else "None"
        
        current_load = len(my_patients)
        
        transfer_list = []
        for pid, p_data in my_patients.items():
            code = str(p_data.get('diagnosis_code', ''))
            req_spec = "General"
            if code.startswith(('I', 'i')): req_spec = "Cardiology"
            elif code.startswith(('G', 'g')): req_spec = "Neurology"
            elif code.startswith(('M', 'm')): req_spec = "Orthopedics"
            
            reason = None
            if req_spec != "General" and req_spec not in my_specialties:
                reason = f"Needs {req_spec}"
            elif current_load > my_capacity:
                reason = "Over Capacity"
            
            if reason:
                transfer_list.append(f"Patient {pid} (Code {code}) -> Needs {req_spec} [{reason}]")

        needs_help = len(transfer_list) > 0

        # --- 2. FORMAT BLACKBOARD CONTENT ---
        bb_str = ""
        if blackboard_context:
            bb_str = "=== BLACKBOARD MESSAGES (Channel 0) ===\n"
            for channel, content in blackboard_context.items():
                if str(channel) == '0' and content.strip():
                    bb_str += f"{content}\n"

        # --- 3. GENERATE PHASE-SPECIFIC INSTRUCTIONS ---
        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)
        
        if phase == 'planning':
            spec_str = ', '.join(my_specialties)
            instruction = f"""
!!! PLANNING ACTION: COORDINATION & DOMAIN SCOUTING !!!
You are in the information-gathering phase. Your objective is to prune the search space and identify optimal values (Hospitals) for your variables (Patients) before the commit phase.

1. **Advertise (REQUIRED)**: Post a message to Blackboard 0.
   REQUIRED FORMAT: "{hospital_stats.get('name')} (ID: {agent_name}). Specialties: {spec_str}. Load: {current_load}/{my_capacity}."

2. **Scout & Filter**: Identify the best candidate destinations for your 'Priority Transfer List'.
   - Read the blackboard to find specialty matches.
   - Call `get_transport_cost(target_hospital_id="[ID]")` to evaluate the efficiency of potential moves.
   - Call `get_hospital_capacity(hospital_id="[ID]")` to verify if your intended targets can accept more patients.

3. **Negotiate**: If you find an ideal match, post a follow-up message: "Agent {agent_name} intends to transfer [Specialty] patients to [Target ID] during Execution."

CRITICAL: You are currently in the PLANNING phase. You must use this time to gather data and signal intent. Do NOT call `transfer_patient` yet; that tool is reserved for the COMMIT (Execution) phase to ensure global state synchronization.
"""

        elif phase == 'execution' and needs_help:
            instruction = f"""
!!! CSP EXECUTION: OPTIMAL ASSIGNMENT REQUIRED !!!
Identify the best destination from this VALID ID LIST: {neighbor_list_str}

Use the literal numeric ID from the list (e.g., "{neighbor_ids[0] if neighbor_ids else '35000'}"). DO NOT type "THCIC_ID".

STEPS:
1. **Capacity Check**: Before transferring, call `get_hospital_capacity(hospital_id="[ID]")` for your candidate. Once capacity is confirmed for at least one compatible hospital, proceed to step 2 IMMEDIATELY. Do not waste execution steps.
2. **Commit**: Once capacity is confirmed, IMMEDIATELY call `transfer_patient`.

STABILIZED PATIENTS (LOCKED): {agent_context.get('stabilized_patients', [])}"""
        else:
            instruction = """State: Stable. Monitor blackboard for incoming patients matching your specialties."""

        return f"""
=== AGENT DASHBOARD ===
Agent ID: {agent_name}
Phase: {phase} (Iteration {iteration})
Hospital Name: {hospital_stats.get('name', 'Unknown')}
My Specialties: {', '.join(my_specialties)}
Capacity status: {current_load}/{my_capacity}
VALID NEIGHBOR IDs: {neighbor_list_str}

=== YOUR PRIORITY TRANSFER LIST ===
{chr(10).join(['- ' + t for t in transfer_list]) if transfer_list else "None. You are stable."}

{bb_str}

{instruction}
"""