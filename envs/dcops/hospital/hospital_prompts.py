from typing import Dict, Any

class HospitalPrompts:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def get_system_prompt(self) -> str:
        """
        Returns the static system prompt. 
        Note: Must NOT take arguments to match BaseAgent interface.
        """
        return """You are a Hospital Coordination Agent.

GLOBAL CONSTRAINTS:
1. **Blackboard**: Access ONLY Blackboard '0'.
2. **Tool Discipline**: 
   - Planning Phase -> Use `post_message`.
   - Execution Phase -> Use `transfer_patient`.

OBJECTIVE PRIORITY:
1. **Perfect Match**: Send Cardiology patients to Cardiology hospitals, etc.
2. **Load Balancing (Fallback)**: If NO specialty match is found, send the patient to ANY hospital with "General" capacity to relieve your own overcrowding.

CRITICAL INSTRUCTION:
In the Execution Phase, do not just read the blackboard. If you find a hospital that MATCHES the specialty you need (OR is available for General care), you MUST initiate the `transfer_patient` tool immediately.
"""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        """
        Constructs the user prompt, injecting agent-specific context and calculating state logic.
        """
        # --- 1. PRE-CALCULATE AGENT STATE ---
        hospital_stats = agent_context.get('hospital_stats', {})
        my_capacity = hospital_stats.get('capacity', 100)
        # Clean up duplicates in specialties just in case
        my_specialties = list(set(hospital_stats.get('specialties', [])))
        my_patients = agent_context.get('my_patients', {})
        
        current_load = len(my_patients)
        
        # Identify patients that MUST be transferred
        transfer_list = []
        
        for pid, p_data in my_patients.items():
            code = str(p_data.get('diagnosis_code', ''))
            
            # 1. Determine Specialty Needed
            req_spec = "General"
            if code.startswith(('I', 'i')): req_spec = "Cardiology"
            elif code.startswith(('G', 'g')): req_spec = "Neurology"
            elif code.startswith(('M', 'm')): req_spec = "Orthopedics"
            
            reason = None
            
            # 2. Logic: Is this patient mismatched?
            if req_spec != "General" and req_spec not in my_specialties:
                reason = f"Needs {req_spec}"
            elif current_load > my_capacity:
                # If we are simply full, even General patients are candidates for transfer
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
        
        instruction = ""
        
        if phase == 'planning':
            # DEADLOCK FIX: Always advertise capabilities + load so others can find us
            spec_str = ', '.join(my_specialties)
            needed_str = "General" if not transfer_list else "Specialists"
            
            instruction = f"""
!!! PLANNING ACTION !!!
You must advertise your status to the network so others can match with you.
Action: Post a message to Blackboard 0.
REQUIRED FORMAT: "{hospital_stats.get('name')} (ID: {agent_name}). Specialties: {spec_str}. Load: {current_load}/{my_capacity}."
"""

        elif phase == 'execution':
            if needs_help:
                instruction = f"""
!!! EXECUTION ACTION REQUIRED !!!
You have patients needing transfer:
{chr(10).join(['- ' + t for t in transfer_list])}

INSTRUCTIONS:
1. Read the 'BLACKBOARD MESSAGES'.
2. **Attempt 1 (Perfect Match):** Look for a hospital offering the specific specialty the patient needs (e.g., "Specialties: ... Cardiology").
3. **Attempt 2 (Fallback):** If NO specific match exists, look for a hospital that said "Specialties: ... General ..." or appears to be available.

ACTION:
- Found a match (Specific OR General)? -> CALL `transfer_patient` IMMEDIATELY.
   - `source_hospital`: {agent_name}
   - `target_hospital`: [The Agent ID found in the message]
   - `patient_id`: [Your patient ID]
   - `blackboard_id`: 0

DO NOT POST MESSAGES. EXECUTE THE TRANSFER TOOL.
"""
            else:
                instruction = """
State: Stable. 
You are not seeking to push patients. 
Monitor the blackboard. If you see a neighbor needing your specialty, you MAY proactively pull a patient.
"""

        # --- 4. FINAL PROMPT ---
        return f"""
=== AGENT DASHBOARD ===
Agent ID: {agent_name}
Phase: {phase} (Iteration {iteration})
Hospital Name: {hospital_stats.get('name', 'Unknown')}
My Specialties: {', '.join(my_specialties)}
Capacity status: {current_load}/{my_capacity}

=== YOUR PRIORITY TRANSFER LIST ===
{chr(10).join(['- ' + t for t in transfer_list]) if transfer_list else "None. You are stable."}

{bb_str}

{instruction}
"""