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
        return """You are a hospital administrator participating in a regional patient transfer coordination system.

PHASES:
- Planning Phase: Communicate with other hospitals via the blackboard. Discuss capacity, specialties (e.g., Cardiology, Neurology), and patient needs.
- Execution Phase: Execute transfers using the `transfer_patient` tool.

OBJECTIVE:
Your goal is to maximize the JOINT health outcomes of the region, not just your own metrics.
1. Match patients to hospitals that have the specific specialty for their condition.
   - ICD-10 'I' -> Cardiology
   - ICD-10 'G' -> Neurology
   - ICD-10 'M' -> Orthopedics
   - ICD-10 'P' -> Pediatrics
2. Minimize transportation costs (distance between hospitals).
3. Strictly adhere to hospital capacity limits. Overcrowding incurs massive penalties.

DATA CONTEXT:
- Your ID corresponds to your THCIC_ID.
- Patient conditions are derived from their PRINC_DIAG_CODE.

Coordinate efficiently. If you are over capacity, ask neighbors for help. If you have open beds and a specialty, advertise it."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        """
        Constructs the user prompt, injecting agent-specific context.
        """
        patients_str = ""
        my_patients = agent_context.get('my_patients', {})
        current_assignments = agent_context.get('current_assignments', {})
        
        for pid, p_data in my_patients.items():
            current_target = current_assignments.get(pid, "Not Assigned")
            patients_str += f"- Patient {pid}: Condition={p_data['condition']} (Code: {p_data['diagnosis_code']}), Severity={p_data['severity']}, Current Target={current_target}\n"

        hospital_stats = agent_context.get('hospital_stats', {})
        known_neighbors = agent_context.get('known_neighbors', [])
        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)

        # Build Blackboard Context String
        bb_str = ""
        if blackboard_context:
            bb_str = "=== BLACKBOARD MESSAGES ===\n"
            for channel, content in blackboard_context.items():
                if content.strip():
                    bb_str += f"Channel {channel}:\n{content}\n\n"

        return f"""
=== TURN INFORMATION ===
Phase: {phase}
Iteration: {iteration}
You are Administrator for Hospital: {agent_name}

=== YOUR HOSPITAL STATS ===
Name: {hospital_stats.get('name', 'Unknown')}
Location Index: {hospital_stats.get('location', 'Unknown')}
Capacity: {hospital_stats.get('capacity', 'Unknown')}
Specialties: {', '.join(hospital_stats.get('specialties', []))}

=== YOUR PATIENTS ===
{patients_str if patients_str else "No patients currently at your facility."}

=== KNOWN NEIGHBORS ===
{', '.join(known_neighbors)}

{bb_str}
Use the blackboard to negotiate transfers for patients who need specialties you do not possess, or to offload patients if you are over capacity.
"""