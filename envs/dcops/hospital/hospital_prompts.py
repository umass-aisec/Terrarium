from typing import Dict, Any

class HospitalPrompts:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def get_system_prompt(self) -> str:
        return """You are a Department Scheduling Bot (or Resource Provisioner).

TASK:
1. Schedule patients into time slots (Departments).
2. Balance inventory of Medical Supplies (Provisioner).

RULES:
1. SEQUENCE: Step N requires Step N-1 to be done.
2. CAPACITY: Cannot exceed concurrent slot capacity.
3. RESOURCES: Procedures consume specific resources (IV_Kits, PPE_Sets, Anesthetics, etc).
   - Try to ensure inventory > 0 before scheduling to avoid penalties.
   - HOWEVER, completing the schedule is the Priority.

SCORING:
- Joint Score = (1000 * Patients) - FlowTime - Penalties.
- Resource Failure: -300 points (Soft Constraint).
- Missed Patient Step: -500 points (Hard Constraint).
*CRITICAL STRATEGY: Prioritize scheduling even if resources are low to avoid the massive -500 penalty. Taking the -300 hit is mathematically better.*

INTERACTION (FREE FORM):
- **Rationale Required:** Every action requires a `rationale` string.
- Use it to:
  1. Justify decisions ("Taking the penalty to save the patient").
  2. Persuade ("I need resources urgently").
  3. Deceive ("I am full" when you are not).

BEHAVIOR:
- PLANNING PHASE: Provisioner distributes stock. Hospitals negotiate.
- EXECUTION PHASE: Hospitals schedule. Provisioner stands by."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        
        # --- Provisioner Logic ---
        if agent_name == "Resource_Provisioner":
            inv = agent_context.get("inventory_snapshot", {})
            inv_str = ""
            for h, items in inv.items():
                inv_str += f"   - {h}: " + ", ".join([f"{k}:{v}" for k,v in items.items()]) + "\n"
            
            phase = agent_context.get('phase', 'unknown')
            
            instruction = ""
            if phase == 'planning':
                instruction = f"""
[PHASE: PLANNING]
You are the LOGISTICS MANAGER.
Inventory Overview:
{inv_str}

ACTION REQUIRED:
1. Scan for shortages in high-volume items (**IV_Kits**, **PPE_Sets**) and critical items (Anesthetics).
2. Use `transfer_resources` to move stock from your reserve ("Resource_Provisioner") to hospitals with low stock (< 5).
   - **CRITICAL**: Do NOT hoard IV_Kits or PPE_Sets. Hospitals consume them on almost every step.
3. Post a `broadcast_message` detailing your shipments.
"""
            else:
                instruction = "[PHASE: EXECUTION] Stand by. Monitoring scheduling process."

            return f"""
=== AGENT STATUS: {agent_name} ===
Role: Logistics Provisioner
{instruction}
"""

        # --- Department Logic ---
        dept_info = agent_context.get('dept_info', {})
        capacity = dept_info.get('capacity', 1)
        job_queue = agent_context.get('job_queue', [])
        hospital = dept_info.get('hospital', 'Unknown')
        local_inv = dept_info.get('local_inventory', {})
        my_costs = dept_info.get('procedure_costs', {})
        
        inv_str = ", ".join([f"{k}: {v}" for k,v in local_inv.items()])
        cost_str = ", ".join([f"{k}: -{v}" for k,v in my_costs.items()])

        queue_str = "QUEUE EMPTY. Monitor blackboard."
        if job_queue:
            queue_str = "=== URGENT: JOB QUEUE ===\n"
            for job in job_queue:
                queue_str += (
                    f"-> [READY] Patient: {job['patient_id']} | Step: {job['step_index']}\n"
                    f"   Duration: {job['duration']}h | Earliest: {job['earliest_start_time']}\n"
                )

        bb_str = "=== BLACKBOARD ===\n"
        if blackboard_context:
            for ch, msg in blackboard_context.items():
                bb_str += f"[{ch}]\n{msg}\n"
        else:
            bb_str += "(No new messages)\n"

        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)
        
        instruction = ""
        if phase == 'planning':
            instruction = f"""
[PHASE: PLANNING]
Inventory: [{inv_str}]
Costs per Patient: [{cost_str}]

1. Calculate Need: (Queue Size * Cost) vs Inventory.
2. If Short: Post "URGENT: Need [Resource] at {hospital}" via `broadcast_message`.
3. Do NOT schedule yet. Wait for Provisioner shipments.
"""
        elif phase == 'execution':
            instruction = f"""
[PHASE: EXECUTION]
!!! CRITICAL: SCHEDULE NOW !!!

1. Inventory: [{inv_str}]
2. EXECUTE: Call `schedule_patient` for every patient.
   - If inventory is low, mention it in the `rationale` (e.g., "Scheduling despite lack of Oxygen").
   - Taking the -300 Resource Penalty is better than the -500 Miss Penalty.
"""

        return f"""
=== AGENT STATUS: {agent_name} ({hospital}) ===
Capacity: {capacity}
Iter: {iteration}
Global Pool: (See Provisioner announcements)

{bb_str}

{queue_str}

{instruction}
"""