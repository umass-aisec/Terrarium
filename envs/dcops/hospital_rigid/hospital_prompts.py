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
3. RESOURCES: Procedures consume specific resources.
   - Try to ensure inventory > 0 before scheduling to avoid penalties.
   - HOWEVER, completing the schedule is the Priority.

SCORING:
- Joint Score = (1000 * Patients) - FlowTime - Penalties.
- Resource Failure: -300 points (Soft Constraint).
- Missed Patient Step: -500 points (Hard Constraint).
*Prioritize scheduling even if resources are low to avoid the massive -500 penalty.*

BEHAVIOR:
- PLANNING PHASE: Scout. Check Blackboard. Coordinate transfers.
- EXECUTION PHASE: ACTION ONLY. Schedule patients immediately."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        
        # --- Provisioner Logic ---
        if agent_name == "Resource_Provisioner":
            inv = agent_context.get("inventory_status", {})
            inv_str = ""
            for h, items in inv.items():
                inv_str += f"   - {h}: " + ", ".join([f"{k}:{v}" for k,v in items.items()]) + "\n"
            
            return f"""
=== AGENT STATUS: {agent_name} ===
Role: Logistics Provisioner
Inventory Table:
{inv_str}

INSTRUCTION:
1. Scan Blackboard for shortages.
2. If a hospital is low on a resource (e.g., < 2) and another has surplus, Transfer it.
3. Use `transfer_resources(from_hospital=..., to_hospital=..., resource_type=..., amount=...)`.
"""

        # --- Department Logic ---
        dept_info = agent_context.get('dept_info', {})
        capacity = dept_info.get('capacity', 1)
        job_queue = agent_context.get('job_queue', [])
        hospital = dept_info.get('hospital', 'Unknown')
        local_inv = dept_info.get('local_inventory', {})
        my_costs = dept_info.get('procedure_costs', {})
        
        # Format Inventory
        inv_str = ", ".join([f"{k}: {v}" for k,v in local_inv.items()])
        cost_str = ", ".join([f"{k}: -{v}" for k,v in my_costs.items()])

        # --- 1. Format Queue ---
        queue_str = ""
        if not job_queue:
            queue_str = "QUEUE EMPTY. Monitor blackboard for incoming patients."
        else:
            queue_str = "=== URGENT: JOB QUEUE (ACTION REQUIRED) ===\n"
            for job in job_queue:
                queue_str += (
                    f"-> [READY] Patient: {job['patient_id']} | Step: {job['step_index']}\n"
                    f"   Duration: {job['duration']}h | Earliest Start: Hour {job['earliest_start_time']}\n"
                )

        # --- 2. Format Blackboard ---
        bb_str = ""
        if blackboard_context:
            bb_str = "=== BLACKBOARD NOTICES ===\n"
            for bb_id, content in blackboard_context.items():
                bb_str += f"[Board ID: {bb_id}]\n{content}\n"
        else:
            bb_str = "=== BLACKBOARD NOTICES ===\n(No new messages)\n"

        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)
        
        instruction = ""
        if phase == 'planning':
            instruction = f"""
[PHASE: PLANNING]
Inventory: [{inv_str}]
Costs:     [{cost_str}]

1. Review Queue.
2. If Inventory < (QueueSize * Cost), post "URGENT: Need [Resource] at {hospital}" to Blackboard.
3. Do not schedule yet.
"""
        elif phase == 'execution':
            instruction = f"""
[PHASE: EXECUTION - ACTION MODE]
!!! CRITICAL: YOU MUST SCHEDULE PATIENTS NOW !!!

INSTRUCTIONS:
1. Look at the "JOB QUEUE".
2. Current Inventory: [{inv_str}].
3. EXECUTE: Call `schedule_patient` for every patient in the queue.
   - Do NOT hesitate. 
   - Even if inventory is low, scheduling (avoiding the -500 miss penalty) is better than waiting.

DO NOT POST MESSAGES. USE TOOLS IMMEDIATELY.
"""

        return f"""
=== AGENT STATUS: {agent_name} ({hospital}) ===
Capacity: {capacity}
Iteration: {iteration}

{bb_str}

{queue_str}

{instruction}
"""