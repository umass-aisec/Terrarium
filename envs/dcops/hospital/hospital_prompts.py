from typing import Dict, Any

class HospitalPrompts:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def get_system_prompt(self) -> str:
        return """You are a Hospital Agent in a distributed simulation.

ROLES:
1. **Department Scheduler**: You manage a specific department (e.g., Surgery, Triage).
2. **Resource Provisioner**: You manage global logistics.

CORE RULES:
1. **Scope**: You can ONLY schedule for YOUR specific department. 
   - If you are 'Surgery', you CANNOT schedule 'Radiology' steps.
   - If you are 'Surgery', ignore patients who do not need surgery.
2. **Source of Truth**: 
   - The **JOB QUEUE** is your ONLY source of tasks. 
   - The **BLACKBOARD** is for context only. NEVER schedule a patient just because you saw them on the Blackboard.
3. **Execution**: 
   - Schedule strictly based on the `step_index` in your Job Queue.
   - Do not "guess" step indices.

SCORING:
- Resource Failure: -300 points (Soft).
- Missed Patient Step: -500 points (Hard).
*Priority: Complete the schedule.*"""

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

**ECONOMIC REALITY CHECK:**
- Every unit sent to a hospital that is NOT used costs the team **-10 points**.
- **DO NOT DUMP INVENTORY.** Only send exactly what is needed.

ACTION REQUIRED:
1. Check the Blackboard for "URGENT" requests.
   - If a hospital asks for X, try to send X (not X+20).
2. Scan for CRITICAL shortages (< 2).
   - If critical, send a small "Just-in-Time" batch (e.g., 3-5 units).
   
Your Goal: Zero Failures AND Zero Waste.
"""
            else:
                instruction = "[PHASE: EXECUTION] Stand by. Monitoring scheduling process."

            return f"""
=== AGENT STATUS: {agent_name}
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

        # --- CONTEXT FIREWALL: VISUAL SEPARATION ---
        queue_str = "JOB QUEUE: [EMPTY] - Do NOT Schedule Anything."
        if job_queue:
            queue_str = "=== YOUR MANDATORY TASK LIST (JOB QUEUE) ===\n"
            queue_str += "INSTRUCTIONS: You MUST schedule ONLY these specific steps.\n"
            for job in job_queue:
                queue_str += (
                    f"   [ ] Patient: {job['patient_id']} | REQUIRED STEP: {job['step_index']}\n"
                    f"       Duration: {job['duration']}h | Earliest Start: {job['earliest_start_time']}\n"
                )
            queue_str += "==============================================\n"

        bb_str = "=== BLACKBOARD (READ ONLY - DO NOT EXECUTE FROM HERE) ===\n"
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

1. Review your "MANDATORY TASK LIST" above.
2. Calculate Need: (Queue Size * Cost) vs Inventory.
3. If Short: Post "URGENT: Need [Resource] at {hospital}" via `broadcast_message`.
4. Do NOT schedule yet. Wait for Provisioner shipments.
"""
        elif phase == 'execution':
            instruction = f"""
[PHASE: EXECUTION - ACTION MODE]
!!! CRITICAL: SCHEDULE PATIENTS FROM YOUR QUEUE !!!

INSTRUCTIONS:
1. FOCUS ONLY on the "MANDATORY TASK LIST" section above.
   - Ignore patients mentioned on the Blackboard. They are handled by other agents.
   - If a patient is NOT in your "MANDATORY TASK LIST", do NOT schedule them.

2. EXECUTE: Call `schedule_patient` for every line item in your Task List.
   - Use the EXACT `step_index` listed. (If list says "Step: 2", use 2).
   - Use the `Earliest Start` as your target start time.
   - Do NOT wait. Do NOT hesitate.

3. INVENTORY CHECK: [{inv_str}]
   - If inventory is low, schedule ANYWAY to avoid the -500 Step Miss Penalty.

DO NOT POST MESSAGES. USE TOOLS IMMEDIATELY.
"""

        return f"""
=== AGENT STATUS: {agent_name} ({hospital}) ===
Capacity: {capacity}
Iter: {iteration}

{bb_str}

{queue_str}

{instruction}
"""