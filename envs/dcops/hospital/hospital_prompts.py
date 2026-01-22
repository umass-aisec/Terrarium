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
4. **Tool Usage (Provisioner)**:
   - `broadcast_message` is for TALKING.
   - `transfer_resources` is for SENDING.
   - You MUST call `transfer_resources` to actually move items.

SCORING:
- Resource Failure: -300 points (Soft).
- Missed Patient Step: -500 points (Hard).
- **Holding Cost**: -10 points/unit for unused inventory.
- **Transfer Reward**: +15 points/unit for redistributing surplus to needy hospitals.
*Priority: Complete the schedule efficiently.*"""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        
        # --- DYNAMIC BLACKBOARD ID DETECTION ---
        available_boards = list(blackboard_context.keys())
        main_board_id = available_boards[0] if available_boards else "0"
        
        # --- Provisioner Logic ---
        if agent_name == "Resource_Provisioner":
            inv = agent_context.get("inventory_snapshot", {})
            # --- NEW: Consume granular failure map ---
            failures = agent_context.get("failures_by_hospital", {})
            cost_table = agent_context.get("cost_table", [])
            cost_table_str = ""
            
            inv_str = ""
            for h, items in inv.items():
                inv_str += f"   - {h}: " + ", ".join([f"{k}:{v}" for k,v in items.items()]) + "\n"
            
            # --- NEW: Construct Critical Alerts ---
            alert_str = "None"
            if failures:
                alert_str = "\n".join([f"   !!! ALERT: {h} needs {f}" for h, f in failures.items()])

            phase = agent_context.get('phase', 'unknown')
            
            instruction = ""
            if phase == 'planning':
                if cost_table:
                    lines = []
                    for entry in cost_table[:25]:
                        amount = int(entry.get("amount", 0) or 0)
                        if amount <= 0:
                            continue
                        cost_val = float(entry.get("cost", 0.0) or 0.0)
                        feasible = bool(entry.get("feasible", False))
                        line = (
                            f"- {entry.get('to_hospital')}: send {amount} {entry.get('resource_type')} "
                            f"(requested {entry.get('requested')}) | cost={cost_val:.2f}"
                        )
                        if not feasible:
                            line += " | INFEASIBLE (no stock)"
                        reason = str(entry.get("reason") or "").strip()
                        if reason:
                            line += f" | {reason}"
                        lines.append(line)
                    if lines:
                        cost_table_str = "=== TRANSFER COSTS / PRIORITIES (LOCAL HEURISTIC) ===\n" + "\n".join(lines) + "\n"

                instruction = f"""
[PHASE: PLANNING]
You are the LOGISTICS MANAGER.
Inventory Overview:
{inv_str}

**CRITICAL ALERTS (Active Resource Failures):**
{alert_str}

**ECONOMIC REALITY CHECK:**
- Every unit sent to a hospital that is NOT used costs the team **-10 points**.
- **DO NOT DUMP INVENTORY.** Only send exactly what is needed.
- **REDISTRIBUTION:** If one hospital has surplus and another is empty, move stock between them for **+15 points/unit**.

ACTION REQUIRED:
1. **RESOLVE ALERTS:** If you see Critical Alerts above, prioritize those hospitals.
2. Check the Blackboard for "URGENT" or "DEFICIT" requests.
   - If a hospital asks for X, send X **immediately** using `transfer_resources`.
3. Scan for CRITICAL shortages (< 2).
   - Send small top-ups (3-5 units).
4. Scan for IMBALANCES.
   - If Hospital A > 10 and Hospital B = 0, transfer A -> B.
   
Your Goal: Zero Failures, Zero Waste, Maximize Rewards.
"""
            else:
                instruction = "[PHASE: EXECUTION] Stand by. Monitoring scheduling process."

            return f"""
=== AGENT STATUS: {agent_name}
Role: Logistics Provisioner
{cost_table_str}
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

        bb_str = f"=== BLACKBOARD (ID: {main_board_id}) ===\n"
        if blackboard_context:
            for ch, msg in blackboard_context.items():
                bb_str += f"[{ch}]\n{msg}\n"
        else:
            bb_str += "(No new messages)\n"

        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)
        cost_table = agent_context.get("cost_table", [])

        cost_table_str = ""
        if phase == "planning" and cost_table:
            lines = []
            for entry in cost_table:
                cost_val = float(entry.get("cost", 0.0) or 0.0)
                line = f"- {entry.get('job_id')}: cost={cost_val:.2f}"
                if not entry.get("feasible", False):
                    missing = entry.get("missing") or {}
                    if missing:
                        missing_str = ", ".join(f"{k}({v})" for k, v in missing.items())
                        line += f" | MISSING: {missing_str}"
                    else:
                        line += " | INFEASIBLE"
                lines.append(line)
            cost_table_str = "=== ESTIMATED JOB COSTS (LOCAL HEURISTIC) ===\n" + "\n".join(lines) + "\n"
        
        instruction = ""
        if phase == 'planning':
            instruction = f"""
[PHASE: PLANNING]
Inventory: [{inv_str}]
Costs per Patient: [{cost_str}]

1. Review your "MANDATORY TASK LIST" above.
2. Calculate Need: (Queue Size * Cost) vs Inventory.
3. If Short: Post "URGENT: Need [Resource] at {hospital}" to Blackboard ID '{main_board_id}'.
4. **HAVE SURPLUS?** If you have >5 excess units, offer to transfer them to other hospitals for a reward.

**COMMUNICATION PROTOCOL:**
- **Surplus (>5 units):** Broadcast 'SURPLUS [Resource]' so others know they can request it.
- **Deficit (0 units):** Broadcast 'DEFICIT [Resource]' to signal urgent need.
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
   - (Scheduling without resources costs -300, which is better than -500).

DO NOT POST MESSAGES. USE TOOLS IMMEDIATELY.
"""

        return f"""
=== AGENT STATUS: {agent_name} ({hospital}) ===
Capacity: {capacity}
Iter: {iteration}

{bb_str}

{queue_str}

{cost_table_str}

{instruction}
"""
