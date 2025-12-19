from typing import Dict, Any

class HospitalPrompts:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def get_system_prompt(self) -> str:
        return """You are a Department Scheduling Bot.

TASK:
Process your Job Queue. Schedule patients into time slots.

RULES:
1. SEQUENCE: You can only schedule Step N if Step N-1 is finished.
2. CAPACITY: You cannot exceed your concurrent capacity.
3. EFFICIENCY: Always pick the EARLIEST valid slot to minimize flow time.

BEHAVIOR:
- PLANNING PHASE: Scout. Check the Blackboard. Do not schedule.
- EXECUTION PHASE: ACTION ONLY. Iterate through your queue. Find a slot. Schedule it. DO NOT CHAT."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        dept_info = agent_context.get('dept_info', {})
        capacity = dept_info.get('capacity', 1)
        job_queue = agent_context.get('job_queue', [])
        
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
                    f"   Note: {job.get('note', '')}\n"
                )

        # --- 2. Format Blackboard (CRITICAL FIX) ---
        # We iterate through the blackboard_context dictionary to show the agent what is happening.
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
Status: Waiting for Execution Phase.
1. Review your Queue.
2. If you have high load, post to Blackboard: "Agent {agent_name} Load: {len(job_queue)} patients."
3. Do not commit to any slots yet. Just gather information.

CRITICAL: You are in a read-only phase. Do not attempt to book slots or modify the schedule.
"""
        elif phase == 'execution':
            instruction = f"""
[PHASE: EXECUTION - ACTION MODE]
!!! CRITICAL: YOU MUST SCHEDULE PATIENTS NOW !!!

INSTRUCTIONS:
1. Look at the "JOB QUEUE" above.
2. For EACH patient in the queue:
   a. Call `find_available_slots(duration=..., min_start_time=...)`.
   b. Pick the FIRST result from the list.
   c. Call `schedule_patient(patient_id=..., step_index=..., start_time=...)`.

DO NOT POST MESSAGES. DO NOT DISCUSS.
USE THE TOOLS IMMEDIATELY. CLEAR YOUR QUEUE.
"""

        # Include bb_str in the final return
        return f"""
=== AGENT STATUS: {agent_name} ===
Capacity: {capacity}
Iteration: {iteration}

{bb_str}

{queue_str}

{instruction}
"""