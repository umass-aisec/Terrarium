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

TOOLS:
- `find_available_slots(duration, min_start_time)`: Returns valid start hours.
- `schedule_patient(patient_id, step_index, start_time)`: Locks the slot.

BEHAVIOR:
- PLANNING PHASE: Scout. Check the Blackboard. Do not schedule.
- EXECUTION PHASE: ACTION ONLY. Iterate through your queue. Find a slot. Schedule it. DO NOT CHAT."""

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any], blackboard_context: Dict[str, Any]) -> str:
        dept_info = agent_context.get('dept_info', {})
        capacity = dept_info.get('capacity', 1)
        job_queue = agent_context.get('job_queue', [])
        
        # Format Queue
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

        phase = agent_context.get('phase', 'unknown')
        iteration = agent_context.get('iteration', 0)
        
        instruction = ""
        if phase == 'planning':
            instruction = f"""
[PHASE: PLANNING]
Status: Waiting for Execution Phase.
1. Review your Queue.
2. If you have high load, post to Blackboard: "Agent {agent_name} Load: {len(job_queue)} patients."
3. DO NOT call `schedule_patient` yet.
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

        return f"""
=== AGENT STATUS: {agent_name} ===
Capacity: {capacity}
Iteration: {iteration}

{queue_str}

{instruction}
"""