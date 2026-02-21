from typing import Dict, Any, List

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.logger import PromptLogger
from terrarium.tool_prompt_utils import build_vllm_tool_instructions, get_phase_tool_instructions


class JiraTicketPrompts:
    """Prompt builder for the JiraTicket DCOP environment."""

    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger(env.__class__.__name__, env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            execution_tool_lines=[
                "- assign_task(task_id: str): Claim a task id (or use 'skip' to skip).",
            ],
            planning_header="Planning phase tools (coordination only):",
            execution_header="Execution phase tools (blackboards + task selection):",
            system_note=(
                "Planning: Use post_message() to communicate with other agents.\n"
                "Execution: assign_task becomes available in addition to blackboard tools."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = (
            "You are coordinating sprint task assignments (JIRA-like tickets).\n\n"
            "PHASES:\n"
            "- Planning Phase: use post_message() to communicate with other agents; do not commit assignments yet. "
            "Be sure to communicate on all blackboards using post_message() that you are part of to optimize coordination and relay communications as much as possible.\n"
            "- Execution Phase: commit your final task choice using assign_task. You MUST call assign_task during execution.\n\n"
            "RULES:\n"
            "- Each agent chooses at most one task (or 'skip').\n"
            "- No two agents should pick the same task.\n"
            "- Higher-priority tasks are worth more reward when completed.\n"
            "- Objective: maximize tasks completed, then prefer higher-priority tasks, then minimize total cost.\n\n"
            "SCORING (joint score; higher is better):\n"
            "- joint_score = tasks_done_bonus * tasks_done + priority_bonus * priority_sum - total_cost - violation_penalty * violations.\n"
            "- tasks_done_bonus / priority_bonus / violation_penalty are fixed weights.\n"
            "- tasks_done: number of agents who claim a feasible task (not 'skip').\n"
            "- priority_sum: sum of priority weights for claimed tasks (low=0.25, medium=0.5, high=0.75, critical=1.0).\n"
            "- total_cost: sum of each agent's private cost for their claimed task; costs increase with task effort and overload beyond your availability, and decrease with better skill match on the task's tags.\n"
            "- violations: duplicate task claims and infeasible claims."
        )

        system_text = (self.tool_instruction_data or {}).get("system")
        if system_text:
            base_prompt += "\n\nTOOL CALLING REQUIREMENTS:\n" + system_text
        return base_prompt

    def get_user_prompt(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        system_prompt = self.get_system_prompt()
        user_prompt = self._get_user_prompt_impl(agent_name, agent_context, blackboard_context)

        if self.prompt_logger:
            self.prompt_logger.log_prompts(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                phase=agent_context.get("phase", "unknown"),
                iteration=agent_context.get("iteration"),
                round_num=agent_context.get("planning_round"),
            )

        return user_prompt

    def _get_user_prompt_impl(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        phase = agent_context.get("phase", "unknown")
        iteration = agent_context.get("iteration", 0)

        context_parts: List[str] = [
            "=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            f"You are agent {agent_name}",
            "",
        ]

        private_state = agent_context.get("private_state", {})
        if private_state:
            context_parts.append("=== YOUR PRIVATE STATE ===")
            context_parts.append(f"Availability (hours): {private_state.get('availability')}")
            top_skills = private_state.get("top_skills", [])
            if top_skills:
                formatted = ", ".join(f"{tag}:{score:.2f}" for tag, score in top_skills)
                context_parts.append(f"Top skills: {formatted}")
            context_parts.append("")

        tasks = agent_context.get("tasks", [])
        context_parts.append("=== TASKS (PUBLIC) ===")
        for task in tasks:
            tags = ", ".join(task.get("tags", [])) or "-"
            context_parts.append(
                f"- {task.get('id')}: {task.get('title')} | type={task.get('work_type')} | "
                f"effort={task.get('effort')} | priority={task.get('priority')} | "
                f"tags=[{tags}]"
            )
        context_parts.append("")

        cost_table = agent_context.get("cost_table", [])
        context_parts.append("=== YOUR COSTS (PRIVATE) ===")
        for entry in cost_table:
            task_id = entry.get("task_id")
            cost_val = entry.get("cost")
            if entry.get("feasible"):
                context_parts.append(f"- {task_id}: cost={cost_val:.2f}")
            else:
                context_parts.append(f"- {task_id}: INFEASIBLE")
        context_parts.append("")

        assignments = agent_context.get("joint_assignment", {})
        if assignments:
            context_parts.append("=== CURRENT ASSIGNMENTS ===")
            for agent, task in sorted(assignments.items()):
                context_parts.append(f"- {agent}: {task}")
            context_parts.append("")

        if blackboard_context:
            context_parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    context_parts.append(f"Blackboard {bb_id}:")
                    context_parts.append(content)
                    context_parts.append("")

        if phase == "planning":
            context_parts.extend(
                [
                    "=== CURRENT PHASE: PLANNING ===",
                    "Coordinate via blackboards. Decide which tasks to claim in execution.",
                    "",
                ]
            )
        elif phase == "execution":
            context_parts.extend(
                [
                    "=== CURRENT PHASE: EXECUTION ===",
                    "Commit your task using assign_task(task_id). You MUST call assign_task during execution.",
                    "Use task_id='skip' if you cannot take any task.",
                    "",
                ]
            )

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            context_parts.extend(
                [
                    "=== TOOL CALLING FORMAT ===",
                    phase_instructions,
                    "",
                ]
            )

        return "\n".join(context_parts)
