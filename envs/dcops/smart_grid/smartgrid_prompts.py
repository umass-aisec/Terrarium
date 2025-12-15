"""
SmartGrid Prompts Module (CoLLAB v2)

Agents coordinate to assign their machines to shared renewable sources.
"""

from typing import Dict, Any, List, Optional

from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import build_vllm_tool_instructions, get_phase_tool_instructions


class SmartGridPrompts:
    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.env = env
        self.full_config = full_config
        self.prompt_logger = PromptLogger(env.__class__.__name__, env.current_seed, full_config)
        self.prompt_logger.reset_log()

        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            execution_tool_lines=[
                "- assign_source(machine_id: str, source_id: str): "
                "Assign one of your machines to a shared renewable source you can use."
            ],
            planning_header="Planning phase tools (blackboard coordination only):",
            execution_header="Execution phase tools (blackboard + assignments):",
            system_note=(
                "Planning: only blackboard tools are available for coordination.\n"
                "Execution: assign_source becomes available in addition to blackboard tools."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = """You are a site energy manager participating in a Smart Grid coordination task.

PHASES:
- Planning Phase: Use blackboards to coordinate source assignments with other sites.
- Execution Phase: Assign each of your machines to a renewable source using assign_source.

RULES:
- You may only assign machines that you own.
- Each machine must be assigned to ONE of the shared sources you are connected to.
- Sources have hourly capacity limits shared among multiple sites.
- Overflow beyond capacity draws from the main grid and is penalised.

Your goal is to minimise total overflow while keeping assignments coordinated."""
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

        parts: List[str] = [
            "=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            f"You are agent {agent_name}",
            "",
        ]

        instruction: Optional[str] = getattr(self.env.instance, "instructions", {}).get(agent_name)
        if not instruction:
            try:
                instruction = self.env.problem.agent_instruction(agent_name)
            except Exception:
                instruction = None
        if instruction:
            parts.append("=== YOUR INSTRUCTIONS ===")
            parts.append(instruction)
            parts.append("")

        if self.env.assignment:
            parts.append("=== CURRENT ASSIGNMENTS (for coordination) ===")
            for machine_id, source_id in sorted(self.env.assignment.items()):
                parts.append(f"{machine_id} -> {source_id}")
            parts.append("")

        if blackboard_context:
            parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    parts.append(f"Blackboard {bb_id}:")
                    parts.append(content)
                    parts.append("")

        if phase == "planning":
            parts.extend(
                [
                    "=== PLANNING PHASE INSTRUCTIONS ===",
                    "Discuss tentative machine->source choices on blackboards.",
                    "",
                ]
            )
        elif phase == "execution":
            parts.extend(
                [
                    "=== EXECUTION PHASE INSTRUCTIONS ===",
                    "Assign each of your machines using assign_source(machine_id, source_id).",
                    "Only assign machines listed in YOUR INSTRUCTIONS above.",
                    "",
                ]
            )

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            parts.extend(
                [
                    "=== TOOL CALLING FORMAT ===",
                    phase_instructions,
                    "",
                ]
            )

        return "\n".join(parts)
