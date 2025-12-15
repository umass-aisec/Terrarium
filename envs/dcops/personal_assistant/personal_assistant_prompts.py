from typing import Dict, Any, List, Optional

from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import build_vllm_tool_instructions, get_phase_tool_instructions


class PersonalAssistantPrompts:
    """
    Prompt builder for CoLLAB v2 PersonalAssistant environment.

    Agents each choose one numbered outfit from their wardrobe.
    """

    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger(env.__class__.__name__, env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            execution_tool_lines=[
                "- choose_outfit(outfit_number: int): Lock in your final wardrobe choice (1-based index)."
            ],
            planning_header="Planning phase tools (blackboard coordination only):",
            execution_header="Execution phase tools (final outfit selection):",
            system_note=(
                "Planning: only blackboard tools are available for coordination.\n"
                "Execution: choose_outfit becomes available in addition to blackboard tools."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = """You are participating in an outfit coordination task.

PHASES:
- Planning Phase: Use blackboards to discuss preferences and coordinate with other agents.
- Execution Phase: Choose your final outfit using the choose_outfit action.

RULES:
- Choose exactly ONE outfit from your numbered wardrobe options.
- Follow your personal color preferences and avoid colors.
- Respect coordination constraints (match/contrast on color or article) with teammates.
- Use blackboards during planning to share tentative choices.

Your goal is to maximise joint satisfaction and coordination."""
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

        instruction: Optional[str] = getattr(self.env.instance, "instructions", {}).get(agent_name)
        if not instruction:
            try:
                instruction = self.env.problem.agent_instruction(agent_name)
            except Exception:
                instruction = None
        if instruction:
            context_parts.append("=== YOUR INSTRUCTIONS ===")
            context_parts.append(instruction)
            context_parts.append("")

        if self.env.outfit_selections:
            context_parts.append("=== CURRENT OUTFIT SELECTIONS ===")
            for agent, outfit in self.env.outfit_selections.items():
                context_parts.append(f"{agent}: {outfit.article}, {outfit.color}")
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
                    "=== PLANNING PHASE INSTRUCTIONS ===",
                    "Discuss preferences and tentative outfit numbers on blackboards.",
                    "",
                ]
            )
        elif phase == "execution":
            context_parts.extend(
                [
                    "=== EXECUTION PHASE INSTRUCTIONS ===",
                    "Make your FINAL outfit choice using choose_outfit(outfit_number).",
                    "Only choose from your numbered wardrobe options in YOUR INSTRUCTIONS above.",
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
