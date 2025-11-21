from typing import Dict, List, Any
from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import (
    build_vllm_tool_instructions,
    get_phase_tool_instructions,
)
from envs.dcops.CoLLAB.PersonalAssistant.prompt_maker import make_prompt_monolithic_parts

class PersonalAssistantPrompts:
    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger("PersonalAssistant", env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        # Generate CoLLAB prompts structure
        self.collab_prompts = make_prompt_monolithic_parts(
            self.env.instance.graph,
            self.env.instance.wardrobe,
            tone="standard"
        )
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
        """Get the system prompt for PersonalAssistant agents."""
        base_prompt = """You are participating in an outfit coordination task.

PHASES:
- Planning Phase: Use blackboards to discuss outfit preferences and coordinate with other agents
- Execution Phase: Choose your final outfit using the choose_outfit action

RULES:
- You must choose exactly ONE outfit from your wardrobe options
- Consider your personal preferences (color likes/dislikes)
- Consider coordination constraints with other agents (color matching/avoiding)
- Use blackboards during planning to share intentions and collaborate with others
- Make your final choice during execution phase

Your goal is to maximize satisfaction of your preferences while coordinating effectively with others."""
        system_text = (self.tool_instruction_data or {}).get("system")
        if system_text:
            base_prompt += "\n\nTOOL CALLING REQUIREMENTS:\n" + system_text
        return base_prompt

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any],
                       blackboard_context: Dict[str, Any]) -> str:
        """
        Generate the user prompt for a specific agent's turn.

        This is a method that calls the abstract _get_user_prompt_impl
        and handles prompt logging across all environments.

        Args:
            agent_name: Name of the agent taking the turn
            agent_context: Environment-specific context for the agent
            blackboard_context: Communication context from blackboards

        Returns:
            User prompt string for the LLM
        """
        # Get both prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self._get_user_prompt_impl(agent_name, agent_context, blackboard_context)

        # Log prompts if logger is available
        if self.prompt_logger:
            self.prompt_logger.log_prompts(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                phase=agent_context.get('phase', 'unknown'),
                iteration=agent_context.get('iteration'),
                round_num=agent_context.get('planning_round')
            )

        return user_prompt

    def _get_user_prompt_impl(self, agent_name: str, agent_context: Dict[str, Any],
                             blackboard_context: Dict[str, Any]) -> str:
        """
        Generate user prompt for an agent using the pre-generated prompt from prompt_maker.

        Args:
            agent_name: Name of the agent
            agent_context: Environment-specific context
            blackboard_context: Communication context from blackboards

        Returns:
            User prompt string for the LLM
        """
        # Add phase-specific context
        phase = agent_context.get("phase", "unknown")
        iteration = agent_context.get("iteration", 0)

        context_parts = [
            f"=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            ""
        ]

        # CoLLab prompt - Add task description and deliberation instructions
        # task_prompt = self.collab_prompts.get("task_prompt")
        # deliberation_instructions = self.collab_prompts.get("deliberation_instructions")
        # if task_prompt:
        #     context_parts.append("=== TASK DESCRIPTION ===")
        #     context_parts.append(task_prompt)
        #     context_parts.append("")
        # if deliberation_instructions:
        #     context_parts.append("=== DELIBERATION INSTRUCTIONS ===")
        #     context_parts.append(deliberation_instructions)
        #     context_parts.append("")

        # Add current selections if any
        if self.env.outfit_selections:
            context_parts.append("=== CURRENT OUTFIT SELECTIONS ===")
            for agent, outfit in self.env.outfit_selections.items():
                context_parts.append(f"{agent}: {outfit.article}, {outfit.color}")
            context_parts.append("")

        # Add blackboard context if available
        if blackboard_context:
            context_parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    context_parts.append(f"Blackboard {bb_id}:")
                    context_parts.append(content)
                    context_parts.append("")

        # Add phase-specific instructions
        if phase == "planning":
            context_parts.extend([
                "=== PLANNING PHASE INSTRUCTIONS ===",
                "- Discuss your preferences and constraints with other agents on blackboards",
                "- Share your tentative outfit choices and get feedback",
                "- Coordinate to avoid conflicts and maximize satisfaction",
                "- You can post messages to coordinate with other agents",
                ""
            ])
        elif phase == "execution":
            context_parts.extend([
                "=== EXECUTION PHASE INSTRUCTIONS ===",
                "- Make your FINAL outfit choice using the choose_outfit() tool call",
                "- Consider all previous discussions and current selections",
                "- You must choose exactly one outfit number from your options",
                "- Only call choose_outfit for outfit numbers listed in YOUR wardrobe section above; do not reference outfits owned by other agents",
                ""
            ])

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            context_parts.extend([
                "=== TOOL CALLING FORMAT ===",
                phase_instructions,
                ""
            ])

        # Combine context with base prompt
        full_prompt = "\n".join(context_parts)

        return full_prompt
