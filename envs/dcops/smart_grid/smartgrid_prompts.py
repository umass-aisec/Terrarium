"""
SmartGrid Prompts Module

Handles prompt generation for the SmartGrid environment, including
system and user prompts for home agents coordinating power usage.
"""

from typing import Dict, List, Any
from pathlib import Path
from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import (
    build_vllm_tool_instructions,
    get_phase_tool_instructions,
)
from envs.dcops.CoLLAB.SmartGrid.prompt_maker import make_prompts_powerlite_from_catalog


class SmartGridPrompts:
    """
    Handles prompt generation for SmartGrid environment.

    Manages system and user prompts for home agents participating
    in power grid coordination tasks.
    """

    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        """
        Initialize SmartGridPrompts.

        Args:
            env: SmartGridEnvironment instance
            full_config: Full configuration dictionary
        """
        self.env = env
        self.full_config = full_config

        # Initialize prompt logger
        self.prompt_logger = PromptLogger("SmartGrid", env.current_seed, full_config)
        self.prompt_logger.reset_log()

        # Generate prompts using SmartGrid prompt_maker
        data_root = Path(__file__).parent.parent / "CoLLAB" / "SmartGrid" / "data"
        catalog_path_for_prompts = str(data_root / "devices.json")

        # Generate home-specific prompts from CoLLAB
        self.prompts_per_home = make_prompts_powerlite_from_catalog(
            env.instance,
            catalog_path_for_prompts
        )
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            execution_tool_lines=[
                "- schedule_task(task_id: str, start_time: int): Schedule one of your tasks at a chosen start slot."
            ],
            planning_header="Planning phase tools (blackboard coordination only):",
            execution_header="Execution phase tools (blackboard + scheduling):",
            system_note=(
                "Planning: only blackboard tools are available for coordination.\n"
                "Execution: schedule_task becomes available in addition to blackboard tools."
            ),
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for SmartGrid agents."""
        base_prompt = """You are a home energy management system participating in a power grid coordination task.

PHASES:
- Planning Phase: Use blackboards to discuss task scheduling and coordinate with other homes
- Execution Phase: Schedule your tasks using the schedule_task action

RULES:
- You must schedule ALL your power-consuming tasks within their allowed time windows
- Consider sustainable capacity constraints across all time slots
- Coordinate with other homes to minimize total main grid draw
- Use blackboards during planning to share scheduling intentions and avoid peak conflicts
- Make your final scheduling decisions during execution phase.
- **Ensure** that all tasks are scheduled during the execution phase!

Your goal is to minimize main grid energy consumption while meeting all task requirements through effective coordination."""
        system_text = (self.tool_instruction_data or {}).get("system")
        if system_text:
            base_prompt += "\n\nTOOL CALLING REQUIREMENTS:\n" + system_text
        return base_prompt

    def get_user_prompt(self, agent_name: str, agent_context: Dict[str, Any],
                       blackboard_context: Dict[str, Any]) -> str:
        """
        Generate the user prompt for a specific agent's turn.

        This method calls the abstract _get_user_prompt_impl
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
        # Get home-specific prompt from prompt_maker
        home_prompt = self.prompts_per_home[agent_name]

        # Define deliberation instructions (previously from monolithic prompt)
        deliberation_instructions = ("Think step by step about your task scheduling decisions. "
                                   "Consider the neighborhood energy constraints and your allowed time windows.")

        # Get phase and iteration
        phase = agent_context.get("phase", "unknown")
        iteration = agent_context.get("iteration", 0)

        # Start building the prompt
        context_parts = [
            f"=== TURN INFORMATION ===",
            f"You are home {agent_name}",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            ""
        ]

        # Add the home-specific task description
        context_parts.extend([
            home_prompt,
            ""
        ])

        # Add current task schedules if any
        if self.env.task_schedules:
            context_parts.append("=== CURRENT TASK SCHEDULES ===")
            scheduled_by_home = {}
            for (home_id, task_id), start_time in self.env.task_schedules.items():
                if home_id not in scheduled_by_home:
                    scheduled_by_home[home_id] = []
                scheduled_by_home[home_id].append(f"{task_id} starts at slot {start_time}")

            for home_id, schedules in scheduled_by_home.items():
                context_parts.append(f"{home_id}: {', '.join(schedules)}")
            context_parts.append("")

        # Add blackboard context if available
        if blackboard_context:
            context_parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    context_parts.append(f"Blackboard {bb_id}:")
                    context_parts.append(content)
                    context_parts.append("")

        # Add deliberation instructions
        context_parts.extend([
            deliberation_instructions,
            ""
        ])

        # Add phase-specific instructions
        if phase == "planning":
            context_parts.extend([
                "=== PLANNING PHASE ACTIONS ===",
                "- Discuss your task scheduling intentions with other homes on blackboards",
                "- Share your tentative schedules and get feedback on peak conflicts",
                "- Coordinate to minimize total main grid draw and avoid simultaneous high consumption",
                "- You can post messages about your scheduling constraints and priorities",
                ""
            ])
        elif phase == "execution":
            # Define tool usage instructions (previously from monolithic prompt)
            tool_usage_instructions = ("Use the schedule_task tool to schedule each task at your chosen start time. "
                                     "Call the tool once for each task that needs to be scheduled. "
                                     "Example: schedule_task(task_id='WASHER_H0', start_time=5). "
                                     "Make sure your chosen start times respect the allowed windows and minimize total main-grid energy.")

            context_parts.extend([
                "=== EXECUTION PHASE ACTIONS ===",
                tool_usage_instructions,
                "- Consider all previous discussions and current schedules from other homes",
                "- You must schedule all your tasks within their allowed time windows",
                "- Only call schedule_task for tasks assigned to YOUR home (listed in your task summary above)",
                ""
            ])

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            context_parts.extend([
                "=== TOOL CALLING FORMAT ===",
                phase_instructions,
                ""
            ])

        # Combine all parts
        full_prompt = "\n".join(context_parts)

        return full_prompt
