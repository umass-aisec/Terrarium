from typing import Dict, List, Any
from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from envs.dcops.CoLLAB.MeetingScheduling.prompt_maker  import MonolithicMeetingPrompter
from envs.dcops.CoLLAB.MeetingScheduling.data_structure  import SLOT_LABELS

class MeetingSchedulingPrompts:
    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger("MeetingScheduling", env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        self.prompter = MonolithicMeetingPrompter(
            self.env.instance.graph,
            self.env.instance.preferences,
            self.env.instance.coords
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for MeetingScheduling agents."""
        return """You are a meeting coordinator responsible for scheduling meetings to optimize attendee satisfaction and coordination.

PHASES:
- Planning Phase: Use blackboards to discuss scheduling preferences and coordinate with other meeting organizers
- Execution Phase: Schedule your meetings using the schedule_meeting action

RULES:
- You can only schedule meetings that you OWN (you are the organizer)
- You must schedule meetings to time slots 1-10 (8:00-17:00, one hour each)
- Consider attendee time preferences for maximum satisfaction
- For PHYSICAL meetings, consider travel time between buildings
- Agents have priority rankings for meetings they attend - higher priority meetings are more important
- Use blackboards during planning to coordinate with other organizers and avoid conflicts
- Make your final scheduling decisions during execution phase

Your goal is to maximize the overall satisfaction score by considering:
1. Time preferences of attendees (MEETING_TIME_MATCH factors)
2. Feasibility constraints ensuring attendees can actually attend based on priority and travel (FEASIBILITY_AGENT factors)"""

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
            available_tools: List of available tools for this turn

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
        Generate duser prompt for an agent using the pre-generated agent-specific prompt.

        Args:
            agent_name: Name of the agent (meeting owner)
            agent_context: Environment-specific context
            blackboard_context: Communication context from blackboards
            available_tools: Available tools for this turn

        Returns:
            User prompt string for the LLM
        """

        phase = agent_context.get("phase", "unknown")
        iteration = agent_context.get("iteration", 0)
        # Build dynamic context (information that changes during simulation)
        context_parts = [
            f"=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            f"You are agent {agent_name}"
        ]

        # CoLLab prompt
        # Get agent-specific base prompt
        # prompts_dict = self.prompter.make_prompts_monolithic_parts()
        # task_prompt = prompts_dict.get("task_prompt")
        # deliberation_instructions = prompts_dict.get("deliberation_instructions")
        # context_parts.append("=== TASK DESCRIPTION ===")
        # context_parts.append(task_prompt)
        # context_parts.append("")
        # context_parts.append("=== DELIBERATION INSTRUCTIONS ===")
        # context_parts.append(deliberation_instructions)
        # context_parts.append("")

        # Modified CoLLab prompts
        # Add scheduling status updates (what's been scheduled since base prompt was created)
        agent_owned_meetings = [mid for mid, meeting in self.env.instance.graph.meetings.items()
                               if meeting.owner == agent_name]
        if agent_owned_meetings:
            scheduled_meetings = [mid for mid in agent_owned_meetings if mid in self.env.meeting_schedules]
            unscheduled_meetings = [mid for mid in agent_owned_meetings if mid not in self.env.meeting_schedules]

            context_parts.append("=== YOUR SCHEDULING STATUS ===")
            if scheduled_meetings:
                context_parts.append("SCHEDULED:")
                for mid in scheduled_meetings:
                    slot = self.env.meeting_schedules[mid]
                    context_parts.append(f"  {mid}: Slot {slot + 1} ({SLOT_LABELS[slot]})")

            if unscheduled_meetings:
                context_parts.append("STILL TO SCHEDULE:")
                for mid in unscheduled_meetings:
                    context_parts.append(f"  {mid}")
            context_parts.append("")

        # Add all current schedules for coordination awareness
        if self.env.meeting_schedules:
            context_parts.append("=== ALL CURRENT SCHEDULES (for coordination) ===")
            for mid, slot in self.env.meeting_schedules.items():
                meeting = self.env.instance.graph.meetings[mid]
                context_parts.append(f"{mid}: Slot {slot + 1} ({SLOT_LABELS[slot]}) - {meeting.owner} - {meeting.mode}")
            context_parts.append("")

        # Add blackboard communications
        if blackboard_context:
            context_parts.append("=== BLACKBOARD COMMUNICATIONS ===")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    context_parts.append(f"Blackboard {bb_id}:")
                    context_parts.append(content)
                    context_parts.append("")

        # Add brief phase reminder (detailed instructions are in base prompt)
        if phase == "planning":
            context_parts.extend([
                "=== CURRENT PHASE: PLANNING ===",
                "Use blackboards to coordinate with other organizers before making final decisions.",
                ""
            ])
        elif phase == "execution":
            context_parts.extend([
                "=== CURRENT PHASE: EXECUTION ===",
                "Time to make your final scheduling decisions using the schedule_meeting tool.",
                ""
            ])

        # Combine dynamic context with base prompt
        full_prompt = "\n".join(context_parts)

        return full_prompt