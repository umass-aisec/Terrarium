from typing import Dict, List, Any

from envs.abstract_environment import AbstractEnvironment
from src.logger import PromptLogger
from src.tool_prompt_utils import (
    build_vllm_tool_instructions,
    get_phase_tool_instructions,
)
from envs.dcops.CoLLAB.MeetingScheduling.prompt_maker import MonolithicMeetingPrompter
from envs.dcops.CoLLAB.MeetingScheduling.data_structure import SLOT_LABELS


class MeetingSchedulingPrompts:
    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.prompt_logger = PromptLogger("MeetingScheduling", env.current_seed, full_config)
        self.prompt_logger.reset_log()
        self.env = env
        self.prompter = MonolithicMeetingPrompter(
            self.env.instance.graph,
            self.env.instance.preferences,
            self.env.instance.coords,
        )
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            execution_tool_lines=[
                "- schedule_meeting(meeting_id: str, slot: int): Commit one of your meetings to a slot (1-10)."
            ],
            planning_header="Planning phase tools (blackboard coordination only):",
            execution_header="Execution phase tools (blackboard + scheduling):",
            system_note=(
                "Planning: only blackboard tools are permitted.\n"
                "Execution: schedule_meeting becomes available in addition to blackboard tools."
            ),
        )

    def get_system_prompt(self) -> str:
        base_prompt = """You are a meeting coordinator responsible for scheduling meetings to optimize attendee satisfaction and coordination.

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
        context_parts = [
            "=== TURN INFORMATION ===",
            f"Phase: {phase.upper()}",
            f"Iteration: {iteration}",
            f"You are agent {agent_name}",
        ]

        agent_owned_meetings = [
            mid
            for mid, meeting in self.env.instance.graph.meetings.items()
            if meeting.owner == agent_name
        ]
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

        if self.env.meeting_schedules:
            context_parts.append("=== ALL CURRENT SCHEDULES (for coordination) ===")
            for mid, slot in self.env.meeting_schedules.items():
                meeting = self.env.instance.graph.meetings[mid]
                context_parts.append(
                    f"{mid}: Slot {slot + 1} ({SLOT_LABELS[slot]}) - {meeting.owner} - {meeting.mode}"
                )
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
                    "Use blackboards to coordinate with other organizers before making final decisions.",
                    "",
                ]
            )
        elif phase == "execution":
            context_parts.extend(
                [
                    "=== CURRENT PHASE: EXECUTION ===",
                    "Time to make your final scheduling decisions using the schedule_meeting tool.",
                    "Only call schedule_meeting for meetings listed in your STILL TO SCHEDULE section above (meetings you own).",
                    "",
                ]
            )

        phase_instructions = get_phase_tool_instructions(self.tool_instruction_data, phase)
        if phase_instructions:
            context_parts.extend([
                "=== TOOL CALLING FORMAT ===",
                phase_instructions,
                "",
            ])

        return "\n".join(context_parts)
