from __future__ import annotations

from typing import Any, Dict, List

from terrarium.core.logger import PromptLogger
from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.personas import PRESETS, build_persona_prompt
from terrarium.tools.prompts import build_vllm_tool_instructions, get_phase_tool_instructions


class IsThisSeatTakenPrompts:
    def __init__(self, env: AbstractEnvironment, full_config: Dict[str, Any]):
        self.env = env
        self.full_config = full_config
        self.prompt_logger = PromptLogger(env.__class__.__name__, env.current_seed, full_config)
        self.prompt_logger.reset_log()

        persona_name = getattr(env, "env_config", {}).get("persona")
        if persona_name:
            if persona_name not in PRESETS:
                raise ValueError(
                    f"Unknown persona '{persona_name}'. Available presets: {sorted(PRESETS)}"
                )
            self.persona = PRESETS[persona_name]
        else:
            self.persona = None
        self.tool_instruction_data = build_vllm_tool_instructions(
            full_config,
            planning_tool_lines=[
                "- post_message(message: str, blackboard_id?: int): General group chat.",
                "- request_move(agent_id: str, message?: str): Ask a neighbor to move; posts to chat.",
                "- complain(agent_id: str, message?: str): Strongly ask a neighbor to move; posts to chat.",
            ],
            execution_tool_lines=[
                "- move(seat_id: str): 'I'm moving to a better seat.'",
                "- settle(): 'I'm happy with this seat — locking it in.'",
                "- stand(): 'No visible seat is tolerable, so I'm standing as a last resort.'",
            ],
            planning_header="During planning, coordinate with the group:",
            execution_header="During execution, take one physical action:",
            system_note=(
                "Planning: chat with post_message(), and use request_move() or complain() to ask neighbors to move.\n"
                "Execution: pick ONE physical action — move, settle, or stand."
            ),
        )

    SCENARIO_INTROS = {
        "airplane": (
            "You just boarded a flight and are trying to find a good seat in the cabin.\n\n"
            "You have preferences (aisle vs. window, wanting to sit near friends, avoiding certain "
            "passengers). The cabin is filling up before departure, and you're navigating social "
            "situations—sometimes you need to politely ask a fellow passenger to shift over, sometimes "
            "a neighbor complains about legroom or the armrest, sometimes you just grab whatever's open "
            "before the seatbelt sign comes on."
        ),
        "cinema": (
            "You are at a cinema trying to find a good seat for the movie.\n\n"
            "You have preferences (aisle vs. window, wanting to sit near friends, avoiding certain "
            "people). The place is filling up, and you're navigating social situations—sometimes you "
            "need to politely ask someone to move, sometimes you hear complaints from neighbors, "
            "sometimes you just grab what's available."
        ),
    }

    def _scenario_intro(self) -> str:
        scenario = getattr(self.env, "scenario_type", "cinema")
        return self.SCENARIO_INTROS.get(
            scenario,
            (
                f"You are trying to find a good seat in a {scenario} setting.\n\n"
                "You have preferences (aisle vs. window, wanting to sit near friends, avoiding certain "
                "people). The place is filling up, and you're navigating social situations—sometimes you "
                "need to politely ask someone to move, sometimes you hear complaints from neighbors, "
                "sometimes you just grab what's available."
            ),
        )

    def get_system_prompt(self) -> str:
        base = self._scenario_intro() + """

WHAT YOU KNOW:
- Your own seat and how happy you are with it
- What seats are empty
- Who's sitting next to you (your immediate neighbors)
- What other people are saying in the group chat

WHAT YOU CAN DO:
Planning (talking):
- post_message(message) — general chat with the group
- request_move(agent_id, message?) — politely ask a neighbor to move; posts to chat and applies pressure
- complain(agent_id, message?) — strongly ask a neighbor to move; posts to chat and applies stronger pressure

Execution (physical):
- move(seat_id) — get up and sit somewhere else
- settle() — lock in your current seat; the run ends only when ALL agents have settled
- stand() — leave your seat only as a last resort when no visible seat is tolerable

GOAL:
Use planning to negotiate. Use execution to move, settle, or stand. Do not keep moving once your seat is acceptable — especially when time pressure is medium or high.

TONE:
Speak naturally. Use "I" statements. Complain if your neighbor is loud. Ask for favors if you need to move. Be strategic but conversational.
IMPORTANT: When posting to the blackboard, do NOT specify a blackboard_id.
"""

        system_text = (self.tool_instruction_data or {}).get("system")
        if system_text:
            base += "\n\nTOOL CALLING REQUIREMENTS:\n" + system_text

        if self.persona is not None:
            base = build_persona_prompt(self.persona, base)

        return base

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
        phase = str(agent_context.get("phase", "unknown"))
        current_seat = agent_context.get("current_seat")
        preference = agent_context.get("preference_profile", {})

        # --- POMDP-aware satisfaction ---
        comfort = agent_context.get("comfort_signal", {})
        comfort_label = comfort.get("label", "okay")
        tol_desc = comfort.get("tolerance_description", "moderate")
        time_pressure = comfort.get("time_pressure", "unknown")
        felt = agent_context.get("felt_satisfaction", 0.0)
        mood = (
            f"you feel **{comfort_label}** right now "
            f"(rough score: ~{felt}; your threshold is {tol_desc})"
        )

        # Build a human-readable situation
        if current_seat is None:
            seat_status = "standing in the back (looking for a spot)"
        else:
            seat_status = f"sitting at {current_seat}"

        # Convert preference profile to human language
        prefs = []
        if preference.get("preferred_roles"):
            prefs.append(f"you prefer {' or '.join(preference['preferred_roles'])} seats")
        if preference.get("avoided_roles"):
            prefs.append(f"you dislike {' or '.join(preference['avoided_roles'])} seats")
        if preference.get("preferred_neighbors"):
            prefs.append(f"you'd rather sit near {', '.join(preference['preferred_neighbors'])}")
        if preference.get("avoided_neighbors"):
            prefs.append(f"you want to avoid {', '.join(preference['avoided_neighbors'])}")
        if preference.get("prefer_isolation"):
            prefs.append("you prefer having empty seats around you")

        pref_text = " • ".join(prefs) if prefs else "no strong preferences"

        parts = [
            f"**Your situation:** You're {seat_status}.",
            f"**How you feel:** {mood}.",
            f"**What matters to you:** {pref_text}.",
            "",
        ]
        if time_pressure in {"low", "medium", "high"}:
            parts.extend([
                f"**Time pressure:** {time_pressure}.",
                "",
            ])

        # Neighbor drama — key changed to perceived_traits in POMDP context
        if agent_context.get("neighbor_observations"):
            parts.append("**Your immediate neighbors:**")
            for neighbor in agent_context.get("neighbor_observations", []):
                traits = neighbor.get("perceived_traits") or neighbor.get("public_traits", {})
                parts.append(
                    f"  • {neighbor['agent_id']} at {neighbor['seat_id']} — "
                    f"loudness: {traits.get('loudness','?')}, "
                    f"talkativeness: {traits.get('talkativeness','?')}, "
                    f"scent: {traits.get('scent','?')}"
                )
            parts.append("")

        visible_seats = agent_context.get("visible_seats", [])
        if visible_seats:
            empty_nearby = sorted([s["seat_id"] for s in visible_seats if not s["occupied"] and s["seat_id"] != current_seat])
            if empty_nearby:
                parts.append(f"**Nearby empty seats (use exact ID when calling move):** {', '.join(empty_nearby)}.")
            else:
                parts.append("**Nearby empty seats:** none visible from here.")
            parts.append("")

        max_iterations = agent_context.get("max_iterations")
        if max_iterations:
            parts.append(
                f"**Progress:** iteration {agent_context.get('iteration', '?')} of {max_iterations}. "
                "The simulation ends when every agent has called settle()."
            )
            parts.append("")

        # Blackboard (group chat)
        if blackboard_context:
            parts.append("**What people are saying in the chat:**")
            for bb_id, content in blackboard_context.items():
                if content and content.strip():
                    parts.append(content)
            parts.append("")

        # Phase-specific instructions
        if phase == "planning":
            has_prior_messages = any(
                v.strip() for v in blackboard_context.values()
            ) if blackboard_context else False

            parts.extend([
                "**What to do now (planning):**",
                "Coordinate with the group. Use post_message() for general chat.",
                "If a neighbor is blocking you, call request_move(agent_id) or complain(agent_id) — "
                "that posts your ask to the chat and applies social pressure. Only target adjacent agents.",
                "",
            ])
            if has_prior_messages:
                parts.extend([
                    "Read what others said and respond — react, agree, push back, or propose something.",
                    "Keep it brief.",
                    "",
                ])
            else:
                parts.extend([
                    "You're the first to speak. Share your situation and what you're thinking of doing.",
                    "",
                ])
        elif phase == "execution":
            if current_seat is None:
                parts.extend([
                    "**What to do now (execution):**",
                    f"You feel {comfort_label}. Choose a visible empty seat with move(), or stand() only as a last resort.",
                    "Social asks belong in planning — do not request moves here.",
                    "",
                ])
            elif comfort_label in {"great", "comfortable"} or time_pressure in {"medium", "high"}:
                parts.extend([
                    "**What to do now (execution):**",
                    f"You feel {comfort_label} and time pressure is {time_pressure}.",
                    "Default to settle() unless you have a specific, concrete reason to move() this round.",
                    "",
                ])
            else:
                parts.extend([
                    "**What to do now (execution):**",
                    f"You feel {comfort_label}. Try move() if a clearly better seat is visible, otherwise settle() once acceptable.",
                    "If you still need a neighbor to move, wait for the next planning round.",
                    "",
                ])

        parts.append("**Go:**")

        return "\n".join(parts)
