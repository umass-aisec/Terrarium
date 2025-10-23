from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

# Use relative import since data_structure.py is in the same package
from .data_structure import (
    AgentId, MeetingId, Meeting, Factor, FactorGraph, MeetingEnvInstance, SLOT_LABELS
)

# ----------------------------
# Small utilities
# ----------------------------

def _slot_nums_human(slots: List[int]) -> str:
    """
    Render slots with 1-based numbering and HH:MM labels:
      e.g., 01(08:00), 03(10:00)
    Assumes 'slots' are 0-based internally.
    """
    return ", ".join(f"{s+1:02d}({SLOT_LABELS[s]})" for s in sorted(slots))

def _all_building_pairs_distances(
    coords: Dict[str, Tuple[int, int]],
    buildings: List[str],
) -> List[str]:
    """
    Pretty strings like 'CS ↔ Library = 17m' for unique building pairs.
    Integer Euclidean distance used as minutes.
    """
    uniq = sorted(set(b for b in buildings if b in coords))
    out: List[str] = []
    from math import dist
    for i, b1 in enumerate(uniq):
        for b2 in uniq[i+1:]:
            (x1, y1) = coords[b1]; (x2, y2) = coords[b2]
            d = int(dist((x1, y1), (x2, y2)))
            out.append(f"{b1} ↔ {b2} = {d}m")
    return out

def _collect_all_physical_buildings(graph: FactorGraph) -> List[str]:
    bs: List[str] = []
    for m in graph.meetings.values():
        if getattr(m, "mode", None) == "PHYSICAL" and getattr(m, "location", None):
            bs.append(m.location)
    return bs

# ----------------------------
# NEW: Extract env knobs from factors
# ----------------------------

def _extract_time_match_weights(graph: FactorGraph) -> Dict[MeetingId, float]:
    """
    Returns per-meeting weight w_m for MEETING_TIME_MATCH. If absent, defaults to 1.0.
    """
    w: Dict[MeetingId, float] = {mid: 1.0 for mid in graph.meetings.keys()}
    for f in getattr(graph, "factors", []):
        if getattr(f, "ftype", "") == "MEETING_TIME_MATCH":
            var_scope = list(getattr(f, "var_scope", []) or getattr(f, "scope", []) or [])
            wm = float(getattr(f, "payload", {}).get("w", 1.0))
            for mid in var_scope:
                w[mid] = wm
    return w

def _extract_agent_priorities(graph: FactorGraph) -> Dict[AgentId, Dict[MeetingId, int]]:
    """
    Returns {agent: {mid: priority}} from FEASIBILITY_AGENT payloads.
    If missing, falls back to empty dict (treated as equal priority).
    """
    pr: Dict[AgentId, Dict[MeetingId, int]] = {}
    for f in getattr(graph, "factors", []):
        if getattr(f, "ftype", "") == "FEASIBILITY_AGENT":
            agent_scope = list(getattr(f, "agent_scope", []) or [])
            if len(agent_scope) != 1:
                continue
            a = agent_scope[0]
            pr_map = dict(getattr(f, "payload", {}).get("priority", {}))
            if pr_map:
                pr[a] = {str(k): int(v) for k, v in pr_map.items()}
    return pr

# ============================================================
# Monolithic (by-parts) prompt builder for Meeting Scheduling
# ============================================================

class MonolithicMeetingPrompter:
    """
    Builds THREE distinct prompt strings for a monolithic, two-stage agent:
      1) task_prompt               -- problem statement + full instance context
      2) deliberation_instructions -- how to reason step-by-step for THIS env's objective
      3) json_mode_instructions    -- strict output contract for JSON-only emission

    Usage:
        prompter = MonolithicMeetingPrompter(graph, preferences, coords)
        parts = prompter.make_prompts_monolithic_parts()
    """

    def __init__(
        self,
        graph: FactorGraph,
        preferences: Dict[AgentId, set],
        coords: Dict[str, Tuple[int, int]],
        *,
        viz_map: Optional[Dict[AgentId, str]] = None,  # optional calendar+map image path per agent (not displayed here)
        tone: str = "standard",
    ):
        self.graph = graph
        self.preferences = preferences
        self.coords = coords
        self.viz_map = viz_map or {}
        self.tone = tone

        # Extract env parameters from factors
        self.time_match_w = _extract_time_match_weights(graph)          # {mid: w_m}
        self.agent_priorities = _extract_agent_priorities(graph)        # {agent: {mid: prio}}

    # -------------------------
    # Part 1: Task prompt
    # -------------------------
    def _build_task_prompt(self) -> str:
        lines: List[str] = []
        lines.append("## TASK PROMPT")
        lines.append("You are the global coordinator for today’s schedule.")
        lines.append("There are 10 one-hour slots from 08:00 to 18:00 (slots 1..10).")
        lines.append("")
        lines.append("Assign exactly ONE time slot to EACH meeting to maximize the environment’s score:")
        lines.append("")
        lines.append("Score components")
        lines.append("  • Meeting-Time-Match (per meeting M):  + w_M × (# of its attendees who prefer the chosen slot).")
        lines.append("  • Feasibility-Agent (per agent A):     + (number of A’s meetings they can actually attend),")
        lines.append("      computed by a greedy priority rule:")
        lines.append("        – Sort A’s meetings by priority (higher first), then by earlier start time.")
        lines.append("        – Traverse that order and keep a meeting if it does not overlap with already kept ones")
        lines.append("          and has enough travel gap between consecutive PHYSICAL locations.")
        lines.append("        – Otherwise, skip the lower-priority conflicting meeting.")
        lines.append("")
        lines.append("Travel feasibility")
        lines.append("  • For two PHYSICAL meetings in different buildings, the start-time gap in minutes must be")
        lines.append("    ≥ the integer Euclidean distance between buildings (given below).")
        lines.append("  • Zoom meetings have zero travel time.")
        lines.append("")
        lines.append("Slots and labels:")
        lines.append("  " + ", ".join(f"{i+1:02d}({lab})" for i, lab in enumerate(SLOT_LABELS)))
        lines.append("")

        # --- Meetings summary (owner + attendees) ---
        if not self.graph.meetings:
            lines.append("There are no meetings in this instance.")
        else:
            lines.append("### Meetings")
            for m in sorted(self.graph.meetings.values(), key=lambda x: x.mid):
                mode = getattr(m, "mode", "ZOOM")
                loc_str = "ZOOM" if mode != "PHYSICAL" else f"PHYSICAL @ {getattr(m, 'location', None)}"
                owner = getattr(m, "owner", None)
                attendees = ", ".join(m.attendees) if getattr(m, "attendees", None) else "(none)"
                w_m = self.time_match_w.get(m.mid, 1.0)
                lines.append(f"- {m.mid}: {loc_str}; owner={owner}; attendees=[{attendees}]; w_M={w_m:g}")

        lines.append("")

        # --- Preferences by agent ---
        lines.append("### Agent Preferred Slots")
        if not self.graph.agents:
            lines.append("- (no agents)")
        else:
            for a in sorted(self.graph.agents):
                prefs = sorted(list(self.preferences.get(a, set())))
                if prefs:
                    lines.append(f"- {a}: { _slot_nums_human(prefs) }")
                else:
                    lines.append(f"- {a}: (no declared preferences; may choose any slot 1..10)")

        lines.append("")

        # --- Priorities by agent (if provided) ---
        if self.agent_priorities:
            lines.append("### Agent Meeting Priorities (higher is more important)")
            for a in sorted(self.graph.agents):
                pr_map = self.agent_priorities.get(a, {})
                if not pr_map:
                    lines.append(f"- {a}: (no explicit priorities; treat all equal)")
                else:
                    # Stable order by meeting id
                    items = ", ".join(f"{mid}:{prio}" for mid, prio in sorted(pr_map.items()))
                    lines.append(f"- {a}: {items}")
            lines.append("")
        else:
            lines.append("### Agent Meeting Priorities")
            lines.append("- (no explicit priorities provided; treat all equal)")
            lines.append("")

        # --- Building distances (global table) ---
        buildings = _collect_all_physical_buildings(self.graph)
        dist_rows = _all_building_pairs_distances(self.coords, buildings)
        if dist_rows:
            lines.append("### Building-to-Building Distances (minutes required as start-time gap)")
            for s in dist_rows:
                lines.append(f"- {s}")

        lines.append("")
        lines.append("When you are done reasoning, output ONLY a single JSON object mapping meeting_id -> slot_number (1..10).")

        return "\n".join(lines)

    # -------------------------
    # Part 2: Deliberation
    # -------------------------
    def _build_deliberation_instructions(self) -> str:
        lines: List[str] = []
        lines.append("## INSTRUCTIONS")
        lines.append("- Optimize the stated score (Meeting-Time-Match + Feasibility-Agent).")
        lines.append("- Meeting-Time-Match: a meeting contributes w_M × (count of attendees whose prefs include the chosen slot).")
        lines.append("- Feasibility-Agent (priority-greedy attendance per agent):")
        lines.append("  1) For each agent, sort their meetings by (priority desc, start time asc).")
        lines.append("  2) Keep a meeting if it does not overlap earlier kept meetings and the travel gap is sufficient")
        lines.append("     for any physical-to-physical transition; otherwise skip it.")
        lines.append("- There is no hard requirement to pick only preferred slots; picking a preferred slot just increases the score.")
        lines.append("- When conflicts arise, prefer keeping higher-priority meetings and those that yield larger time-match gains.")
        return "\n".join(lines)

    # -------------------------
    # Part 3: JSON mode
    # -------------------------
    def _build_json_mode_instructions(self) -> str:
        lines: List[str] = []
        lines.append("## INSTRUCTIONS")
        lines.append("Emit EXACTLY ONE JSON object with NO extra text, mapping every meeting id to a 1-based slot number in [1..10].")
        return "\n".join(lines)

    # -------------------------
    # Public API
    # -------------------------
    def make_prompts_monolithic_parts(self) -> Dict[str, str]:
        return {
            "task_prompt": self._build_task_prompt(),
            "deliberation_instructions": self._build_deliberation_instructions(),
            "json_mode_instructions": self._build_json_mode_instructions(),
        }

    def as_dict_for_pipeline(self) -> Dict[str, str]:
        parts = self.make_prompts_monolithic_parts()
        return {
            "__TASK__": parts["task_prompt"],
            "__DELIBERATION__": parts["deliberation_instructions"],
            "__JSON_MODE__": parts["json_mode_instructions"],
        }

# ============================================================
# Functional adapter (drop-in for generate_instances(...))
# ============================================================

def make_prompts_monolithic_parts_as_dict(
    graph: FactorGraph,
    preferences: Dict[AgentId, set],
    coords: Dict[str, Tuple[int, int]],
    *,
    viz_map: Optional[Dict[AgentId, str]] = None,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter so you can pass this directly to your dataset generator:
        generate_instances(..., make_prompts_fn=make_prompts_monolithic_parts_as_dict)
    """
    prompter = MonolithicMeetingPrompter(
        graph=graph,
        preferences=preferences,
        coords=coords,
        viz_map=viz_map,
        tone=tone,
    )
    return prompter.as_dict_for_pipeline()



# If your helpers are defined elsewhere, they will be used; otherwise these fallbacks kick in.
def _slot_nums_human(slots: List[int]) -> str:
    try:
        return ", ".join(f"{s+1:02d}({SLOT_LABELS[s]})" for s in sorted(slots))
    except Exception:
        return ", ".join(f"{s+1:02d}" for s in sorted(slots))

def _all_building_pairs_distances(coords: Dict[str, Tuple[int, int]], buildings: List[str]) -> List[str]:
    try:
        from math import dist
        uniq = sorted(set(b for b in buildings if b in coords))
        out: List[str] = []
        for i, b1 in enumerate(uniq):
            for b2 in uniq[i+1:]:
                (x1, y1) = coords[b1]; (x2, y2) = coords[b2]
                d = int(dist((x1, y1), (x2, y2)))
                out.append(f"{b1} ↔ {b2} = {d}m")
        return out
    except Exception:
        return []

def _extract_time_match_weights(graph: FactorGraph) -> Dict[MeetingId, float]:
    w: Dict[MeetingId, float] = {mid: 1.0 for mid in graph.meetings.keys()}
    for f in getattr(graph, "factors", []):
        if getattr(f, "ftype", "") == "MEETING_TIME_MATCH":
            var_scope = list(getattr(f, "var_scope", []) or getattr(f, "scope", []) or [])
            wm = float(getattr(f, "payload", {}).get("w", 1.0))
            for mid in var_scope:
                w[mid] = wm
    return w

def _extract_agent_priorities(graph: FactorGraph) -> Dict[AgentId, Dict[MeetingId, int]]:
    pr: Dict[AgentId, Dict[MeetingId, int]] = {}
    for f in getattr(graph, "factors", []):
        if getattr(f, "ftype", "") == "FEASIBILITY_AGENT":
            agent_scope = list(getattr(f, "agent_scope", []) or [])
            if len(agent_scope) != 1:
                continue
            a = agent_scope[0]
            pr_map = dict(getattr(f, "payload", {}).get("priority", {}))
            if pr_map:
                pr[a] = {str(k): int(v) for k, v in pr_map.items()}
    return pr

import json 

def make_prompts_ms_DT(
    graph: FactorGraph,
    preferences: Dict[AgentId, set],
    coords: Dict[str, Tuple[int, int]],
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Revised prompt maker (prompt-only changes).

    Builds:
      - deliberation_prompt (CENTRAL COORDINATOR): robust parsing + normalization rules,
        unit-correct feasibility, and a concrete search plan with an audit step.

      - json_mode_prompt: strict, sorted, flat JSON contract.

      - personal_instructions: per-agent block that yields a STRICT JSON payload
        (no prose) the central can reliably parse.
    """
    # ---------- small local helpers (prompt-only formatting) ----------
    def _hour_to_slot_map_str() -> str:
        return "{ " + ", ".join(f"{8+i}:{i+1}" for i in range(10)) + " }"

    def _slots_line() -> str:
        return ", ".join(f"{i+1:02d}({lab})" for i, lab in enumerate(SLOT_LABELS))

    def _agent_meeting_json_snippets(a: AgentId) -> List[str]:
        """Produce one-line JSON examples for meetings attended by agent a."""
        out: List[str] = []
        tw = _extract_time_match_weights(graph)
        for m in sorted((m for m in graph.meetings.values() if a in m.attendees), key=lambda x: x.mid):
            mode = "PHYSICAL" if getattr(m, "mode", "ZOOM") == "PHYSICAL" else "ZOOM"
            loc = getattr(m, "location", None) if mode == "PHYSICAL" else None
            owner = getattr(m, "owner", None)
            attendees = list(m.attendees)
            w = float(tw.get(m.mid, 1.0))
            # Compact one-line JSON object as a string for copy-paste
            loc_json = f"\"{loc}\"" if loc is not None else "null"
            out.append(
                "{"
                f"\"mid\":\"{m.mid}\","
                f"\"mode\":\"{mode}\","
                f"\"location\":{loc_json},"
                f"\"owner\":{json.dumps(owner)},"
                f"\"attendees\":{json.dumps(attendees)},"
                f"\"w\":{w:g}"
                "}"
            )
        return out

    def _agent_travel_minutes(a: AgentId) -> List[Tuple[str, str, int]]:
        """All pairwise distances (minutes) between buildings relevant to agent a."""
        # buildings the agent actually visits physically
        bs: List[str] = []
        for m in graph.meetings.values():
            if getattr(m, "mode", None) == "PHYSICAL" and a in m.attendees and getattr(m, "location", None):
                bs.append(m.location)
        uniq = sorted(set(bs))
        triplets: List[Tuple[str, str, int]] = []
        from math import dist
        for i, b1 in enumerate(uniq):
            for b2 in uniq[i+1:]:
                (x1, y1) = coords[b1]; (x2, y2) = coords[b2]
                triplets.append((b1, b2, int(dist((x1, y1), (x2, y2)))))
        return triplets

    # ---------- 1) CENTRAL DELIBERATION (instance-agnostic) ----------
    dl: List[str] = []
    dl.append("## ROLE")
    dl.append("You are the CENTRAL COORDINATOR. Assign exactly one slot (1..10) to every meeting whose id matches ^M\\d{3}$ to maximize the environment score.")

    dl.append("")
    dl.append("## WHAT YOU RECEIVE")
    dl.append("Multiple AGENT blocks, each starting with:")
    dl.append("  [AGENT_ID=<Name>]")
    dl.append("followed by noisy text lines (not guaranteed to be valid JSON). You MUST robustly extract facts using the rules below.")

    dl.append("")
    dl.append("## ROBUST PARSING RULES (MANDATORY)")
    dl.append("- Meeting tokens: Any token matching ^M\\d{3}$ is a meeting id. Collect all such ids across all blocks to form the canonical meeting set.")
    dl.append("- Mode/location: If a token like PHYSICAL@<Location> appears next to a meeting in ANY block, that meeting is PHYSICAL with that location; otherwise it is ZOOM (location = null).")
    dl.append("- Attendees/owner: If you see owner=<Name> for a meeting, treat it as a candidate owner. Attendees may appear in lists like others=[..., ...] or A/B/C or Name,Name. Aggregate attendees across blocks (union). Owner = plurality winner among candidates; break ties lexicographically. If no owner appears, it is unspecified (does not affect scoring).")
    dl.append("- Weight: If w_M=<num> or w=<num> appears for a meeting in ANY block, set w_m to that numeric value; else default w_m=1.")
    dl.append("- Preferred slots (by agent): Accept any of these and CONVERT to slots:")
    dl.append("  • preferred_slots: [1..10]")
    dl.append("  • phrases such as “I prefer slots 01, 02, 10”.")
    dl.append(f"  • prefer_hours / PREFER_HOURS: map hours→slots via { _hour_to_slot_map_str() }")
    dl.append("- Priorities (by agent): Accept priority:{...} or PRIORITY:{...}. Higher number = higher priority. If missing, treat all that agent’s meetings as equal priority.")
    dl.append("- Travel times: Accept A-B(<minutes>m) patterns, travel_time:{{A-B:<minutes>}}, or travel_minutes=[[A,B,<minutes>],...]. Travel is symmetric. If missing, travel(A,B)=0.")
    dl.append("- Ignore non-meeting tokens (e.g., T_*). Only ids ^M\\d{3}$ matter for the final JSON keys.")

    dl.append("")
    dl.append("## TIME SLOTS")
    dl.append(f"There are 10 one-hour slots with starts: { _slots_line() }")
    dl.append("StartMinute(slot) = 480 + 60*(slot-1).  (01→08:00, …, 10→17:00)")

    dl.append("")
    dl.append("## OBJECTIVE (ENVIRONMENT SCORE)")
    dl.append("TotalScore =")
    dl.append("  ∑_m [ w_m × (# of m’s attendees whose preferred_slots contain chosen_slot(m)) ]")
    dl.append("  + ∑_agent [ # of that agent’s meetings they can attend after the greedy feasibility filter ].")

    dl.append("")
    dl.append("## FEASIBILITY FILTER (PER AGENT, GREEDY; MANDATORY)")
    dl.append("1) Build that agent’s candidate list = all meetings they attend, with chosen slots.")
    dl.append("2) Sort by (priority DESC, start slot ASC).")
    dl.append("3) Traverse: keep a meeting if (a) its slot ≠ any kept meeting’s slot (no overlaps), and")
    dl.append("   (b) if both current and previous kept are PHYSICAL in different buildings:")
    dl.append("       StartMinute(curr) − StartMinute(prev) ≥ travel_minutes(prev_loc, curr_loc).")
    dl.append("   ZOOM has 0 travel; same building PHYSICAL has 0 travel.")
    dl.append("4) The agent’s feasibility contribution is the count of kept meetings.")

    dl.append("")
    dl.append("## SEARCH PLAN (MANDATORY; PERFORM THIS IN YOUR SCRATCHPAD)")
    dl.append("Phase A — Initialization")
    dl.append("• For each meeting m, compute prefCount_m[s] = number of its attendees who prefer slot s (1..10).")
    dl.append("• Choose initial slot s* = argmax_s ( w_m × prefCount_m[s] ).")
    dl.append("• Tie-breaker for s*: earlier slot → covers more high-priority attendees (sum of their priorities) → larger raw prefCount → lower meeting id.")
    dl.append("Phase B — First-Improve Local Search")
    dl.append("• Repeat until one full pass yields no improvement:")
    dl.append("  – For each meeting m (ascending by id), evaluate its top-3 alternative slots by Δscore, where:")
    dl.append("      Δscore = Δ(Meeting-Time-Match) + Δ(Feasibility-Agent)  [recompute feasibility for affected agents only].")
    dl.append("  – Apply the single best move with Δscore > 0 (first-improve).")
    dl.append("• Keep the best schedule seen.")

    dl.append("")
    dl.append("## NORMALIZATION & SAFETY")
    dl.append("- If a meeting is ever PHYSICAL in any block, treat it as PHYSICAL overall.")
    dl.append("- If a PHYSICAL meeting has conflicting locations across blocks, choose the plurality location; if tied, pick lexicographically first.")
    dl.append("- If a PHYSICAL meeting ends up without a location, treat it as ZOOM.")
    dl.append("- Ignore any token not matching ^M\\d{3}$ for meeting ids.")

    dl.append("")
    dl.append("## PRE-JSON OUTPUT FEASIBILITY AUDIT (SCRATCHPAD ONLY)")
    dl.append("Before emitting JSON, list one line per agent:")
    dl.append("  Agent A: [Mxxx@S.. keep/skip …] gaps: [..minutes..]")
    dl.append("to verify greedy ordering and travel gaps. Do NOT include this audit in the final output.")

    dl.append("")
    dl.append("## JSON OUTPUT RULES")
    dl.append("• Output EXACTLY one flat JSON object with EVERY canonical meeting id mapped to a slot integer 1..10.")
    dl.append("• Keys must be the discovered meeting ids sorted ascending (M001, M002, …).")
    deliberation_prompt = "\n".join(dl)

    # ---------- 2) JSON-MODE (flat; meeting -> 1..10) ----------
    jm: List[str] = []
    jm.append("## INSTRUCTIONS (JSON-MODE)")
    jm.append("Emit EXACTLY one JSON object mapping EVERY discovered meeting id (matching ^M\\d{3}$) to a 1-based slot in [1..10].")
    jm.append("")
    jm.append("Validation checklist (MANDATORY):")
    jm.append("1) Flat object only; no nesting.")
    jm.append("2) Keys: exactly the canonical meeting ids you discovered, sorted ascending (M001..).")
    jm.append("3) Values: integers 1..10 only.")
    jm.append("4) No extra keys, no comments, no trailing text. Valid JSON only.")
    json_mode_prompt = "\n".join(jm)

    # ---------- 3) PERSONAL INSTRUCTIONS (per agent, STRICT JSON BLOCKS) ----------
    time_w = _extract_time_match_weights(graph)
    agent_prio = _extract_agent_priorities(graph)

    personal_instructions: Dict[AgentId, str] = {}
    for a in graph.agents:
        lines: List[str] = []
        lines.append(f"# PERSONAL INSTRUCTION FOR {a}")
        lines.append("")
        lines.append("Paste the header, then emit EXACTLY ONE fenced JSON object (no prose).")
        lines.append(f"[AGENT_ID={a}]")
        lines.append("")
        lines.append("## SLOTS")
        lines.append(f"SLOTS (1..10): { _slots_line() }")
        lines.append(f"Hour→Slot map: { _hour_to_slot_map_str() }")
        lines.append("")
        lines.append("## WHAT TO OUTPUT (STRICT)")
        lines.append("Output EXACTLY this JSON schema (single object):")
        lines.append("```json")
        lines.append("{")
        lines.append('  "meetings": [')
        # include one-line JSON objects for this agent's meetings, as copyable examples
        meeting_objs = _agent_meeting_json_snippets(a)
        if meeting_objs:
            for i, obj in enumerate(meeting_objs):
                comma = "," if i < len(meeting_objs) - 1 else ""
                lines.append(f"    {obj}{comma}")
        lines.append("  ],")
        # preferred slots from our ground truth preferences (1-based)
        prefs_1b = sorted([(s+1) for s in preferences.get(a, set())])
        lines.append(f'  "preferred_slots": {json.dumps(prefs_1b)},')
        # priority map (if any)
        pr_map = agent_prio.get(a, {})
        if pr_map:
            # stable order by meeting id
            pr_sorted = {mid: pr_map[mid] for mid in sorted(pr_map.keys())}
            lines.append(f'  "priority": {json.dumps(pr_sorted)},')
        else:
            lines.append('  "priority": {},')
        # travel minutes list for this agent
        tm = _agent_travel_minutes(a)
        if tm:
            # [["B1","B2",min], ...]
            tm_list = [[b1, b2, m] for (b1, b2, m) in tm]
            lines.append(f'  "travel_minutes": {json.dumps(tm_list)}')
        else:
            lines.append('  "travel_minutes": []')
        lines.append("}")
        lines.append("```")
        lines.append("")
        lines.append("Rules:")
        lines.append("- No extra keys. No trailing text outside the JSON fence.")
        lines.append("- Use integers 1..10 for preferred_slots.")
        lines.append("- Keep meetings exactly as listed; do not invent or rename ids.")
        # Light reminder of weights for visibility (not required if all 1.0)
        # Also keep w_M explicit for clarity in central parsing
        if any(time_w.get(m.mid, 1.0) != 1.0 for m in graph.meetings.values() if a in m.attendees):
            lines.append("- Each meeting object includes its weight as the field \"w\" (w_M).")
        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


def make_prompts_ms_DT_as_dict(
    graph: FactorGraph,
    preferences: Dict[AgentId, set],
    coords: Dict[str, Tuple[int, int]],
    *,
    viz_map: Optional[Dict[AgentId, str]] = None,  # <-- accept for API compatibility (ignored)
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter exposing sentinel keys for pipelines.
    `viz_map` is accepted for compatibility with save_meeting_env_instance_viz_only(...)
    and is ignored by default.
    """
    parts = make_prompts_ms_DT(graph, preferences, coords, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out



import json
from typing import Any, Dict, List, Optional, Tuple, Set

AgentId = str
MeetingId = str

def _mode(m) -> str:
    return "PHYSICAL" if getattr(m, "mode", "ZOOM") == "PHYSICAL" else "ZOOM"

def _loc(m) -> Optional[str]:
    return getattr(m, "location", None) if _mode(m) == "PHYSICAL" else None

def _pref_1b(preferences: Dict[AgentId, Set[int]], a: AgentId) -> List[int]:
    # convert 0-based slot ids to 1..10
    return sorted([s + 1 for s in preferences.get(a, set()) if 0 <= s < 10])

def make_prompts_ms_DPT_descriptive(
    graph,                                  # FactorGraph with .meetings: Dict[MeetingId, Meeting]
    preferences: Dict[AgentId, Set[int]],   # per-agent preferred slots (0-based)
    coords: Dict[str, Tuple[int, int]],     # unused; kept for API symmetry
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    DPT prompt maker (description-only).
    For each AGENT, produce a two-section descriptive block:

      SECTION 1: List of meetings the agent attends.
        - (meeting id, mode/location, attendees, agent's own preferred slots 1..10)

      SECTION 2: List of meetings the agent controls (is the owner of).
        - (meeting id, mode/location, attendees)

    No instructions, no JSON contract, just plain descriptions.
    """
    # Empty shared prompts (description-only)
    deliberation_prompt = ""
    json_mode_prompt = ""

    # Build attendee index for quick lookup
    meetings_sorted = sorted(graph.meetings.values(), key=lambda m: m.mid)

    # Agents list: prefer graph.agents if available; else derive from attendees
    agents: List[AgentId]
    if hasattr(graph, "agents") and isinstance(graph.agents, list):
        agents = list(graph.agents)
    else:
        pool: Set[AgentId] = set()
        for m in meetings_sorted:
            pool.update(m.attendees)
        agents = sorted(pool)

    personal_instructions: Dict[AgentId, str] = {}

    for a in agents:
        # Section 1 — meetings this agent attends
        attends = [m for m in meetings_sorted if a in m.attendees]

        # Section 2 — meetings this agent owns/controls
        controls = [m for m in meetings_sorted if getattr(m, "owner", None) == a]

        lines: List[str] = []
        lines.append(f"SECTION 1 — Meetings attended by {a}")
        if attends:
            for m in attends:
                mode = _mode(m)
                loc  = _loc(m)
                loc_str = f"{mode} @ {loc}" if mode == "PHYSICAL" else "ZOOM @ null"
                lines.append(
                    f"- {m.mid} | {loc_str} | attendees={json.dumps(m.attendees)} | "
                    f"your_preferred_slots={_pref_1b(preferences, a)}"
                )
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append(f"SECTION 2 — Meetings controlled by {a}")
        if controls:
            for m in controls:
                mode = _mode(m)
                loc  = _loc(m)
                loc_str = f"{mode} @ {loc}" if mode == "PHYSICAL" else "ZOOM @ null"
                lines.append(
                    f"- {m.mid} | {loc_str} | attendees={json.dumps(m.attendees)}"
                )
        else:
            lines.append("- (none)")

        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,   # intentionally empty
        "json_mode_prompt": json_mode_prompt,         # intentionally empty
        "personal_instructions": personal_instructions,
    }

def make_prompts_ms_DPT_descriptive_as_dict(
    graph,
    preferences: Dict[AgentId, Set[int]],
    coords: Dict[str, Tuple[int, int]],
    *,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that flattens to sentinel-key dict for pipelines.
    Each AGENT block is exposed as __AGENT__<AgentId>.
    """
    parts = make_prompts_ms_DPT_descriptive(graph, preferences, coords, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out




def make_prompts_ms_DPI_descriptive(
    graph,                                  # FactorGraph with .meetings: Dict[MeetingId, Meeting]
    preferences: Dict[AgentId, Set[int]],   # per-agent preferred slots (0-based)
    coords: Dict[str, Tuple[int, int]],     # unused; kept for API symmetry
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    DPT prompt maker (description-only).
    For each AGENT, produce a two-section descriptive block:

      SECTION 1: List of meetings the agent attends.
        - (meeting id, mode/location, attendees, agent's own preferred slots 1..10)

      SECTION 2: List of meetings the agent controls (is the owner of).
        - (meeting id, mode/location, attendees)

    No instructions, no JSON contract, just plain descriptions.
    """
    # Empty shared prompts (description-only)
    deliberation_prompt = ""
    json_mode_prompt = ""

    # Build attendee index for quick lookup
    meetings_sorted = sorted(graph.meetings.values(), key=lambda m: m.mid)

    # Agents list: prefer graph.agents if available; else derive from attendees
    agents: List[AgentId]
    if hasattr(graph, "agents") and isinstance(graph.agents, list):
        agents = list(graph.agents)
    else:
        pool: Set[AgentId] = set()
        for m in meetings_sorted:
            pool.update(m.attendees)
        agents = sorted(pool)

    personal_instructions: Dict[AgentId, str] = {}

    for a in agents:
        # Section 1 — meetings this agent attends
        attends = [m for m in meetings_sorted if a in m.attendees]

        # Section 2 — meetings this agent owns/controls
        controls = [m for m in meetings_sorted if getattr(m, "owner", None) == a]

        lines: List[str] = []
        lines.append(f"SECTION 1 — Meetings attended by {a}")
        if attends:
            for m in attends:
                mode = _mode(m)
                loc  = _loc(m)
                loc_str = f"{mode} @ {loc}" if mode == "PHYSICAL" else "ZOOM @ null"
                lines.append(
                    f"- {m.mid} | {loc_str} | attendees={json.dumps(m.attendees)} | "
                )
            lines.append(f"your_preferred_slots and building maps can be viewed in the visualization files provided separately.")
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append(f"SECTION 2 — Meetings controlled by {a}")
        if controls:
            for m in controls:
                mode = _mode(m)
                loc  = _loc(m)
                loc_str = f"{mode} @ {loc}" if mode == "PHYSICAL" else "ZOOM @ null"
                lines.append(
                    f"- {m.mid} | {loc_str} | attendees={json.dumps(m.attendees)}"
                )
        else:
            lines.append("- (none)")

        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,   # intentionally empty
        "json_mode_prompt": json_mode_prompt,         # intentionally empty
        "personal_instructions": personal_instructions,
    }

def make_prompts_ms_DPI_descriptive_as_dict(
    graph,
    preferences: Dict[AgentId, Set[int]],
    coords: Dict[str, Tuple[int, int]],
    *,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that flattens to sentinel-key dict for pipelines.
    Each AGENT block is exposed as __AGENT__<AgentId>.
    """
    parts = make_prompts_ms_DPI_descriptive(graph, preferences, coords, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out
