
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json, random, math, itertools
# ---------------- Core datatypes ----------------

AgentId   = str
MeetingId = str
Building  = str
SlotId    = int  # 0..9 (0=8am, ..., 9=5pm start)
SLOT_LABELS = [f"{8 + i}:00" for i in range(10)]  # 8:00..17:00 start (1h each)

# ---------------- Meetings (variables) ----------------

@dataclass
class Meeting:
    """
    A meeting variable M_i has:
      - attendees: the set of agents who must attend
      - owner: one of the attendees (organizer/owner variable)
      - mode/location
      - duration in 1-hour slots (keep 1 for simplicity)
    """
    mid: MeetingId
    attendees: List[AgentId]
    owner: AgentId                       # MUST be in attendees
    mode: str                            # "ZOOM" or "PHYSICAL"
    location: Optional[Building] = None  # required if PHYSICAL
    duration_slots: int = 1

    def __post_init__(self):
        self.mode = self.mode.upper()
        if self.owner not in self.attendees:
            raise ValueError(f"Owner {self.owner} is not an attendee of {self.mid}")
        if self.mode not in {"ZOOM", "PHYSICAL"}:
            raise ValueError(f"Meeting.mode must be ZOOM or PHYSICAL, got {self.mode}")
        if self.mode == "PHYSICAL" and not self.location:
            raise ValueError(f"PHYSICAL meeting {self.mid} must have a location")

# ---------------- Factors (dual-scope) ----------------

@dataclass
class Factor:
    """
    Dual-scope factors:
      - var_scope: list of MeetingId variables referenced by the factor
      - agent_scope: list of AgentId participating in this factor
      - ftype: one of {"MEETING_TIME_MATCH", "FEASIBILITY_AGENT"}
      - payload: extra knobs (e.g., weight)

    eval(...) returns a *float* score for this factor instance.
    """
    fid: str
    var_scope: List[MeetingId]
    agent_scope: List[AgentId]
    ftype: str
    payload: Dict[str, Any] = field(default_factory=dict)

    # --- helpers ---
    @staticmethod
    def _travel_minutes(m_i: Meeting, m_j: Meeting,
                        coords: Dict[Building, Tuple[int, int]]) -> int:
        if m_i.mode != "PHYSICAL" or m_j.mode != "PHYSICAL":
            return 0
        if (m_i.location not in coords) or (m_j.location not in coords):
            # If we can't resolve coords, be conservative (no travel credit)
            return 0
        (x1, y1) = coords[m_i.location]
        (x2, y2) = coords[m_j.location]
        return int(math.dist((x1, y1), (x2, y2)))  # minutes

    @staticmethod
    def _slot_to_interval(slot: SlotId, duration_slots: int) -> Tuple[int, int]:
        start_min = 60 * slot
        end_min   = start_min + 60 * duration_slots
        return start_min, end_min

    def eval(self,
             schedule: Dict[MeetingId, SlotId],
             preferences: Dict[AgentId, set],
             coords: Dict[Building, Tuple[int, int]],
             meeting_lut: Dict[MeetingId, Meeting]) -> float:

        # If any scoped meeting unscheduled, this factor contributes 0 (conservative)
        if any(mid not in schedule for mid in self.var_scope):
            return 0.0

        if self.ftype == "MEETING_TIME_MATCH":
            # Single meeting in var_scope; agent_scope are the attendees we check.
            assert len(self.var_scope) == 1, "MEETING_TIME_MATCH must have a single var in scope"
            mid = self.var_scope[0]
            slot = schedule[mid]
            weight = float(self.payload.get("w", 1.0))
            # Score = weight * (# agents in agent_scope who prefer this slot)
            count = sum(1 for a in self.agent_scope if slot in preferences.get(a, set()))
            return weight * float(count)

        # --- in Factor.eval(), replace the FEASIBILITY_AGENT block with this ---

        if self.ftype == "FEASIBILITY_AGENT":
            # One agent in scope; var_scope = all meetings they *could* attend
            assert len(self.agent_scope) == 1, "FEASIBILITY_AGENT must have a single agent in scope"
            a = self.agent_scope[0]
            prio_map: Dict[MeetingId, int] = self.payload.get("priority", {})
            weight = float(self.payload.get("w", 1.0))

            # Meetings that actually include this agent
            ms = [meeting_lut[m] for m in self.var_scope if a in meeting_lut[m].attendees]
            if not ms:
                return 0.0

            # Helper: minutes interval for a meeting given the schedule
            def interval(m: Meeting) -> Tuple[int, int]:
                s, e = self._slot_to_interval(schedule[m.mid], m.duration_slots)
                return s, e

            # Check pairwise compatibility with already selected meetings
            def conflicts(m: Meeting, selected: List[Meeting]) -> bool:
                s_m, e_m = interval(m)
                for x in selected:
                    s_x, e_x = interval(x)

                    # True time overlap?
                    if (e_m > s_x) and (e_x > s_m):
                        return True

                    # No overlap: ensure enough travel time in the actual order
                    if e_x <= s_m:  # x -> m
                        gap = s_m - e_x
                        if self._travel_minutes(x, m, coords) > gap:
                            return True
                    elif e_m <= s_x:  # m -> x
                        gap = s_x - e_m
                        if self._travel_minutes(m, x, coords) > gap:
                            return True
                return False

            # Sort by priority (desc), then by start time (asc) as a tiebreaker
            ms_sorted = sorted(
                ms,
                key=lambda m: (-int(prio_map.get(m.mid, 0)), schedule[m.mid], m.duration_slots, m.mid)
            )

            attended: List[Meeting] = []
            for m in ms_sorted:
                if not conflicts(m, attended):
                    attended.append(m)

            # Reward = number of meetings the agent can actually attend
            return weight * float(len(attended))


        # Unknown factor type
        return 0.0

# ---------------- Factor graph ----------------

@dataclass
class FactorGraph:
    """
    Variables are Meetings (each with an owner).
    An assignment is a dict {MeetingId -> SlotId}.
    """
    agents: List[AgentId]
    meetings: Dict[MeetingId, Meeting]
    factors: List[Factor] = field(default_factory=list)

    def add_factor(self, f: Factor) -> None:
        self.factors.append(f)

    def validate_owners(self):
        for m in self.meetings.values():
            if m.owner not in m.attendees:
                raise ValueError(f"Owner {m.owner} not in attendees for {m.mid}")

    def global_score(self,
                     schedule: Dict[MeetingId, SlotId],
                     preferences: Dict[AgentId, set],
                     coords: Dict[Building, Tuple[int, int]]) -> float:
        meeting_lut = self.meetings
        return sum(f.eval(schedule, preferences, coords, meeting_lut) for f in self.factors)

# ---------------- Instance container ----------------

@dataclass
class MeetingEnvInstance:
    graph: FactorGraph
    buildings: List[Building]
    coords: Dict[Building, Tuple[int, int]]
    preferences: Dict[AgentId, set]  # preferred slots per agent (set of SlotId)
    time_slots: List[SlotId] = field(default_factory=lambda: list(range(10)))
