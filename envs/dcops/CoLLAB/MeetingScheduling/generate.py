from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json, random, math, itertools
# Use relative import since data_structure.py is in the same package
from .data_structure import AgentId, MeetingId, Building, SlotId, SLOT_LABELS, Meeting, Factor, FactorGraph, MeetingEnvInstance


# ---------------- IO helpers ----------------

def _as_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

# ---------------- Data loading ----------------

def _load_names(root: Path) -> List[str]:
    p = root / "names.json"
    if p.exists():
        return _read_json(p)["names"]
    return ["Alice", "Bob", "Charlie", "Dinesh", "Eve", "Fatima", "Grace", "Henry", "Ivy", "Jamal"]

def _load_buildings(root: Path) -> List[Building]:
    p = root / "buildings_umass.json"
    if p.exists():
        return _read_json(p)["buildings"]
    return ["Computer Science Building", "Morrill", "W.E.B. Du Bois Library", "Hasbrouck"]

def _make_agents(n_agents: int, names: List[str]) -> List[AgentId]:
    agents: List[AgentId] = []
    i = 0
    while len(agents) < n_agents:
        suffix = f"_{i // len(names) + 1}" if i >= len(names) else ""
        agents.append(names[i % len(names)] + suffix)
        i += 1
    return agents

def _make_coords(buildings: List[Building], rng: random.Random) -> Dict[Building, Tuple[int, int]]:
    return {b: (rng.randint(0, 60), rng.randint(0, 60)) for b in buildings}

def _new_fid_gen():
    i = 0
    def new():
        nonlocal i; i += 1
        return f"F{i:05d}"
    return new

# ---------------- Generation under new spec ----------------

def build_meeting_env(
    n_agents: int,
    n_meetings: int,
    data_root: Path | str,
    rng_seed: int = 7,
    max_attendees_per_meeting: int = 4,
    p_zoom: float = 0.4,
    min_prefs_per_agent: int = 4,
    max_prefs_per_agent: int = 7,
    time_match_weight: float = 1.0,
) -> MeetingEnvInstance:
    """
    Create a meeting scheduling instance:
      - Variables are meetings with *owners* (owners are sampled from attendees).
      - Factor types:
          * MEETING_TIME_MATCH per meeting (var_scope=[M_i], agent_scope=attendees)
          * FEASIBILITY_AGENT per agent (var_scope=all meetings that agent attends, agent_scope=[agent])
      - Preferences: each agent likes a random subset of slots.
      - Travel time = integer Euclidean between buildings (minutes); Zoom => 0.
    """
    rng = random.Random(rng_seed)
    root = _as_path(data_root)

    names = _load_names(root)
    buildings = _load_buildings(root)
    agents = _make_agents(n_agents, names)
    coords = _make_coords(buildings, rng)

    # Build meetings (variables with owners)
    mtgs: Dict[MeetingId, Meeting] = {}
    for i in range(n_meetings):
        k = rng.randint(2, min(max_attendees_per_meeting, n_agents))
        attendees = rng.sample(agents, k)
        owner = rng.choice(attendees)             # <-- owner is one of the attendees
        if rng.random() < p_zoom:
            mode, loc = "ZOOM", None
        else:
            mode, loc = "PHYSICAL", rng.choice(buildings)
        mid = f"M{i+1:03d}"
        mtgs[mid] = Meeting(mid=mid, attendees=attendees, owner=owner, mode=mode, location=loc)

    # Preferences: each agent likes some slots (0..9)
    slots = list(range(10))
    preferences: Dict[AgentId, set] = {
        a: set(rng.sample(slots, rng.randint(min_prefs_per_agent, max_prefs_per_agent)))
        for a in agents
    }

    # ---- NEW: per-agent priority maps over their meetings ----
    agent_priority: Dict[AgentId, Dict[MeetingId, int]] = {}
    for a in agents:
        ms_for_a = [mid for mid, m in mtgs.items() if a in m.attendees]
        # Random strict order; highest priority = largest integer
        order = rng.sample(ms_for_a, len(ms_for_a))
        agent_priority[a] = {mid: (len(order) - i) for i, mid in enumerate(order)}

    # Factors
    new_fid = _new_fid_gen()
    factors: List[Factor] = []

    # (1) MEETING_TIME_MATCH per meeting (unchanged)
    for mid, m in mtgs.items():
        factors.append(Factor(
            fid=new_fid(),
            var_scope=[mid],
            agent_scope=list(m.attendees),
            ftype="MEETING_TIME_MATCH",
            payload={"w": float(time_match_weight)}
        ))

    # (2) FEASIBILITY_AGENT per agent, now with priority map
    for a in agents:
        agent_meetings = [mid for mid, m in mtgs.items() if a in m.attendees]
        if agent_meetings:
            factors.append(Factor(
                fid=new_fid(),
                var_scope=agent_meetings,
                agent_scope=[a],
                ftype="FEASIBILITY_AGENT",
                payload={"priority": agent_priority[a]}  # <-- key change
            ))

    graph = FactorGraph(agents=agents, meetings=mtgs, factors=factors)
    graph.validate_owners()
    return MeetingEnvInstance(graph=graph, buildings=buildings, coords=coords, preferences=preferences)

# ---------------- Assignment helpers ----------------

def schedule_from_choices(meetings: Dict[MeetingId, Meeting],
                          choices: Dict[MeetingId, int]) -> Dict[MeetingId, SlotId]:
    """
    Build a schedule from 1-based user choices in [1..10].
    """
    sched: Dict[MeetingId, SlotId] = {}
    for mid, choice in choices.items():
        if mid not in meetings:
            raise KeyError(f"Unknown meeting in choices: {mid}")
        i = int(choice)
        if not (1 <= i <= 10):
            raise ValueError(f"Choice for {mid} out of range: {i} (valid 1..10)")
        sched[mid] = i - 1
    return sched

def make_preferred_or_random_schedule(meetings: Dict[MeetingId, Meeting],
                                      preferences: Dict[AgentId, set],
                                      rng: Optional[random.Random] = None) -> Dict[MeetingId, SlotId]:
    """
    For each meeting, try to pick a slot maximizing attendee preference matches.
    If ties or none, pick a random slot.
    """
    r = rng or random.Random(0)
    sched: Dict[MeetingId, SlotId] = {}
    for mid, m in meetings.items():
        # Score each slot by # of attendees who prefer it; pick best (break ties randomly)
        best_slots, best_score = [], -1
        for slot in range(10):
            s = sum(1 for a in m.attendees if slot in preferences.get(a, set()))
            if s > best_score:
                best_slots, best_score = [slot], s
            elif s == best_score:
                best_slots.append(slot)
        sched[mid] = r.choice(best_slots) if best_slots else r.randint(0, 9)
    return sched

# ---------------- Visualization (unchanged APIs) ----------------

def draw_dummy_calendar(graph: FactorGraph,
                        preferences: Dict[AgentId, set],
                        figsize=(12, 2.2)):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    for slot in range(10):
        ax.add_patch(plt.Rectangle((slot, 0), 1, 1, fill=False, linewidth=1.5))
        preferred_meetings = []
        for m in graph.meetings.values():
            inter_ok = all(slot in preferences.get(a, set()) for a in m.attendees)
            if inter_ok:
                preferred_meetings.append(m.mid)
        label = "\n".join(preferred_meetings)
        ax.text(slot + 0.5, 0.7, SLOT_LABELS[slot], ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(slot + 0.5, 0.35, label, ha="center", va="center", fontsize=10)
    return fig

def draw_agent_building_map(instance: MeetingEnvInstance,
                            agent: AgentId,
                            figsize=(12, 7),
                            node_fontsize=16,
                            edge_fontsize=14,
                            node_size=1800):
    try:
        import matplotlib.pyplot as plt, networkx as nx
    except Exception:
        return None

    buildings = []
    for m in instance.graph.meetings.values():
        if m.mode == "PHYSICAL" and agent in m.attendees and m.location:
            buildings.append(m.location)
    buildings = sorted(set(buildings))
    if not buildings:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No physical meetings for {agent}", ha="center", va="center", fontsize=14)
        return fig

    G = nx.Graph()
    for b in buildings:
        (x, y) = instance.coords[b]
        G.add_node(b, pos=(x, y))
    for i, b1 in enumerate(buildings):
        for b2 in buildings[i+1:]:
            (x1, y1) = instance.coords[b1]
            (x2, y2) = instance.coords[b2]
            d = int(math.dist((x1, y1), (x2, y2)))
            G.add_edge(b1, b2, weight=d)
    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=node_fontsize, ax=ax)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels={(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)},
                                 font_size=edge_fontsize, ax=ax)
    ax.set_title(f"{agent} • Physical meeting map", fontsize=node_fontsize, pad=12)
    ax.axis("off")
    return fig

def combine_calendar_and_map(cal_fig, map_fig, out_path: Path | str = "meeting_calendar_map.png") -> Optional[str]:
    if cal_fig is None or map_fig is None:
        return None
    try:
        import io
        from PIL import Image
    except Exception:
        return None
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    cal_fig.savefig(buf1, format="png", bbox_inches="tight", dpi=150)
    map_fig.savefig(buf2, format="png", bbox_inches="tight", dpi=150)
    buf1.seek(0); buf2.seek(0)
    img1, img2 = Image.open(buf1).convert("RGBA"), Image.open(buf2).convert("RGBA")
    width = max(img1.width, img2.width)
    canvas = Image.new("RGBA", (width, img1.height + img2.height), (255, 255, 255, 0))
    canvas.paste(img1, (0, 0), img1)
    canvas.paste(img2, (0, img1.height), img2)
    out = _as_path(out_path)
    canvas.save(out)
    return str(out)

# ---------------- Tiny demo ----------------

def demo_instance():
    data_root = "MeetingAssistant/data/"
    inst = build_meeting_env(
        n_agents=6,
        n_meetings=8,
        data_root=data_root,
        rng_seed=42,
        max_attendees_per_meeting=4,
        p_zoom=0.35,
        min_prefs_per_agent=4,
        max_prefs_per_agent=7,
        time_match_weight=1.0,
    )

    # Heuristic schedule
    rng = random.Random(0)
    schedule = make_preferred_or_random_schedule(inst.graph.meetings, inst.preferences, rng=rng)

    # Score
    score = inst.graph.global_score(schedule, inst.preferences, inst.coords)

    # Visuals
    first_agent = inst.graph.agents[0]
    cal_fig = draw_dummy_calendar(inst.graph, inst.preferences)
    map_fig = draw_agent_building_map(inst, first_agent)
    combined_path = combine_calendar_and_map(cal_fig, map_fig, out_path="meeting_demo.png")

    print("Agents:", ", ".join(inst.graph.agents))
    print("Meetings (owner shown):")
    for m in inst.graph.meetings.values():
        loc = m.location if m.mode == "PHYSICAL" else m.mode
        attendees = ", ".join(m.attendees)
        chosen = SLOT_LABELS[schedule[m.mid]]
        print(f"  {m.mid}: {loc} | owner={m.owner} | attendees=[{attendees}] | slot={chosen}")
    print("\nGlobal score:", score)
    if combined_path:
        print("Combined calendar+map image saved to:", combined_path)
    else:
        print("Combined image could not be created (missing matplotlib/networkx/PIL?).")

if __name__ == "__main__":
    demo_instance()
