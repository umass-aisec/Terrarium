from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import json
import math
import random
from pathlib import Path
import itertools

# Use relative imports since these files are in the same package
from .data_structure import InstanceSpec
from .generate import load_device_catalog_json, _contiguous_runs


def make_prompts_powerlite_from_catalog(instance: InstanceSpec, catalog_path: str) -> Dict[str, str]:
    devices = load_device_catalog_json(catalog_path)
    lookup = {d["id"]: d for d in devices}

    header = [
        "You are an agent for a single home in a neighborhood.",
        "Choose ONE start time for each task (start-time-only).",
        "Any neighborhood demand over S[t] pulls from the main grid (coal).",
        "Goal: minimize total main-grid energy while honoring allowed starts.",
        "Use tools to schedule tasks.",
        "S_cap:",
        " " + json.dumps(instance.S_cap),
    ]
    header_txt = "\n".join(header)

    def _intervals(allowed: List[int]) -> List[Tuple[int, int]]:
        return _contiguous_runs(sorted(set(allowed)))

    prompts: Dict[str, str] = {}
    for h in instance.homes:
        lines = ["Tasks:"]
        for t in h.tasks:
            base = t.id.split("_")[0]
            desc = lookup.get(base, {}).get("description", "")
            intervals_str = ", ".join([f"[{a},{b}]" for a, b in _intervals(t.allowed_starts)])
            lines.append(
                f" - id={t.id}; desc=\"{desc}\"; consumption={t.consumption}; duration={t.duration}; allowed={intervals_str}"
            )
        prompts[h.id] = header_txt + "\n\nHome: " + h.id + "\n" + "\n".join(lines) + ". Use tools."

    return prompts


# =========================
# SmartGrid: monolithic prompts by parts
# =========================
def make_prompts_powerlite_monolithic_parts(
    instance: InstanceSpec, catalog_path: str, *, tone: str = "standard"
) -> Dict[str, str]:
    """
    Returns three distinct prompts for Neighborhood-PowerLite (SmartGrid):
    - 'task_prompt' (problem statement + full instance details)
    - 'deliberation_instructions' (think-first instructions)
    - 'json_mode_instructions' (minimal parser-facing emission rules)

    Output contract (JSON-MODE): An object mapping each home id to a nested object of {task_id: start_time}, e.g.,
    {
      "H0": {"WASHER": 5, "DRYER": 9},
      "H1": {"EV": 3}
    }
    """

    # Load catalog for human-readable device descriptions
    devices = load_device_catalog_json(catalog_path)
    base_lookup = {d["id"]: d for d in devices}

    # --- TASK PROMPT ---
    tp_lines: List[str] = []
    tp_lines.append("## TASK PROMPT:")
    tp_lines.append("You are a smart home schedular.")
    tp_lines.append("You control start times for ALL homes' tasks. Pick EXACTLY ONE start time per task.")
    tp_lines.append("Objective: minimize total main-grid energy over the horizon.")
    tp_lines.append("")
    tp_lines.append("Scoring model:")
    tp_lines.append(" - Let D[t] be total neighborhood demand at slot t from all running tasks.")
    tp_lines.append(" - Main-grid draw G[t] = max(0, D[t] - S_cap[t]).")
    tp_lines.append(" - Minimize Sum_t G[t].")
    tp_lines.append("")
    tp_lines.append(f"Horizon T = {instance.T}")
    tp_lines.append("S_cap (per-slot sustainable capacity):")
    tp_lines.append(" " + json.dumps(list(instance.S_cap)))
    tp_lines.append("")
    tp_lines.append("Constraints:")
    tp_lines.append(" - Each task has fixed duration L (consecutive slots).")
    tp_lines.append(" - You must choose a start time t0 within its allowed window(s).")
    tp_lines.append(" - A task occupies slots [t0, t0+L-1].")
    tp_lines.append("")
    tp_lines.append("Homes and tasks (with allowed start intervals):")

    def _intervals(allowed: List[int]) -> List[Tuple[int, int]]:
        return _contiguous_runs(sorted(set(allowed)))

    for h in instance.homes:
        tp_lines.append(f"- Home {h.id}:")
        for t in h.tasks:
            base = t.id.split("_")[0]
            desc = base_lookup.get(base, {}).get("description", "")
            intervals_str = ", ".join([f"[{a},{b}]" for a, b in _intervals(t.allowed_starts)])
            tp_lines.append(
                f" • id={t.id}; desc=\"{desc}\"; consumption={t.consumption}; duration={t.duration}; allowed={intervals_str}"
            )

    task_prompt = "\n".join(tp_lines)

    # --- DELIBERATION INSTRUCTIONS ---
    delib_lines: List[str] = []
    delib_lines.append("## INSTRUCTIONS:")
    delib_lines.append("- Think step-by-step about how start times change neighborhood peaks.")
    delib_lines.append("- Respect allowed start intervals.")
    delib_lines.append("- After finishing your reasoning, use the schedule_task tool to schedule your chosen tasks.")
    deliberation_instructions = "\n".join(delib_lines)

    # --- TOOL USAGE INSTRUCTIONS ---
    tool_lines: List[str] = []
    tool_lines.append("## INSTRUCTIONS")
    tool_lines.append("Use the schedule_task tool to schedule each task at your chosen start time.")
    tool_lines.append("Call the tool once for each task that needs to be scheduled.")
    tool_lines.append("Example: schedule_task(task_id='WASHER_H0', start_time=5)")
    tool_lines.append("Make sure your chosen start times respect the allowed windows and minimize total main-grid energy.")
    tool_usage_instructions = "\n".join(tool_lines)

    return {
        "task_prompt": task_prompt,
        "deliberation_instructions": deliberation_instructions,
        "tool_usage_instructions": tool_usage_instructions,
    }


# =========================
# Adapter for batch writers (sentinel keys)
# =========================
def make_prompts_powerlite_monolithic_parts_as_dict(
    instance: InstanceSpec, catalog_path: str, *, tone: str = "standard"
) -> Dict[str, str]:
    """
    Adapter for dataset writers expecting a dict of prompts. Stores the three parts under sentinel keys consistent with your PA pipeline.
    """
    parts = make_prompts_powerlite_monolithic_parts(instance, catalog_path, tone=tone)
    return {
        "__TASK__": parts["task_prompt"],
        "__DELIBERATION__": parts["deliberation_instructions"],
        "__TOOL_USAGE__": parts["tool_usage_instructions"],
    }






def _intervals(allowed: List[int]) -> List[Tuple[int, int]]:
    """Collapse allowed starts into closed intervals [a,b]."""
    return _contiguous_runs(sorted(set(allowed)))


def make_prompts_sg_DT(
    instance: InstanceSpec,
    catalog_path: str,
    *,
    agent_id_map: Optional[Dict[str, str]] = None,  # optional map home_id -> agent_id
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Builds three parts for the new protocol:

      - deliberation_prompt (CENTRAL SCHEDULER):
          * NO instance-specific homes or tasks.
          * Only global scoring info, S_cap, and explicit instructions that the model must
            discover ALL homes and tasks from the attached agent blocks.
          * Includes an EXAMPLE format (placeholders only).

      - json_mode_prompt:
          * Strict, parser-friendly emission rules.
          * Must produce assignments for EVERY home/task discovered from the attached blocks.

      - personal_instructions:
          * Dict mapping agent_id -> per-agent prompt.
          * Each agent knows everything about its own home and tasks and provides a
            machine-readable TASK CATALOG plus free-form preferences.
    """
    devices = load_device_catalog_json(catalog_path)
    base_lookup = {d["id"]: d for d in devices}

    # ---------------------------------------------------------------------
    # 1) DELIBERATION PROMPT (CENTRAL; learns all structure from attachments)
    # ---------------------------------------------------------------------
    dl: List[str] = []
    dl.append("## ROLE")
    dl.append("You are the CENTRAL SCHEDULER for a neighborhood smart grid.")

    dl.append("")
    dl.append("## WHAT YOU WILL RECEIVE")
    dl.append("AFTER this prompt, you will receive a LIST of AGENT PREFERENCE BLOCKS.")
    dl.append("Each block corresponds to exactly one home and begins with a header line, e.g.:")
    dl.append("  [AGENT_ID=AGENT_HX][HOME_ID=HX]")
    dl.append("The block then contains:")
    dl.append("  1) A machine-readable TASK CATALOG listing every schedulable task for that home; and")
    dl.append("  2) Free-form first-person preferences (and optional semi-structured hints).")
    dl.append("")
    dl.append("You MUST discover all home IDs, task IDs, durations, consumptions, and allowed start intervals")
    dl.append("exclusively from these attached blocks. Do not assume any information not present in them.")

    dl.append("")
    dl.append("## EXAMPLE FORMAT (PLACEHOLDERS ONLY — NOT REAL DATA)")
    dl.append("[AGENT_ID=AGENT_HX][HOME_ID=HX]")
    dl.append("TASK CATALOG:")
    dl.append('  - task_id=DRYER_HX; device_base="DRYER"; consumption=3; duration=2; allowed=[ [6,9], [18,20] ]')
    dl.append('  - task_id=DISHWASHER_HX; device_base="DISHWASHER"; consumption=2; duration=3; allowed=[ [10,14] ]')
    dl.append("PREFERENCES:")
    dl.append("  I prefer to avoid late-night noise. If possible, finish DRYER before 21.")
    dl.append('  avoid_hours: [22,23,0,1,2,3,4,5]')
    dl.append('  deadline: {"task_id":"DRYER_HX","finish_by":21}')
    dl.append("----")
    dl.append("You may receive many such blocks, one per home/agent.")

    dl.append("")
    dl.append("## PRIMARY OBJECTIVE")
    dl.append("Minimize total main-grid draw while respecting feasibility (one valid start per task within its allowed windows).")

    dl.append("")
    dl.append("## SCORING MODEL")
    dl.append("Let D[t] be total neighborhood demand at slot t from all running tasks.")
    dl.append("Main-grid draw G[t] = max(0, D[t] - S_cap[t]).")
    dl.append("Minimize Sum_t G[t].")

    dl.append("")
    dl.append(f"Horizon T = {instance.T}")
    dl.append("S_cap (per-slot sustainable capacity):")
    dl.append("  " + json.dumps(list(instance.S_cap)))

    dl.append("")
    dl.append("## DELIBERATION STEPS")
    dl.append("1) Parse each attached block; extract HOME_ID and enumerate its TASK CATALOG.")
    dl.append("2) For each task, derive the feasible start times from the provided allowed intervals.")
    dl.append("3) Construct a schedule aligning heavier loads to greener (higher S_cap) slots.")
    dl.append("4) Integrate preferences (soft) and explicit hard constraints from the text/hints.")
    dl.append("5) Ensure exactly one valid start for every discovered task.")
    dl.append("6) Emit the final JSON.")

    dl.append("")
    dl.append("## PROHIBITIONS")
    dl.append("• Do NOT invent any homes or tasks not present in the attached blocks.")
    dl.append("• Do NOT rely on any per-home details beyond what is stated in the blocks.")
   
    deliberation_prompt = "\n".join(dl)

    # ---------------------------------------------------------------------
    # 2) JSON-MODE PROMPT (strict emission rules)
    # ---------------------------------------------------------------------
    SEP = "::"  # flat key separator
    jm = []
    jm.append("## INSTRUCTIONS (JSON-MODE)")
    jm.append(f'Emit EXACTLY one JSON object mapping "<HOME_ID>{SEP}<TASK_ID>" to integer start times.')
    jm.append("You must cover EVERY task discovered from the ATTACHED AGENT BLOCKS (Stage 0 outputs).")
    jm.append("")
    jm.append("Example:")
    jm.append(f'  {{"H0{SEP}WASHER": 5, "H0{SEP}DRYER": 12}}')
    jm.append("")
    jm.append("Validation checklist (MANDATORY):")
    jm.append("1) Flat object only; do NOT nest by home.")
    jm.append(f"2) Keys are strings <HOME_ID>{SEP}<TASK_ID> that EXACTLY match IDs in the blocks.")
    jm.append("3) Values are integer start_time.")
    jm.append("4) Each start_time lies within the union of allowed intervals listed for that task in its catalog.")
    jm.append("5) No extra keys, comments, or trailing text. Valid JSON only.")
    json_mode_prompt = "\n".join(jm)

    # ---------------------------------------------------------------------
    # 3) PERSONAL INSTRUCTIONS (one per agent/home; contains full catalog + prefs)
    # ---------------------------------------------------------------------
    personal_instructions: Dict[str, str] = {}

    def default_agent_id_for(home_id: str) -> str:
        return f"AGENT_{home_id}"

    for h in instance.homes:
        agent_id = agent_id_map[h.id] if agent_id_map and h.id in agent_id_map else default_agent_id_for(h.id)

        pi: List[str] = []
        pi.append(f"#INSTRUCTION FOR {agent_id} (HOME {h.id})")
        pi.append("")
        pi.append("Put this EXACT header at the very top of your reply so the central scheduler can link your text:")
        pi.append(f"[AGENT_ID={agent_id}][HOME_ID={h.id}]")
        pi.append("")
        pi.append("## TASK CATALOG (MANDATORY — machine-readable lines, one per task)")
        pi.append("List every schedulable task using exactly this format:")
        pi.append('  - task_id=<TASK_ID>; device_base="<DEVICE_BASE>"; consumption=<kW:int|float>; duration=<L:int>; allowed=[ [a,b], [c,d], ... ]')
        pi.append("Now fill it with YOUR home’s actual tasks below:")
        for t in h.tasks:
            base = t.id.split("_")[0]
            desc = base_lookup.get(base, {}).get("description", "")
            intervals_str = ", ".join([f"[{a},{b}]" for a, b in _intervals(t.allowed_starts)])
            pi.append(
                f'  - task_id={t.id}; device_base="{base}"; consumption={t.consumption}; '
                f'duration={t.duration}; allowed=[ {intervals_str} ]'
            )
        pi.append("")
        pi.append("## PREFERENCES (optional, free-form, first person)")
        pi.append("You may describe scheduling preferences in words. IMPORTANT: Feasibility is fully determined by the TASK CATALOG (allowed start times, durations, consumption).")
        pi.append('Use statements like "I prefer...", "I need...", "Please avoid...". Examples (soft guidance only):')
        pi.append("- Reasons you favor certain times that are ALREADY allowed (e.g., evenings for laundry).")
        pi.append("- Willingness to shift within the allowed window to reduce neighborhood peaks.")
        pi.append("")
        pi.append("Do NOT request start times outside the allowed windows. Do NOT change durations or consumptions.")
        pi.append("")
        pi.append("Do NOT output JSON; send only this header line, the TASK CATALOG, and your preference text.")

        personal_instructions[agent_id] = "\n".join(pi)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


# =========================
# Adapter (sentinel keys) — optional, mirrors your existing pattern
# =========================
def make_prompts_sg_DT_as_dict(
    instance: InstanceSpec,
    catalog_path: str,
    *,
    agent_id_map: Optional[Dict[str, str]] = None,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Stores the three parts under sentinel keys and flattens per-agent instructions using
    keys like '__AGENT__<AGENT_ID>' for convenient batch writing.
    """
    parts = make_prompts_sg_DT(instance, catalog_path, agent_id_map=agent_id_map, tone=tone)

    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out



from typing import Dict, List, Tuple, Optional, Any
import json

# Assumed available in your codebase:
# - InstanceSpec with fields: T:int, S_cap:List[float], homes:List[Home]
# - Home with fields: id:str, tasks:List[Task]
# - Task with fields: id:str, consumption:float|int, duration:int, allowed_starts:List[int]
# - load_device_catalog_json(catalog_path:str) -> List[Dict[str, Any]]
# If not present, provide your local implementations or stubs.

# ---------- small helpers (safe stand-ins) ----------
def _contiguous_runs(sorted_vals: List[int]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if not sorted_vals:
        return runs
    start = prev = sorted_vals[0]
    for v in sorted_vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return runs

def _intervals(allowed: List[int]) -> List[Tuple[int, int]]:
    """Collapse allowed starts into closed intervals [a,b]."""
    return _contiguous_runs(sorted(set(allowed)))



# =========================
# DPT (Agent -> Factor-connected Agents) — TASKS ONLY + GLOBAL CONTEXT
# =========================
from typing import Dict, List, Optional, Any

def make_prompts_sg_DPT(
    instance: "InstanceSpec",
    catalog_path: str,  # kept for API symmetry; UNUSED
    *,
    agent_id_map: Optional[Dict[str, str]] = None,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Minimal DPT with a global context paragraph and per-home TASK CATALOG only.

    Global context (applies to all homes):
      • You are a smart home assistant in a neighborhood smart grid.
      • Your job is to select ONE legal integer start time for each task in your home.
      • Objective: avoid neighborhood demand peaks so main-grid draw is reduced and
        sustainable capacity (S_cap) is maximally utilized.
      • Hard constraints come solely from each task’s allowed windows; durations and
        consumptions are fixed. Coordination/messaging is handled by the agent script.

    Per-home blocks contain ONLY:
      [AGENT_ID=...][HOME_ID=...]
      TASK CATALOG:
        - task_id=<ID>; device_base="<BASE>"; consumption=<kW>; duration=<L>; allowed=[ [a,b], ... ]
    """
    # Global description (kept protocol-agnostic; no emission rules here)
    deliberation_prompt = (
        "## ROLE\n"
        "You are a SMART HOME ASSISTANT for a single home in a neighborhood smart grid.\n\n"
        "## TASK\n"
        "Select exactly ONE legal integer start time for each of your home’s tasks, where legality is defined by the task’s allowed windows.\n\n"
        "## OBJECTIVE\n"
        "Reduce neighborhood peaks and thereby minimize main-grid draw by aligning demand with sustainable capacity (S_cap), maximizing the use of green energy.\n\n"
        "## NOTES\n"
        "Per-home blocks list ONLY task data. Messaging, coordination, and decision mechanics are handled by the agent/runtime."
    )
    json_mode_prompt = ""  # emission format decided by downstream pipeline

    personal_instructions: Dict[str, str] = {}

    def _agent_for(home_id: str) -> str:
        return agent_id_map.get(home_id, f"AGENT_{home_id}") if agent_id_map else f"AGENT_{home_id}"

    for h in instance.homes:
        agent_id = _agent_for(h.id)
        lines: List[str] = []
        # Exact header used downstream for linking
        lines.append(f"[AGENT_ID={agent_id}][HOME_ID={h.id}]")
        lines.append("TASK CATALOG:")
        for t in h.tasks:
            base = t.id.split("_")[0]
            intervals_str = ", ".join([f"[{a},{b}]" for a, b in _intervals(t.allowed_starts)])
            lines.append(
                f'  - task_id={t.id}; device_base="{base}"; consumption={t.consumption}; '
                f'duration={t.duration}; allowed=[ {intervals_str} ]'
            )
        personal_instructions[agent_id] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


def make_prompts_sg_DPT_as_dict(
    instance: "InstanceSpec",
    catalog_path: str,
    *,
    agent_id_map: Optional[Dict[str, str]] = None,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that flattens per-agent blocks under sentinel keys.
    """
    parts = make_prompts_sg_DPT(instance, catalog_path, agent_id_map=agent_id_map, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out

# =========================
# DPT (Agent -> Factor-connected Agents) — TASKS ONLY + GLOBAL CONTEXT
# =========================
from typing import Dict, List, Optional, Any

def make_prompts_sg_DPI(
    instance: "InstanceSpec",
    catalog_path: str,  # kept for API symmetry; UNUSED
    *,
    agent_id_map: Optional[Dict[str, str]] = None,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    DPI (image-primacy) variant: produce ONLY per-agent personal prompts.

    Per-agent prompt format:
      [AGENT_ID=...][HOME_ID=...]
      NOTE: Allowed start windows and the sustainable capacity profile (S_cap) are provided in the IMAGE.
      TASKS:
        - task_id=<ID>; consumption=<kW>
        - ...

    No deliberation or JSON-mode prompts are returned.
    """
    personal_instructions: Dict[str, str] = {}

    def _agent_for(home_id: str) -> str:
        return agent_id_map.get(home_id, f"AGENT_{home_id}") if agent_id_map else f"AGENT_{home_id}"

    for h in instance.homes:
        agent_id = _agent_for(h.id)
        lines: List[str] = []
        lines.append(f"[AGENT_ID={agent_id}][HOME_ID={h.id}]")
        lines.append("NOTE: Allowed start windows and the sustainable capacity profile (S_cap) are provided in the IMAGE.")
        lines.append("TASKS:")
        for t in h.tasks:
            lines.append(f"  - task_id={t.id}; consumption={t.consumption}")
        personal_instructions[agent_id] = "\n".join(lines)

    # Return only personal prompts (no deliberation/json fields)
    return {"personal_instructions": personal_instructions}


def make_prompts_sg_DPI_as_dict(
    instance: "InstanceSpec",
    catalog_path: str,
    *,
    agent_id_map: Optional[Dict[str, str]] = None,
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Flatten per-agent prompts under sentinel keys only (no __DELIBERATION__/__JSON_MODE__).
    """
    parts = make_prompts_sg_DPI(instance, catalog_path, agent_id_map=agent_id_map, tone=tone)
    out: Dict[str, str] = {}
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out
