from typing import Dict, List, Tuple, Optional, Any

# Use relative import since data_structure.py is in the same package
from .data_structure import (
    AgentId,
    Outfit,
    Wardrobe,
    Factor,
    FactorGraph,
    PersonalEnvInstance,
)


# =========================
# Per-agent TEXT-ONLY prompts
# =========================
def make_prompts_vanilla(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # kept for API compatibility, ignored
    tone: str = "standard",
) -> Dict[AgentId, str]:
    """ Build prompts for a given instance using ONLY text (article, color). No images. No collage references. """
    # factors incident on each agent
    inc: Dict[AgentId, List[Factor]] = {a: [] for a in graph.agents}
    for f in graph.factors:
        for a in f.agent_scope:
            inc[a].append(f)

    prompts: Dict[AgentId, str] = {}
    for a in graph.agents:
        lines: List[str] = []
        lines.append("You are dressing up for a party. Choose exactly ONE outfit from your options.")
        lines.append("Your goal is to satisfy as many of your preferences as possible.")

        unary_bits, friends_bits = [], []
        for f in inc[a]:
            if len(f.agent_scope) == 1:
                if f.ftype == "PREF_COLOR":
                    unary_bits.append(f"prefer wearing color {f.payload['color']}")
                elif f.ftype == "AVOID_COLOR":
                    unary_bits.append(f"avoid wearing color {f.payload['color']}")
                else:
                    pass
            else:
                other = f.agent_scope[1] if f.agent_scope[0] == a else f.agent_scope[0]
                if f.ftype == "MATCH_COLOR":
                    friends_bits.append(f"match color with {other}")
                elif f.ftype == "NOT_MATCH_COLOR":
                    friends_bits.append(f"do NOT match color with {other}")

        if unary_bits:
            lines.append("Personal preferences: " + "; ".join(unary_bits) + ".")
        if friends_bits:
            lines.append("Friend constraints: " + "; ".join(friends_bits) + ".")

        lines.append("Options:")
        for idx, o in enumerate(wardrobe.options[a], 1):
            # TEXT ONLY: article + color
            lines.append(f"{idx}. article={o.article}, color={o.color}")

        lines.append("Reply ONLY with the number of the option you choose.")
        prompts[a] = "\n".join(lines)

    return prompts


# =========================
# Monolithic TEXT-ONLY prompt
# =========================
def make_prompt_monolithic(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # kept for API compatibility, ignored
    tone: str = "standard",
) -> str:
    """ One prompt describing ALL agents, constraints, and wardrobe options using ONLY text. """
    lines: List[str] = []
    lines.append("Coordinate outfits for ALL agents. Each agent must pick EXACTLY ONE option.")
    lines.append("Maximize satisfied constraints (preferences and friend color relations).")

    # Constraints
    lines.append("\n--- Constraints ---")
    for f in graph.factors:
        if len(f.agent_scope) == 1:
            a = f.agent_scope[0]
            if f.ftype == "PREF_COLOR":
                lines.append(f"- {a} prefers color {f.payload['color']}.")
            elif f.ftype == "AVOID_COLOR":
                lines.append(f"- {a} avoids color {f.payload['color']}.")
            else:
                pass
        else:
            a, b = f.agent_scope
            if f.ftype == "MATCH_COLOR":
                lines.append(f"- {a} should MATCH color with {b}.")
            elif f.ftype == "NOT_MATCH_COLOR":
                lines.append(f"- {a} should NOT match color with {b}.")

    # Wardrobes (TEXT ONLY)
    lines.append("\n--- Wardrobes ---")
    for a in graph.agents:
        lines.append(f"Agent {a}:")
        for idx, o in enumerate(wardrobe.options[a], 1):
            lines.append(f" {idx}. article={o.article}, color={o.color}")

    # Output contract
    lines.append("\n--- Output format ---")
    lines.append("Return ONLY a JSON object mapping each agent to the chosen option index (1-based).")
    # Example: { "Alice": 2, "Bob": 1, ... }
    return "\n".join(lines)


# =========================
# Adapter for your batch maker
# =========================
def make_prompts_monolithic_as_dict(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # kept for API compatibility, ignored
    tone: str = "standard",
) -> Dict[AgentId, str]:
    """ Adapter so you can reuse benchmark_maker.generate_instances(..., make_prompts_fn=...). Stores the single TEXT-ONLY prompt under a sentinel key. """
    mono = make_prompt_monolithic(graph, wardrobe, collage_map=None, tone=tone)
    return {"__ALL__": mono}


from typing import Dict, List, Optional

# =========================
# Three separate prompts (no concatenation)
# =========================
def make_prompt_monolithic_parts(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # kept for API compatibility, ignored
    tone: str = "standard",
) -> Dict[str, str]:
    """ Returns three distinct prompts:
    - 'task_prompt'
    - 'deliberation_instructions'
    - 'json_mode_instructions'
    """

    # --- TASK PROMPT ---
    task_lines: List[str] = []
    task_lines.append("## TASK PROMPT")
    task_lines.append("Coordinate outfits for ALL agents. Each agent must pick EXACTLY ONE option.")
    task_lines.append("Objective: maximize satisfied constraints (agent preferences and friend color relations).")
    task_lines.append("\n### Constraints")
    for f in graph.factors:
        if len(f.agent_scope) == 1:
            a = f.agent_scope[0]
            if f.ftype == "PREF_COLOR":
                task_lines.append(f"- {a} prefers color {f.payload['color']}.")
            elif f.ftype == "AVOID_COLOR":
                task_lines.append(f"- {a} avoids color {f.payload['color']}.")
            else:
                pass
        else:
            a, b = f.agent_scope
            if f.ftype == "MATCH_COLOR":
                task_lines.append(f"- {a} should MATCH color with {b}.")
            elif f.ftype == "NOT_MATCH_COLOR":
                task_lines.append(f"- {a} should NOT match color with {b}.")

    task_lines.append("\n### Wardrobes")
    for a in graph.agents:
        task_lines.append(f"Agent {a}:")
        for idx, o in enumerate(wardrobe.options[a], 1):
            task_lines.append(f" {idx}. article={o.article}, color={o.color}")

    task_prompt = "\n".join(task_lines)

    # --- DELIBERATION INSTRUCTIONS ---
    delib_lines: List[str] = []
    delib_lines.append("## INSTRUCTIONS")
    delib_lines.append("- First, think step-by-step about how to maximize satisfied constraints.")
    delib_lines.append("- Use reasoning to resolve conflicts and choose one option per agent.")
    delib_lines.append("- When finished, switch INTERNALLY to JSON-MODE and return ONLY a JSON object mapping each agent to the chosen option index (1-based).")
    deliberation_instructions = "\n".join(delib_lines)

    # --- JSON-MODE INSTRUCTIONS ---
    json_lines: List[str] = []
    json_lines.append("## INSTRUCTIONS")
    json_lines.append("PARSE the thinking and emit EXACTLY one JSON object mapping each agent name to a 1-based option index.")
    json_mode_instructions = "\n".join(json_lines)

    return {
        "task_prompt": task_prompt,
        "deliberation_instructions": deliberation_instructions,
        "json_mode_instructions": json_mode_instructions,
    }


# =========================
# Adapter for your batch maker
# =========================
def make_prompts_monolithic_parts_as_dict(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # kept for API compatibility, ignored
    tone: str = "standard",
) -> Dict[AgentId, str]:
    """ For benchmark_maker.generate_instances(..., make_prompts_fn=...). Exposes the three prompts under sentinel keys so your pipeline can store them distinctly. """
    parts = make_prompt_monolithic_parts(graph, wardrobe, collage_map=None, tone=tone)
    return {
        # Sentinel keys; adjust if your loader expects different names.
        "__TASK__": parts["task_prompt"],
        "__DELIBERATION__": parts["deliberation_instructions"],
        "__JSON_MODE__": parts["json_mode_instructions"],
    }

def make_prompts_pa_DT(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Builds three parts for the new protocol:

      - deliberation_prompt (CENTRAL COORDINATOR):
          * NO instance-specific wardrobes or agent lists.
          * The central model must discover ALL agents, their options, and constraints
            exclusively from the attached agent blocks.

      - json_mode_prompt:
          * Flat JSON mapping: { "<AgentId>": <option_index:int>, ... }
          * 1-based indices, exactly as shown in the agent blocks.

      - personal_instructions:
          * Dict mapping AgentId -> per-agent instruction.
          * Each agent receives: header + WARDROBE OPTIONS (numbered) + CONSTRAINTS
            and is asked to write a short first-person preference paragraph.
    """
    # ---------- 1) CENTRAL DELIBERATION (instance-agnostic wrt agents/options) ----------
    dl: List[str] = []
    dl.append("## ROLE")
    dl.append("You are the CENTRAL COORDINATOR choosing outfits for multiple people.")
    dl.append("")
    dl.append("## WHAT YOU WILL RECEIVE")
    dl.append("A LIST of AGENT blocks. Each block begins with:")
    dl.append("  [AGENT_ID=<AgentName>]")
    dl.append("and then contains:")
    dl.append("  1) WARDROBE OPTIONS — a numbered list of outfits, each 'article' and 'color';")
    dl.append("  2) CONSTRAINTS — color preferences and pairwise color-match relations with friends;")
    dl.append("  3) A short first-person PREFERENCES paragraph.")
    dl.append("")
    dl.append("You MUST discover all agents, options, and constraints strictly from these blocks.")
    dl.append("Do not assume anything not present in them.")
    dl.append("")
    dl.append("## EXAMPLE BLOCK (PLACEHOLDER — NOT REAL DATA)")
    dl.append("[AGENT_ID=AX]")
    dl.append("WARDROBE OPTIONS:")
    dl.append("  1. article=t-shirt, color=blue")
    dl.append("  2. article=jeans,  color=black")
    dl.append("CONSTRAINTS:")
    dl.append("  - prefers color blue")
    dl.append("  - should MATCH color with BY")
    dl.append("PREFERENCES:")
    dl.append("  I like wearing blue. If possible, match with BY.")
    dl.append("----")
    dl.append("You may receive many such blocks, one per agent.")
    dl.append("")
    dl.append("## PRIMARY OBJECTIVE")
    dl.append("Each agent must pick EXACTLY ONE option (1-based index).")
    dl.append("Maximize satisfied constraints: personal color preferences and friend color relations.")
    dl.append("If constraints conflict, choose assignments that satisfy as many as possible overall.")
    dl.append("")
    dl.append("## DELIBERATION STEPS")
    dl.append("1) Parse each block: extract AGENT_ID, numbered options, and constraints.")
    dl.append("2) Consider pairwise color-match relations across agents.")
    dl.append("3) Resolve conflicts to maximize total satisfied constraints.")
    dl.append("4) Emit the final JSON (see JSON-MODE).")
    dl.append("")
    dl.append("## PROHIBITIONS")
    dl.append("• Do NOT invent agents or options not present in the blocks.")
    dl.append("• Do NOT output anything except the final JSON in the JSON-MODE step.")
    deliberation_prompt = "\n".join(dl)

    # ---------- 2) JSON-MODE (flat; agent -> 1-based index) ----------
    jm: List[str] = []
    jm.append("## INSTRUCTIONS (JSON-MODE)")
    jm.append('Emit EXACTLY one JSON object mapping each agent name to a 1-based option index (integer).')
    jm.append("Cover EVERY agent present in the attached blocks.")
    jm.append("Validation checklist (MANDATORY):")
    jm.append("1) Flat object only; do NOT nest.")
    jm.append("2) Keys EXACTLY match the AGENT_ID strings from the blocks.")
    jm.append("3) Values are integers in the valid range for that agent’s numbered list.")
    jm.append("4) No extra keys, comments, or trailing text. Valid JSON only.")
    json_mode_prompt = "\n".join(jm)

    # ---------- 3) PERSONAL INSTRUCTIONS (per agent; numbered options + constraints) ----------
    # Build incident-factor lists for each agent
    inc: Dict[AgentId, List[Factor]] = {a: [] for a in graph.agents}
    for f in graph.factors:
        for a in f.agent_scope:
            inc[a].append(f)

    personal_instructions: Dict[AgentId, str] = {}
    for a in graph.agents:
        lines: List[str] = []
        lines.append(f"# PERSONAL INSTRUCTION FOR {a}")
        lines.append("")
        lines.append("Put this EXACT header at the very top so the central coordinator can link your text:")
        lines.append(f"[AGENT_ID={a}]")
        lines.append("")
        lines.append("WARDROBE OPTIONS:")
        for idx, o in enumerate(wardrobe.options[a], 1):
            # TEXT ONLY: article + color
            lines.append(f"  {idx}. article={o.article}, color={o.color}")
        lines.append("")
        lines.append("CONSTRAINTS:")
        unary_any, pair_any = False, False
        for f in inc[a]:
            if len(f.agent_scope) == 1:
                if f.ftype == "PREF_COLOR":
                    lines.append(f"  - prefers color {f.payload['color']}")
                    unary_any = True
                elif f.ftype == "AVOID_COLOR":
                    lines.append(f"  - avoids color {f.payload['color']}")
                    unary_any = True
            else:
                other = f.agent_scope[1] if f.agent_scope[0] == a else f.agent_scope[0]
                if f.ftype == "MATCH_COLOR":
                    lines.append(f"  - should MATCH color with {other}")
                    pair_any = True
                elif f.ftype == "NOT_MATCH_COLOR":
                    lines.append(f"  - should NOT match color with {other}")
                    pair_any = True
        if not (unary_any or pair_any):
            lines.append("  - (no explicit constraints)")
        lines.append("")
        lines.append("Write a SHORT paragraph in first person summarizing your preferences.")
        lines.append("Do not output JSON. Keep it concise.")
        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


def make_prompts_pa_DT_as_dict(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # <-- accept for API compatibility (ignored)
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that exposes the three parts under sentinel keys for your pipelines.
    """
    parts = make_prompts_pa_DT(graph, wardrobe, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out






def make_prompts_pa_DPT(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Builds three parts for the new protocol:

      - personal_instructions:
          * Dict mapping AgentId -> per-agent instruction.
          * Each agent receives: header + WARDROBE OPTIONS (numbered) + CONSTRAINTS
            and is asked to write a short first-person preference paragraph.
    """
    # ---------- 1) CENTRAL DELIBERATION (instance-agnostic wrt agents/options) ----------
    dl: List[str] = []
    dl.append("")
    deliberation_prompt = "\n".join(dl)

    # ---------- 2) JSON-MODE (flat; agent -> 1-based index) ----------
    jm: List[str] = []
    json_mode_prompt = "\n".join(jm)

    # ---------- 3) PERSONAL INSTRUCTIONS (per agent; numbered options + constraints) ----------
    # Build incident-factor lists for each agent
    inc: Dict[AgentId, List[Factor]] = {a: [] for a in graph.agents}
    for f in graph.factors:
        for a in f.agent_scope:
            inc[a].append(f)

    personal_instructions: Dict[AgentId, str] = {}
    for a in graph.agents:
        lines: List[str] = []
        lines.append(f"# PERSONAL Preferance {a}")
        lines.append("")
        lines.append(f"[AGENT_ID={a}]")
        lines.append("")
        lines.append("WARDROBE OPTIONS:")
        for idx, o in enumerate(wardrobe.options[a], 1):
            # TEXT ONLY: article + color
            lines.append(f"  {idx}. article={o.article}, color={o.color}")
        lines.append("")
        lines.append("CONSTRAINTS:")
        unary_any, pair_any = False, False
        for f in inc[a]:
            if len(f.agent_scope) == 1:
                if f.ftype == "PREF_COLOR":
                    lines.append(f"  - prefers color {f.payload['color']}")
                    unary_any = True
                elif f.ftype == "AVOID_COLOR":
                    lines.append(f"  - avoids color {f.payload['color']}")
                    unary_any = True
            else:
                other = f.agent_scope[1] if f.agent_scope[0] == a else f.agent_scope[0]
                if f.ftype == "MATCH_COLOR":
                    lines.append(f"  - should MATCH color with {other}")
                    pair_any = True
                elif f.ftype == "NOT_MATCH_COLOR":
                    lines.append(f"  - should NOT match color with {other}")
                    pair_any = True
        if not (unary_any or pair_any):
            lines.append("  - (no explicit constraints)")
        lines.append("")
        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


def make_prompts_pa_DPT_as_dict(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # <-- accept for API compatibility (ignored)
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that exposes the three parts under sentinel keys for your pipelines.
    """
    parts = make_prompts_pa_DPT(graph, wardrobe, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out




def make_prompts_pa_DPI(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    tone: str = "standard",
) -> Dict[str, Any]:
    """
    Builds three parts for the new protocol:

      - personal_instructions:
          * Dict mapping AgentId -> per-agent instruction.
          * Each agent receives: header + WARDROBE OPTIONS (numbered) + CONSTRAINTS
            and is asked to write a short first-person preference paragraph.
    """
    # ---------- 1) CENTRAL DELIBERATION (instance-agnostic wrt agents/options) ----------
    dl: List[str] = []
    dl.append("")
    deliberation_prompt = "\n".join(dl)

    # ---------- 2) JSON-MODE (flat; agent -> 1-based index) ----------
    jm: List[str] = []
    json_mode_prompt = "\n".join(jm)

    # ---------- 3) PERSONAL INSTRUCTIONS (per agent; numbered options + constraints) ----------
    # Build incident-factor lists for each agent
    inc: Dict[AgentId, List[Factor]] = {a: [] for a in graph.agents}
    for f in graph.factors:
        for a in f.agent_scope:
            inc[a].append(f)

    personal_instructions: Dict[AgentId, str] = {}
    for a in graph.agents:
        lines: List[str] = []
        lines.append(f"# PERSONAL Preferance {a}")
        lines.append("")
        lines.append(f"[AGENT_ID={a}]")
        lines.append("")
        lines.append("CONSTRAINTS:")
        unary_any, pair_any = False, False
        for f in inc[a]:
            if len(f.agent_scope) == 1:
                if f.ftype == "PREF_COLOR":
                    lines.append(f"  - prefers color {f.payload['color']}")
                    unary_any = True
                elif f.ftype == "AVOID_COLOR":
                    lines.append(f"  - avoids color {f.payload['color']}")
                    unary_any = True
            else:
                other = f.agent_scope[1] if f.agent_scope[0] == a else f.agent_scope[0]
                if f.ftype == "MATCH_COLOR":
                    lines.append(f"  - should MATCH color with {other}")
                    pair_any = True
                elif f.ftype == "NOT_MATCH_COLOR":
                    lines.append(f"  - should NOT match color with {other}")
                    pair_any = True
        if not (unary_any or pair_any):
            lines.append("  - (no explicit constraints)")
        lines.append("")
        personal_instructions[a] = "\n".join(lines)

    return {
        "deliberation_prompt": deliberation_prompt,
        "json_mode_prompt": json_mode_prompt,
        "personal_instructions": personal_instructions,
    }


def make_prompts_pa_DPI_as_dict(
    graph: FactorGraph,
    wardrobe: Wardrobe,
    *,
    collage_map: Optional[Dict[AgentId, str]] = None,  # <-- accept for API compatibility (ignored)
    tone: str = "standard",
) -> Dict[str, str]:
    """
    Adapter that exposes the three parts under sentinel keys for your pipelines.
    """
    parts = make_prompts_pa_DPI(graph, wardrobe, tone=tone)
    out: Dict[str, str] = {
        "__DELIBERATION__": parts["deliberation_prompt"],
        "__JSON_MODE__": parts["json_mode_prompt"],
    }
    for agent_id, text in parts["personal_instructions"].items():
        out[f"__AGENT__{agent_id}"] = text
    return out