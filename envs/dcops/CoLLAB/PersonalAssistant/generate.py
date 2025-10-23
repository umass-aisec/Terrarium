"""
Collab (Coordinating LLM Agent Benchmark) - Personal Agent: Clothes Matching
STYLE REMOVED: outfits are (article, color) (+ optional image). No style anywhere.

- Reads colors.json, articles.json (flat list), and figs/images.json
- Problem instance = FactorGraph + Wardrobe (no prompts inside)
- Optional collages can be generated later; prompts are built from a utility
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json, random

# Use relative import since data_structure.py is in the same package
from .data_structure import AgentId, Outfit, Wardrobe, Factor, FactorGraph, PersonalEnvInstance
from .prompt_maker import make_prompts_vanilla
# --------------- IO helpers ---------------

def _as_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _load_image_index_from_figs(figs_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    figs/images.json:
      {"images":[{"color":"blue","article":"t-shirt","filename":"blue_t-shirt.png"}, ...]}
    Returns: {article: {color: "figs/<filename>.png"}}
    """
    data = _read_json(figs_dir / "images.json")
    m: Dict[str, Dict[str, str]] = {}
    for rec in data.get("images", []):
        art, col, fn = rec["article"], rec["color"], rec["filename"]
        m.setdefault(art, {})[col] = str(Path("figs") / fn)
    return m

# --------------- Assignment helper ---------------

def assignment_from_choices(wardrobe: Wardrobe, choices: Dict[AgentId, int]) -> Dict[AgentId, Outfit]:
    assign: Dict[AgentId, Outfit] = {}
    for agent, idx in choices.items():
        if agent not in wardrobe.options:
            raise KeyError(f"Unknown agent in choices: {agent}")
        i = int(idx)
        opts = wardrobe.options[agent]
        if not (1 <= i <= len(opts)):
            raise ValueError(f"Choice for {agent} out of range: {i} (have {len(opts)} options)")
        assign[agent] = opts[i - 1]
    return assign

# --------------- Problem generator (no prompts, no collages) ---------------

def build_personal_env(
    n_agents: int,
    max_degree: int,
    data_root: Path | str,
    rng_seed: int = 7,
    min_outfits_per_agent: int = 5,
    max_outfits_per_agent: int = 8,
    p_add_unary_color: float = 0.7,
) -> PersonalEnvInstance:
    """
    Pure problem instance (graph + wardrobes). No styles anywhere. No prompts or collages.
    """
    rng = random.Random(rng_seed)
    root = _as_path(data_root)

    names   = _read_json(root / "names.json")["names"]
    colors  = _read_json(root / "colors.json")["colors"]
    articles: List[str] = _read_json(root / "articles.json")["articles"]

    image_index = _load_image_index_from_figs(root)

    # Agents (cycle names if needed)
    agents: List[AgentId] = []
    i = 0
    while len(agents) < n_agents:
        suffix = f"_{i//len(names)+1}" if i >= len(names) else ""
        agents.append(names[i % len(names)] + suffix)
        i += 1

    # Sparse graph with max degree
    edges: List[Tuple[AgentId, AgentId]] = []
    degree = {a: 0 for a in agents}
    pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i+1:]]
    rng.shuffle(pairs)
    for (u, v) in pairs:
        if degree[u] < max_degree and degree[v] < max_degree and rng.random() < 0.6:
            edges.append((u, v)); degree[u] += 1; degree[v] += 1
    if not edges and len(agents) > 1:
        for i in range(len(agents)-1):
            edges.append((agents[i], agents[i+1]))
            degree[agents[i]] += 1; degree[agents[i+1]] += 1

    # Factors: only color-based
    factors: List[Factor] = []
    fid = 0
    def new_fid() -> str:
        nonlocal fid; fid += 1; return f"F{fid:05d}"

    for (u, v) in edges:
        ftype = rng.choice(["MATCH_COLOR", "NOT_MATCH_COLOR"])
        factors.append(Factor(fid=new_fid(), agent_scope=[u, v], ftype=ftype, payload={}))

    for a in agents:
        if rng.random() < p_add_unary_color:
            col = rng.choice(colors)
            ftype = "PREF_COLOR" if rng.random() < 0.5 else "AVOID_COLOR"
            factors.append(Factor(fid=new_fid(), agent_scope=[a], ftype=ftype, payload={"color": col}))

    graph = FactorGraph(agents=agents, factors=factors)

    # Wardrobes (article + color [+ image if available])
    wardrobe_map: Dict[AgentId, List[Outfit]] = {}
    for a in agents:
        k = rng.randint(min_outfits_per_agent, max_outfits_per_agent)
        opts: List[Outfit] = []
        for _ in range(k):
            art = rng.choice(articles)
            col = rng.choice(colors)
            img_rel = image_index.get(art, {}).get(col)  # 'figs/<file>.png' or None
            opts.append(Outfit(article=art, color=col, image=img_rel))
        wardrobe_map[a] = opts

    return PersonalEnvInstance(graph=graph, wardrobe=Wardrobe(options=wardrobe_map))

# --------------- Optional presentation-time helpers ---------------

def generate_collages(instance, data_root, out_subdir="figs/wardrobes", cols=3, thumb_size=(256,256), pad=8):
    try:
        from PIL import Image
    except Exception:
        return {}

    root = Path(data_root).resolve()
    out_path = Path(out_subdir)
    # If an absolute output dir is provided, use it as-is. Otherwise, make it relative to root.
    out_dir = out_path if out_path.is_absolute() else (root / out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for agent, outfits in instance.wardrobe.options.items():
        abs_imgs = []
        for o in outfits:
            if getattr(o, "image", None):
                p = (root / o.image).resolve()
                if p.exists():
                    abs_imgs.append(p)
        if not abs_imgs:
            continue

        rows = (len(abs_imgs) + cols - 1) // cols
        W, H = thumb_size
        canvas_w = cols * W + (cols + 1) * pad
        canvas_h = rows * H + (rows + 1) * pad
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        for i, p in enumerate(abs_imgs):
            try:
                im = Image.open(p).convert("RGBA").resize((W, H))
            except Exception:
                continue
            r, c = divmod(i, cols)
            x = pad + c * (W + pad)
            y = pad + r * (H + pad)
            canvas.paste(im, (x, y), im)

        out_file = out_dir / f"{agent}.png"
        canvas.save(out_file)
        result[agent] = str(out_file)  # absolute path; caller will relativize
    return result


# ---------------- Tiny demo ----------------

def demo_instance():
    data_root = "PersonalAssistant/data/"
    inst = build_personal_env(
        n_agents=6,
        max_degree=3,
        data_root=data_root,
        rng_seed=42,
        min_outfits_per_agent=5,
        max_outfits_per_agent=8,
    )
    rng = random.Random(0)
    choices = {a: rng.randint(1, len(inst.wardrobe.options[a])) for a in inst.graph.agents}
    assign = assignment_from_choices(inst.wardrobe, choices)

    # Presentation-time: optionally build collages and prompts
    collages = generate_collages(inst, data_root=data_root, cols=3, thumb_size=(192,192))
    prompts = make_prompts_vanilla(inst.graph, inst.wardrobe, collage_map=collages, tone="standard")

    print("Agents:", ", ".join(inst.graph.agents))
    first = inst.graph.agents[0]
    print("\nSample prompt for first agent:\n")
    print(prompts[first])
    print("\nGlobal score:", inst.graph.global_score(assign))

if __name__ == "__main__":
    demo_instance()
