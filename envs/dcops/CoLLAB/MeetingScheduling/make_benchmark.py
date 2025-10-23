from __future__ import annotations
import os, json, pickle, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# Use relative imports since these files are in the same package
from .generate import (
    build_meeting_env,
    draw_dummy_calendar,
    draw_agent_building_map,
    combine_calendar_and_map,
)
from .data_structure import MeetingEnvInstance, SLOT_LABELS

# ---------- helpers ----------

def _hash_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:12]

def _rel_to(base: Path, target: Path) -> str:
    """
    Robust relative path computation. Works even if `target` wasn't created under `base`.
    """
    base_abs = base.resolve()
    tgt_abs = target.resolve()
    rel = os.path.relpath(tgt_abs.as_posix(), start=base_abs.as_posix())
    return rel.replace("\\", "/")

def _safe_close(fig) -> None:
    try:
        import matplotlib.pyplot as plt
        if fig is not None:
            plt.close(fig)
    except Exception:
        pass

# ---------- core: save-one (viz only, optional prompts) ----------

def save_meeting_env_instance_viz_only(
    inst: MeetingEnvInstance,
    out_dir: str | Path,
    *,
    name: Optional[str] = None,
    # prompts
    make_prompts_fn: Optional[Callable[..., Dict[str, str]]] = None,
    prompts_kwargs: Optional[Dict[str, Any]] = None,
    prompts_filename: str = "prompts.json",
) -> Path:
    """
    Creates a self-contained folder:
      <out_dir>/<name or auto>/
        instance.pkl
        manifest.json
        images/viz/<Agent>.png     # combined calendar (top) + per-agent map (bottom)
        [prompts.json]             # if make_prompts_fn is provided (viz paths embedded)
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Name/signature (stable hash if not provided)
    if name is None:
        sig = {
            "agents": inst.graph.agents,
            "meetings": [
                {
                    "mid": m.mid,
                    "mode": m.mode,
                    "loc": m.location,
                    "owner": getattr(m, "owner", None),
                    "attendees": sorted(m.attendees),
                }
                for m in sorted(inst.graph.meetings.values(), key=lambda x: x.mid)
            ],
        }
        name = f"inst_{_hash_bytes(json.dumps(sig, sort_keys=True).encode())}"

    inst_dir = (out_dir / name).resolve()
    viz_dir = (inst_dir / "images" / "viz").resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- Render visuals ---
    cal_fig = draw_dummy_calendar(inst.graph, inst.preferences)

    viz_rel_map: Dict[str, str] = {}  # {agent -> 'images/viz/<agent>.png'}
    for agent in inst.graph.agents:
        map_fig = draw_agent_building_map(inst, agent)
        combined_path = combine_calendar_and_map(
            cal_fig, map_fig, out_path=(viz_dir / f"{agent}.png")
        )
        _safe_close(map_fig)
        if combined_path:
            viz_rel_map[agent] = _rel_to(inst_dir, Path(combined_path))

    # Close calendar fig once
    _safe_close(cal_fig)

    # --- Optionally emit prompts (with visualization paths) ---
    if make_prompts_fn is not None:
        kwargs = {"viz_map": viz_rel_map}
        if prompts_kwargs:
            kwargs.update(prompts_kwargs)
        # Convention: pass graph-level objects; builder can ignore extras
        prompts = make_prompts_fn(inst.graph, inst.preferences, inst.coords, **kwargs)
        (inst_dir / prompts_filename).write_text(
            json.dumps({"prompts": prompts}, indent=2)
        )

    # Save the pure problem instance
    with open(inst_dir / "instance.pkl", "wb") as f:
        pickle.dump(inst, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Manifest (slightly richer for debugging)
    manifest = {
        "schema_version": "ms:1.1-priority-dual-scope",
        "name": name,
        "saved_at_unix": time.time(),
        "num_agents": len(inst.graph.agents),
        "num_meetings": len(inst.graph.meetings),
        "num_factors": len(inst.graph.factors),
        "visualizations": viz_rel_map,               # {agent -> 'images/viz/<agent>.png'}
        "has_prompts": make_prompts_fn is not None,
        "prompts_file": (prompts_filename if make_prompts_fn is not None else None),
        "slot_labels": SLOT_LABELS,
        "meetings_meta": [
            {
                "mid": m.mid,
                "owner": getattr(m, "owner", None),
                "mode": m.mode,
                "location": m.location,
                "attendees": sorted(m.attendees),
                "duration_slots": getattr(m, "duration_slots", 1),
            }
            for m in sorted(inst.graph.meetings.values(), key=lambda x: x.mid)
        ],
    }
    (inst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return inst_dir

# ---------- batch: make-N (viz only, optional prompts) ----------

def generate_instances(
    n_instances: int,
    cfg: dict,
    dataset_dir: str | Path,
    *,
    base_seed: int = 123,
    make_prompts_fn: Optional[Callable[..., Dict[str, str]]] = None,
    prompts_kwargs: Optional[Dict[str, Any]] = None,
    prompts_filename: str = "prompts.json",
    progress: bool = True,
) -> Path:
    """
    Builds N instances with build_meeting_env(**cfg, rng_seed=...).
    Saves each in dataset_dir/seed_<seed>/ with ONLY visualizations (no raw building images),
    and writes dataset_dir/index.json. If `make_prompts_fn` is provided, also writes prompts.json.
    """
    dataset_dir = Path(dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    for i in range(n_instances):
        seed = base_seed + i
        cfg_i = dict(cfg)
        cfg_i["rng_seed"] = seed

        inst = build_meeting_env(**cfg_i)
        name = f"seed_{seed:04d}"
        inst_path = save_meeting_env_instance_viz_only(
            inst,
            out_dir=dataset_dir,
            name=name,
            make_prompts_fn=make_prompts_fn,
            prompts_kwargs=prompts_kwargs,
            prompts_filename=prompts_filename,
        )
        index.append({"seed": seed, "path": str(inst_path)})

        if progress:
            print(f"[{i+1}/{n_instances}] saved {name}")

    (dataset_dir / "index.json").write_text(
        json.dumps({"count": n_instances, "instances": index}, indent=2)
    )

    return dataset_dir
