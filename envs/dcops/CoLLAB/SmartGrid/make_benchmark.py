from __future__ import annotations
from dataclasses import dataclass, asdict  # noqa: F401 (kept for consistency)
from typing import Dict, List, Tuple, Optional, Callable
import json
from pathlib import Path

# Use relative imports since these files are in the same package
from .generate import generate_instance_from_catalog, greedy_peak_shaving, save_all_home_viz
from .prompt_maker import make_prompts_powerlite_from_catalog
from .data_structure import NeighborhoodPowerLiteEnv


PromptMakerFn = Callable[[object, str], Dict[str, str]]  # (instance, catalog_path) -> {home_id: prompt}


def write_instances_and_prompts(
    *,
    n_instances: int,
    cfg: dict,
    catalog_path: str,
    out_dir: str,
    base_seed: int = 0,
    # Prompt control (PA-style)
    make_prompts_fn: Optional[PromptMakerFn] = make_prompts_powerlite_from_catalog,
    prompts_kwargs: Optional[dict] = None,          # reserved for future prompt makers
    prompts_filename: str = "prompts.json",
    include_prompts: bool = True,
    # Viz control
    include_viz: bool = True,
) -> None:
    """
    Build N catalog-driven PowerLite instances and, for each i, create a **self-contained folder**:
      <out_dir>/instance_<i:03d>/
        instance.json          # problem instance
        {prompts_filename}     # prompts (if include_prompts=True)
        starts.json            # greedy baseline starts ({ "H::task": t0 })
        eval.json              # {"MainGridEnergy": ..., "D": [...], "G": [...]}
        images/homes/*.png     # per-home viz (if include_viz=True)

    Also writes a top-level <out_dir>/index.json manifest.

    Notes:
      - No post-processing / patching required. Filenames in index.json match outputs.
      - If include_prompts=False, the 'prompts' field in index.json is set to None.
      - If include_viz=False, the 'viz_dir' field in index.json is set to None.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index: List[dict] = []

    for i in range(n_instances):
        seed_i = base_seed + i
        cfg_i = {**cfg, "seed": seed_i}

        # ---- build instance
        inst = generate_instance_from_catalog(cfg_i, catalog_path)
        env = NeighborhoodPowerLiteEnv(inst)

        # ---- per-instance folder
        inst_dir = out / f"instance_{i:03d}"
        if include_viz:
            (inst_dir / "images" / "homes").mkdir(parents=True, exist_ok=True)
        else:
            inst_dir.mkdir(parents=True, exist_ok=True)

        # instance.json
        with (inst_dir / "instance.json").open("w") as f:
            json.dump(inst.to_dict(), f, indent=2)

        # prompts (optional)
        prompts_path_str: Optional[str] = None
        if include_prompts and make_prompts_fn is not None:
            prompts = make_prompts_fn(inst, catalog_path) if not prompts_kwargs else make_prompts_fn(inst, catalog_path, **prompts_kwargs)  # type: ignore[arg-type]
            prompts_path = inst_dir / prompts_filename
            with prompts_path.open("w") as f:
                json.dump({"prompts": prompts, "catalog_path": catalog_path}, f, indent=2)
            prompts_path_str = prompts_filename

        # baseline starts + evaluation (FINAL NAMES, no 'baseline_*')
        starts = greedy_peak_shaving(inst, iters=2, seed=seed_i)
        starts_serializable = {f"{hid}::{tid}": t0 for (hid, tid), t0 in starts.items()}
        with (inst_dir / "starts.json").open("w") as f:
            json.dump(starts_serializable, f, indent=2)

        eval_res = env.evaluate(starts)
        with (inst_dir / "eval.json").open("w") as f:
            json.dump({k: v for k, v in eval_res.items() if k != "ok"}, f, indent=2)

        # per-home viz (optional)
        viz_dir_str: Optional[str] = None
        if include_viz:
            save_all_home_viz(inst, starts=starts, out_dir=str(inst_dir / "images" / "homes"), show=False)
            viz_dir_str = "images/homes"

        # index entry
        files = {
            "instance": "instance.json",
            "prompts": prompts_path_str,  # may be None
            "starts": "starts.json",
            "eval": "eval.json",
            "viz_dir": viz_dir_str,       # may be None
        }
        index.append({
            "i": i,
            "seed": seed_i,
            "dir": f"instance_{i:03d}",
            "files": files,
        })
        print(f"[{i+1}/{n_instances}] wrote {inst_dir.name} (seed={seed_i})")

    # top-level index
    with (out / "index.json").open("w") as f:
        json.dump({"count": n_instances, "instances": index}, f, indent=2)


# --------------------
# CLI demo
# --------------------
if __name__ == "__main__":
    # Config
    cfg = dict(T=24, n_homes=6, tasks_per_home=(2, 4), allowed_window_len=(2, 6))

    # External catalog path
    catalog_path = "SmartGrid/data/devices.json"

    # Quick sanity: one instance + score
    inst = generate_instance_from_catalog(cfg, catalog_path)
    env = NeighborhoodPowerLiteEnv(inst)
    starts = greedy_peak_shaving(inst, iters=2, seed=7)
    res = env.evaluate(starts)
    print("MainGridEnergy:", res.get("MainGridEnergy"))

    # Batch save (self-contained folders) — WITH prompts
    out_dir = "datasets/powerlite"
    write_instances_and_prompts(
        n_instances=3,
        cfg=cfg,
        catalog_path=catalog_path,
        out_dir=out_dir,
        base_seed=100,
        include_prompts=True,
        prompts_filename="prompts.json",
        include_viz=True,
    )
    print(f"Wrote instances under {out_dir}/ (each with prompts + viz)")
