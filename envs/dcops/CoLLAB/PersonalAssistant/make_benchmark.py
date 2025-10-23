import os, json, pickle, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# Use relative imports since generate.py is in the same package
from .generate import build_personal_env, generate_collages

# ---------- helpers ----------

def _hash_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()[:12]

def _rel_to(base: Path, target: Path) -> str:
    """
    Robust relative path computation. Works even if `target` wasn't created under `base`
    (e.g., when a callee wrote to an unexpected folder). We still prefer to write collages
    under the instance dir so this path stays nice and short.
    """
    base_abs = base.resolve()
    tgt_abs = target.resolve()
    rel = os.path.relpath(tgt_abs.as_posix(), start=base_abs.as_posix())
    return rel.replace("\\", "/")

# ---------- core: save-one (collages only, optional prompts) ----------

def save_personal_env_instance_collages_only(
    inst: Any,
    out_dir: str | Path,
    data_root: str | Path,
    name: Optional[str] = None,
    *,
    make_collages: bool = True,
    collage_cols: int = 3,
    collage_thumb: Tuple[int, int] = (192, 192),
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
        images/wardrobes/*.png      # ONLY collages (no outfit images)
        [prompts.json]               # if make_prompts_fn is provided
    """
    out_dir = Path(out_dir).resolve()
    data_root = Path(data_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        sig = {
            "agents": inst.graph.agents,
            "factors": [{"fid": f.fid, "scope": f.agent_scope, "ftype": f.ftype} for f in inst.graph.factors],
        }
        name = f"inst_{_hash_bytes(json.dumps(sig, sort_keys=True).encode())}"

    inst_dir = (out_dir / name).resolve()
    wardrobes_dir = (inst_dir / "images" / "wardrobes").resolve()
    wardrobes_dir.mkdir(parents=True, exist_ok=True)

    # Generate collages INSIDE the instance folder
    collage_rel_map: Dict[str, str] = {}
    if make_collages and generate_collages is not None:
        # Ask the generator to read outfit images from data_root but WRITE collages to inst_dir/images/wardrobes
        coll_ret = generate_collages(
            inst,
            data_root=data_root,                  # read source outfit images here
            out_subdir=str(wardrobes_dir),        # write collages HERE (absolute)
            cols=collage_cols,
            thumb_size=collage_thumb,
            pad=8,
        )

        # Normalize whatever the collage fn returned (absolute or relative)
        for agent, p in (coll_ret or {}).items():
            p_path = Path(p)
            if not p_path.is_absolute():
                # Try relative to instance dir first (preferred), else data_root (legacy behavior)
                cand1 = (inst_dir / p_path)
                cand2 = (data_root / p_path)
                p_path = cand1 if cand1.exists() else cand2
            if p_path.exists():
                collage_rel_map[agent] = _rel_to(inst_dir, p_path)

    # Optionally emit prompts
    if make_prompts_fn is not None:
        kwargs = {"collage_map": collage_rel_map}
        if prompts_kwargs:
            kwargs.update(prompts_kwargs)
        prompts = make_prompts_fn(inst.graph, inst.wardrobe, **kwargs)
        with open(inst_dir / prompts_filename, "w") as f:
            json.dump({"prompts": prompts}, f, indent=2)

    # Save the pure problem instance
    with open(inst_dir / "instance.pkl", "wb") as f:
        pickle.dump(inst, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Manifest
    manifest = {
        "name": name,
        "saved_at_unix": time.time(),
        "num_agents": len(inst.graph.agents),
        "num_factors": len(inst.graph.factors),
        "collages": collage_rel_map,                 # {agent -> 'images/wardrobes/<agent>.png'}
        "has_prompts": make_prompts_fn is not None,
        "prompts_file": (prompts_filename if make_prompts_fn is not None else None),
    }
    with open(inst_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return inst_dir

# ---------- batch: make-N (collages only, optional prompts) ----------

def generate_instances(
    n_instances: int,
    cfg: dict,
    dataset_dir: str | Path,
    data_root: str | Path,
    *,
    base_seed: int = 123,
    make_collages: bool = True,
    collage_cols: int = 3,
    collage_thumb: Tuple[int, int] = (192, 192),
    # prompts
    make_prompts_fn: Optional[Callable[..., Dict[str, str]]] = None,
    prompts_kwargs: Optional[Dict[str, Any]] = None,
    prompts_filename: str = "prompts.json",
    progress: bool = True,
) -> Path:
    """
    Builds N instances with build_personal_env(**cfg, rng_seed=...).
    Saves each in dataset_dir/seed_<seed>/ with ONLY collages (no outfit images),
    and writes dataset_dir/index.json. If `make_prompts_fn` is provided, also writes prompts.json.
    """
    dataset_dir = Path(dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    for i in range(n_instances):
        seed = base_seed + i
        cfg_i = dict(cfg)
        cfg_i["rng_seed"] = seed

        inst = build_personal_env(**cfg_i, data_root=data_root)
        name = f"seed_{seed:04d}"
        inst_path = save_personal_env_instance_collages_only(
            inst,
            out_dir=dataset_dir,
            data_root=data_root,
            name=name,
            make_collages=make_collages,
            collage_cols=collage_cols,
            collage_thumb=collage_thumb,
            make_prompts_fn=make_prompts_fn,
            prompts_kwargs=prompts_kwargs,
            prompts_filename=prompts_filename,
        )
        index.append({"seed": seed, "path": str(inst_path)})

        if progress:
            print(f"[{i+1}/{n_instances}] saved {name}")

    with open(dataset_dir / "index.json", "w") as f:
        json.dump({"count": n_instances, "instances": index}, f, indent=2)

    return dataset_dir
