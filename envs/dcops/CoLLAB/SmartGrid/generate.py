from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import math
import random
from pathlib import Path
import itertools
# Use relative import since data_structure.py is in the same package
from .data_structure import InstanceSpec, HomeSpec, TaskSpec, SustainableCapFactor, NeighborhoodPowerLiteEnv


# Optional plotting (used by helper functions)
try:
    import matplotlib.pyplot as plt  # noqa: F401
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

# =============================================================
# Neighborhood-PowerLite (Catalog-driven)
# -------------------------------------------------------------
# - Agents are HOMES. Each home has tasks drawn from an external
#   device catalog JSON (see schema below).
# - Only decision: a start time for each task, within allowed set.
# - Sustainable cap S[t] per slot. Demand over S[t] pulls from
#   the main grid automatically. Score = sum_t G[t].
# - Single global factor connecting all tasks across homes.
# =============================================================




# --------------------
# Catalog helpers (external JSON only; no unused fields)
# --------------------

def load_device_catalog_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    devices = data.get("devices", [])
    if not isinstance(devices, list) or not devices:
        raise ValueError("Device catalog JSON must contain non-empty 'devices' list")
    for d in devices:
        if not {"id", "description", "consumption_kw", "duration_slots"} <= set(d.keys()):
            raise ValueError(f"Device entry missing fields: {d}")
        a, b = d["duration_slots"]
        if not (isinstance(a, int) and isinstance(b, int) and 1 <= a <= b):
            raise ValueError(f"duration_slots must be [min,max] ints for {d['id']}")
    return devices


def _sample_allowed_starts(T: int, L: int, wlen_rng: Tuple[int, int], rng: random.Random) -> List[int]:
    latest = T - L
    if latest < 0:
        return []
    wmin, wmax = wlen_rng
    wlen = max(1, min(rng.randint(wmin, wmax), latest + 1))
    start0 = rng.randint(0, latest - (wlen - 1)) if latest - (wlen - 1) >= 0 else 0
    return list(range(start0, start0 + wlen))


def sample_tasks_from_catalog(
    T: int,
    devices: List[dict],
    *,
    rng: random.Random,
    tasks_per_home: Tuple[int, int],
    allowed_window_len: Tuple[int, int],
) -> List[TaskSpec]:
    m = rng.randint(tasks_per_home[0], tasks_per_home[1])
    chosen = [rng.choice(devices) for _ in range(m)]
    id_counts: Dict[str, int] = {}
    tasks: List[TaskSpec] = []
    for spec in chosen:
        base = spec["id"]
        id_counts[base] = id_counts.get(base, 0) + 1
        suffix = id_counts[base] - 1
        task_id = base if suffix == 0 else f"{base}_{suffix}"
        cons = float(spec["consumption_kw"])
        dmin, dmax = spec["duration_slots"]
        L = rng.randint(int(dmin), int(dmax))
        allowed = _sample_allowed_starts(T, L, allowed_window_len, rng)
        if not allowed and L <= T:
            allowed = [0]
        tasks.append(TaskSpec(id=task_id, consumption=cons, duration=L, allowed_starts=allowed))
    return tasks


# --------------------
# Instance generation (from external catalog)
# --------------------
DEFAULT_CFG = dict(
    T=24,
    n_homes=5,
    tasks_per_home=(2, 4),
    allowed_window_len=(2, 6),
    S_pattern="sin",  # or "flat"
    S_base=6.0,
    S_amp=2.5,
    S_min_clip=0.0,
    seed=None,
)


def _make_S_cap(T: int, cfg: dict, rng: random.Random) -> List[float]:
    pattern = cfg.get("S_pattern", "sin")
    if pattern == "flat":
        base = float(cfg.get("S_base", 6.0))
        return [base] * T
    base = float(cfg.get("S_base", 6.0))
    amp = float(cfg.get("S_amp", 2.5))
    S = []
    for t in range(T):
        val = base + amp * math.sin(2 * math.pi * (t / max(1, T)))
        val += random.uniform(-0.4, 0.4)
        S.append(max(cfg.get("S_min_clip", 0.0), round(val, 3)))
    return S


def generate_instance_from_catalog(cfg: dict, catalog_path: str) -> InstanceSpec:
    cfg2 = {**DEFAULT_CFG, **(cfg or {})}
    rng = random.Random(cfg2.get("seed"))
    T = int(cfg2["T"])
    S_cap = _make_S_cap(T, cfg2, rng)
    devices = load_device_catalog_json(catalog_path)

    homes: List[HomeSpec] = []
    for h_idx in range(int(cfg2["n_homes"])):
        tasks = sample_tasks_from_catalog(
            T,
            devices,
            rng=rng,
            tasks_per_home=cfg2["tasks_per_home"],
            allowed_window_len=cfg2["allowed_window_len"],
        )
        homes.append(HomeSpec(id=f"H{h_idx}", tasks=tasks))

    meta = {"name": "Neighborhood-PowerLite", "single_factor": True, "catalog_path": catalog_path}
    return InstanceSpec(T=T, S_cap=S_cap, homes=homes, meta=meta)


# --------------------
# Baseline: greedy peak shaving (local search)
# --------------------

def greedy_peak_shaving(instance: InstanceSpec, iters: int = 2, seed: Optional[int] = None) -> Dict[Tuple[str, str], int]:
    env = NeighborhoodPowerLiteEnv(instance)
    rng = random.Random(seed)

    starts: Dict[Tuple[str, str], int] = {}
    for h in instance.homes:
        for t in h.tasks:
            starts[(h.id, t.id)] = t.allowed_starts[0]

    def score(temp: Dict[Tuple[str, str], int]) -> float:
        res = env.evaluate(temp)
        assert res["ok"], res.get("errors", [])
        return float(res["MainGridEnergy"])

    order: List[Tuple[str, str, float]] = []
    for h in instance.homes:
        for t in h.tasks:
            order.append((h.id, t.id, t.consumption * t.duration))
    order.sort(key=lambda x: x[2], reverse=True)

    for _ in range(max(1, iters)):
        for hid, tid, _ in order:
            task = env.task_index[(hid, tid)]
            best_t = starts[(hid, tid)]
            best_val = math.inf
            for cand in task.allowed_starts:
                starts[(hid, tid)] = cand
                val = score(starts)
                if val < best_val or (val == best_val and rng.random() < 0.5):
                    best_val, best_t = val, cand
            starts[(hid, tid)] = best_t
    return starts


# --------------------
# Visualization: per-home (cap on top; allowed periods below)
# --------------------

def _contiguous_runs(sorted_ints: List[int]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    for _, grp in itertools.groupby(sorted_ints, key=lambda n, c=itertools.count(): n - next(c)):
        chunk = list(grp)
        runs.append((chunk[0], chunk[-1]))
    return runs


def plot_home_caps_and_allowed(
    instance: InstanceSpec,
    home_id: str,
    *,
    starts: Optional[Dict[Tuple[str, str], int]] = None,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = True,
):
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available in this environment")

    home = next((h for h in instance.homes if h.id == home_id), None)
    if home is None:
        raise ValueError(f"Unknown home_id: {home_id}")

    import matplotlib.pyplot as plt  # local import for runtime availability

    T = instance.T
    S = instance.S_cap

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, max(1, len(home.tasks) * 0.5)], hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(range(T), S)
    ax0.set_ylabel("S_cap")
    ax0.set_xticks(range(T))
    ax0.set_xticklabels([])
    ax0.set_title(f"Home {home_id} — S_cap (top) and Allowed Start Periods (bottom)")

    ax1 = fig.add_subplot(gs[1, 0])
    yticks, ylabels = [], []
    for idx, task in enumerate(home.tasks):
        y = idx
        yticks.append(y)
        ylabels.append(f"{task.id}  (L={task.duration}, c={task.consumption})")
        allowed = sorted(set(task.allowed_starts))
        if allowed:
            for a, b in _contiguous_runs(allowed):
                ax1.plot([a, b + 1], [y, y], linewidth=6)
        if starts is not None and (home_id, task.id) in starts:
            ax1.scatter([starts[(home_id, task.id)]], [y], s=50, zorder=3)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)
    ax1.set_xlabel("Time slot")
    ax1.set_xlim(0, T)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_all_home_viz(
    instance: InstanceSpec,
    *,
    starts: Optional[Dict[Tuple[str, str], int]] = None,
    out_dir: str = "viz/homes",
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 150,
    show: bool = False,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for h in instance.homes:
        path = str(Path(out_dir) / f"{h.id}.png")
        plot_home_caps_and_allowed(
            instance,
            h.id,
            starts=starts,
            figsize=figsize,
            dpi=dpi,
            save_path=path,
            show=show,
        )



