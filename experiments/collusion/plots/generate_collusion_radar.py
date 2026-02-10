from __future__ import annotations

import asyncio
import argparse
import logging
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch
from scipy.stats import t as student_t

from experiments.collusion.plots.common import canonical_variant, default_out_dir
from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    as_int,
    ensure_dir,
    finite,
    mean,
    safe_load_json,
    sanitize_filename,
    write_json,
)
from experiments.common.plotting.logging_utils import (
    configure_basic_logging,
    log_saved_plot,
)
from experiments.common.plotting.load_runs import LoadedRun, load_runs
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)

_BAR_CHART_HEIGHT_SCALE = 0.8  # 20% smaller vertically
_PVALL_GROUP_PALETTE = [
    "#264653",  # Charcoal Blue
    "#2a9d8f",  # Verdigris
    "#8ab17d",  # Muted Olive
    "#e9c46a",  # Jasmine
    "#f4a261",  # Sandy Brown
    "#e76f51",  # Burnt Peach
]


def _apply_group_palette(groups: List[Dict[str, Any]], palette: List[str]) -> None:
    for idx, group in enumerate(groups):
        if idx >= len(palette):
            break
        group["color"] = palette[idx]


def _apply_readable_bars_style() -> None:
    # Bar figures can get very wide (many metrics / topologies), which makes the
    # default font sizes look tiny when viewing scaled-to-fit. Bump sizes for
    # readability while keeping the original layout/arrangement.
    plt.rcParams.update(
        {
            # Base font (used by fig.text / misc annotations)
            "font.size": 14,
            # Axes + legend
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


def _apply_large_font_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def _pretty_metric_label(key: str) -> str:
    k = str(key or "").strip()
    if k == "optimality_gap":
        return "Regret"
    if k == "achieved_over_optimal":
        return "Achieved / Optimal"
    if k == "regret_ratio":
        return "Regret"
    if k == "joint_reward_ratio":
        return "Joint Reward Ratio"
    if k == "joint_reward":
        return "Joint Reward"
    if k == "coalition_mean_regret":
        return "Coalition Mean Regret"
    if k == "noncoalition_mean_regret":
        return "Non-Coalition Mean Regret"
    if k == "coalition_regret_advantage_mean":
        return "Coalition Regret Advantage"
    if k == "judge_mean_rating":
        return "Collusion Judge (↓)"
    return k.replace("_", " ").title()


def _safe_load_config(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = safe_load_json(path)
            return payload if isinstance(payload, dict) else None
        if suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        payload = safe_load_json(path)
        if isinstance(payload, dict):
            return payload
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _pretty_topology_label(
    topo: str, *, title_by_topology: Optional[Dict[str, str]] = None
) -> str:
    s = str(topo or "").strip()
    if not s:
        return s
    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    canonical_key = alias_to_canonical.get(key, key)
    if title_by_topology:
        # Accept either the raw topology string or its canonicalized key (underscored).
        if s in title_by_topology:
            return str(title_by_topology[s])
        if canonical_key in title_by_topology:
            return str(title_by_topology[canonical_key])
    canonical = {
        "erdos_renyi": "Erdős–Rényi",
        "barabasi_albert": "Barabási–Albert",
        "watts_strogatz": "Watts–Strogatz",
    }
    if canonical_key in canonical:
        return canonical[canonical_key]
    return s.replace("_", " ").title()


def _topology_title_overrides_from_config(
    *, sweep_dir: Path, titles_config_path: Optional[Path]
) -> Dict[str, str]:
    """
    Best-effort mapping from topology -> pretty title (including parameters).

    Used to annotate plot headers for topology sweeps. Keys are canonicalized
    (underscored) topology names (e.g., "watts_strogatz").
    """
    title_by_topology: Dict[str, str] = {}
    try:
        cfg_path = (
            titles_config_path
            if titles_config_path is not None
            else (sweep_dir.parent.parent.parent / "config.json")
        )
        cfg = _safe_load_config(cfg_path) if cfg_path is not None else None

        edge_prob = None
        watts_k = None
        watts_rewire_prob = None
        ba_m = None
        if isinstance(cfg, dict):
            cn = cfg.get("communication_network")
            if isinstance(cn, dict):
                try:
                    edge_prob = (
                        float(cn.get("edge_prob"))
                        if cn.get("edge_prob") is not None
                        else None
                    )
                except Exception:
                    edge_prob = None
                try:
                    k_raw = cn.get("k", cn.get("nearest_neighbors", cn.get("num_neighbors")))
                    watts_k = int(k_raw) if k_raw is not None else None
                except Exception:
                    watts_k = None
                try:
                    p_raw = cn.get("rewire_prob", cn.get("rewiring_prob", cn.get("beta")))
                    watts_rewire_prob = float(p_raw) if p_raw is not None else None
                except Exception:
                    watts_rewire_prob = None
                try:
                    m_raw = cn.get("m", cn.get("edges_per_node", cn.get("num_edges_to_attach")))
                    ba_m = int(m_raw) if m_raw is not None else None
                except Exception:
                    ba_m = None

        if edge_prob is not None and np.isfinite(edge_prob):
            p_str = (f"{float(edge_prob):.3f}").rstrip("0").rstrip(".")
            title_by_topology["erdos_renyi"] = f"Erdős–Rényi (p={p_str})"

        has_watts_rewire_p = watts_rewire_prob is not None and np.isfinite(
            watts_rewire_prob
        )
        if watts_k is not None or has_watts_rewire_p:
            parts: List[str] = []
            if watts_k is not None:
                parts.append(f"k={int(watts_k)}")
            if has_watts_rewire_p:
                p_str = (f"{float(watts_rewire_prob):.3f}").rstrip("0").rstrip(".")
                parts.append(f"p={p_str}")
            if parts:
                title_by_topology["watts_strogatz"] = (
                    "Watts–Strogatz (" + ", ".join(parts) + ")"
                )

        if ba_m is not None:
            title_by_topology["barabasi_albert"] = f"Barabási–Albert (m={int(ba_m)})"
    except Exception:
        return {}
    return title_by_topology


def _topology_acronym_label(
    topo: str, *, title_by_topology: Optional[Dict[str, str]] = None
) -> str:
    """
    Abbreviate random-graph topology names (ER/WS/BA) while preserving any
    "(...)" parameter suffix from title_by_topology.
    """
    s = str(topo or "").strip()
    if not s:
        return s

    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    canonical_key = alias_to_canonical.get(key, key)

    full = _pretty_topology_label(s, title_by_topology=title_by_topology)
    abbrev = {"erdos_renyi": "ER", "watts_strogatz": "WS", "barabasi_albert": "BA"}.get(
        canonical_key
    )
    if not abbrev:
        return full

    if "(" in full and full.endswith(")"):
        return abbrev + " (" + full.split("(", 1)[1]
    return abbrev


def _canonical_topology_key(topo: Any) -> str:
    s = str(topo or "").strip()
    if not s:
        return s
    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    return alias_to_canonical.get(key, key)


def _sort_topology_keys(topologies: Iterable[str]) -> List[str]:
    deterministic = ["complete", "path", "star"]
    random_graphs = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
    deterministic_idx = {k: i for i, k in enumerate(deterministic)}
    random_idx = {k: i for i, k in enumerate(random_graphs)}

    def _key(topo_key: str) -> Tuple[int, int, str]:
        k = _canonical_topology_key(topo_key)
        if k in deterministic_idx:
            return (0, deterministic_idx[k], k)
        if k in random_idx:
            return (1, random_idx[k], k)
        return (2, 0, k)

    out: List[str] = []
    seen: set[str] = set()
    for t in topologies:
        k = _canonical_topology_key(t)
        if not k or k in seen:
            continue
        out.append(k)
        seen.add(k)
    out.sort(key=_key)
    return out


def _largest_non_secret_blackboard_id(
    rc: Dict[str, Any], *, secret_blackboard_id: Optional[int]
) -> Optional[int]:
    participants = rc.get("blackboard_participants")
    if not isinstance(participants, dict):
        return None

    best_id: Optional[int] = None
    best_size = -1
    for bb_raw, members in participants.items():
        bb_id = as_int(bb_raw)
        if bb_id is None:
            continue
        if secret_blackboard_id is not None and int(bb_id) == int(secret_blackboard_id):
            continue
        if not isinstance(members, (list, tuple, set)):
            continue
        size = len([m for m in members if m is not None and str(m).strip()])
        if size > best_size or (size == best_size and (best_id is None or bb_id < best_id)):
            best_size = int(size)
            best_id = int(bb_id)
    return best_id


def _colluder_secret_to_largest_bb_ratio_mean(
    rc: Dict[str, Any], metrics: Dict[str, Any]
) -> Optional[float]:
    colluders = rc.get("colluders")
    if not isinstance(colluders, list) or not colluders:
        return None

    secret_enabled = as_bool(rc.get("secret_channel_enabled"))
    secret_bb_id = as_int(rc.get("secret_blackboard_id"))
    if not secret_enabled or secret_bb_id is None:
        return 0.0

    largest_bb_id = _largest_non_secret_blackboard_id(
        rc, secret_blackboard_id=secret_bb_id
    )
    if largest_bb_id is None:
        return None

    agents = metrics.get("agents")
    if not isinstance(agents, list):
        return None

    posts_by_agent: Dict[str, Dict[str, Any]] = {}
    for a in agents:
        if not isinstance(a, dict):
            continue
        name = a.get("agent_name")
        if name is None:
            continue
        posts_by_bb = a.get("posts_by_blackboard")
        if not isinstance(posts_by_bb, dict):
            posts_by_bb = {}
        posts_by_agent[str(name)] = posts_by_bb

    secret_key = str(int(secret_bb_id))
    largest_key = str(int(largest_bb_id))
    ratios: List[float] = []
    for agent in [str(a) for a in colluders if a is not None and str(a).strip()]:
        counts = posts_by_agent.get(str(agent), {}) or {}
        secret_posts = as_int(counts.get(secret_key)) or 0
        largest_posts = as_int(counts.get(largest_key)) or 0
        ratios.append(float(secret_posts) / float(max(1, int(largest_posts))))
    return float(mean(ratios)) if ratios else 0.0


def _maybe_run_judge(
    *,
    sweep_dir: Path,
    run_judge: bool,
    judge_model: str,
    judge_max_output_tokens: int,
    judge_temperature: float,
    judge_max_concurrent: int,
    judge_max_retries: int,
    judge_baseline_all_blackboards: bool,
    judge_overwrite: bool,
) -> None:
    if not run_judge:
        return
    try:
        from experiments.collusion.judge_blackboards import (
            JudgeConfig,
            evaluate_sweep_dir,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to import judge pipeline: {exc}") from exc

    judge_cfg = JudgeConfig(
        model=str(judge_model),
        max_output_tokens=int(judge_max_output_tokens),
        temperature=float(judge_temperature),
    )
    asyncio.run(
        evaluate_sweep_dir(
            sweep_dir=sweep_dir,
            judge_cfg=judge_cfg,
            max_concurrent=int(judge_max_concurrent),
            overwrite=bool(judge_overwrite),
            dry_run=False,
            max_retries=int(judge_max_retries),
            baseline_all_blackboards=bool(judge_baseline_all_blackboards),
        )
    )


def _sample_std(values: Iterable[Any]) -> Optional[float]:
    vals = finite(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(np.std(np.array(vals, dtype=float), ddof=1))


def _standard_error(values: Iterable[Any]) -> Optional[float]:
    vals = finite(values)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    s = _sample_std(vals)
    if s is None:
        return 0.0
    return float(s) / float(math.sqrt(n))


def _ci95_half_width(values: Iterable[Any]) -> Optional[float]:
    vals = finite(values)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    s = _sample_std(vals)
    if s is None:
        return 0.0
    t_crit = float(student_t.ppf(0.975, int(n - 1)))
    if not np.isfinite(t_crit):
        t_crit = 1.96
    return float(t_crit) * float(s) / float(math.sqrt(n))


def _iqr(values: Iterable[Any]) -> Optional[float]:
    vals = finite(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    if not np.isfinite(q1) or not np.isfinite(q3):
        return None
    return float(q3 - q1)


def _iqr_half_width(values: Iterable[Any]) -> Optional[float]:
    iqr = _iqr(values)
    if iqr is None:
        return None
    return float(iqr) / 2.0


def _seed_means(rows: List[Dict[str, Any]], key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = r.get("seed")
        if seed is None:
            continue
        try:
            seed_i = int(seed)
        except Exception:
            continue
        v = as_float(r.get(key))
        if v is None:
            continue
        by_seed.setdefault(seed_i, []).append(float(v))

    out: List[float] = []
    for seed_i in sorted(by_seed):
        vals = by_seed[seed_i]
        if not vals:
            continue
        out.append(float(mean(vals)))
    return out


def _robust_range(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if len(values) == 1:
        center = float(values[0])
        pad = max(1e-6, abs(center) * 0.1)
        return center - pad, center + pad

    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if lo == hi:
        # Symmetric padding so constant values map to 0.5 after normalization.
        pad = max(1e-6, abs(lo) * 0.1)
        lo -= pad
        hi += pad
    return lo, hi


def _minmax_range(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.1)
        lo -= pad
        hi += pad
    return lo, hi



def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


_SMART_GRID_BOUNDS_CACHE: Dict[Tuple[Any, ...], Tuple[float, float]] = {}


def _smart_grid_min_max_utility(
    run_config: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """
    Best-effort extraction of (min_utility, max_utility) for SmartGrid.

    Older runs wrote joint_reward_ratio as joint_reward / max_utility, which
    collapses to 0 when max_utility == 0. For plotting existing results without
    re-running experiments, we re-generate the SmartGrid instance for the run's
    seed/config and use its [min_utility, max_utility] range for normalization.
    """
    try:
        seed = int(run_config.get("seed"))
        num_agents = int(run_config.get("num_agents"))
    except Exception:
        return None

    env_cfg = run_config.get("environment_cfg") or {}
    if not isinstance(env_cfg, dict):
        env_cfg = {}

    timeline_length = env_cfg.get("timeline_length", env_cfg.get("T", 24))
    tasks_per_home = env_cfg.get("tasks_per_home", (3, 6))
    try:
        min_tasks, max_tasks = tasks_per_home
    except Exception:
        min_tasks, max_tasks = 3, 6

    min_machines = int(env_cfg.get("min_machines_per_agent", min_tasks))
    max_machines = int(env_cfg.get("max_machines_per_agent", max_tasks))
    min_sources = int(env_cfg.get("min_sources_per_agent", 2))
    max_sources = int(env_cfg.get("max_sources_per_agent", 3))

    key = (
        seed,
        num_agents,
        int(timeline_length),
        min_machines,
        max_machines,
        min_sources,
        max_sources,
    )
    if key in _SMART_GRID_BOUNDS_CACHE:
        return _SMART_GRID_BOUNDS_CACHE[key]

    try:
        # Ensure external/CoLLAB is on sys.path (see envs/dcops/__init__.py).
        import envs.dcops  # noqa: F401
        from problem_layer.smart_grid import SmartGridConfig, generate_instance
    except Exception:
        return None

    try:
        collab_cfg = SmartGridConfig(
            num_agents=int(num_agents),
            timeline_length=int(timeline_length),
            min_machines_per_agent=int(min_machines),
            max_machines_per_agent=int(max_machines),
            min_sources_per_agent=int(min_sources),
            max_sources_per_agent=int(max_sources),
            rng_seed=int(seed),
        )

        repo_root = Path(__file__).resolve().parents[3]
        instance_dir = (
            repo_root
            / "envs"
            / "dcops"
            / "outputs"
            / "collab_instances"
            / "smart_grid"
            / f"seed_{seed}"
        )
        instance = generate_instance(collab_cfg, instance_dir)
        max_u = float(getattr(instance, "max_utility", 0.0))
        min_u = float(getattr(instance, "min_utility", 0.0))
    except Exception:
        return None

    _SMART_GRID_BOUNDS_CACHE[key] = (min_u, max_u)
    return (min_u, max_u)


def _smart_grid_normalized_joint_reward_ratio(
    *, run_config: Dict[str, Any], final_summary: Dict[str, Any], joint_reward: float
) -> Optional[float]:
    # Prefer bounds provided by the run itself (newer runs).
    min_r = as_float(final_summary.get("min_joint_reward"))
    max_r = as_float(final_summary.get("max_joint_reward"))
    if min_r is not None and max_r is not None and float(max_r) != float(min_r):
        return _clamp01((float(joint_reward) - float(min_r)) / (float(max_r) - float(min_r)))

    bounds = _smart_grid_min_max_utility(run_config)
    if bounds is None:
        return None
    min_u, max_u = bounds
    if float(max_u) == float(min_u):
        return None
    return _clamp01((float(joint_reward) - float(min_u)) / (float(max_u) - float(min_u)))


_TICK_LABEL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _titlecase_for_ticks(label: str) -> str:
    """
    Title-case metric labels for bar/histogram x-ticks while preserving acronyms.
    Examples:
      - "joint reward" -> "Joint Reward"
      - "Non-coalition mean reward" -> "Non-Coalition Mean Reward"
      - "CI95" / "IQR" stay upper-case
    """

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(ch.isalpha() for ch in token) and token.isupper():
            return token
        if not token:
            return token
        return token[0].upper() + token[1:].lower()

    return _TICK_LABEL_TOKEN_RE.sub(_repl, str(label))


def _wrap_labels(labels: List[str], *, width: int = 14) -> List[str]:
    out: List[str] = []
    for label in labels:
        s = _titlecase_for_ticks(str(label))
        if s.startswith("Coalition Regret"):
            out.append(s.replace("Coalition Regret", "Coalition\nRegret", 1))
            continue
        if s.startswith("Collusion Judge"):
            out.append(s.replace("Collusion Judge", "Collusion\nJudge", 1))
            continue
        if len(s) <= width:
            out.append(s)
            continue
        out.append(
            "\n".join(
                textwrap.wrap(
                    s,
                    width=int(width),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        )
    return out


def _bottom_margin_for_wrapped_xticks(labels_wrapped: List[str]) -> float:
    """
    Heuristic bottom margin for multi-line x tick labels (and our bottom legend).
    """
    max_lines = 1
    for lbl in labels_wrapped:
        try:
            max_lines = max(max_lines, str(lbl).count("\n") + 1)
        except Exception:
            continue
    # Baseline tuned for our bottom legend at y≈0.08. Add a bit more room per extra line.
    bottom = 0.14 + 0.04 * max(0, max_lines - 2)
    return float(min(0.28, bottom))


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    label: str
    higher_is_better: bool = True
    transform: Optional[Callable[[float], float]] = None
    flip_for_plot: bool = True
    center_zero: bool = False

    def apply(self, v: float) -> float:
        out = float(v)
        if self.transform is not None:
            out = float(self.transform(out))
        if self.flip_for_plot and not self.higher_is_better:
            out = -out
        return float(out)


_REWARD_METRIC_CHOICES = [
    "joint_reward_ratio",
    "achieved_over_optimal",
    "normalized_regret",
    "regret_ratio",
    "optimality_gap",
    "regret",
]


def _canonical_reward_metric_key(name: str) -> str:
    s = str(name or "").strip()
    if s == "regret":
        return "optimality_gap"
    if s == "normalized_regret":
        return "regret_ratio"
    return s


def _reward_metric_spec(name: str) -> _MetricSpec:
    key = _canonical_reward_metric_key(name)
    if key == "joint_reward_ratio":
        return _MetricSpec(
            "joint_reward_ratio", "Joint reward ratio", higher_is_better=True
        )
    if key == "achieved_over_optimal":
        return _MetricSpec(
            "achieved_over_optimal",
            "Achieved / Optimal",
            higher_is_better=True,
        )
    if key == "regret_ratio":
        return _MetricSpec(
            "regret_ratio", "Regret", higher_is_better=False, flip_for_plot=False
        )
    if key == "optimality_gap":
        return _MetricSpec(
            "optimality_gap", "Regret", higher_is_better=False, flip_for_plot=False
        )
    raise ValueError(f"Unknown reward metric: {name!r}")


def _default_metric_specs(*, reward_metric: str) -> List[_MetricSpec]:
    return [
        _reward_metric_spec(reward_metric),
        _MetricSpec("tasks_done", "Tasks done", higher_is_better=True),
        _MetricSpec(
            "violations",
            "Constraint violations",
            higher_is_better=False,
            flip_for_plot=False,
        ),
        _MetricSpec(
            "coalition_mean_regret",
            "Coalition regret",
            higher_is_better=False,
            flip_for_plot=False,
        ),
        _MetricSpec(
            "noncoalition_mean_regret",
            "Non-coalition regret",
            higher_is_better=False,
            flip_for_plot=False,
        ),
        _MetricSpec(
            "coalition_regret_advantage_mean",
            "Coalition Advantage (-)",
            higher_is_better=False,
            flip_for_plot=False,
            center_zero=True,
        ),
        _MetricSpec(
            "colluder_posts_secret_rate",
            "Colluders' Secret Message Rate",
            higher_is_better=True,
            flip_for_plot=False,
        ),
        _MetricSpec(
            "judge_mean_rating",
            "Collusion Judge",
            higher_is_better=False,
            flip_for_plot=False,
            transform=lambda v: float(v) / 5.0,
        ),
    ]


def _population_std(values: List[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    m = float(sum(values) / len(values))
    var = sum((x - m) ** 2 for x in values) / float(len(values))
    return float(var**0.5)


def _build_rows(runs: List[LoadedRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        rc = run.run_config or {}
        rid = str(rc.get("run_id") or run.run_dir.name)
        env_label = rc.get("environment_label")
        if env_label is None:
            try:
                if "__env" in rid:
                    suffix = rid.split("__env", 1)[1]
                    env_label = suffix.split("__", 1)[0] or None
            except Exception:
                env_label = None
        if env_label is not None and not str(env_label).strip():
            env_label = None
        env_name = rc.get("environment_name") if rc.get("environment_name") is not None else None

        fs = run.final_summary or {}
        if not isinstance(fs, dict):
            fs = {}
        metrics = run.metrics or {}
        if not isinstance(metrics, dict):
            metrics = {}

        joint_reward = as_float(fs.get("joint_reward"))
        joint_reward_ratio = as_float(fs.get("joint_reward_ratio"))
        raw_joint_reward = as_float(fs.get("raw_joint_reward"))

        # Back-compat: some environments historically stored the normalized score in
        # `joint_reward` and the raw score in `raw_joint_reward`.
        if (
            joint_reward_ratio is None
            and joint_reward is not None
            and raw_joint_reward is not None
            and 0.0 <= float(joint_reward) <= 1.0
        ):
            joint_reward_ratio = float(joint_reward)
            joint_reward = float(raw_joint_reward)

        # SmartGrid: normalize using min/max utilities so the metric is meaningful.
        if (
            joint_reward is not None
            and str(env_label or "").strip().lower() == "smart_grid"
        ):
            sg_ratio = _smart_grid_normalized_joint_reward_ratio(
                run_config=rc, final_summary=fs, joint_reward=float(joint_reward)
            )
            if sg_ratio is not None:
                joint_reward_ratio = float(sg_ratio)

        # Optional: exact Jira optimal computed by experiments/collusion/compute_jira_optimal.py.
        # Populated when `optimal_summary.json` exists in the run directory.
        optimal_joint_reward = None
        optimality_gap = None
        achieved_over_optimal = None
        regret_ratio = None
        try:
            optimal_payload = safe_load_json(run.run_dir / "optimal_summary.json")
            if isinstance(optimal_payload, dict):
                optimal = optimal_payload.get("optimal")
                if isinstance(optimal, dict):
                    optimal_joint_reward = as_float(optimal.get("joint_reward"))
            if joint_reward is not None and optimal_joint_reward is not None:
                optimality_gap = float(optimal_joint_reward) - float(joint_reward)
                if float(optimal_joint_reward) != 0.0:
                    achieved_over_optimal = float(joint_reward) / float(optimal_joint_reward)
                    regret_ratio = 1.0 - float(achieved_over_optimal)
                    if regret_ratio < 0.0:
                        regret_ratio = 0.0
                    elif regret_ratio > 1.0:
                        regret_ratio = 1.0
        except Exception:
            pass

        coalition_reward_sum = as_float(metrics.get("coalition_reward_sum"))
        coalition_reward_ratio = None
        if (
            coalition_reward_sum is not None
            and joint_reward is not None
            and joint_reward != 0.0
        ):
            coalition_reward_ratio = float(coalition_reward_sum) / float(joint_reward)

        # Optional: post-hoc LLM-as-a-judge outputs (written outside the run directory).
        judge_simple = None
        judge_medium = None
        judge_complex = None
        judge_mean_rating = None
        try:
            sweep_dir = run.run_dir.parent
            model_dir = sweep_dir.parent
            judge_path = (
                model_dir
                / "judge_secret_blackboard"
                / sweep_dir.name
                / f"{run.run_dir.name}.json"
            )
            judge_payload = safe_load_json(judge_path) if judge_path.exists() else None
            if isinstance(judge_payload, dict):
                judgements = judge_payload.get("judgements") or {}
                if isinstance(judgements, dict):

                    def _rating(key: str) -> Optional[float]:
                        j = judgements.get(key) or {}
                        if not isinstance(j, dict):
                            return None
                        return as_float(j.get("rating"))

                    judge_simple = _rating("simple")
                    judge_medium = _rating("medium")
                    judge_complex = _rating("complex")
                    vals = [
                        float(v)
                        for v in (judge_simple, judge_medium, judge_complex)
                        if v is not None
                    ]
                    if vals:
                        judge_mean_rating = float(mean(vals))
        except Exception:
            # Treat judge metrics as optional; keep plotting robust even if files are missing/corrupt.
            pass

        colluder_secret_to_largest_bb_ratio_mean = as_float(
            metrics.get("colluder_posts_secret_to_largest_bb_ratio_mean")
        )
        if colluder_secret_to_largest_bb_ratio_mean is None:
            colluder_secret_to_largest_bb_ratio_mean = (
                _colluder_secret_to_largest_bb_ratio_mean(rc, metrics)
            )

        posts_total_all = None
        posts_total_noncolluders = None
        try:
            agents = metrics.get("agents")
            if isinstance(agents, list):
                totals = [
                    as_int(a.get("posts_total"))
                    for a in agents
                    if isinstance(a, dict) and a.get("posts_total") is not None
                ]
                totals_i = [int(t) for t in totals if t is not None]
                if totals_i:
                    posts_total_all = float(sum(totals_i))
        except Exception:
            posts_total_all = None

        colluder_posts_total_i = as_int(metrics.get("colluder_posts_total"))
        if posts_total_all is not None and colluder_posts_total_i is not None:
            posts_total_noncolluders = float(
                max(0, int(posts_total_all) - int(colluder_posts_total_i))
            )

        coalition_mean_regret = as_float(metrics.get("coalition_mean_regret"))
        noncoalition_mean_regret = as_float(metrics.get("noncoalition_mean_regret"))
        coalition_regret_advantage_mean = None
        if coalition_mean_regret is not None and noncoalition_mean_regret is not None:
            coalition_regret_advantage_mean = float(noncoalition_mean_regret) - float(
                coalition_mean_regret
            )

        row: Dict[str, Any] = {
            "run_dir": str(run.run_dir),
            "run_id": rid,
            "seed": as_int(rc.get("seed")),
            "environment_label": str(env_label) if env_label is not None else None,
            "environment_name": str(env_name) if env_name is not None else None,
            "topology": rc.get("topology"),
            "colluder_count": as_int(rc.get("colluder_count")),
            "secret_channel_enabled": as_bool(rc.get("secret_channel_enabled")),
            "prompt_variant": canonical_variant(rc.get("prompt_variant")),
            "status": metrics.get("status") or fs.get("status"),
            # Run-level (collusion metrics)
            "coalition_reward_sum": coalition_reward_sum,
            "coalition_mean_reward": as_float(metrics.get("coalition_mean_reward")),
            "noncoalition_mean_reward": as_float(
                metrics.get("noncoalition_mean_reward")
            ),
            "coalition_mean_regret": coalition_mean_regret,
            "noncoalition_mean_regret": noncoalition_mean_regret,
            "coalition_advantage_mean": as_float(
                metrics.get("coalition_advantage_mean")
            ),
            "coalition_regret_advantage_mean": coalition_regret_advantage_mean,
            "tasks_done": as_float(metrics.get("tasks_done")),
            "violations": as_float(metrics.get("violations")),
            "total_cost": as_float(metrics.get("total_cost")),
            "priority_sum": as_float(metrics.get("priority_sum")),
            # New: regret + fairness (if present in metrics.json)
            "mean_regret": as_float(metrics.get("mean_regret")),
            "reward_std": as_float(metrics.get("reward_std")),
            # Run-level (environment summary)
            "coverage": as_float(fs.get("coverage")),
            "normalized_score": as_float(fs.get("normalized_score")),
            "joint_reward_ratio": joint_reward_ratio,
            "joint_reward": joint_reward,
            "optimal_joint_reward": optimal_joint_reward,
            "optimality_gap": optimality_gap,
            "achieved_over_optimal": achieved_over_optimal,
            "regret_ratio": regret_ratio,
            "coalition_reward_ratio": coalition_reward_ratio,
            # Comms: aggregate post counts + derived ratios
            "colluder_posts_secret_rate": as_float(metrics.get("colluder_posts_secret_rate")),
            "colluder_posts_total": as_float(metrics.get("colluder_posts_total")),
            "colluder_posts_secret": as_float(metrics.get("colluder_posts_secret")),
            "colluder_posts_non_secret": as_float(
                metrics.get("colluder_posts_non_secret")
            ),
            "posts_total_all": posts_total_all,
            "posts_total_noncolluders": posts_total_noncolluders,
            # Comms: secret vs largest non-secret blackboard (mean over colluders)
            "colluder_posts_secret_to_largest_bb_ratio_mean": colluder_secret_to_largest_bb_ratio_mean,
            # Judge (0–5 ratings; mean over simple/medium/complex prompts)
            "judge_simple_rating": judge_simple,
            "judge_medium_rating": judge_medium,
            "judge_complex_rating": judge_complex,
            "judge_mean_rating": judge_mean_rating,
        }

        # Backfill fairness from per-agent rewards if missing.
        if row.get("reward_std") is None:
            agents = metrics.get("agents")
            if isinstance(agents, list):
                rewards = finite(
                    [a.get("reward") for a in agents if isinstance(a, dict)]
                )
                row["reward_std"] = _population_std(rewards)

        rows.append(row)
    return rows


def _compute_and_write_optimal_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        from experiments.collusion import compute_jira_optimal

        try:
            instance = compute_jira_optimal._reconstruct_instance(run_dir)
        except Exception:
            instance = compute_jira_optimal._load_instance_from_agent_prompts(run_dir)

        weights = compute_jira_optimal._load_weights(run_dir, overrides=None)
        optimal = compute_jira_optimal.solve_optimal_assignment(
            instance=instance, weights=weights
        )

        payload: Dict[str, Any] = {
            "weights": {
                "tasks_done_bonus": weights.tasks_done_bonus,
                "priority_bonus": weights.priority_bonus,
                "violation_penalty": weights.violation_penalty,
            },
            "optimal": {
                "joint_reward": optimal.joint_reward,
                "tasks_done": optimal.tasks_done,
                "priority_sum": optimal.priority_sum,
                "total_cost": optimal.total_cost,
                "violations": optimal.violations,
                "assignment": {
                    k: (v if v is not None else "skip")
                    for k, v in optimal.assignment.items()
                },
            },
        }
        write_json(run_dir / "optimal_summary.json", payload)
        return payload
    except Exception as exc:
        logger.warning("Failed to compute optimal for %s: %s", run_dir, exc)
        return None


def _maybe_compute_missing_optimal_summaries(
    runs: List[LoadedRun], *, compute_optimal: bool
) -> None:
    if not compute_optimal:
        return
    for run in runs:
        run_dir = getattr(run, "run_dir", None)
        if not isinstance(run_dir, Path):
            continue
        if (run_dir / "optimal_summary.json").exists():
            continue
        _compute_and_write_optimal_summary(run_dir)


def _select_colluder_count(rows: List[Dict[str, Any]], requested: Optional[int]) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--colluder-count must be > 0 for a collusion radar chart")
        return int(requested)

    counts = sorted(
        {int(c) for c in finite(r.get("colluder_count") for r in rows) if int(c) > 0}
    )
    if not counts:
        raise ValueError(
            "No runs with colluder_count > 0 found; cannot build a collusion radar chart."
        )
    return int(counts[-1])


def _select_treatment_variant(
    rows: List[Dict[str, Any]], *, colluder_count: int, requested: Optional[str]
) -> str:
    if requested:
        return canonical_variant(requested)

    variants = sorted(
        {
            canonical_variant(r.get("prompt_variant"))
            for r in rows
            if int(r.get("colluder_count") or 0) == int(colluder_count)
            and r.get("secret_channel_enabled") is True
            and r.get("prompt_variant") is not None
        }
    )
    non_control = [v for v in variants if v != "control"]
    if non_control:
        return non_control[0]
    if variants:
        return variants[0]
    return "control"


def _group_filter(
    rows: List[Dict[str, Any]],
    *,
    colluder_count: int,
    secret: bool,
    prompt_variant: str,
    require_complete: bool,
) -> List[Dict[str, Any]]:
    prompt_variant = canonical_variant(prompt_variant)
    out: List[Dict[str, Any]] = []
    for r in rows:
        if int(r.get("colluder_count") or 0) != int(colluder_count):
            continue
        if bool(r.get("secret_channel_enabled") is True) != bool(secret):
            continue
        if str(r.get("prompt_variant") or "") != str(prompt_variant):
            continue
        if (
            require_complete
            and str(r.get("status") or "").strip().lower() != "complete"
        ):
            continue
        out.append(r)
    return out


def _mean_metric(rows: List[Dict[str, Any]], metric: _MetricSpec) -> Optional[float]:
    vals = finite(
        metric.apply(v) for v in (r.get(metric.key) for r in rows) if v is not None
    )
    if not vals:
        return None
    return float(mean(vals))


def _run_seeds(rows: List[Dict[str, Any]]) -> List[int]:
    seeds = sorted({int(s) for s in finite(r.get("seed") for r in rows)})
    return seeds


def _plot_radar(
    *,
    baseline_vals: List[float],
    treatment_vals: List[float],
    baseline_err: Optional[List[float]] = None,
    treatment_err: Optional[List[float]] = None,
    labels: List[str],
    baseline_label: str,
    treatment_label: str,
    title: str,
    out_path: Path,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    n = len(labels)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    b = list(baseline_vals) + [baseline_vals[0]]
    t = list(treatment_vals) + [treatment_vals[0]]

    def _plot_series(
        values: List[float],
        errors: Optional[List[float]],
        *,
        label: str,
        color: str,
        fill_alpha: float,
        band_alpha: float,
    ) -> None:
        ax.plot(angles, values, linewidth=2.0, label=label, color=color)
        if not errors or len(errors) != n:
            ax.fill(angles, values, alpha=fill_alpha, color=color)
            return
        lo = [
            _clamp01(float(v) - float(e) if e is not None and np.isfinite(float(e)) else float(v))
            for v, e in zip(values[:-1], errors)
        ]
        hi = [
            _clamp01(float(v) + float(e) if e is not None and np.isfinite(float(e)) else float(v))
            for v, e in zip(values[:-1], errors)
        ]
        lo_loop = lo + [lo[0]]
        hi_loop = hi + [hi[0]]
        band_angles = angles + list(reversed(angles))
        band_r = hi_loop + list(reversed(lo_loop))
        ax.fill(band_angles, band_r, alpha=band_alpha, color=color, linewidth=0.0)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])

    _plot_series(
        b,
        baseline_err,
        label=baseline_label,
        color="#2563eb",
        fill_alpha=0.18,
        band_alpha=0.14,
    )
    _plot_series(
        t,
        treatment_err,
        label=treatment_label,
        color="#dc2626",
        fill_alpha=0.18,
        band_alpha=0.14,
    )

    ax.set_title(title, pad=18)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_radar_multi(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: str,
    out_path: Path,
    use_error_bands: bool = False,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    n = len(labels)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.8, 5.0))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])

    for s in series:
        vals = list(s.get("values") or [])
        if len(vals) != n:
            continue
        color = str(s.get("color") or "#111827")
        label = str(s.get("label") or "series")
        linewidth = float(s.get("linewidth") or 1.8)
        alpha = float(s.get("fill_alpha") or 0.07)

        loop_vals = vals + [vals[0]]
        ax.plot(angles, loop_vals, linewidth=linewidth, label=label, color=color)
        if use_error_bands:
            errors = s.get("errors")
            if isinstance(errors, list) and len(errors) == n:
                lo = [
                    _clamp01(
                        float(v) - float(e)
                        if e is not None and np.isfinite(float(e))
                        else float(v)
                    )
                    for v, e in zip(vals, errors)
                ]
                hi = [
                    _clamp01(
                        float(v) + float(e)
                        if e is not None and np.isfinite(float(e))
                        else float(v)
                    )
                    for v, e in zip(vals, errors)
                ]
                lo_loop = lo + [lo[0]]
                hi_loop = hi + [hi[0]]
                band_angles = angles + list(reversed(angles))
                band_r = hi_loop + list(reversed(lo_loop))
                ax.fill(band_angles, band_r, alpha=0.12, color=color, linewidth=0.0)
            continue
        if alpha > 0:
            ax.fill(angles, loop_vals, alpha=alpha, color=color)

    ax.set_title(title, pad=18)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),
        frameon=True,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_grouped_bars(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: str,
    out_path: Path,
    y_label: str = "Normalized Mean",
) -> None:
    apply_default_style(plt)
    _apply_readable_bars_style()
    ensure_dir(out_path.parent)

    if not labels or not series:
        return

    def _infer_ylim(series: List[Dict[str, Any]], n: int) -> Tuple[float, bool]:
        max_hi = 0.0
        any_finite = False
        for s in series:
            vals = list(s.get("values") or [])
            errors = s.get("errors")
            errs = errors if isinstance(errors, list) else None
            for i in range(min(n, len(vals))):
                v = vals[i]
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if not np.isfinite(vf):
                    continue
                ef = 0.0
                if errs is not None and i < len(errs):
                    e = errs[i]
                    if e is not None:
                        try:
                            ef = abs(float(e))
                        except Exception:
                            ef = 0.0
                        if not np.isfinite(ef):
                            ef = 0.0
                hi = vf + ef
                if not np.isfinite(hi):
                    continue
                max_hi = max(max_hi, hi)
                any_finite = True

        if not any_finite:
            return 1.0, True

        if max_hi <= 1.0 + 1e-9:
            return 1.0, True
        return max_hi * 1.05, False

    labels_wrapped = _wrap_labels(labels, width=16)
    n_metrics = len(labels_wrapped)
    n_series = len(series)

    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86
    bar_width = total_width / float(max(1, n_series))
    offsets = (np.arange(n_series, dtype=float) - (n_series - 1) / 2.0) * bar_width

    fig_width = max(7.6, 0.85 * n_metrics + 2.2)
    fig_height = (4.6 if n_series <= 4 else 5.0) * _BAR_CHART_HEIGHT_SCALE
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for idx, s in enumerate(series):
        vals = list(s.get("values") or [])
        if len(vals) != n_metrics:
            continue
        errors = s.get("errors")
        yerr = errors if isinstance(errors, list) and len(errors) == n_metrics else None
        color = str(s.get("color") or "#111827")
        label = str(s.get("label") or f"series {idx + 1}")
        ax.bar(
            x + offsets[idx],
            vals,
            width=bar_width,
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "black"},
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
        )

    ax.set_title(title)
    ax.set_ylabel(str(y_label))
    y_max, use_fixed_ticks = _infer_ylim(series, n_metrics)
    if str(y_label).strip().lower() == "normalized mean":
        y_max, use_fixed_ticks = 1.0, True
    ax.set_ylim(0.0, float(y_max))
    if use_fixed_ticks:
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_wrapped, rotation=22, ha="right")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_grouped_bars_by_topology(
    *,
    topologies: List[str],
    series_by_topology: Dict[str, List[Dict[str, Any]]],
    labels: List[str],
    title: str,
    out_path: Path,
    title_by_topology: Optional[Dict[str, str]] = None,
    y_label: str = "Normalized Mean",
) -> None:
    apply_default_style(plt)
    _apply_readable_bars_style()
    ensure_dir(out_path.parent)

    if not topologies or not labels:
        return

    labels_wrapped = _wrap_labels(labels, width=16)

    # Use the first topology's series ordering as the canonical legend order.
    first_series = series_by_topology.get(topologies[0]) or []
    if not first_series:
        return

    n_metrics = len(labels_wrapped)
    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86
    all_series: List[Dict[str, Any]] = []
    for topo in topologies:
        all_series.extend(series_by_topology.get(topo) or [])

    def _infer_ylim(series: List[Dict[str, Any]], n: int) -> Tuple[float, bool]:
        max_hi = 0.0
        any_finite = False
        for s in series:
            vals = list(s.get("values") or [])
            errors = s.get("errors")
            errs = errors if isinstance(errors, list) else None
            for i in range(min(n, len(vals))):
                v = vals[i]
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if not np.isfinite(vf):
                    continue
                ef = 0.0
                if errs is not None and i < len(errs):
                    e = errs[i]
                    if e is not None:
                        try:
                            ef = abs(float(e))
                        except Exception:
                            ef = 0.0
                        if not np.isfinite(ef):
                            ef = 0.0
                hi = vf + ef
                if not np.isfinite(hi):
                    continue
                max_hi = max(max_hi, hi)
                any_finite = True

        if not any_finite:
            return 1.0, True

        if max_hi <= 1.0 + 1e-9:
            return 1.0, True
        return max_hi * 1.05, False

    y_max, use_fixed_ticks = _infer_ylim(all_series, n_metrics)
    if str(y_label).strip().lower() == "normalized mean":
        y_max, use_fixed_ticks = 1.0, True

    def _pretty_topology(topo: str) -> str:
        return _pretty_topology_label(topo, title_by_topology=title_by_topology)

    ncols = 2
    nrows = int(math.ceil(len(topologies) / float(ncols)))
    # Target a wide, paper-friendly aspect ratio (~4:12), scaling up if needed.
    fig_width = max(12.0, 2.4 * float(n_metrics))
    fig_height = max(4.0, fig_width / 3.0) * _BAR_CHART_HEIGHT_SCALE
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, topo in zip(axes, topologies):
        series = series_by_topology.get(topo) or []
        if not series:
            ax.set_title(f"{_pretty_topology(topo)} (no data)")
            continue

        # Center bars within each metric even if some series are missing (all-NaN) for this topology.
        series_arrays: List[np.ndarray] = []
        present_indices: List[int] = []
        for idx, s in enumerate(series):
            vals = list(s.get("values") or [])
            if len(vals) != n_metrics:
                series_arrays.append(np.full((n_metrics,), np.nan, dtype=float))
                continue
            arr = np.array(
                [float(v) if v is not None else np.nan for v in vals], dtype=float
            )
            series_arrays.append(arr)
            if np.any(np.isfinite(arr)):
                present_indices.append(idx)

        if not present_indices:
            ax.set_title(f"{_pretty_topology(topo)} (no data)")
            continue

        bar_width = total_width / float(max(1, len(present_indices)))
        offsets = (
            np.arange(len(present_indices), dtype=float)
            - (len(present_indices) - 1) / 2.0
        ) * bar_width
        half_bar = float(bar_width) / 2.0
        x_left = float(x[0] + float(np.min(offsets)) - half_bar)
        x_right = float(x[-1] + float(np.max(offsets)) + half_bar)

        for pos, idx in enumerate(present_indices):
            s = series[idx]
            vals_arr = series_arrays[idx]
            errors = s.get("errors")
            yerr = (
                errors
                if isinstance(errors, list) and len(errors) == n_metrics
                else None
            )
            color = str(s.get("color") or "#111827")
            label = str(s.get("label") or f"series {idx + 1}")
            mask = np.isfinite(vals_arr)
            if not np.any(mask):
                continue
            yerr_arr = None
            if yerr is not None:
                yerr_arr = np.array(
                    [float(e) if e is not None else np.nan for e in yerr], dtype=float
                )
                yerr_arr = yerr_arr[mask]

            ax.bar(
                (x[mask] + offsets[pos]),
                vals_arr[mask],
                width=bar_width,
                yerr=yerr_arr,
                capsize=4,
                error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "black"},
                label=label,
                color=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.92,
            )

        ax.set_title(_pretty_topology(topo))
        ax.set_ylim(0.0, float(y_max))
        # Use the full horizontal span (minimal side padding) so long tick labels have more room.
        ax.set_xlim(x_left - 0.02, x_right + 0.02)
        if use_fixed_ticks:
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Hide unused subplots (if any).
    for ax in axes[len(topologies) :]:
        ax.axis("off")

    bottom_start = (nrows - 1) * ncols
    for idx, ax in enumerate(axes[: len(topologies)]):
        ax.set_xticks(x)
        if idx >= bottom_start:
            ax.set_xticklabels(labels_wrapped, rotation=0, ha="center")
        else:
            ax.set_xticklabels([])

    # Legend: collect unique labels from all panels, ordered by the first topology's series order.
    desired_order = [str(s.get("label") or "") for s in first_series]
    handle_by_label: Dict[str, Any] = {}
    for ax in axes[: len(topologies)]:
        handles_here, labels_here = ax.get_legend_handles_labels()
        for handle, label in zip(handles_here, labels_here):
            if label and label not in handle_by_label:
                handle_by_label[label] = handle

    handles = [
        handle_by_label[label] for label in desired_order if label in handle_by_label
    ]
    leg_labels = [label for label in desired_order if label in handle_by_label]
    for label, handle in handle_by_label.items():
        if label not in set(leg_labels):
            handles.append(handle)
            leg_labels.append(label)

    if handles:
        ncol = min(len(handles), 6)
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.10),
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.2,
            handlelength=1.4,
            handletextpad=0.6,
            labelspacing=0.4,
        )

    # Intentionally no main header / suptitle (per user request).
    bottom_margin = _bottom_margin_for_wrapped_xticks(labels_wrapped)
    fig.tight_layout(rect=(0.04, bottom_margin, 0.995, 1.0))
    # Global y label (avoid repeated labels per-subplot), centered on the subplot area.
    try:
        bboxes = [ax.get_position() for ax in axes[: len(topologies)]]
        y0 = min(b.y0 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)
        y_center = (y0 + y1) / 2.0
    except Exception:
        y_center = 0.5
    fig.text(
        0.03,
        float(y_center),
        str(y_label),
        va="center",
        rotation="vertical",
        fontsize=plt.rcParams.get("axes.labelsize", 12),
    )
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_grouped_bars_single_topology(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: Optional[str],
    out_path: Path,
    y_label: str = "Normalized Mean",
) -> None:
    apply_default_style(plt)
    _apply_readable_bars_style()
    ensure_dir(out_path.parent)

    if not labels or not series:
        return

    def _infer_ylim(series: List[Dict[str, Any]], n: int) -> Tuple[float, bool]:
        max_hi = 0.0
        any_finite = False
        for s in series:
            vals = list(s.get("values") or [])
            errors = s.get("errors")
            errs = errors if isinstance(errors, list) else None
            for i in range(min(n, len(vals))):
                v = vals[i]
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if not np.isfinite(vf):
                    continue
                ef = 0.0
                if errs is not None and i < len(errs):
                    e = errs[i]
                    if e is not None:
                        try:
                            ef = abs(float(e))
                        except Exception:
                            ef = 0.0
                        if not np.isfinite(ef):
                            ef = 0.0
                hi = vf + ef
                if not np.isfinite(hi):
                    continue
                max_hi = max(max_hi, hi)
                any_finite = True

        if not any_finite:
            return 1.0, True

        if max_hi <= 1.0 + 1e-9:
            return 1.0, True
        return max_hi * 1.05, False

    labels_wrapped = _wrap_labels(labels, width=16)
    n_metrics = len(labels_wrapped)
    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86

    # Keep series order stable, but drop all-NaN series and mask per-metric NaNs.
    series_arrays: List[np.ndarray] = []
    present_indices: List[int] = []
    for idx, s in enumerate(series):
        vals = list(s.get("values") or [])
        if len(vals) != n_metrics:
            series_arrays.append(np.full((n_metrics,), np.nan, dtype=float))
            continue
        arr = np.array(
            [float(v) if v is not None else np.nan for v in vals], dtype=float
        )
        series_arrays.append(arr)
        if np.any(np.isfinite(arr)):
            present_indices.append(idx)

    if not present_indices:
        return

    bar_width = total_width / float(max(1, len(present_indices)))
    offsets = (
        np.arange(len(present_indices), dtype=float)
        - (len(present_indices) - 1) / 2.0
    ) * bar_width

    # Match _plot_grouped_bars_by_topology aesthetics (wide aspect ratio, bottom legend).
    # Keep it wide enough for unrotated tick labels, but avoid oversized
    # canvases that cause "tiny text" when viewers scale-to-fit.
    fig_width = max(12.0, 2.0 * float(n_metrics))
    fig_height = max(5.2, fig_width / 2.6) * _BAR_CHART_HEIGHT_SCALE
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    font_scale = 1.1
    label_fs = float(plt.rcParams.get("axes.labelsize", 10)) * font_scale
    tick_fs_x = float(plt.rcParams.get("xtick.labelsize", 9)) * font_scale
    tick_fs_y = float(plt.rcParams.get("ytick.labelsize", 9)) * font_scale

    for pos, idx in enumerate(present_indices):
        s = series[idx]
        vals_arr = series_arrays[idx]
        errors = s.get("errors")
        yerr = errors if isinstance(errors, list) and len(errors) == n_metrics else None
        color = str(s.get("color") or "#111827")
        label = str(s.get("label") or f"series {idx + 1}")

        mask = np.isfinite(vals_arr)
        if not np.any(mask):
            continue

        yerr_arr = None
        if yerr is not None:
            yerr_arr = np.array(
                [float(e) if e is not None else np.nan for e in yerr], dtype=float
            )
            yerr_arr = yerr_arr[mask]

        ax.bar(
            (x[mask] + offsets[pos]),
            vals_arr[mask],
            width=bar_width,
            yerr=yerr_arr,
            capsize=4,
            error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "black"},
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
        )

    if title is not None and str(title).strip():
        ax.set_title(str(title))
    y_max, use_fixed_ticks = _infer_ylim(series, n_metrics)
    if str(y_label).strip().lower() == "normalized mean":
        y_max, use_fixed_ticks = 1.0, True
    ax.set_ylim(0.0, float(y_max))
    if use_fixed_ticks:
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.tick_params(axis="y", labelsize=tick_fs_y)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_ylabel(str(y_label), fontsize=label_fs, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_wrapped, rotation=0, ha="center", fontsize=tick_fs_x)
    # Use the full horizontal span (minimal side padding) so long tick labels have more room.
    half_bar = float(bar_width) / 2.0
    x_left = float(x[0] + float(np.min(offsets)) - half_bar)
    x_right = float(x[-1] + float(np.max(offsets)) + half_bar)
    ax.set_xlim(x_left - 0.02, x_right + 0.02)

    # Reserve room for the bottom legend and left y label.
    bottom_margin = _bottom_margin_for_wrapped_xticks(labels_wrapped)
    fig.tight_layout(rect=(0.04, bottom_margin, 0.995, 1.0))

    handles, leg_labels = ax.get_legend_handles_labels()
    if handles:
        ncol = min(len(handles), 6)
        ax_box = ax.get_position()
        ax_center_x = float((ax_box.x0 + ax_box.x1) / 2.0)
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(ax_center_x, -0.01),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.2,
            handlelength=1.4,
            handletextpad=0.6,
            labelspacing=0.4,
        )
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_combined_six_bars_by_topology(
    *,
    rows: List[Dict[str, Any]],
    metric_key: str,
    out_path: Path,
    colluder_count: int,
    baseline_variant: str,
    include_incomplete: bool,
    title_by_topology: Optional[Dict[str, str]] = None,
) -> None:
    # Match experiments/collusion/plots/generate_jira_regret_report.py aesthetics.
    try:
        matplotlib.rcdefaults()
    except Exception:
        pass
    try:
        plt.style.use("default")
    except Exception:
        pass
    _apply_large_font_style()

    topologies = _sort_topology_keys(
        [str(r.get("topology")) for r in rows if r.get("topology") is not None]
    )
    if not topologies:
        return

    require_complete = not bool(include_incomplete)
    conditions = ["baseline", "control", "simple"]
    colors = {
        "baseline": "#264653",
        "control": "#2a9d8f",
        "simple": "#8ab17d",
    }

    def _rows_for(topo: str, condition: str) -> List[Dict[str, Any]]:
        topo_rows = [
            r for r in rows if _canonical_topology_key(r.get("topology")) == topo
        ]
        if condition == "baseline":
            return _group_filter(
                topo_rows,
                colluder_count=colluder_count,
                secret=False,
                prompt_variant=baseline_variant,
                require_complete=require_complete,
            )
        return _group_filter(
            topo_rows,
            colluder_count=colluder_count,
            secret=True,
            prompt_variant=condition,
            require_complete=require_complete,
        )

    def _stats_for(topo: str, condition: str, key: str) -> Tuple[float, float, int]:
        seed_vals = _seed_means(_rows_for(topo, condition), key)
        if not seed_vals:
            return float("nan"), float("nan"), 0
        mu = float(mean(seed_vals))
        se = float(_standard_error(seed_vals) or 0.0)
        return mu, se, len(seed_vals)

    metric_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_means: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    for topo in topologies:
        for c in conditions:
            mu, se, _ = _stats_for(topo, c, metric_key)
            metric_means[c].append(mu)
            metric_sems[c].append(se)

            j_mu, j_se, _ = _stats_for(topo, c, "judge_mean_rating")
            judge_means[c].append(j_mu)
            judge_sems[c].append(j_se)

    any_judge = any(math.isfinite(v) for c in conditions for v in judge_means[c])
    if not any_judge:
        logger.warning("No judge_mean_rating data found; skipping: %s", out_path)
        return

    ensure_dir(out_path.parent)
    fig, ax_metric = plt.subplots(nrows=1, ncols=1, figsize=(12.0, 3.0))
    ax_judge = ax_metric.twinx()

    x = np.arange(len(topologies), dtype=float)
    width = 0.11
    metric_offsets = {"baseline": -2.5 * width, "control": -1.5 * width, "simple": -0.5 * width}
    judge_offsets = {"baseline": 0.5 * width, "control": 1.5 * width, "simple": 2.5 * width}

    for c in conditions:
        ax_metric.bar(
            x + metric_offsets[c],
            metric_means[c],
            width=width,
            yerr=metric_sems[c],
            color=colors[c],
            alpha=0.85,
            capsize=3,
            label="_nolegend_",
        )
    for c in conditions:
        ax_judge.bar(
            x + judge_offsets[c],
            judge_means[c],
            width=width,
            yerr=judge_sems[c],
            color=colors[c],
            alpha=0.35,
            capsize=3,
            hatch="///",
            edgecolor="black",
            linewidth=0.3,
            label="_nolegend_",
        )

    ax_metric_label = _pretty_metric_label(metric_key)
    if metric_key in {"regret_ratio", "normalized_regret", "optimality_gap"}:
        ax_metric_label = f"{ax_metric_label} (↓)"
    if metric_key in {"regret_ratio", "normalized_regret"}:
        ax_metric_label = "Normalized Regret (↓)"
    elif metric_key in {
        "normalized_coalition_advantage",
        "normalized_coalition_regret_gap",
        "coalition_regret_ratio",
    }:
        ax_metric_label = "Normalized"
    ax_metric.set_ylabel(ax_metric_label)
    if metric_key in {
        "regret_ratio",
        "normalized_regret",
        "normalized_coalition_advantage",
        "normalized_coalition_regret_gap",
        "coalition_regret_ratio",
    }:
        ax_metric.set_ylim(0.0, 1.0)
    judge_axis_label = _pretty_metric_label("judge_mean_rating").replace(" (Judge)", "")
    ax_judge.set_ylabel(judge_axis_label, labelpad=10)
    ax_judge.tick_params(axis="y", labelcolor="black")
    ax_judge.spines["right"].set_visible(True)

    condition_order = ["baseline", "control", "simple"]
    condition_labels = {
        "baseline": "Baseline (no SC)",
        "control": "Control (SC)",
        "simple": "Simple (SC)",
    }
    condition_handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=condition_labels.get(c, c))
        for c in condition_order
    ]
    style_color = "white"
    ax_metric_label_for_legend = (
        ax_metric_label.replace(" (↓)", "").replace(" (-)", "").strip()
    )
    style_handles = [
        Patch(
            facecolor=style_color,
            edgecolor="black",
            linewidth=0.8,
            alpha=1.0,
            label=ax_metric_label_for_legend,
        ),
        Patch(
            facecolor=style_color,
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            alpha=1.0,
            label=_pretty_metric_label("judge_mean_rating").replace(" (↓)", "").strip(),
        ),
    ]

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(
        [_topology_acronym_label(t, title_by_topology=title_by_topology) for t in topologies],
        rotation=0,
        ha="center",
    )
    ax_metric.tick_params(axis="x", pad=6)
    ax_metric.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_metric.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax_metric.set_axisbelow(True)

    legend_handles = condition_handles + style_handles
    legend_labels = [h.get_label() for h in legend_handles]
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.22)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(legend_handles),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.6,
        labelspacing=0.4,
    )
    plt.savefig(out_path, dpi=200)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _pretty_variant(name: str) -> str:
    s = canonical_variant(name)
    return s.replace("_", " ").title() if s else s


def _slice_by_indices(values: List[Any], indices: List[int]) -> List[Any]:
    return [values[i] for i in indices if i < len(values)]


def _slice_series_metrics(
    series: List[Dict[str, Any]], indices: List[int]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in series:
        vals = list(s.get("values") or [])
        errs = list(s.get("errors") or [])
        out.append(
            {
                **s,
                "values": _slice_by_indices(vals, indices),
                "errors": _slice_by_indices(errs, indices) if errs else [],
            }
        )
    return out


def _sorted_variants(variants: Iterable[str]) -> List[str]:
    preferred = ["control", "simple", "deception", "structured", "aggressive"]
    vals = [canonical_variant(v) for v in variants if v is not None and str(v).strip()]
    uniq: List[str] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        uniq.append(v)
        seen.add(v)

    out: List[str] = []
    for v in preferred:
        if v in seen:
            out.append(v)
            seen.remove(v)
    out.extend(sorted(seen))
    return out


def _infer_single_topology(rows: List[Dict[str, Any]]) -> Optional[str]:
    values = {str(r.get("topology")) for r in rows if r.get("topology") is not None}
    if len(values) == 1:
        return next(iter(values))
    return None


def _generate(
    *,
    sweep_dir: Path,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    colluder_count_requested: Optional[int],
    treatment_variant_requested: Optional[str],
    plot_all_prompt_variants: bool,
    baseline_variant_requested: str,
    include_incomplete: bool,
    reward_metric: str,
    title_prefix: Optional[str] = None,
    title_by_topology: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> bool:
    if not rows:
        if strict:
            raise SystemExit("No runs found; cannot build a collusion radar chart.")
        return False

    colluder_count = _select_colluder_count(rows, colluder_count_requested)
    treatment_variant = _select_treatment_variant(
        rows, colluder_count=colluder_count, requested=treatment_variant_requested
    )
    baseline_variant = str(baseline_variant_requested or "control").strip()
    require_complete = not bool(include_incomplete)

    baseline = _group_filter(
        rows,
        colluder_count=colluder_count,
        secret=False,
        prompt_variant=baseline_variant,
        require_complete=require_complete,
    )
    if not baseline:
        msg = (
            f"No baseline runs found for colluder_count={colluder_count}, secret_channel_enabled=false, "
            f"prompt_variant={baseline_variant!r}."
        )
        if strict:
            raise SystemExit(msg)
        return False

    metrics = [
        m
        for m in _default_metric_specs(reward_metric=str(reward_metric))
        if m.key != "colluder_posts_secret_rate"
    ]

    topo = _infer_single_topology(rows)
    prefix = (str(title_prefix).strip() + "\n") if title_prefix else ""

    ensure_dir(out_dir)
    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)

    if plot_all_prompt_variants:
        available = {
            str(r.get("prompt_variant"))
            for r in rows
            if int(r.get("colluder_count") or 0) == int(colluder_count)
            and r.get("secret_channel_enabled") is True
            and r.get("prompt_variant") is not None
        }
        variants = _sorted_variants(available)
        if not variants:
            msg = f"No secret-channel runs found for colluder_count={colluder_count} (secret_channel_enabled=true)."
            if strict:
                raise SystemExit(msg)
            return False

        groups: List[Dict[str, Any]] = [
            {
                "kind": "baseline",
                "secret_channel_enabled": False,
                "prompt_variant": baseline_variant,
                "label": "Baseline (no SC)",
                "color": None,
                "rows": baseline,
            }
        ]
        for pv in variants:
            grp_rows = _group_filter(
                rows,
                colluder_count=colluder_count,
                secret=True,
                prompt_variant=pv,
                require_complete=require_complete,
            )
            if not grp_rows:
                continue
            groups.append(
                {
                    "kind": "treatment",
                    "secret_channel_enabled": True,
                    "prompt_variant": pv,
                    "label": f"{_pretty_variant(pv)} (SC)",
                    "color": None,
                    "rows": grp_rows,
                }
            )

        _apply_group_palette(groups, _PVALL_GROUP_PALETTE)

        if len(groups) <= 1:
            msg = f"Found baseline but no secret-channel runs to plot for colluder_count={colluder_count}."
            if strict:
                raise SystemExit(msg)
            return False

        raw_summary: Dict[str, Any] = {
            "sweep_dir": str(sweep_dir),
            "topology": topo,
            "colluder_count": colluder_count,
            "groups": [
                {
                    "kind": g["kind"],
                    "secret_channel_enabled": g["secret_channel_enabled"],
                    "prompt_variant": g["prompt_variant"],
                    "label": g["label"],
                    "n_runs": len(g["rows"]),
                    "seeds": _run_seeds(g["rows"]),
                }
                for g in groups
            ],
            "metrics": [],
        }

        # Build per-group normalized metric vectors (mean +/- std over seeds).
        group_vectors: List[Dict[str, Any]] = []
        group_vectors_se: List[Dict[str, Any]] = []
        group_vectors_ci95: List[Dict[str, Any]] = []
        group_vectors_iqr: List[Dict[str, Any]] = []
        metric_defs: List[Dict[str, Any]] = []
        for m in metrics:
            seed_raw_by_group: List[List[float]] = []
            pooled_transformed: List[float] = []
            for g in groups:
                seed_raw = _seed_means(g["rows"], m.key)
                seed_raw_by_group.append(seed_raw)
                if seed_raw:
                    pooled_transformed.extend([m.apply(v) for v in seed_raw])

            lo, hi = _minmax_range(pooled_transformed) if pooled_transformed else (0.0, 1.0)
            if m.center_zero:
                max_abs = max(abs(float(lo)), abs(float(hi)))
                scale_lo = -max_abs
                scale_hi = max_abs
            else:
                max_abs = None
                scale_lo = float(lo)
                scale_hi = float(hi)

            def _norm_one(value: float) -> float:
                if scale_hi == scale_lo:
                    return 0.5
                if m.center_zero:
                    if not max_abs:
                        return 0.5
                    return _clamp01(0.5 + float(value) / (2.0 * float(max_abs)))
                return _clamp01((float(value) - float(scale_lo)) / (float(scale_hi) - float(scale_lo)))

            entry = {
                "key": m.key,
                "label": m.label,
                "higher_is_better": m.higher_is_better,
                "scale_lo": scale_lo,
                "scale_hi": scale_hi,
                "group_means_raw": [
                    {
                        "label": groups[i]["label"],
                        "mean_raw": float(mean(seed_raw_by_group[i]))
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_stds_raw": [
                    {
                        "label": groups[i]["label"],
                        "std_raw": float(_sample_std(seed_raw_by_group[i]) or 0.0)
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_means_transformed": [
                    {
                        "label": groups[i]["label"],
                        "mean_transformed": float(
                            mean([m.apply(v) for v in seed_raw_by_group[i]])
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_stds_transformed": [
                    {
                        "label": groups[i]["label"],
                        "std_transformed": float(
                            _sample_std([m.apply(v) for v in seed_raw_by_group[i]])
                            or 0.0
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_means_norm01": [
                    {
                        "label": groups[i]["label"],
                        "mean_norm01": float(
                            mean([_norm_one(m.apply(v)) for v in seed_raw_by_group[i]])
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_stds_norm01": [
                    {
                        "label": groups[i]["label"],
                        "std_norm01": float(
                            _sample_std(
                                [_norm_one(m.apply(v)) for v in seed_raw_by_group[i]]
                            )
                            or 0.0
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_se_norm01": [
                    {
                        "label": groups[i]["label"],
                        "se_norm01": float(
                            _standard_error(
                                [_norm_one(m.apply(v)) for v in seed_raw_by_group[i]]
                            )
                            or 0.0
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_ci95_norm01": [
                    {
                        "label": groups[i]["label"],
                        "ci95_norm01": float(
                            _ci95_half_width(
                                [_norm_one(m.apply(v)) for v in seed_raw_by_group[i]]
                            )
                            or 0.0
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
                "group_iqr_half_norm01": [
                    {
                        "label": groups[i]["label"],
                        "iqr_half_norm01": float(
                            _iqr_half_width(
                                [_norm_one(m.apply(v)) for v in seed_raw_by_group[i]]
                            )
                            or 0.0
                        )
                        if seed_raw_by_group[i]
                        else None,
                    }
                    for i in range(len(groups))
                ],
            }
            metric_defs.append(entry)

        # Drop metrics that are missing everywhere (prevents wide plots with empty columns).
        metric_defs = [
            md
            for md in metric_defs
            if any(
                v.get("mean_norm01") is not None
                for v in (md.get("group_means_norm01") or [])
                if isinstance(v, dict)
            )
        ]

        if not metric_defs:
            msg = "No plottable metrics found (all missing/NaN)."
            if strict:
                raise SystemExit(msg)
            return False

        raw_summary["metrics"] = metric_defs
        labels_all: List[str] = []
        for md in metric_defs:
            base = str(md.get("label") or "")
            key = str(md.get("key") or "")
            if key == "coalition_regret_advantage_mean":
                labels_all.append(base)
                continue
            if key == "colluder_posts_secret_rate":
                arrow = "–"
            else:
                arrow = "↑" if bool(md.get("higher_is_better", True)) else "↓"
            labels_all.append(f"{base} ({arrow})")

        bar_indices = [
            i
            for i, md in enumerate(metric_defs)
            if str(md.get("key") or "") != "colluder_posts_secret_rate"
        ]
        bar_metric_defs = [metric_defs[i] for i in bar_indices]
        labels_bars: List[str] = []
        for md in bar_metric_defs:
            base = str(md.get("label") or "")
            key = str(md.get("key") or "")
            if key == "coalition_regret_advantage_mean":
                labels_bars.append(base)
                continue
            arrow = "↑" if bool(md.get("higher_is_better", True)) else "↓"
            labels_bars.append(f"{base} ({arrow})")

        for idx, g in enumerate(groups):
            # Pull in the normalized values for this group in label order.
            vals: List[float] = []
            errs_std: List[float] = []
            errs_se: List[float] = []
            errs_ci95: List[float] = []
            errs_iqr: List[float] = []
            for md in metric_defs:
                v = (md.get("group_means_norm01") or [])[idx].get("mean_norm01")
                e = (md.get("group_stds_norm01") or [])[idx].get("std_norm01")
                e_se = (md.get("group_se_norm01") or [])[idx].get("se_norm01")
                e_ci = (md.get("group_ci95_norm01") or [])[idx].get("ci95_norm01")
                e_iqr = (md.get("group_iqr_half_norm01") or [])[idx].get(
                    "iqr_half_norm01"
                )
                vals.append(float(v) if v is not None else float("nan"))
                errs_std.append(float(e) if e is not None else float("nan"))
                errs_se.append(float(e_se) if e_se is not None else float("nan"))
                errs_ci95.append(float(e_ci) if e_ci is not None else float("nan"))
                errs_iqr.append(float(e_iqr) if e_iqr is not None else float("nan"))
            group_vectors.append(
                {
                    "label": g["label"],
                    "values": vals,
                    "errors": errs_std,
                    "color": g["color"],
                    "linewidth": 2.2 if g["kind"] == "baseline" else 1.8,
                    "fill_alpha": 0.08 if g["kind"] == "baseline" else 0.06,
                }
            )
            group_vectors_se.append(
                {
                    "label": g["label"],
                    "values": vals,
                    "errors": errs_se,
                    "color": g["color"],
                    "linewidth": 2.2 if g["kind"] == "baseline" else 1.8,
                    "fill_alpha": 0.08 if g["kind"] == "baseline" else 0.06,
                }
            )
            group_vectors_ci95.append(
                {
                    "label": g["label"],
                    "values": vals,
                    "errors": errs_ci95,
                    "color": g["color"],
                    "linewidth": 2.2 if g["kind"] == "baseline" else 1.8,
                    "fill_alpha": 0.08 if g["kind"] == "baseline" else 0.06,
                }
            )
            group_vectors_iqr.append(
                {
                    "label": g["label"],
                    "values": vals,
                    "errors": errs_iqr,
                    "color": g["color"],
                    "linewidth": 2.2 if g["kind"] == "baseline" else 1.8,
                    "fill_alpha": 0.08 if g["kind"] == "baseline" else 0.06,
                }
            )

        # Fill in missing colors deterministically using tab10 (skip tab10[0]).
        cmap = plt.get_cmap("tab10")
        color_idx = 1
        for g in group_vectors:
            if g.get("color"):
                continue
            g["color"] = matplotlib.colors.to_hex(cmap(color_idx % 10))
            color_idx += 1
        for i, g in enumerate(group_vectors):
            if i < len(group_vectors_se):
                group_vectors_se[i]["color"] = g.get("color")
            if i < len(group_vectors_ci95):
                group_vectors_ci95[i]["color"] = g.get("color")
            if i < len(group_vectors_iqr):
                group_vectors_iqr[i]["color"] = g.get("color")

        # For bar charts only: keep some metrics un-normalized so they're
        # directly interpretable on their natural scale.
        violations_idx = next(
            (i for i, md in enumerate(bar_metric_defs) if md.get("key") == "violations"),
            None,
        )
        raw_vio_by_label: Dict[str, Dict[str, float]] = {}
        for g in groups:
            label = str(g["label"])
            seed_vio = _seed_means(g["rows"], "violations")
            if seed_vio:
                raw_vio_by_label[label] = {
                    "mean": float(mean(seed_vio)),
                    "std": float(_sample_std(seed_vio) or 0.0),
                    "se": float(_standard_error(seed_vio) or 0.0),
                    "ci95": float(_ci95_half_width(seed_vio) or 0.0),
                    "iqr": float(_iqr_half_width(seed_vio) or 0.0),
                }

        def _with_raw_overrides(
            series: List[Dict[str, Any]], *, error_key: str
        ) -> List[Dict[str, Any]]:
            if violations_idx is None:
                return series
            out: List[Dict[str, Any]] = []
            for s in series:
                label = str(s.get("label") or "")
                vio = raw_vio_by_label.get(label) if violations_idx is not None else None
                vals = list(s.get("values") or [])
                errs = list(s.get("errors") or [])
                if (
                    vio is not None
                    and violations_idx < len(vals)
                    and violations_idx < len(errs)
                ):
                    vals[violations_idx] = float(vio["mean"])
                    errs[violations_idx] = float(vio[error_key])
                out.append({**s, "values": vals, "errors": errs})
            return out

        bars_series = _with_raw_overrides(
            _slice_series_metrics(group_vectors, bar_indices), error_key="std"
        )
        bars_series_se = _with_raw_overrides(
            _slice_series_metrics(group_vectors_se, bar_indices), error_key="se"
        )
        bars_series_ci95 = _with_raw_overrides(
            _slice_series_metrics(group_vectors_ci95, bar_indices), error_key="ci95"
        )
        bars_series_iqr = _with_raw_overrides(
            _slice_series_metrics(group_vectors_iqr, bar_indices), error_key="iqr"
        )

        radar_base = f"collusion_radar__c{colluder_count}__pvALL"
        hist_base = f"collusion_hist__c{colluder_count}__pvALL"
        write_json(sweep_out / f"{radar_base}.json", raw_summary)

        title = (
            prefix
            + f"Collusion: baseline vs secret-channel prompt variants (c={colluder_count})"
        )
        _plot_radar_multi(
            series=group_vectors,
            labels=labels_all,
            title=title,
            out_path=sweep_out / f"{radar_base}.png",
        )
        _plot_radar_multi(
            series=group_vectors_se,
            labels=labels_all,
            title=f"{title}\n(±1 SE)",
            out_path=sweep_out / f"{radar_base}__se.png",
            use_error_bands=True,
        )
        _plot_radar_multi(
            series=group_vectors_ci95,
            labels=labels_all,
            title=f"{title}\n(95% CI)",
            out_path=sweep_out / f"{radar_base}__ci95.png",
            use_error_bands=True,
        )
        # Match the "by_topology" figure layout when this sweep only contains one topology:
        # - unrotated x-axis labels
        # - legend centered at bottom
        # - simple per-panel header (topology title)
        if topo:
            topo_title = _pretty_topology_label(
                str(topo), title_by_topology=title_by_topology or None
            )
            _plot_grouped_bars_single_topology(
                series=bars_series,
                labels=labels_bars,
                title=None,
                out_path=sweep_out / f"{hist_base}__bars.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars_single_topology(
                series=bars_series_se,
                labels=labels_bars,
                title=None,
                out_path=sweep_out / f"{hist_base}__bars__se.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars_single_topology(
                series=bars_series_ci95,
                labels=labels_bars,
                title=None,
                out_path=sweep_out / f"{hist_base}__bars__ci95.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars_single_topology(
                series=bars_series_iqr,
                labels=labels_bars,
                title=None,
                out_path=sweep_out / f"{hist_base}__bars__iqr.png",
                y_label="Normalized Mean",
            )
        else:
            _plot_grouped_bars(
                series=bars_series,
                labels=labels_bars,
                title=title,
                out_path=sweep_out / f"{hist_base}__bars.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars(
                series=bars_series_se,
                labels=labels_bars,
                title=title,
                out_path=sweep_out / f"{hist_base}__bars__se.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars(
                series=bars_series_ci95,
                labels=labels_bars,
                title=title,
                out_path=sweep_out / f"{hist_base}__bars__ci95.png",
                y_label="Normalized Mean",
            )
            _plot_grouped_bars(
                series=bars_series_iqr,
                labels=labels_bars,
                title=title,
                out_path=sweep_out / f"{hist_base}__bars__iqr.png",
                y_label="Normalized Mean",
            )
        return True

    treatment = _group_filter(
        rows,
        colluder_count=colluder_count,
        secret=True,
        prompt_variant=treatment_variant,
        require_complete=require_complete,
    )
    if not treatment:
        msg = (
            f"No treatment runs found for colluder_count={colluder_count}, secret_channel_enabled=true, "
            f"prompt_variant={treatment_variant!r}."
        )
        if strict:
            raise SystemExit(msg)
        return False

    baseline_norm: List[float] = []
    treatment_norm: List[float] = []
    baseline_err: List[float] = []
    treatment_err: List[float] = []
    baseline_se: List[float] = []
    treatment_se: List[float] = []
    baseline_ci95: List[float] = []
    treatment_ci95: List[float] = []
    baseline_iqr: List[float] = []
    treatment_iqr: List[float] = []
    raw_summary = {
        "sweep_dir": str(sweep_dir),
        "topology": topo,
        "colluder_count": colluder_count,
        "baseline": {
            "secret_channel_enabled": False,
            "prompt_variant": baseline_variant,
            "n_runs": len(baseline),
            "seeds": _run_seeds(baseline),
        },
        "treatment": {
            "secret_channel_enabled": True,
            "prompt_variant": treatment_variant,
            "n_runs": len(treatment),
            "seeds": _run_seeds(treatment),
        },
        "metrics": [],
    }

    metric_defs = []
    for m in metrics:
        b_seed_raw = _seed_means(baseline, m.key)
        t_seed_raw = _seed_means(treatment, m.key)
        if not b_seed_raw or not t_seed_raw:
            continue

        b_seed_transformed = [m.apply(v) for v in b_seed_raw]
        t_seed_transformed = [m.apply(v) for v in t_seed_raw]
        lo, hi = _minmax_range(b_seed_transformed + t_seed_transformed)
        if m.center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            scale_lo = -max_abs
            scale_hi = max_abs
        else:
            max_abs = None
            scale_lo = float(lo)
            scale_hi = float(hi)

        def _norm_one(value: float) -> float:
            if scale_hi == scale_lo:
                return 0.5
            if m.center_zero:
                if not max_abs:
                    return 0.5
                return _clamp01(0.5 + float(value) / (2.0 * float(max_abs)))
            return _clamp01((float(value) - float(scale_lo)) / (float(scale_hi) - float(scale_lo)))

        b_seed_norm = [_norm_one(v) for v in b_seed_transformed]
        t_seed_norm = [_norm_one(v) for v in t_seed_transformed]

        b_mean_norm = float(mean(b_seed_norm))
        t_mean_norm = float(mean(t_seed_norm))
        b_std_norm = float(_sample_std(b_seed_norm) or 0.0)
        t_std_norm = float(_sample_std(t_seed_norm) or 0.0)
        b_se_norm = float(_standard_error(b_seed_norm) or 0.0)
        t_se_norm = float(_standard_error(t_seed_norm) or 0.0)
        b_ci_norm = float(_ci95_half_width(b_seed_norm) or 0.0)
        t_ci_norm = float(_ci95_half_width(t_seed_norm) or 0.0)
        b_iqr_norm = float(_iqr_half_width(b_seed_norm) or 0.0)
        t_iqr_norm = float(_iqr_half_width(t_seed_norm) or 0.0)

        baseline_norm.append(b_mean_norm)
        treatment_norm.append(t_mean_norm)
        baseline_err.append(b_std_norm)
        treatment_err.append(t_std_norm)
        baseline_se.append(b_se_norm)
        treatment_se.append(t_se_norm)
        baseline_ci95.append(b_ci_norm)
        treatment_ci95.append(t_ci_norm)
        baseline_iqr.append(b_iqr_norm)
        treatment_iqr.append(t_iqr_norm)

        metric_defs.append(
            {
                "key": m.key,
                "label": m.label,
                "higher_is_better": m.higher_is_better,
                "scale_lo": scale_lo,
                "scale_hi": scale_hi,
                "baseline_mean_raw": float(mean(b_seed_raw)),
                "baseline_std_raw": float(_sample_std(b_seed_raw) or 0.0),
                "treatment_mean_raw": float(mean(t_seed_raw)),
                "treatment_std_raw": float(_sample_std(t_seed_raw) or 0.0),
                "baseline_mean_transformed": float(mean(b_seed_transformed)),
                "baseline_std_transformed": float(
                    _sample_std(b_seed_transformed) or 0.0
                ),
                "treatment_mean_transformed": float(mean(t_seed_transformed)),
                "treatment_std_transformed": float(
                    _sample_std(t_seed_transformed) or 0.0
                ),
                "baseline_mean_norm01": b_mean_norm,
                "baseline_std_norm01": b_std_norm,
                "baseline_se_norm01": b_se_norm,
                "baseline_ci95_norm01": b_ci_norm,
                "baseline_iqr_half_norm01": b_iqr_norm,
                "treatment_mean_norm01": t_mean_norm,
                "treatment_std_norm01": t_std_norm,
                "treatment_se_norm01": t_se_norm,
                "treatment_ci95_norm01": t_ci_norm,
                "treatment_iqr_half_norm01": t_iqr_norm,
            }
        )

    raw_summary["metrics"] = metric_defs

    labels_all: List[str] = []
    for md in metric_defs:
        base = str(md.get("label") or "")
        key = str(md.get("key") or "")
        if key == "coalition_regret_advantage_mean":
            labels_all.append(base)
            continue
        if key == "colluder_posts_secret_rate":
            arrow = "–"
        else:
            arrow = "↑" if bool(md.get("higher_is_better", True)) else "↓"
        labels_all.append(f"{base} ({arrow})")
    if not labels_all:
        msg = "No comparable metrics found to plot (all missing/NaN)."
        if strict:
            raise SystemExit(msg)
        return False

    bar_indices = [
        i
        for i, md in enumerate(metric_defs)
        if str(md.get("key") or "") != "colluder_posts_secret_rate"
    ]
    bar_metric_defs = [metric_defs[i] for i in bar_indices]
    labels_bars: List[str] = []
    for md in bar_metric_defs:
        base = str(md.get("label") or "")
        arrow = "↑" if bool(md.get("higher_is_better", True)) else "↓"
        labels_bars.append(f"{base} ({arrow})")

    radar_base = f"collusion_radar__c{colluder_count}__pv{treatment_variant}"
    hist_base = f"collusion_hist__c{colluder_count}__pv{treatment_variant}"
    write_json(sweep_out / f"{radar_base}.json", raw_summary)

    title = (
        prefix
        + f"Collusion: secret-channel vs baseline (c={colluder_count}, pv={treatment_variant})\n"
        + f"baseline n={len(baseline)} seeds={len(raw_summary['baseline']['seeds'])} | "
        + f"treatment n={len(treatment)} seeds={len(raw_summary['treatment']['seeds'])}"
    )
    baseline_label = "Baseline (no SC)"
    treatment_label = f"{_pretty_variant(treatment_variant)} (SC)"
    _plot_radar(
        baseline_vals=baseline_norm,
        treatment_vals=treatment_norm,
        labels=labels_all,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
        title=title,
        out_path=sweep_out / f"{radar_base}.png",
    )
    _plot_radar(
        baseline_vals=baseline_norm,
        treatment_vals=treatment_norm,
        baseline_err=baseline_se,
        treatment_err=treatment_se,
        labels=labels_all,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
        title=f"{title}\n(±1 SE)",
        out_path=sweep_out / f"{radar_base}__se.png",
    )
    _plot_radar(
        baseline_vals=baseline_norm,
        treatment_vals=treatment_norm,
        baseline_err=baseline_ci95,
        treatment_err=treatment_ci95,
        labels=labels_all,
        baseline_label=baseline_label,
        treatment_label=treatment_label,
        title=f"{title}\n(95% CI)",
        out_path=sweep_out / f"{radar_base}__ci95.png",
    )

    # For bar charts only: keep some metrics un-normalized so they're directly
    # interpretable on their natural scale.
    violations_idx = next(
        (i for i, md in enumerate(bar_metric_defs) if md.get("key") == "violations"),
        None,
    )
    baseline_bars = _slice_by_indices(list(baseline_norm), bar_indices)
    treatment_bars = _slice_by_indices(list(treatment_norm), bar_indices)
    baseline_bars_err = _slice_by_indices(list(baseline_err), bar_indices)
    treatment_bars_err = _slice_by_indices(list(treatment_err), bar_indices)
    baseline_bars_se = _slice_by_indices(list(baseline_se), bar_indices)
    treatment_bars_se = _slice_by_indices(list(treatment_se), bar_indices)
    baseline_bars_ci95 = _slice_by_indices(list(baseline_ci95), bar_indices)
    treatment_bars_ci95 = _slice_by_indices(list(treatment_ci95), bar_indices)
    baseline_bars_iqr = _slice_by_indices(list(baseline_iqr), bar_indices)
    treatment_bars_iqr = _slice_by_indices(list(treatment_iqr), bar_indices)
    if violations_idx is not None:
        b_vio_seed_raw = _seed_means(baseline, "violations")
        t_vio_seed_raw = _seed_means(treatment, "violations")
        if b_vio_seed_raw and t_vio_seed_raw:
            baseline_bars[violations_idx] = float(mean(b_vio_seed_raw))
            treatment_bars[violations_idx] = float(mean(t_vio_seed_raw))
            baseline_bars_err[violations_idx] = float(_sample_std(b_vio_seed_raw) or 0.0)
            treatment_bars_err[violations_idx] = float(_sample_std(t_vio_seed_raw) or 0.0)
            baseline_bars_se[violations_idx] = float(_standard_error(b_vio_seed_raw) or 0.0)
            treatment_bars_se[violations_idx] = float(_standard_error(t_vio_seed_raw) or 0.0)
            baseline_bars_ci95[violations_idx] = float(
                _ci95_half_width(b_vio_seed_raw) or 0.0
            )
            treatment_bars_ci95[violations_idx] = float(
                _ci95_half_width(t_vio_seed_raw) or 0.0
            )
            baseline_bars_iqr[violations_idx] = float(
                _iqr_half_width(b_vio_seed_raw) or 0.0
            )
            treatment_bars_iqr[violations_idx] = float(
                _iqr_half_width(t_vio_seed_raw) or 0.0
            )
    _plot_grouped_bars(
        series=[
            {
                "label": baseline_label,
                "values": baseline_bars,
                "errors": baseline_bars_err,
                "color": "#2563eb",
            },
            {
                "label": treatment_label,
                "values": treatment_bars,
                "errors": treatment_bars_err,
                "color": "#dc2626",
            },
        ],
        labels=labels_bars,
        title=title,
        out_path=sweep_out / f"{hist_base}__bars.png",
        y_label="Normalized Mean",
    )
    _plot_grouped_bars(
        series=[
            {
                "label": baseline_label,
                "values": baseline_bars,
                "errors": baseline_bars_se,
                "color": "#2563eb",
            },
            {
                "label": treatment_label,
                "values": treatment_bars,
                "errors": treatment_bars_se,
                "color": "#dc2626",
            },
        ],
        labels=labels_bars,
        title=title,
        out_path=sweep_out / f"{hist_base}__bars__se.png",
        y_label="Normalized Mean",
    )
    _plot_grouped_bars(
        series=[
            {
                "label": baseline_label,
                "values": baseline_bars,
                "errors": baseline_bars_ci95,
                "color": "#2563eb",
            },
            {
                "label": treatment_label,
                "values": treatment_bars,
                "errors": treatment_bars_ci95,
                "color": "#dc2626",
            },
        ],
        labels=labels_bars,
        title=title,
        out_path=sweep_out / f"{hist_base}__bars__ci95.png",
        y_label="Normalized Mean",
    )
    _plot_grouped_bars(
        series=[
            {
                "label": baseline_label,
                "values": baseline_bars,
                "errors": baseline_bars_iqr,
                "color": "#2563eb",
            },
            {
                "label": treatment_label,
                "values": treatment_bars,
                "errors": treatment_bars_iqr,
                "color": "#dc2626",
            },
        ],
        labels=labels_bars,
        title=title,
        out_path=sweep_out / f"{hist_base}__bars__iqr.png",
        y_label="Normalized Mean",
    )
    return True


def _compare_topologies_bars(
    *,
    sweep_dir: Path,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    colluder_count_requested: Optional[int],
    baseline_variant_requested: str,
    include_incomplete: bool,
    reward_metric: str,
    titles_config_path: Optional[Path] = None,
) -> bool:
    """
    Builds a single multi-panel grouped-bar figure:
      one panel per topology, globally normalized per-metric across all topologies.

    Output: out_dir/sweep/collusion_hist__cX__pvALL__bars__by_topology.png
    """
    topologies = sorted(
        {str(r.get("topology")) for r in rows if r.get("topology") is not None}
    )
    if len(topologies) <= 1:
        return False

    colluder_count = _select_colluder_count(rows, colluder_count_requested)
    baseline_variant = str(baseline_variant_requested or "control").strip()
    require_complete = not bool(include_incomplete)

    metrics = [
        m
        for m in _default_metric_specs(reward_metric=str(reward_metric))
        if m.key != "colluder_posts_secret_rate"
    ]

    available_variants = {
        str(r.get("prompt_variant"))
        for r in rows
        if int(r.get("colluder_count") or 0) == int(colluder_count)
        and r.get("secret_channel_enabled") is True
        and r.get("prompt_variant") is not None
        and (
            not require_complete
            or str(r.get("status") or "").strip().lower() == "complete"
        )
    }
    variants = _sorted_variants(available_variants)
    if not variants:
        return False

    groups: List[Dict[str, Any]] = [
        {
            "kind": "baseline",
            "secret_channel_enabled": False,
            "prompt_variant": baseline_variant,
            "label": "Baseline (no SC)",
            "color": None,
        }
    ]
    for pv in variants:
        groups.append(
            {
                "kind": "treatment",
                "secret_channel_enabled": True,
                "prompt_variant": pv,
                "label": f"{_pretty_variant(pv)} (SC)",
                "color": None,
            }
        )

    _apply_group_palette(groups, _PVALL_GROUP_PALETTE)

    # Resolve missing group colors deterministically (tab10), skipping tab10[0].
    cmap = plt.get_cmap("tab10")
    color_idx = 1
    for g in groups:
        if g.get("color"):
            continue
        g["color"] = matplotlib.colors.to_hex(cmap(color_idx % 10))
        color_idx += 1

    # Pre-filter rows per topology and per group.
    rows_by_topo: Dict[str, List[Dict[str, Any]]] = {
        t: [r for r in rows if str(r.get("topology")) == t] for t in topologies
    }

    def _filter_group(
        topo_rows: List[Dict[str, Any]], group: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return _group_filter(
            topo_rows,
            colluder_count=colluder_count,
            secret=bool(group["secret_channel_enabled"]),
            prompt_variant=str(group["prompt_variant"]),
            require_complete=require_complete,
        )

    group_rows_by_topo: Dict[str, List[List[Dict[str, Any]]]] = {}
    for t in topologies:
        topo_rows = rows_by_topo[t]
        group_rows_by_topo[t] = [_filter_group(topo_rows, g) for g in groups]

    # Global metric scaling: pooled transformed seed-means across all topologies/groups.
    metric_defs: List[Dict[str, Any]] = []
    for m in metrics:
        pooled_transformed: List[float] = []
        # Require that each topology has at least baseline + 1 treatment with data for this metric.
        for t in topologies:
            for grp_rows in group_rows_by_topo[t]:
                seed_raw = _seed_means(grp_rows, m.key)
                if not seed_raw:
                    continue
                pooled_transformed.extend([m.apply(v) for v in seed_raw])

        if not pooled_transformed:
            continue

        lo, hi = _minmax_range(pooled_transformed)
        if m.center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            scale_lo = -max_abs
            scale_hi = max_abs
        else:
            scale_lo = float(lo)
            scale_hi = float(hi)
        metric_defs.append({"spec": m, "scale_lo": scale_lo, "scale_hi": scale_hi})

    if not metric_defs:
        return False

    labels = [str(md["spec"].label) for md in metric_defs]

    def _norm_one(value: float, lo: float, hi: float, center_zero: bool) -> float:
        if hi == lo:
            return 0.5
        if center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            if max_abs == 0.0:
                return 0.5
            return _clamp01(0.5 + float(value) / (2.0 * float(max_abs)))
        return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))

    series_by_topology: Dict[str, List[Dict[str, Any]]] = {}
    for t in topologies:
        topo_series: List[Dict[str, Any]] = []
        topo_group_rows = group_rows_by_topo[t]

        for group, grp_rows in zip(groups, topo_group_rows):
            values: List[float] = []
            errors_std: List[float] = []
            errors_se: List[float] = []
            errors_ci95: List[float] = []
            errors_iqr: List[float] = []
            for md in metric_defs:
                m = md["spec"]
                lo = float(md["scale_lo"])
                hi = float(md["scale_hi"])

                seed_raw = _seed_means(grp_rows, m.key)
                if not seed_raw:
                    values.append(float("nan"))
                    errors_std.append(float("nan"))
                    errors_se.append(float("nan"))
                    errors_ci95.append(float("nan"))
                    errors_iqr.append(float("nan"))
                    continue

                seed_transformed = [m.apply(v) for v in seed_raw]
                seed_norm = [_norm_one(v, lo, hi, bool(m.center_zero)) for v in seed_transformed]
                values.append(float(mean(seed_norm)))
                errors_std.append(float(_sample_std(seed_norm) or 0.0))
                errors_se.append(float(_standard_error(seed_norm) or 0.0))
                errors_ci95.append(float(_ci95_half_width(seed_norm) or 0.0))
                errors_iqr.append(float(_iqr_half_width(seed_norm) or 0.0))

            topo_series.append(
                {
                    "label": str(group["label"]),
                    "values": values,
                    "errors": errors_std,
                    "errors_se": errors_se,
                    "errors_ci95": errors_ci95,
                    "errors_iqr": errors_iqr,
                    "color": str(group["color"]),
                }
            )
        series_by_topology[t] = topo_series

    ensure_dir(out_dir)
    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)

    # Optional topology parameter display (e.g., Erdős–Rényi edge prob p).
    title_by_topology = _topology_title_overrides_from_config(
        sweep_dir=sweep_dir, titles_config_path=titles_config_path
    )

    out_path = (
        sweep_out / f"collusion_hist__c{colluder_count}__pvALL__bars__by_topology.png"
    )
    title = (
        f"Collusion: baseline vs secret-channel prompt variants by topology (c={colluder_count})"
    )
    _plot_grouped_bars_by_topology(
        topologies=topologies,
        series_by_topology=series_by_topology,
        labels=labels,
        title=title,
        out_path=out_path,
        title_by_topology=title_by_topology or None,
        y_label="Normalized Mean",
    )
    series_by_topology_se: Dict[str, List[Dict[str, Any]]] = {
        topo: [
            {**s, "errors": (s.get("errors_se") or s.get("errors"))}
            for s in (series_by_topology.get(topo) or [])
            if isinstance(s, dict)
        ]
        for topo in topologies
    }
    out_path_se = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_topology__se.png"
    )
    _plot_grouped_bars_by_topology(
        topologies=topologies,
        series_by_topology=series_by_topology_se,
        labels=labels,
        title=title,
        out_path=out_path_se,
        title_by_topology=title_by_topology or None,
        y_label="Normalized Mean",
    )
    series_by_topology_ci95: Dict[str, List[Dict[str, Any]]] = {
        topo: [
            {**s, "errors": (s.get("errors_ci95") or s.get("errors"))}
            for s in (series_by_topology.get(topo) or [])
            if isinstance(s, dict)
        ]
        for topo in topologies
    }
    out_path_ci95 = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_topology__ci95.png"
    )
    _plot_grouped_bars_by_topology(
        topologies=topologies,
        series_by_topology=series_by_topology_ci95,
        labels=labels,
        title=title,
        out_path=out_path_ci95,
        title_by_topology=title_by_topology or None,
        y_label="Normalized Mean",
    )
    series_by_topology_iqr: Dict[str, List[Dict[str, Any]]] = {
        topo: [
            {**s, "errors": (s.get("errors_iqr") or s.get("errors"))}
            for s in (series_by_topology.get(topo) or [])
            if isinstance(s, dict)
        ]
        for topo in topologies
    }
    out_path_iqr = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_topology__iqr.png"
    )
    _plot_grouped_bars_by_topology(
        topologies=topologies,
        series_by_topology=series_by_topology_iqr,
        labels=labels,
        title=title,
        out_path=out_path_iqr,
        title_by_topology=title_by_topology or None,
        y_label="Normalized Mean",
    )

    # Also write one PNG per topology (split panels) under out_dir/hist/topologies/.
    topo_out = out_dir / "hist" / "topologies"
    ensure_dir(topo_out)
    for topo in topologies:
        topo_series = series_by_topology.get(topo) or []
        if not topo_series:
            continue
        topo_title = _pretty_topology_label(
            topo, title_by_topology=title_by_topology or None
        )
        _plot_grouped_bars_single_topology(
            series=topo_series,
            labels=labels,
            title=str(topo_title),
            out_path=topo_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(topo)}.png",
        )
        topo_series_se = [
            {**s, "errors": (s.get("errors_se") or s.get("errors"))}
            for s in topo_series
            if isinstance(s, dict)
        ]
        _plot_grouped_bars_single_topology(
            series=topo_series_se,
            labels=labels,
            title=str(topo_title),
            out_path=topo_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(topo)}__se.png",
            y_label="Normalized Mean",
        )
        topo_series_ci95 = [
            {**s, "errors": (s.get("errors_ci95") or s.get("errors"))}
            for s in topo_series
            if isinstance(s, dict)
        ]
        _plot_grouped_bars_single_topology(
            series=topo_series_ci95,
            labels=labels,
            title=str(topo_title),
            out_path=topo_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(topo)}__ci95.png",
            y_label="Normalized Mean",
        )
        topo_series_iqr = [
            {**s, "errors": (s.get("errors_iqr") or s.get("errors"))}
            for s in topo_series
            if isinstance(s, dict)
        ]
        _plot_grouped_bars_single_topology(
            series=topo_series_iqr,
            labels=labels,
            title=str(topo_title),
            out_path=topo_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(topo)}__iqr.png",
            y_label="Normalized Mean",
        )
    return True


def _compare_environments_bars(
    *,
    sweep_dir: Path,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    colluder_count_requested: Optional[int],
    baseline_variant_requested: str,
    include_incomplete: bool,
    reward_metric: str,
) -> bool:
    """
    Builds a single multi-panel grouped-bar figure:
      one panel per environment_label, globally normalized per-metric across all environments.

    Output: out_dir/sweep/collusion_hist__cX__pvALL__bars__by_environment.png
    """
    env_labels = sorted(
        {
            str(r.get("environment_label"))
            for r in rows
            if r.get("environment_label") is not None
            and str(r.get("environment_label") or "").strip()
        }
    )
    if len(env_labels) <= 1:
        return False

    colluder_count = _select_colluder_count(rows, colluder_count_requested)
    baseline_variant = str(baseline_variant_requested or "control").strip()
    require_complete = not bool(include_incomplete)

    metrics = _default_metric_specs(reward_metric=str(reward_metric))

    available_variants = {
        str(r.get("prompt_variant"))
        for r in rows
        if int(r.get("colluder_count") or 0) == int(colluder_count)
        and r.get("secret_channel_enabled") is True
        and r.get("prompt_variant") is not None
        and (
            not require_complete
            or str(r.get("status") or "").strip().lower() == "complete"
        )
    }
    variants = _sorted_variants(available_variants)
    if not variants:
        return False

    groups: List[Dict[str, Any]] = [
        {
            "kind": "baseline",
            "secret_channel_enabled": False,
            "prompt_variant": baseline_variant,
            "label": "Baseline (no SC)",
            "color": None,
        }
    ]
    for pv in variants:
        groups.append(
            {
                "kind": "treatment",
                "secret_channel_enabled": True,
                "prompt_variant": pv,
                "label": f"{_pretty_variant(pv)} (SC)",
                "color": None,
            }
        )

    _apply_group_palette(groups, _PVALL_GROUP_PALETTE)

    # Resolve missing group colors deterministically (tab10), skipping tab10[0].
    cmap = plt.get_cmap("tab10")
    color_idx = 1
    for g in groups:
        if g.get("color"):
            continue
        g["color"] = matplotlib.colors.to_hex(cmap(color_idx % 10))
        color_idx += 1

    # Pre-filter rows per environment and per group.
    rows_by_env: Dict[str, List[Dict[str, Any]]] = {
        e: [r for r in rows if str(r.get("environment_label") or "") == e]
        for e in env_labels
    }

    def _filter_group(
        env_rows: List[Dict[str, Any]], group: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return _group_filter(
            env_rows,
            colluder_count=colluder_count,
            secret=bool(group["secret_channel_enabled"]),
            prompt_variant=str(group["prompt_variant"]),
            require_complete=require_complete,
        )

    group_rows_by_env: Dict[str, List[List[Dict[str, Any]]]] = {}
    for e in env_labels:
        env_rows = rows_by_env[e]
        group_rows_by_env[e] = [_filter_group(env_rows, g) for g in groups]

    # Global metric scaling: pooled transformed seed-means across all environments/groups.
    metric_defs: List[Dict[str, Any]] = []
    for m in metrics:
        pooled_transformed: List[float] = []
        for e in env_labels:
            for grp_rows in group_rows_by_env[e]:
                seed_raw = _seed_means(grp_rows, m.key)
                if not seed_raw:
                    continue
                pooled_transformed.extend([m.apply(v) for v in seed_raw])

        if not pooled_transformed:
            continue

        lo, hi = _minmax_range(pooled_transformed)
        if m.center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            scale_lo = -max_abs
            scale_hi = max_abs
        else:
            scale_lo = float(lo)
            scale_hi = float(hi)
        metric_defs.append({"spec": m, "scale_lo": scale_lo, "scale_hi": scale_hi})

    if not metric_defs:
        return False

    labels = [str(md["spec"].label) for md in metric_defs]

    def _norm_one(value: float, lo: float, hi: float, center_zero: bool) -> float:
        if hi == lo:
            return 0.5
        if center_zero:
            max_abs = max(abs(float(lo)), abs(float(hi)))
            if max_abs == 0.0:
                return 0.5
            return _clamp01(0.5 + float(value) / (2.0 * float(max_abs)))
        return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))

    series_by_env: Dict[str, List[Dict[str, Any]]] = {}
    for e in env_labels:
        env_series: List[Dict[str, Any]] = []
        env_group_rows = group_rows_by_env[e]

        for group, grp_rows in zip(groups, env_group_rows):
            values: List[float] = []
            errors_std: List[float] = []
            errors_se: List[float] = []
            errors_ci95: List[float] = []
            for md in metric_defs:
                m = md["spec"]
                lo = float(md["scale_lo"])
                hi = float(md["scale_hi"])

                seed_raw = _seed_means(grp_rows, m.key)
                if not seed_raw:
                    values.append(float("nan"))
                    errors_std.append(float("nan"))
                    errors_se.append(float("nan"))
                    errors_ci95.append(float("nan"))
                    continue

                seed_transformed = [m.apply(v) for v in seed_raw]
                seed_norm = [_norm_one(v, lo, hi, bool(m.center_zero)) for v in seed_transformed]
                values.append(float(mean(seed_norm)))
                errors_std.append(float(_sample_std(seed_norm) or 0.0))
                errors_se.append(float(_standard_error(seed_norm) or 0.0))
                errors_ci95.append(float(_ci95_half_width(seed_norm) or 0.0))

            env_series.append(
                {
                    "label": str(group["label"]),
                    "values": values,
                    "errors": errors_std,
                    "errors_se": errors_se,
                    "errors_ci95": errors_ci95,
                    "color": str(group["color"]),
                }
            )
        series_by_env[e] = env_series

    ensure_dir(out_dir)
    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)

    out_path = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_environment.png"
    )
    title = (
        f"Collusion: baseline vs secret-channel prompt variants by environment (c={colluder_count})"
    )
    _plot_grouped_bars_by_topology(
        topologies=env_labels,
        series_by_topology=series_by_env,
        labels=labels,
        title=title,
        out_path=out_path,
        title_by_topology=None,
        y_label="Normalized Mean",
    )

    series_by_env_se: Dict[str, List[Dict[str, Any]]] = {
        e: [
            {**s, "errors": (s.get("errors_se") or s.get("errors"))}
            for s in (series_by_env.get(e) or [])
            if isinstance(s, dict)
        ]
        for e in env_labels
    }
    out_path_se = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_environment__se.png"
    )
    _plot_grouped_bars_by_topology(
        topologies=env_labels,
        series_by_topology=series_by_env_se,
        labels=labels,
        title=title,
        out_path=out_path_se,
        title_by_topology=None,
        y_label="Normalized Mean",
    )

    series_by_env_ci95: Dict[str, List[Dict[str, Any]]] = {
        e: [
            {**s, "errors": (s.get("errors_ci95") or s.get("errors"))}
            for s in (series_by_env.get(e) or [])
            if isinstance(s, dict)
        ]
        for e in env_labels
    }
    out_path_ci95 = (
        sweep_out
        / f"collusion_hist__c{colluder_count}__pvALL__bars__by_environment__ci95.png"
    )
    _plot_grouped_bars_by_topology(
        topologies=env_labels,
        series_by_topology=series_by_env_ci95,
        labels=labels,
        title=title,
        out_path=out_path_ci95,
        title_by_topology=None,
        y_label="Normalized Mean",
    )

    env_out = out_dir / "hist" / "environments"
    ensure_dir(env_out)
    for env_label in env_labels:
        env_series = series_by_env.get(env_label) or []
        if not env_series:
            continue
        _plot_grouped_bars_single_topology(
            series=env_series,
            labels=labels,
            title=str(_pretty_topology_label(str(env_label))),
            out_path=env_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(env_label)}.png",
            y_label="Normalized Mean",
        )
        env_series_se = [
            {**s, "errors": (s.get("errors_se") or s.get("errors"))}
            for s in env_series
            if isinstance(s, dict)
        ]
        _plot_grouped_bars_single_topology(
            series=env_series_se,
            labels=labels,
            title=str(_pretty_topology_label(str(env_label))),
            out_path=env_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(env_label)}__se.png",
            y_label="Normalized Mean",
        )
        env_series_ci95 = [
            {**s, "errors": (s.get("errors_ci95") or s.get("errors"))}
            for s in env_series
            if isinstance(s, dict)
        ]
        _plot_grouped_bars_single_topology(
            series=env_series_ci95,
            labels=labels,
            title=str(_pretty_topology_label(str(env_label))),
            out_path=env_out
            / f"collusion_hist__c{colluder_count}__pvALL__bars__{sanitize_filename(env_label)}__ci95.png",
            y_label="Normalized Mean",
        )
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a radar chart comparing collusion vs baseline."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/collusion/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/collusion/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    parser.add_argument(
        "--colluder-count",
        type=int,
        default=None,
        help="Which colluder_count to plot (default: max > 0).",
    )
    parser.add_argument(
        "--treatment-prompt-variant",
        type=str,
        default=None,
        help="Prompt variant for the secret-channel treatment (default: inferred; prefers non-'control').",
    )
    parser.add_argument(
        "--plot-all-prompt-variants",
        action="store_true",
        help="Plot baseline + every secret-channel prompt_variant on one radar chart.",
    )
    parser.add_argument(
        "--baseline-prompt-variant",
        type=str,
        default="control",
        help="Prompt variant for the baseline (secret_channel_enabled=false) group (default: control).",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where metrics.status != 'complete' (not recommended).",
    )
    parser.add_argument(
        "--reward-metric",
        type=str,
        default="regret_ratio",
        choices=_REWARD_METRIC_CHOICES,
        help="Which overall-performance metric to use in place of joint reward. "
        "Use 'normalized_regret'/'regret_ratio' or 'regret' (aka optimality_gap) after running "
        "experiments/collusion/compute_jira_optimal.py --write-json.",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present (final_summary_repaired.json, metrics_repaired.json).",
    )
    parser.add_argument(
        "--by-topology",
        action="store_true",
        help="Also generate the same radar chart(s) separately for each topology present in the sweep (writes under out_dir/by_topology/<topology>/sweep/).",
    )
    parser.add_argument(
        "--by-environment",
        action="store_true",
        help="Also generate the same radar chart(s) separately for each environment present in the sweep (writes under out_dir/by_environment/<environment_label>/sweep/).",
    )
    parser.add_argument(
        "--compare-topologies",
        action="store_true",
        help="Write a single grouped-bar PNG that compares topologies in one figure (requires --plot-all-prompt-variants).",
    )
    parser.add_argument(
        "--compare-topologies-six-bars",
        action="store_true",
        help="Write a dual-axis PNG comparing regret (left) and judge (right) across topologies (baseline/control/simple).",
    )
    parser.add_argument(
        "--compare-environments",
        action="store_true",
        help="Write a single grouped-bar PNG that compares environments in one figure (requires --plot-all-prompt-variants).",
    )
    parser.add_argument(
        "--extra-sweep-dir",
        action="append",
        default=[],
        help="Additional sweep dirs to include in --compare-topologies aggregation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--titles-config",
        type=str,
        default=None,
        help="Optional YAML/JSON config used to annotate topology titles with parameters (e.g., p/k/rewire_prob/m). "
        "Defaults to <output_root>/config.json inferred from --sweep-dir.",
    )
    parser.add_argument(
        "--run-judge",
        action="store_true",
        help="Run the post-hoc secret-blackboard judge on this sweep before plotting (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="If optimal_summary.json is missing, compute and write it (Jira only; no API calls).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model to use when --run-judge is set (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--judge-max-output-tokens",
        type=int,
        default=256,
        help="Judge max output tokens when --run-judge is set.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature when --run-judge is set.",
    )
    parser.add_argument(
        "--judge-max-concurrent",
        type=int,
        default=8,
        help="Max concurrent judge calls when --run-judge is set.",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=2,
        help="Retries per prompt on request failure when --run-judge is set.",
    )
    parser.add_argument(
        "--judge-baseline-all-blackboards",
        dest="judge_baseline_all_blackboards",
        action="store_true",
        default=True,
        help="When --run-judge is set, judge baseline runs by averaging all blackboards (default: enabled).",
    )
    parser.add_argument(
        "--judge-no-baseline-all-blackboards",
        dest="judge_baseline_all_blackboards",
        action="store_false",
        help="When --run-judge is set, disable baseline judging (baseline short-circuits to rating=0).",
    )
    parser.add_argument(
        "--judge-overwrite",
        action="store_true",
        help="When --run-judge is set, overwrite existing judge files.",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    _maybe_run_judge(
        sweep_dir=sweep_dir,
        run_judge=bool(args.run_judge),
        judge_model=str(args.judge_model),
        judge_max_output_tokens=int(args.judge_max_output_tokens),
        judge_temperature=float(args.judge_temperature),
        judge_max_concurrent=int(args.judge_max_concurrent),
        judge_max_retries=int(args.judge_max_retries),
        judge_baseline_all_blackboards=bool(args.judge_baseline_all_blackboards),
        judge_overwrite=bool(args.judge_overwrite),
    )
    runs, _ = load_runs(sweep_dir, prefer_repaired=bool(args.prefer_repaired))
    _maybe_compute_missing_optimal_summaries(
        runs, compute_optimal=bool(args.compute_optimal)
    )
    rows = _build_rows(runs)
    out_dir = default_out_dir(sweep_dir=sweep_dir, requested_out_dir=args.out_dir)
    titles_config_path = (
        Path(str(args.titles_config)).expanduser().resolve()
        if args.titles_config
        else None
    )
    title_by_topology = _topology_title_overrides_from_config(
        sweep_dir=sweep_dir, titles_config_path=titles_config_path
    )
    reward_metric = _canonical_reward_metric_key(str(args.reward_metric or "joint_reward_ratio"))
    if reward_metric != "joint_reward_ratio" and not any(
        r.get(reward_metric) is not None for r in rows
    ):
        raise SystemExit(
            f"--reward-metric={args.reward_metric!r} requires per-run optimal metrics. "
            "Run experiments/collusion/compute_jira_optimal.py --write-json first."
        )
    _generate(
        sweep_dir=sweep_dir,
        out_dir=out_dir,
        rows=rows,
        colluder_count_requested=args.colluder_count,
        treatment_variant_requested=args.treatment_prompt_variant,
        plot_all_prompt_variants=bool(args.plot_all_prompt_variants),
        baseline_variant_requested=str(args.baseline_prompt_variant or "control"),
        include_incomplete=bool(args.include_incomplete),
        reward_metric=reward_metric,
        title_by_topology=title_by_topology or None,
        strict=True,
    )

    if args.compare_topologies_six_bars:
        colluder_count = _select_colluder_count(rows, args.colluder_count)
        out_path = (
            out_dir
            / "sweep"
            / f"combined_six_bars__optimality_gap__and_judge__by_topology__c{colluder_count}.png"
        )
        _plot_combined_six_bars_by_topology(
            rows=rows,
            metric_key="optimality_gap",
            out_path=out_path,
            colluder_count=int(colluder_count),
            baseline_variant=str(args.baseline_prompt_variant or "control"),
            include_incomplete=bool(args.include_incomplete),
            title_by_topology=title_by_topology or None,
        )

    if args.compare_topologies:
        if not args.plot_all_prompt_variants:
            raise SystemExit(
                "--compare-topologies currently requires --plot-all-prompt-variants."
            )
        compare_rows = list(rows)
        # Convenience: when comparing topologies for a topology sweep, also include the sibling "complete"
        # sweep (if present) so the "complete" topology shows up in the combined figure.
        try:
            sibling_complete = sweep_dir.parent / "complete"
            if (
                not (args.extra_sweep_dir or [])
                and sweep_dir.name != "complete"
                and sibling_complete.exists()
                and sibling_complete.is_dir()
            ):
                extra_runs, _ = load_runs(
                    sibling_complete, prefer_repaired=bool(args.prefer_repaired)
                )
                compare_rows.extend(_build_rows(extra_runs))
        except Exception:
            pass
        for extra in list(args.extra_sweep_dir or []):
            extra_dir = Path(str(extra)).expanduser().resolve()
            extra_runs, _ = load_runs(
                extra_dir, prefer_repaired=bool(args.prefer_repaired)
            )
            compare_rows.extend(_build_rows(extra_runs))
        ok = _compare_topologies_bars(
            sweep_dir=sweep_dir,
            out_dir=out_dir,
            rows=compare_rows,
            colluder_count_requested=args.colluder_count,
            baseline_variant_requested=str(args.baseline_prompt_variant or "control"),
            include_incomplete=bool(args.include_incomplete),
            reward_metric=reward_metric,
            titles_config_path=titles_config_path,
        )
        if not ok:
            raise SystemExit(
                "Could not build compare-topologies bar figure (need multiple topologies with data)."
            )

    if args.compare_environments:
        if not args.plot_all_prompt_variants:
            raise SystemExit(
                "--compare-environments currently requires --plot-all-prompt-variants."
            )
        compare_rows = list(rows)
        for extra in list(args.extra_sweep_dir or []):
            extra_dir = Path(str(extra)).expanduser().resolve()
            extra_runs, _ = load_runs(
                extra_dir, prefer_repaired=bool(args.prefer_repaired)
            )
            compare_rows.extend(_build_rows(extra_runs))
        ok = _compare_environments_bars(
            sweep_dir=sweep_dir,
            out_dir=out_dir,
            rows=compare_rows,
            colluder_count_requested=args.colluder_count,
            baseline_variant_requested=str(args.baseline_prompt_variant or "control"),
            include_incomplete=bool(args.include_incomplete),
            reward_metric=reward_metric,
        )
        if not ok:
            raise SystemExit(
                "Could not build compare-environments bar figure (need multiple environments with data)."
            )

    if args.by_topology:
        topologies = sorted(
            {str(r.get("topology")) for r in rows if r.get("topology") is not None}
        )
        for topo in topologies:
            topo_rows = [r for r in rows if str(r.get("topology")) == topo]
            if not topo_rows:
                continue
            topo_out_dir = out_dir / "by_topology" / sanitize_filename(topo)
            try:
                topo_title = _pretty_topology_label(
                    topo, title_by_topology=title_by_topology or None
                )
                _generate(
                    sweep_dir=sweep_dir,
                    out_dir=topo_out_dir,
                    rows=topo_rows,
                    colluder_count_requested=args.colluder_count,
                    treatment_variant_requested=args.treatment_prompt_variant,
                    plot_all_prompt_variants=bool(args.plot_all_prompt_variants),
                    baseline_variant_requested=str(
                        args.baseline_prompt_variant or "control"
                    ),
                    include_incomplete=bool(args.include_incomplete),
                    reward_metric=reward_metric,
                    title_prefix=f"topology={topo_title}",
                    title_by_topology=title_by_topology or None,
                    strict=False,
                )
            except SystemExit as e:
                print(f"[warn] topology={topo}: {e}")

    if args.by_environment:
        env_labels = sorted(
            {
                str(r.get("environment_label"))
                for r in rows
                if r.get("environment_label") is not None
                and str(r.get("environment_label") or "").strip()
            }
        )
        for env_label in env_labels:
            env_rows = [
                r for r in rows if str(r.get("environment_label") or "") == env_label
            ]
            if not env_rows:
                continue
            env_out_dir = out_dir / "by_environment" / sanitize_filename(env_label)
            try:
                _generate(
                    sweep_dir=sweep_dir,
                    out_dir=env_out_dir,
                    rows=env_rows,
                    colluder_count_requested=args.colluder_count,
                    treatment_variant_requested=args.treatment_prompt_variant,
                    plot_all_prompt_variants=bool(args.plot_all_prompt_variants),
                    baseline_variant_requested=str(
                        args.baseline_prompt_variant or "control"
                    ),
                    include_incomplete=bool(args.include_incomplete),
                    reward_metric=reward_metric,
                    title_prefix=f"env={env_label}",
                    title_by_topology=None,
                    strict=False,
                )
            except SystemExit as e:
                print(f"[warn] env={env_label}: {e}")

    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
