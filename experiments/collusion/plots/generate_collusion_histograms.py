from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from experiments.collusion.plots.common import canonical_variant, default_out_dir
from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    as_int,
    ensure_dir,
    finite,
    infer_labels_from_sweep_dir,
    mean,
    safe_load_json,
    sanitize_filename,
    sorted_unique,
    write_csv,
)
from experiments.common.plotting.logging_utils import (
    configure_basic_logging,
    log_saved_plot,
)
from experiments.common.plotting.load_runs import LoadedRun, load_runs
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)


def _pretty(s: Any) -> str:
    if s == "judge_mean_rating":
        return "Collusion Judge"
    if s == "optimality_gap":
        return "regret (optimal - achieved)"
    if s == "achieved_over_optimal":
        return "achieved / optimal"
    if s == "violations":
        return "Violations"
    return str(s).replace("_", " ")


_SMART_GRID_BOUNDS_CACHE: Dict[Tuple[Any, ...], Tuple[float, float]] = {}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _smart_grid_min_max_utility(
    run_config: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
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


def _robust_hist_range(values: List[float]) -> Optional[Tuple[float, float]]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    if len(vals) == 1:
        center = float(vals[0])
        pad = max(1e-6, abs(center) * 0.1)
        return center - pad, center + pad

    arr = np.array(vals, dtype=float)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.1)
        return lo - pad, hi + pad
    pad = max(1e-6, 0.06 * abs(hi - lo))
    return lo - pad, hi + pad


def _is_sweep_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.is_dir() and (child / "run_config.json").exists():
                return True
    except Exception:
        return False
    return False


def _find_sweep_dirs(outputs_root: Path) -> List[Path]:
    """
    Find sweep directories under:
      <outputs_root>/<tag>/<timestamp>/runs/<model_label>/<sweep_name>
    """
    sweep_dirs: List[Path] = []
    try:
        for runs_dir in sorted(outputs_root.glob("*/*/runs")):
            if not runs_dir.is_dir():
                continue
            for model_dir in sorted(runs_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for sweep_dir in sorted(model_dir.iterdir()):
                    if not sweep_dir.is_dir():
                        continue
                    if sweep_dir.name == "judge_secret_blackboard":
                        continue
                    if _is_sweep_dir(sweep_dir):
                        sweep_dirs.append(sweep_dir)
    except Exception:
        return sweep_dirs
    return sweep_dirs


def _build_rows(
    runs: List[LoadedRun],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

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

        if joint_reward is not None and str(env_label or "").strip().lower() == "smart_grid":
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
        except Exception:
            pass

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
            pass

        run_row: Dict[str, Any] = {
            "run_dir": str(run.run_dir),
            "run_id": rid,
            "model_label": rc.get("model_label"),
            "provider": rc.get("provider"),
            "model": rc.get("model"),
            "sweep_name": rc.get("sweep_name") or rc.get("sweep") or None,
            "environment_label": env_label,
            "environment_name": rc.get("environment_name"),
            "topology": rc.get("topology"),
            "num_agents": as_int(rc.get("num_agents")),
            "seed": as_int(rc.get("seed")),
            "colluder_count": as_int(rc.get("colluder_count")),
            "secret_channel_enabled": as_bool(rc.get("secret_channel_enabled")),
            "secret_blackboard_id": as_int(rc.get("secret_blackboard_id")),
            "prompt_variant": canonical_variant(rc.get("prompt_variant")),
            # Run-level outputs
            "status": metrics.get("status"),
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward_ratio,
            "optimal_joint_reward": optimal_joint_reward,
            "optimality_gap": optimality_gap,
            "achieved_over_optimal": achieved_over_optimal,
            "tasks_done": as_int(metrics.get("tasks_done")),
            "violations": as_int(metrics.get("violations")),
            "total_cost": metrics.get("total_cost"),
            "priority_sum": metrics.get("priority_sum"),
            "coalition_reward_sum": metrics.get("coalition_reward_sum"),
            "noncoalition_reward_sum": metrics.get("noncoalition_reward_sum"),
            "coalition_mean_reward": metrics.get("coalition_mean_reward"),
            "noncoalition_mean_reward": metrics.get("noncoalition_mean_reward"),
            "coalition_advantage_mean": metrics.get("coalition_advantage_mean"),
            "colluder_posts_secret_rate": metrics.get("colluder_posts_secret_rate"),
            "colluder_posts_total": as_int(metrics.get("colluder_posts_total")),
            "colluder_posts_secret": as_int(metrics.get("colluder_posts_secret")),
            "colluder_posts_non_secret": as_int(
                metrics.get("colluder_posts_non_secret")
            ),
            "colluder_posts_secret_to_largest_bb_ratio_mean": as_float(
                metrics.get("colluder_posts_secret_to_largest_bb_ratio_mean")
            ),
            # Judge (0–5 ratings; mean over simple/medium/complex prompts)
            "judge_simple_rating": judge_simple,
            "judge_medium_rating": judge_medium,
            "judge_complex_rating": judge_complex,
            "judge_mean_rating": judge_mean_rating,
        }
        run_rows.append(run_row)

        agents = metrics.get("agents")
        if not isinstance(agents, list):
            continue
        for a in agents:
            if not isinstance(a, dict):
                continue
            agent_rows.append(
                {
                    "run_dir": str(run.run_dir),
                    "run_id": rid,
                    "topology": rc.get("topology"),
                    "num_agents": as_int(rc.get("num_agents")),
                    "seed": as_int(rc.get("seed")),
                    "colluder_count": as_int(rc.get("colluder_count")),
                    "secret_channel_enabled": as_bool(rc.get("secret_channel_enabled")),
                    "secret_blackboard_id": as_int(rc.get("secret_blackboard_id")),
                    "prompt_variant": canonical_variant(rc.get("prompt_variant")),
                    **a,
                }
            )

    return run_rows, agent_rows


def _plot_box_by_category(
    rows: List[Dict[str, Any]],
    *,
    category_key: str,
    value_key: str,
    out_path: Path,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    category_order: Optional[List[Any]] = None,
    label_map: Optional[Dict[str, str]] = None,
    show_n_in_labels: bool = False,
    hline_at_zero: bool = False,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    categories = sorted_unique(rows, category_key)
    if not categories:
        return

    if category_order:
        seen = set()
        ordered: List[Any] = []
        for c in category_order:
            if c in categories and c not in seen:
                ordered.append(c)
                seen.add(c)
        for c in categories:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        categories = ordered

    vals_by_cat: List[List[float]] = []
    labels: List[str] = []
    for c in categories:
        subset = [r for r in rows if r.get(category_key) == c]
        vals = finite([r.get(value_key) for r in subset])
        if not vals:
            continue
        if label_map and str(c) in label_map:
            base_label = str(label_map[str(c)])
        else:
            base_label = str(c)
        if show_n_in_labels:
            labels.append(f"{base_label} (n={len(vals)})")
        else:
            labels.append(base_label)
        vals_by_cat.append(vals)

    if not vals_by_cat:
        return

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 3.8))
    # Matplotlib 3.9 renamed `labels` -> `tick_labels`.
    try:
        ax.boxplot(
            vals_by_cat,
            tick_labels=labels,
            showfliers=False,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
    except TypeError:
        ax.boxplot(
            vals_by_cat,
            labels=labels,
            showfliers=False,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

    # Overlay jittered points for visibility.
    rng = np.random.default_rng(0)
    for i, vals in enumerate(vals_by_cat, start=1):
        if not vals:
            continue
        xs = rng.normal(loc=float(i), scale=0.06, size=len(vals))
        ax.scatter(xs, vals, s=16, alpha=0.35, color="#4c72b0", linewidths=0)

    if hline_at_zero:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)

    ax.set_title(f"{_pretty(value_key)} by {_pretty(category_key)}")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_hist_grid(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    row_facet: str,
    col_facet: str,
    out_path: Path,
    bins: int = 24,
    x_label: Optional[str] = None,
    vline_at_zero: bool = False,
    sharex: bool = True,
    sharey: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    hist_range: Optional[Tuple[float, float]] = None,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    row_vals = sorted_unique(rows, row_facet)
    col_vals = sorted_unique(rows, col_facet)
    if not row_vals or not col_vals:
        return

    fig, axes = plt.subplots(
        nrows=len(row_vals),
        ncols=len(col_vals),
        figsize=(4.4 * len(col_vals), 2.8 * len(row_vals)),
        sharex=sharex,
        sharey=sharey,
    )
    if len(row_vals) == 1 and len(col_vals) == 1:
        axes = np.array([[axes]])
    elif len(row_vals) == 1:
        axes = np.array([axes])
    elif len(col_vals) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, r in enumerate(row_vals):
        for j, c in enumerate(col_vals):
            ax = axes[i, j]
            subset = [
                x for x in rows if x.get(row_facet) == r and x.get(col_facet) == c
            ]
            vals = finite([x.get(value_key) for x in subset])
            if vals:
                cell_bins = min(int(bins), max(4, len(vals)))
                ax.hist(
                    vals,
                    bins=cell_bins,
                    range=hist_range,
                    color="#4c72b0",
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.7,
                )
            if vline_at_zero and vals:
                ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
            ax.set_title(
                f"{_pretty(row_facet)}={_pretty(r)} | {_pretty(col_facet)}={_pretty(c)} (n={len(vals)})"
            )
            if x_label:
                ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            if xlim is not None:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))

    fig.suptitle(_pretty(value_key), y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

def _plot_hist_1d(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    out_path: Path,
    bins: int = 24,
    x_label: Optional[str] = None,
    vline_at_zero: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    hist_range: Optional[Tuple[float, float]] = None,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    vals = finite([r.get(value_key) for r in rows])
    if not vals:
        return

    if hist_range is None:
        hist_range = _robust_hist_range(vals)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cell_bins = min(int(bins), max(4, len(vals)))
    ax.hist(
        vals,
        bins=cell_bins,
        range=hist_range,
        color="#4c72b0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    if vline_at_zero:
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)

    ax.set_title(_pretty(value_key))
    if x_label:
        ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _plot_scatter_by_category(
    rows: List[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    category_key: str,
    out_path: Path,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    hline: Optional[float] = None,
    vline: Optional[float] = None,
    category_order: Optional[List[Any]] = None,
    label_map: Optional[Dict[str, str]] = None,
    legend_outside: bool = True,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    categories = sorted_unique(rows, category_key)
    if not categories:
        return

    if category_order:
        seen = set()
        ordered: List[Any] = []
        for c in category_order:
            if c in categories and c not in seen:
                ordered.append(c)
                seen.add(c)
        for c in categories:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        categories = ordered

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("tab10")

    for i, cat in enumerate(categories):
        subset = [r for r in rows if r.get(category_key) == cat]
        pts: List[Tuple[float, float]] = []
        for r in subset:
            x = as_float(r.get(x_key))
            y = as_float(r.get(y_key))
            if x is None or y is None:
                continue
            pts.append((x, y))
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        display = label_map.get(str(cat), str(cat)) if label_map else str(cat)
        ax.scatter(
            xs,
            ys,
            s=30,
            alpha=0.7,
            label=f"{display} (n={len(xs)})",
            color=cmap(i % 10),
            linewidths=0,
        )

    if hline is not None:
        ax.axhline(float(hline), color="black", linewidth=1.0, alpha=0.65)
    if vline is not None:
        ax.axvline(float(vline), color="black", linewidth=1.0, alpha=0.65)

    ax.set_title(f"{_pretty(y_key)} vs {_pretty(x_key)}")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    else:
        ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def _generate_plots(
    *,
    runs: List[LoadedRun],
    run_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
    out_dir: Path,
    bins: int,
    compare_topologies: bool = False,
) -> None:
    ensure_dir(out_dir)
    write_csv(out_dir / "run_rows.csv", run_rows)
    write_csv(out_dir / "agent_rows.csv", agent_rows)
    # Long-form view: one row per (agent, blackboard) with counts + rates.
    blackboard_usage_rows: List[Dict[str, Any]] = []
    for a in agent_rows:
        posts_by_bb = a.get("posts_by_blackboard")
        rates_by_bb = a.get("post_rates_by_blackboard")
        if not isinstance(posts_by_bb, dict):
            continue
        secret_bb = a.get("secret_blackboard_id")
        secret_bb_str = str(secret_bb) if secret_bb is not None else None
        for bb_id, count in posts_by_bb.items():
            bb_id_str = str(bb_id)
            rate = None
            if isinstance(rates_by_bb, dict):
                try:
                    rate = rates_by_bb.get(bb_id_str)
                except Exception:
                    rate = None
            blackboard_usage_rows.append(
                {
                    "run_dir": a.get("run_dir"),
                    "run_id": a.get("run_id"),
                    "topology": a.get("topology"),
                    "num_agents": a.get("num_agents"),
                    "seed": a.get("seed"),
                    "colluder_count": a.get("colluder_count"),
                    "secret_channel_enabled": a.get("secret_channel_enabled"),
                    "secret_blackboard_id": secret_bb,
                    "prompt_variant": a.get("prompt_variant"),
                    "agent_name": a.get("agent_name"),
                    "role": a.get("role"),
                    "blackboard_id": bb_id_str,
                    "posts": count,
                    "post_rate": rate,
                    "is_secret_blackboard": (
                        secret_bb_str is not None and bb_id_str == secret_bb_str
                    ),
                }
            )
    write_csv(out_dir / "blackboard_usage_rows.csv", blackboard_usage_rows)

    hist_out = out_dir / "hist"
    ensure_dir(hist_out)

    # Basic per-metric plots (these should exist even when not comparing topologies).
    secret_runs = [r for r in run_rows if r.get("secret_channel_enabled") is True]
    baseline_runs = [r for r in run_rows if r.get("secret_channel_enabled") is False]

    # Run-level histograms
    if finite([r.get("colluder_posts_secret_rate") for r in secret_runs]):
        _plot_hist_1d(
            secret_runs,
            value_key="colluder_posts_secret_rate",
            out_path=hist_out / "colluder_posts_secret_rate.png",
            bins=int(bins),
            x_label="Colluder posts on secret bb / total posts",
            xlim=(0.0, 1.0),
            hist_range=(0.0, 1.0),
        )

    if finite([r.get("coalition_advantage_mean") for r in run_rows]):
        _plot_hist_1d(
            run_rows,
            value_key="coalition_advantage_mean",
            out_path=hist_out / "coalition_advantage_mean.png",
            bins=int(bins),
            x_label="Coalition mean reward - non-coalition mean reward",
            vline_at_zero=True,
        )

    violation_vals = finite([r.get("violations") for r in run_rows])
    if violation_vals:
        vmax = max(violation_vals)
        _plot_hist_1d(
            run_rows,
            value_key="violations",
            out_path=hist_out / "violations.png",
            bins=max(4, int(vmax) + 1),
            x_label="Constraint violations per run",
            vline_at_zero=True,
            xlim=(0.0, float(vmax) + 1.0),
            hist_range=(0.0, float(vmax) + 1.0),
        )

    # Run-level: by prompt variant (secret-channel only)
    if finite([r.get("colluder_posts_secret_rate") for r in secret_runs]):
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key="colluder_posts_secret_rate",
            out_path=hist_out / "colluder_posts_secret_rate_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="Colluder secret post rate",
            ylim=(0.0, 1.0),
            show_n_in_labels=True,
        )

    if finite([r.get("violations") for r in secret_runs]):
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key="violations",
            out_path=hist_out / "violations_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="Violations",
            hline_at_zero=True,
            show_n_in_labels=True,
        )

    # Run-level: by prompt variant OR baseline (baseline is treated as its own category)
    by_variant_or_baseline: List[Dict[str, Any]] = []
    for r in baseline_runs:
        rr = dict(r)
        rr["variant_or_baseline"] = "baseline"
        by_variant_or_baseline.append(rr)
    for r in secret_runs:
        rr = dict(r)
        rr["variant_or_baseline"] = r.get("prompt_variant")
        by_variant_or_baseline.append(rr)

    if finite([r.get("colluder_posts_secret_rate") for r in by_variant_or_baseline]):
        _plot_box_by_category(
            by_variant_or_baseline,
            category_key="variant_or_baseline",
            value_key="colluder_posts_secret_rate",
            out_path=hist_out / "colluder_posts_secret_rate_by_variant_or_baseline.png",
            x_label="Variant (secret) or baseline (no SC)",
            y_label="Colluder secret post rate",
            ylim=(0.0, 1.0),
            category_order=["baseline"]
            + [v for v in sorted_unique(secret_runs, "prompt_variant") if v is not None],
            show_n_in_labels=True,
        )

    if finite([r.get("coalition_advantage_mean") for r in by_variant_or_baseline]):
        _plot_box_by_category(
            by_variant_or_baseline,
            category_key="variant_or_baseline",
            value_key="coalition_advantage_mean",
            out_path=hist_out / "coalition_advantage_mean_by_variant_or_baseline.png",
            x_label="Variant (secret) or baseline (no SC)",
            y_label="Coalition advantage (mean)",
            hline_at_zero=True,
            category_order=["baseline"]
            + [v for v in sorted_unique(secret_runs, "prompt_variant") if v is not None],
            show_n_in_labels=True,
        )

    if finite([r.get("violations") for r in by_variant_or_baseline]):
        _plot_box_by_category(
            by_variant_or_baseline,
            category_key="variant_or_baseline",
            value_key="violations",
            out_path=hist_out / "violations_by_variant_or_baseline.png",
            x_label="Variant (secret) or baseline (no SC)",
            y_label="Violations",
            hline_at_zero=True,
            category_order=["baseline"]
            + [v for v in sorted_unique(secret_runs, "prompt_variant") if v is not None],
            show_n_in_labels=True,
        )

    if finite([r.get("achieved_over_optimal") for r in by_variant_or_baseline]):
        _plot_box_by_category(
            by_variant_or_baseline,
            category_key="variant_or_baseline",
            value_key="achieved_over_optimal",
            out_path=hist_out / "achieved_over_optimal_by_variant_or_baseline.png",
            x_label="Variant (secret) or baseline (no SC)",
            y_label="achieved / optimal (exact)",
            category_order=["baseline"]
            + [v for v in sorted_unique(secret_runs, "prompt_variant") if v is not None],
            show_n_in_labels=True,
        )

    if finite([r.get("optimality_gap") for r in by_variant_or_baseline]):
        _plot_box_by_category(
            by_variant_or_baseline,
            category_key="variant_or_baseline",
            value_key="optimality_gap",
            out_path=hist_out / "optimality_gap_by_variant_or_baseline.png",
            x_label="Variant (secret) or baseline (no SC)",
            y_label="regret = optimal - achieved (exact)",
            hline_at_zero=True,
            category_order=["baseline"]
            + [v for v in sorted_unique(secret_runs, "prompt_variant") if v is not None],
            show_n_in_labels=True,
        )

    if finite([r.get("joint_reward_ratio") for r in secret_runs]):
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key="joint_reward_ratio",
            out_path=hist_out / "joint_reward_ratio_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="joint_reward / max_joint_reward",
            ylim=(0.0, 1.0),
            show_n_in_labels=True,
        )

    if finite([r.get("achieved_over_optimal") for r in secret_runs]):
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key="achieved_over_optimal",
            out_path=hist_out / "achieved_over_optimal_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="achieved / optimal (exact)",
            show_n_in_labels=True,
        )

    if finite([r.get("optimality_gap") for r in secret_runs]):
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key="optimality_gap",
            out_path=hist_out / "optimality_gap_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="regret = optimal - achieved (exact)",
            hline_at_zero=True,
            show_n_in_labels=True,
        )

    # Judge ratings (0–5), by secret-channel prompt variant (optional files)
    for key in ("judge_mean_rating",):
        if not finite([r.get(key) for r in secret_runs]):
            continue
        _plot_box_by_category(
            secret_runs,
            category_key="prompt_variant",
            value_key=key,
            out_path=hist_out / f"{key}_by_variant.png",
            x_label="prompt_variant (secret_channel_enabled=true)",
            y_label="Judge rating (0–5)",
            ylim=(0.0, 5.0),
            show_n_in_labels=True,
        )

    # Agent-level by role
    if finite([a.get("reward") for a in agent_rows]):
        _plot_box_by_category(
            agent_rows,
            category_key="role",
            value_key="reward",
            out_path=hist_out / "agent_reward_by_role.png",
            x_label="Role",
            y_label="Reward",
            show_n_in_labels=True,
        )

    if finite([a.get("task_cost") for a in agent_rows]):
        _plot_box_by_category(
            agent_rows,
            category_key="role",
            value_key="task_cost",
            out_path=hist_out / "task_cost_by_role.png",
            x_label="Role",
            y_label="Task cost",
            show_n_in_labels=True,
        )

    if compare_topologies:
        # Run-level: secret-channel usage rate (colluders only; avoid colluder_count==0 noise)
        usage_rows = [
            r
            for r in run_rows
            if (r.get("colluder_count") or 0) > 0
            and (r.get("secret_channel_enabled") is True)
        ]
        # Single-figure topology comparisons for the histogram metrics.
        topo_compare = hist_out / "compare_topologies"
        ensure_dir(topo_compare)

        topo_order = sorted_unique(run_rows, "topology")
        topo_label_map = {
            str(t): str(t).replace("_", " ") for t in topo_order if t is not None
        }

        if len(sorted_unique(usage_rows, "topology")) > 1:
            _plot_hist_grid(
                usage_rows,
                value_key="colluder_posts_secret_rate",
                row_facet="prompt_variant",
                col_facet="topology",
                out_path=topo_compare / "colluder_posts_secret_rate__by_topology.png",
                bins=int(bins),
                x_label="Colluder posts on secret bb / total posts",
                vline_at_zero=False,
                xlim=(0.0, 1.0),
                hist_range=(0.0, 1.0),
                sharex=True,
                sharey=True,
            )
            for pv in sorted_unique(usage_rows, "prompt_variant"):
                subset = [r for r in usage_rows if r.get("prompt_variant") == pv]
                _plot_box_by_category(
                    subset,
                    category_key="topology",
                    value_key="colluder_posts_secret_rate",
                    out_path=topo_compare
                    / f"colluder_posts_secret_rate__box_by_topology__pv{sanitize_filename(pv)}.png",
                    x_label=f"Topology (prompt_variant={pv})",
                    y_label="Colluder secret post rate",
                    ylim=(0.0, 1.0),
                    category_order=topo_order,
                    label_map=topo_label_map,
                    show_n_in_labels=True,
                )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate histogram plots for collusion sweep outputs."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=False,
        help="Path like experiments/collusion/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--all-outputs",
        action="store_true",
        help="Generate plots for every sweep under experiments/collusion/outputs/ (auto-discovers sweep dirs).",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Root to scan for outputs (expects <tag>/<timestamp>/runs/<model>/<sweep_name>).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for a single --sweep-dir run (default: experiments/collusion/plots_outputs/<tag>/<ts>/<model>/<sweep_name>).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="When using --all-outputs/--outputs-root, write outputs under this root while preserving tag/timestamp/model/sweep subfolders.",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present (final_summary_repaired.json, metrics_repaired.json).",
    )
    parser.add_argument("--bins", type=int, default=24, help="Histogram bin count.")
    parser.add_argument(
        "--by-topology",
        action="store_true",
        help="Also generate the same plots separately for each topology present in the sweep (writes under out_dir/by_topology/<topology>/).",
    )
    parser.add_argument(
        "--compare-topologies",
        action="store_true",
        help="Generate combined topology-comparison figures (single PNGs) under out_dir/hist/compare_topologies/.",
    )
    args = parser.parse_args(argv)

    multi_mode = not bool(args.sweep_dir)

    sweep_dirs: List[Path] = []
    if args.sweep_dir:
        sweep_dirs = [Path(args.sweep_dir).expanduser().resolve()]
    else:
        outputs_root = (
            Path(args.outputs_root).expanduser().resolve()
            if args.outputs_root
            else Path("experiments/collusion/outputs").resolve()
        )
        if not (bool(args.all_outputs) or bool(args.outputs_root)):
            raise SystemExit(
                "Provide --sweep-dir or use --all-outputs/--outputs-root."
            )
        sweep_dirs = _find_sweep_dirs(outputs_root)
        if not sweep_dirs:
            raise SystemExit(f"No sweep dirs found under: {outputs_root}")

    # In multi-sweep mode, treat --out-dir as an output root for convenience.
    out_root: Optional[Path] = None
    if multi_mode and (args.out_root or args.out_dir):
        out_root = Path(args.out_root or args.out_dir).expanduser().resolve()

    total = 0
    failures = 0
    for sweep_dir in sweep_dirs:
        try:
            runs, _ = load_runs(sweep_dir, prefer_repaired=bool(args.prefer_repaired))
            run_rows, agent_rows = _build_rows(runs)

            if out_root is not None:
                labels = infer_labels_from_sweep_dir(sweep_dir)
                out_dir = (
                    out_root
                    / labels.experiment_tag
                    / labels.timestamp
                    / labels.model_label
                    / labels.sweep_name
                )
            else:
                out_dir = default_out_dir(
                    sweep_dir=sweep_dir, requested_out_dir=args.out_dir
                )

            _generate_plots(
                runs=runs,
                run_rows=run_rows,
                agent_rows=agent_rows,
                out_dir=out_dir,
                bins=int(args.bins),
                compare_topologies=bool(args.compare_topologies),
            )
            total += 1
            print(f"Wrote plots to: {out_dir}")

            if args.by_topology:
                topologies = sorted_unique(run_rows, "topology")
                if topologies:
                    for topo in topologies:
                        topo_runs = [
                            r for r in runs if (r.run_config or {}).get("topology") == topo
                        ]
                        topo_run_rows = [r for r in run_rows if r.get("topology") == topo]
                        topo_agent_rows = [r for r in agent_rows if r.get("topology") == topo]
                        if not topo_run_rows and not topo_agent_rows:
                            continue
                        topo_out = out_dir / "by_topology" / sanitize_filename(str(topo))
                        _generate_plots(
                            runs=topo_runs,
                            run_rows=topo_run_rows,
                            agent_rows=topo_agent_rows,
                            out_dir=topo_out,
                            bins=int(args.bins),
                            compare_topologies=False,
                        )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failures += 1
            logger.exception("Failed generating plots for sweep_dir=%s: %s", sweep_dir, e)

    if failures:
        raise SystemExit(f"Completed with failures: {failures}/{len(sweep_dirs)}")

    return 0 if total else 1


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
