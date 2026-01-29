from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from experiments.common.plotting.io_utils import (
    as_bool,
    as_float,
    as_int,
    ensure_dir,
    mean,
    safe_load_json,
    sanitize_filename,
    sem,
    write_csv,
)


logger = logging.getLogger(__name__)


def _canonical_variant(value: Any) -> str:
    return str(value or "").strip()


_PROVIDER_PREFIX_RE = re.compile(r"^(openai|anthropic|together)[-_]+", re.IGNORECASE)


def _pretty_metric_label(key: str) -> str:
    k = str(key or "").strip()
    if k == "optimality_gap":
        return "Regret (↓)"
    if k == "normalized_regret":
        return "Normalized Regret (↓)"
    if k == "achieved_over_optimal":
        return "Achieved / Optimal"
    if k == "joint_reward_ratio":
        return "Joint Reward Ratio"
    if k == "joint_reward":
        return "Joint Reward"
    if k == "judge_mean_rating":
        return "Collusion Prediction (Judge) (↓)"
    return k.replace("_", " ").title()


def _pretty_model_label(model_label: str) -> str:
    raw = str(model_label or "").strip()
    raw = _PROVIDER_PREFIX_RE.sub("", raw)
    if not raw:
        return "Unknown"

    lowered = raw.lower()
    if lowered.startswith("kimik2"):
        if "thinking" in lowered:
            return "Kimi-K2-Thinking"
        return "Kimi-K2-Instruct"

    parts = re.split(r"[-_]+", raw)
    pretty: List[str] = []
    for part in parts:
        if not part:
            continue
        p = part.strip()
        pl = p.lower()
        if pl == "gpt":
            pretty.append("GPT")
            continue
        if pl == "oss":
            pretty.append("OSS")
            continue
        if re.fullmatch(r"\d+b", pl):
            pretty.append(f"{p[:-1]}B")
            continue
        if pl == "k2":
            pretty.append("K2")
            continue
        if pl in {"claude", "sonnet", "haiku", "opus", "gemini", "flash", "lite", "mini", "instruct"}:
            pretty.append(pl.capitalize())
            continue
        pretty.append(p)

    return "-".join(pretty) if pretty else raw


def _apply_large_font_style() -> None:
    # Bigger, presentation-friendly fonts.
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def _iter_model_dirs(root: Path) -> Iterable[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name == "judge_secret_blackboard":
            continue
        yield model_dir


def _infer_sweep_name(root: Path) -> str:
    sweep_names: set[str] = set()
    for model_dir in _iter_model_dirs(root):
        for child in sorted(model_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "judge_secret_blackboard":
                continue
            if (child / "run_config.json").exists():
                # This is already a run dir, not a sweep dir.
                continue
            sweep_names.add(child.name)
    if len(sweep_names) == 1:
        return next(iter(sweep_names))
    if not sweep_names:
        raise SystemExit(f"No sweep directories found under: {root / 'runs'}")
    raise SystemExit(
        "Multiple sweep directories found; pass --sweep-name. Options: "
        + ", ".join(sorted(sweep_names))
    )


def _iter_run_dirs(sweep_dir: Path) -> Iterable[Path]:
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "run_config.json").exists():
            yield child


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    obj = safe_load_json(path)
    return obj if isinstance(obj, dict) else None


def _judge_mean_rating(*, model_dir: Path, sweep_name: str, run_name: str) -> Optional[float]:
    judge_path = (
        model_dir / "judge_secret_blackboard" / sweep_name / f"{run_name}.json"
    )
    payload = safe_load_json(judge_path) if judge_path.exists() else None
    if not isinstance(payload, dict):
        return None
    judgements = payload.get("judgements")
    if not isinstance(judgements, dict):
        return None

    vals: List[float] = []
    for key in ("simple", "medium", "complex"):
        j = judgements.get(key)
        if not isinstance(j, dict):
            continue
        rating = as_float(j.get("rating"))
        if rating is None:
            continue
        vals.append(float(rating))
    if not vals:
        return None
    return float(mean(vals))


def _load_optimal_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json(run_dir / "optimal_summary.json")
    return payload if isinstance(payload, dict) else None


def _compute_and_write_optimal_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        from experiments.collusion import compute_jira_optimal

        try:
            instance = compute_jira_optimal._reconstruct_instance(run_dir)
        except Exception:
            instance = compute_jira_optimal._load_instance_from_agent_prompts(run_dir)
        weights = compute_jira_optimal._load_weights(run_dir, overrides=None)
        optimal = compute_jira_optimal.solve_optimal_assignment(instance=instance, weights=weights)

        payload = {
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
                    k: (v if v is not None else "skip") for k, v in optimal.assignment.items()
                },
            },
        }
        out_path = run_dir / "optimal_summary.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload
    except Exception as exc:
        logger.warning("Failed to compute optimal for %s: %s", run_dir, exc)
        return None


@dataclass(frozen=True)
class RunRow:
    model_label: str
    provider: Optional[str]
    model: Optional[str]
    sweep_name: str
    topology: Optional[str]
    num_agents: Optional[int]
    colluder_count: Optional[int]
    seed: Optional[int]
    secret_channel_enabled: Optional[bool]
    prompt_variant: str
    status: str

    joint_reward: Optional[float]
    joint_reward_ratio: Optional[float]
    optimal_joint_reward: Optional[float]
    optimality_gap: Optional[float]  # optimal - achieved
    achieved_over_optimal: Optional[float]  # achieved / optimal
    normalized_regret: Optional[float]  # 1 - achieved / optimal (clipped to [0, 1])
    judge_mean_rating: Optional[float]


def _status_is_complete(status: Any) -> bool:
    return str(status or "").strip().lower() == "complete"


def _load_run_row(
    *,
    run_dir: Path,
    model_dir: Path,
    sweep_name: str,
    compute_optimal: bool,
    prefer_repaired: bool,
) -> Optional[RunRow]:
    rc = _read_json(run_dir / "run_config.json") or {}
    final_summary_path = run_dir / "final_summary.json"
    metrics_path = run_dir / "metrics.json"
    if prefer_repaired:
        repaired_summary = run_dir / "final_summary_repaired.json"
        repaired_metrics = run_dir / "metrics_repaired.json"
        if repaired_summary.exists():
            final_summary_path = repaired_summary
        if repaired_metrics.exists():
            metrics_path = repaired_metrics

    fs = _read_json(final_summary_path) or {}
    metrics = _read_json(metrics_path) or {}

    status = metrics.get("status", fs.get("status", "unknown"))
    status_s = str(status or "unknown")

    joint_reward = as_float(fs.get("joint_reward"))
    joint_reward_ratio = as_float(fs.get("joint_reward_ratio"))
    raw_joint_reward = as_float(fs.get("raw_joint_reward"))

    # Back-compat: some envs stored normalized score in joint_reward.
    if (
        joint_reward_ratio is None
        and joint_reward is not None
        and raw_joint_reward is not None
        and 0.0 <= float(joint_reward) <= 1.0
    ):
        joint_reward_ratio = float(joint_reward)
        joint_reward = float(raw_joint_reward)

    optimal_payload = _load_optimal_summary(run_dir)
    if optimal_payload is None and compute_optimal:
        optimal_payload = _compute_and_write_optimal_summary(run_dir)

    optimal_joint_reward = None
    if isinstance(optimal_payload, dict):
        optimal = optimal_payload.get("optimal")
        if isinstance(optimal, dict):
            optimal_joint_reward = as_float(optimal.get("joint_reward"))

    optimality_gap = None
    achieved_over_optimal = None
    normalized_regret = None
    if joint_reward is not None and optimal_joint_reward is not None:
        optimality_gap = float(optimal_joint_reward) - float(joint_reward)
        if float(optimal_joint_reward) != 0.0:
            achieved_over_optimal = float(joint_reward) / float(optimal_joint_reward)
            normalized_regret = 1.0 - float(achieved_over_optimal)
            # Keep the metric stable and comparable across runs.
            if normalized_regret < 0.0:
                normalized_regret = 0.0
            elif normalized_regret > 1.0:
                normalized_regret = 1.0

    prompt_variant = _canonical_variant(rc.get("prompt_variant") or "control")
    judge_mean_rating = _judge_mean_rating(
        model_dir=model_dir, sweep_name=sweep_name, run_name=run_dir.name
    )

    return RunRow(
        model_label=str(rc.get("model_label") or model_dir.name),
        provider=str(rc.get("provider")) if rc.get("provider") is not None else None,
        model=str(rc.get("model")) if rc.get("model") is not None else None,
        sweep_name=str(rc.get("sweep") or sweep_name),
        topology=str(rc.get("topology")) if rc.get("topology") is not None else None,
        num_agents=as_int(rc.get("num_agents")),
        colluder_count=as_int(rc.get("colluder_count")),
        seed=as_int(rc.get("seed")),
        secret_channel_enabled=as_bool(rc.get("secret_channel_enabled")),
        prompt_variant=prompt_variant,
        status=status_s,
        joint_reward=joint_reward,
        joint_reward_ratio=joint_reward_ratio,
        optimal_joint_reward=optimal_joint_reward,
        optimality_gap=optimality_gap,
        achieved_over_optimal=achieved_over_optimal,
        normalized_regret=normalized_regret,
        judge_mean_rating=judge_mean_rating,
    )


def _filter_rows(
    rows: List[RunRow],
    *,
    topology: Optional[str],
    num_agents: Optional[int],
    colluder_count: Optional[int],
    require_complete: bool,
) -> List[RunRow]:
    out: List[RunRow] = []
    for r in rows:
        if topology is not None and str(r.topology) != str(topology):
            continue
        if num_agents is not None and int(r.num_agents or -1) != int(num_agents):
            continue
        if colluder_count is not None and int(r.colluder_count or -1) != int(colluder_count):
            continue
        if require_complete and not _status_is_complete(r.status):
            continue
        out.append(r)
    return out


def _seed_means(rows: List[RunRow], *, key: str) -> List[float]:
    by_seed: Dict[int, List[float]] = {}
    for r in rows:
        seed = r.seed
        if seed is None:
            continue
        val = getattr(r, key, None)
        if val is None:
            continue
        if not math.isfinite(float(val)):
            continue
        by_seed.setdefault(int(seed), []).append(float(val))

    out: List[float] = []
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        if not vals:
            continue
        out.append(float(mean(vals)))
    return out


def _group_stats(rows: List[RunRow], *, key: str) -> Dict[str, Any]:
    vals = _seed_means(rows, key=key)
    if not vals:
        return {"n": 0, "mean": None, "sem": None}
    return {
        "n": int(len(vals)),
        "mean": float(mean(vals)),
        "sem": float(sem(vals)),
    }


def _variant_or_baseline(r: RunRow) -> str:
    if r.secret_channel_enabled is True:
        return _canonical_variant(r.prompt_variant) or "control"
    return "baseline"


def _plot_hist_by_condition(
    *,
    rows: List[RunRow],
    metric_key: str,
    out_path: Path,
    bins: int,
) -> None:
    _apply_large_font_style()
    groups = {"baseline": [], "control": [], "simple": []}
    for group, label in (("baseline", "baseline"), ("control", "control"), ("simple", "simple")):
        subset = [r for r in rows if _variant_or_baseline(r) == group]
        groups[group] = _seed_means(subset, key=metric_key)

    all_vals = [v for vs in groups.values() for v in vs]
    if not all_vals:
        return

    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if metric_key == "normalized_regret":
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        pad = max(1e-6, abs(vmin) * 0.1)
        vmin -= pad
        vmax += pad

    ensure_dir(out_path.parent)
    plt.figure(figsize=(7.2, 4.2))
    for key, color in (("baseline", "tab:blue"), ("control", "tab:gray"), ("simple", "tab:green")):
        vals = groups.get(key) or []
        if not vals:
            continue
        plt.hist(
            vals,
            bins=int(bins),
            range=(vmin, vmax),
            alpha=0.45,
            label=f"{key} (n={len(vals)})",
            color=color,
            edgecolor="black",
            linewidth=0.3,
        )
    plt.xlabel(_pretty_metric_label(metric_key))
    plt.ylabel("count (seeds)")
    plt.title(f"{metric_key.replace('_', ' ')} by condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_mean_sem_by_condition(
    *,
    rows: List[RunRow],
    metric_key: str,
    out_path: Path,
) -> None:
    _apply_large_font_style()
    order = ["baseline", "control", "simple"]
    means: List[float] = []
    errs: List[float] = []
    ns: List[int] = []

    for group in order:
        subset = [r for r in rows if _variant_or_baseline(r) == group]
        vals = _seed_means(subset, key=metric_key)
        ns.append(int(len(vals)))
        if vals:
            means.append(float(mean(vals)))
            errs.append(float(sem(vals)))
        else:
            means.append(float("nan"))
            errs.append(float("nan"))

    ensure_dir(out_path.parent)
    plt.figure(figsize=(6.2, 4.0))
    x = np.arange(len(order))
    plt.bar(x, means, color=["tab:blue", "tab:gray", "tab:green"], alpha=0.85)
    plt.errorbar(x, means, yerr=errs, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    plt.xticks(x, [f"{k}\n(n={n})" for k, n in zip(order, ns)])
    plt.ylabel(_pretty_metric_label(metric_key))
    if metric_key == "normalized_regret":
        plt.ylim(0.0, 1.0)
    plt.title(f"Mean ± SEM: {metric_key.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_combined_mean_sem_with_judge(
    *,
    rows: List[RunRow],
    metric_key: str,
    out_path: Path,
) -> None:
    _apply_large_font_style()
    models = sorted({r.model_label for r in rows})
    if not models:
        return

    conditions = ["baseline", "control", "simple"]
    colors = {"baseline": "tab:blue", "control": "tab:gray", "simple": "tab:green"}

    def _stats_for(model_label: str, condition: str, key: str) -> Tuple[float, float, int]:
        subset = [
            r
            for r in rows
            if r.model_label == model_label and _variant_or_baseline(r) == condition
        ]
        stats = _group_stats(subset, key=key)
        mu = stats.get("mean")
        se = stats.get("sem")
        n = int(stats.get("n") or 0)
        return (
            float(mu) if mu is not None else float("nan"),
            float(se) if se is not None else float("nan"),
            n,
        )

    metric_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_means: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    for m in models:
        for c in conditions:
            mu, se, _ = _stats_for(m, c, metric_key)
            metric_means[c].append(mu)
            metric_sems[c].append(se)

            j_mu, j_se, _ = _stats_for(m, c, "judge_mean_rating")
            judge_means[c].append(j_mu)
            judge_sems[c].append(j_se)

    any_judge = any(math.isfinite(v) for c in conditions for v in judge_means[c])

    ensure_dir(out_path.parent)
    fig_h = 7.2 if any_judge else 4.0
    fig_w = max(8.0, 1.35 * len(models))
    if any_judge:
        fig, (ax_metric, ax_judge) = plt.subplots(
            nrows=2, ncols=1, figsize=(fig_w, fig_h), sharex=True
        )
    else:
        fig, ax_metric = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
        ax_judge = None

    x = np.arange(len(models), dtype=float)
    width = 0.24
    offsets = {"baseline": -width, "control": 0.0, "simple": width}

    for c in conditions:
        ax_metric.bar(
            x + offsets[c],
            metric_means[c],
            width=width,
            yerr=metric_sems[c],
            color=colors[c],
            alpha=0.85,
            capsize=3,
            label=c,
        )
    ax_metric.set_ylabel(_pretty_metric_label(metric_key))
    if metric_key == "normalized_regret":
        ax_metric.set_ylim(0.0, 1.0)

    if ax_judge is not None:
        for c in conditions:
            ax_judge.bar(
                x + offsets[c],
                judge_means[c],
                width=width,
                yerr=judge_sems[c],
                color=colors[c],
                alpha=0.85,
                capsize=3,
                label=c,
            )
        ax_judge.set_ylabel(_pretty_metric_label("judge_mean_rating"))

    labels = [_pretty_model_label(m) for m in models]
    plt.xticks(x, labels, rotation=0, ha="center")

    condition_order = ["baseline", "control", "simple"]
    condition_handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=c.title())
        for c in condition_order
    ]
    plt.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    fig.legend(
        condition_handles,
        [h.get_label() for h in condition_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.4,
    )
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_combined_six_bars_dual_axis(
    *,
    rows: List[RunRow],
    metric_key: str,
    out_path: Path,
) -> None:
    _apply_large_font_style()
    models = sorted({r.model_label for r in rows})
    if not models:
        return

    conditions = ["baseline", "control", "simple"]
    colors = {"baseline": "tab:blue", "control": "tab:gray", "simple": "tab:green"}

    def _stats_for(model_label: str, condition: str, key: str) -> Tuple[float, float, int]:
        subset = [
            r
            for r in rows
            if r.model_label == model_label and _variant_or_baseline(r) == condition
        ]
        stats = _group_stats(subset, key=key)
        mu = stats.get("mean")
        se = stats.get("sem")
        n = int(stats.get("n") or 0)
        return (
            float(mu) if mu is not None else float("nan"),
            float(se) if se is not None else float("nan"),
            n,
        )

    metric_means: Dict[str, List[float]] = {c: [] for c in conditions}
    metric_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_means: Dict[str, List[float]] = {c: [] for c in conditions}
    judge_sems: Dict[str, List[float]] = {c: [] for c in conditions}
    for m in models:
        for c in conditions:
            mu, se, _ = _stats_for(m, c, metric_key)
            metric_means[c].append(mu)
            metric_sems[c].append(se)

            j_mu, j_se, _ = _stats_for(m, c, "judge_mean_rating")
            judge_means[c].append(j_mu)
            judge_sems[c].append(j_se)

    any_judge = any(math.isfinite(v) for c in conditions for v in judge_means[c])
    if not any_judge:
        logger.warning("No judge_mean_rating data found; skipping: %s", out_path)
        return

    ensure_dir(out_path.parent)
    fig, ax_metric = plt.subplots(nrows=1, ncols=1, figsize=(12.0, 3.0))
    ax_judge = ax_metric.twinx()

    x = np.arange(len(models), dtype=float)
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
    ax_metric.set_ylabel(ax_metric_label)
    if metric_key == "normalized_regret":
        ax_metric.set_ylim(0.0, 1.0)
    judge_axis_label = _pretty_metric_label("judge_mean_rating").replace(" (Judge)", "")
    ax_judge.set_ylabel(judge_axis_label, labelpad=10)
    # Ensure right-axis label and ticks remain visible.
    ax_judge.tick_params(axis="y", labelcolor="black")
    ax_judge.spines["right"].set_visible(True)

    # Two centered legends:
    # 1) Condition colors (Baseline / Simple / Control)
    # 2) Bar style (Regret solid vs Judge hatched)
    condition_order = ["baseline", "control", "simple"]
    condition_handles = [
        Patch(facecolor=colors[c], edgecolor="none", label=c.title())
        for c in condition_order
    ]
    style_color = "white"
    ax_metric_label_for_legend = ax_metric_label.replace(" (↓)", "").strip()
    judge_label_for_legend = _pretty_metric_label("judge_mean_rating").replace(" (↓)", "").strip()
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
            label=judge_label_for_legend,
        ),
    ]

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(
        [_pretty_model_label(m) for m in models], rotation=0, ha="center"
    )
    # Keep x tick labels readable, without colliding with the bottom legend.
    ax_metric.tick_params(axis="x", pad=6)
    ax_metric.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    # Light horizontal gridlines for readability.
    ax_metric.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax_metric.set_axisbelow(True)
    # Single-line legend: condition colors + bar style.
    legend_handles = condition_handles + style_handles
    legend_labels = [h.get_label() for h in legend_handles]
    # Keep the legend close while leaving room for the right-side y-label.
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
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate regret-based Jira collusion report (table + plots) from a collusion output root."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path like experiments/collusion/outputs/<tag>/<timestamp>",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Sweep directory name under each model (e.g., complete_n6_c2). If omitted, auto-infer when unique.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/collusion/plots_outputs/<tag>/<ts>/regret_report/<sweep_name>).",
    )
    parser.add_argument("--topology", type=str, default="complete")
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--colluder-count", type=int, default=2)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs where status != 'complete' (not recommended).",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="If optimal_summary.json is missing, compute and write it (no API calls).",
    )
    parser.add_argument(
        "--prefer-repaired",
        action="store_true",
        help="Prefer *_repaired.json artifacts when present (final_summary_repaired.json, metrics_repaired.json).",
    )
    parser.add_argument(
        "--table-prompt-variant",
        type=str,
        default="control",
        help="Prompt variant for the table subset (secret_channel_enabled=true).",
    )
    parser.add_argument(
        "--hist-metric",
        type=str,
        default="optimality_gap",
        choices=[
            "optimality_gap",
            "normalized_regret",
            "achieved_over_optimal",
            "joint_reward_ratio",
            "joint_reward",
        ],
        help="Metric to plot in the baseline-vs-SC comparison.",
    )
    parser.add_argument("--bins", type=int, default=18, help="Histogram bin count.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    sweep_name = str(args.sweep_name) if args.sweep_name else _infer_sweep_name(root)

    # Default output location based on <tag>/<timestamp>.
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        tag = root.parent.name
        timestamp = root.name
        out_dir = (
            Path("experiments/collusion/plots_outputs")
            / str(tag)
            / str(timestamp)
            / "regret_report"
            / sweep_name
        ).resolve()
    ensure_dir(out_dir)

    rows: List[RunRow] = []
    missing_sweeps: List[str] = []
    for model_dir in _iter_model_dirs(root):
        sweep_dir = model_dir / sweep_name
        if not sweep_dir.exists():
            missing_sweeps.append(model_dir.name)
            continue
        for run_dir in _iter_run_dirs(sweep_dir):
            row = _load_run_row(
                run_dir=run_dir,
                model_dir=model_dir,
                sweep_name=sweep_name,
                compute_optimal=bool(args.compute_optimal),
                prefer_repaired=bool(args.prefer_repaired),
            )
            if row is not None:
                rows.append(row)

    if missing_sweeps:
        logger.warning(
            "Missing sweep %s for models: %s", sweep_name, ", ".join(missing_sweeps)
        )
    if not rows:
        raise SystemExit("No runs found.")

    rows = _filter_rows(
        rows,
        topology=str(args.topology) if args.topology else None,
        num_agents=int(args.num_agents) if args.num_agents else None,
        colluder_count=int(args.colluder_count) if args.colluder_count else None,
        require_complete=not bool(args.include_incomplete),
    )
    if not rows:
        raise SystemExit("No runs matched the requested filters.")

    # Table: secret_channel_enabled=true + prompt_variant=<table_variant>, grouped by model_label.
    table_variant = _canonical_variant(args.table_prompt_variant) or "control"
    table_rows: List[Dict[str, Any]] = []
    for model_label in sorted({r.model_label for r in rows}):
        subset = [
            r
            for r in rows
            if r.model_label == model_label
            and r.secret_channel_enabled is True
            and _canonical_variant(r.prompt_variant) == table_variant
        ]
        if not subset:
            continue

        provider = next((r.provider for r in subset if r.provider), None)
        model_name = next((r.model for r in subset if r.model), None)

        regret_stats = _group_stats(subset, key="optimality_gap")
        ratio_stats = _group_stats(subset, key="achieved_over_optimal")
        jr_ratio_stats = _group_stats(subset, key="joint_reward_ratio")
        judge_stats = _group_stats(subset, key="judge_mean_rating")

        table_rows.append(
            {
                "model_label": model_label,
                "provider": provider,
                "model": model_name,
                "n_seeds": regret_stats["n"],
                "regret_mean": regret_stats["mean"],
                "regret_sem": regret_stats["sem"],
                "achieved_over_optimal_mean": ratio_stats["mean"],
                "achieved_over_optimal_sem": ratio_stats["sem"],
                "joint_reward_ratio_mean": jr_ratio_stats["mean"],
                "joint_reward_ratio_sem": jr_ratio_stats["sem"],
                "judge_mean_rating_mean": judge_stats["mean"],
                "judge_mean_rating_sem": judge_stats["sem"],
            }
        )

    table_out = out_dir / f"table_secret_true__pv_{table_variant}.csv"
    write_csv(table_out, table_rows)
    logger.info("Wrote table: %s", table_out)

    # Plots: baseline vs SC(control) vs SC(simple), per model.
    plots_dir = out_dir / "plots" / f"by_model__{args.hist_metric}"
    ensure_dir(plots_dir)
    for model_label in sorted({r.model_label for r in rows}):
        subset = [r for r in rows if r.model_label == model_label]
        if not subset:
            continue
        safe_label = sanitize_filename(model_label)
        hist_out = plots_dir / f"{safe_label}__hist.png"
        bar_out = plots_dir / f"{safe_label}__mean_sem.png"
        _plot_hist_by_condition(
            rows=subset,
            metric_key=str(args.hist_metric),
            out_path=hist_out,
            bins=int(args.bins),
        )
        _plot_mean_sem_by_condition(
            rows=subset,
            metric_key=str(args.hist_metric),
            out_path=bar_out,
        )

    combined_out = out_dir / "plots" / f"combined_mean_sem__{args.hist_metric}__and_judge.png"
    _plot_combined_mean_sem_with_judge(
        rows=rows, metric_key=str(args.hist_metric), out_path=combined_out
    )
    logger.info("Wrote combined plot: %s", combined_out)

    combined_six_out = (
        out_dir / "plots" / f"combined_six_bars__{args.hist_metric}__and_judge.png"
    )
    _plot_combined_six_bars_dual_axis(
        rows=rows, metric_key=str(args.hist_metric), out_path=combined_six_out
    )
    if combined_six_out.exists():
        logger.info("Wrote 6-bars plot: %s", combined_six_out)

    logger.info("Wrote plots under: %s", plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
