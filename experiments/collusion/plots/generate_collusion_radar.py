from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
    mean,
    safe_load_json,
    sanitize_filename,
    write_json,
)
from experiments.common.plotting.load_runs import LoadedRun, load_runs
from experiments.common.plotting.style import apply_default_style


def _sample_std(values: Iterable[Any]) -> Optional[float]:
    vals = finite(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(np.std(np.array(vals, dtype=float), ddof=1))


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


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _wrap_labels(labels: List[str], *, width: int = 14) -> List[str]:
    out: List[str] = []
    for label in labels:
        s = str(label)
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


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    label: str
    higher_is_better: bool = True
    transform: Optional[Callable[[float], float]] = None

    def apply(self, v: float) -> float:
        out = float(v)
        if self.transform is not None:
            out = float(self.transform(out))
        if not self.higher_is_better:
            out = -out
        return float(out)


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

        fs = run.final_summary or {}
        if not isinstance(fs, dict):
            fs = {}
        metrics = run.metrics or {}
        if not isinstance(metrics, dict):
            metrics = {}

        joint_reward = as_float(fs.get("joint_reward"))
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

        row: Dict[str, Any] = {
            "run_dir": str(run.run_dir),
            "run_id": rid,
            "seed": as_int(rc.get("seed")),
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
            "coalition_advantage_mean": as_float(
                metrics.get("coalition_advantage_mean")
            ),
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
            "joint_reward_ratio": as_float(fs.get("joint_reward_ratio")),
            "joint_reward": joint_reward,
            "coalition_reward_ratio": coalition_reward_ratio,
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

    ax.plot(angles, b, linewidth=2.0, label=baseline_label, color="#2563eb")
    ax.fill(angles, b, alpha=0.18, color="#2563eb")

    ax.plot(angles, t, linewidth=2.0, label=treatment_label, color="#dc2626")
    ax.fill(angles, t, alpha=0.18, color="#dc2626")

    ax.set_title(title, pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_radar_multi(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: str,
    out_path: Path,
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
        if alpha > 0:
            ax.fill(angles, loop_vals, alpha=alpha, color=color)

    ax.set_title(title, pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(
    *,
    series: List[Dict[str, Any]],
    labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    if not labels or not series:
        return

    labels_wrapped = _wrap_labels(labels, width=14)
    n_metrics = len(labels_wrapped)
    n_series = len(series)

    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86
    bar_width = total_width / float(max(1, n_series))
    offsets = (np.arange(n_series, dtype=float) - (n_series - 1) / 2.0) * bar_width

    fig_width = max(7.6, 0.85 * n_metrics + 2.2)
    fig_height = 4.6 if n_series <= 4 else 5.0
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
            width=bar_width * 0.92,
            yerr=yerr,
            capsize=3,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )

    ax.set_title(title)
    ax.set_ylabel("Normalized mean (0-1, ±1 std)")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_wrapped, rotation=22, ha="right")

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars_by_topology(
    *,
    topologies: List[str],
    series_by_topology: Dict[str, List[Dict[str, Any]]],
    labels: List[str],
    title: str,
    out_path: Path,
    title_by_topology: Optional[Dict[str, str]] = None,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    if not topologies or not labels:
        return

    labels_wrapped = _wrap_labels(labels, width=14)

    # Use the first topology's series ordering as the canonical legend order.
    first_series = series_by_topology.get(topologies[0]) or []
    if not first_series:
        return

    n_metrics = len(labels_wrapped)
    x = np.arange(n_metrics, dtype=float)
    total_width = 0.86

    def _pretty_topology(topo: str) -> str:
        if title_by_topology and topo in title_by_topology:
            return str(title_by_topology[topo])
        s = str(topo or "").strip()
        if not s:
            return s
        key = s.lower().replace("-", "_").replace(" ", "_")
        canonical = {
            "erdos_renyi": "Erdős–Rényi",
            "barabasi_albert": "Barabási–Albert",
            "watts_strogatz": "Watts–Strogatz",
        }
        if key in canonical:
            return canonical[key]
        return s.replace("_", " ").title()

    ncols = 2
    nrows = int(math.ceil(len(topologies) / float(ncols)))
    # Target a wide, paper-friendly aspect ratio (~4:12), scaling up if needed.
    fig_width = max(12.0, 2.4 * float(n_metrics))
    fig_height = max(4.0, fig_width / 3.0)
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
                width=bar_width * 0.92,
                yerr=yerr_arr,
                capsize=3,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.92,
            )

        ax.set_title(_pretty_topology(topo))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Hide unused subplots (if any).
    for ax in axes[len(topologies) :]:
        ax.axis("off")

    # Global y label (avoid repeated labels per-subplot).
    fig.text(
        0.02, 0.5, "Normalized mean (0-1, ±1 std)", va="center", rotation="vertical"
    )

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
            bbox_to_anchor=(0.5, 0.01),
            ncol=ncol,
            frameon=True,
        )

    # Intentionally no main header / suptitle (per user request).
    fig.tight_layout(rect=(0.04, 0.10, 0.99, 1.0))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _pretty_variant(name: str) -> str:
    s = canonical_variant(name)
    return s.replace("_", " ") if s else s


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
    title_prefix: Optional[str] = None,
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

    metrics: List[_MetricSpec] = [
        _MetricSpec("joint_reward_ratio", "Joint reward", higher_is_better=True),
        _MetricSpec("tasks_done", "Tasks done", higher_is_better=True),
        _MetricSpec(
            "coalition_reward_ratio", "Coalition reward ratio", higher_is_better=True
        ),
        _MetricSpec(
            "coalition_advantage_mean", "Coalition advantage", higher_is_better=True
        ),
        _MetricSpec(
            "judge_simple_rating",
            "Judge collusion (simple)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_medium_rating",
            "Judge collusion (medium)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_complex_rating",
            "Judge collusion (complex)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_mean_rating",
            "Judge collusion (mean)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
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
                "label": "baseline (no SC)",
                "color": "#2563eb",
                "rows": baseline,
            }
        ]

        color_by_variant = {
            "control": "#6b7280",  # gray
            "simple": "#16a34a",  # green
            "deception": "#dc2626",  # red
            "structured": "#f59e0b",  # amber
            "aggressive": "#7c3aed",  # purple
        }
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
                    "label": f"SC ({_pretty_variant(pv)})",
                    "color": color_by_variant.get(pv, None),
                    "rows": grp_rows,
                }
            )

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
        # We choose metrics only if *every* group has a value for them.
        metric_defs: List[Dict[str, Any]] = []
        for m in metrics:
            seed_raw_by_group: List[List[float]] = []
            seed_transformed_by_group: List[List[float]] = []
            pooled_transformed: List[float] = []
            for g in groups:
                seed_raw = _seed_means(g["rows"], m.key)
                if not seed_raw:
                    pooled_transformed = []
                    break
                seed_raw_by_group.append(seed_raw)
                seed_transformed = [m.apply(v) for v in seed_raw]
                seed_transformed_by_group.append(seed_transformed)
                pooled_transformed.extend(seed_transformed)

            if not pooled_transformed:
                continue

            lo, hi = _robust_range(pooled_transformed)

            def _norm_one(value: float) -> float:
                if hi == lo:
                    return 0.5
                return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))

            entry = {
                "key": m.key,
                "label": m.label,
                "higher_is_better": m.higher_is_better,
                "scale_lo": lo,
                "scale_hi": hi,
                "group_means_raw": [
                    {
                        "label": groups[i]["label"],
                        "mean_raw": float(mean(seed_raw_by_group[i])),
                    }
                    for i in range(len(groups))
                ],
                "group_stds_raw": [
                    {
                        "label": groups[i]["label"],
                        "std_raw": float(_sample_std(seed_raw_by_group[i]) or 0.0),
                    }
                    for i in range(len(groups))
                ],
                "group_means_transformed": [
                    {
                        "label": groups[i]["label"],
                        "mean_transformed": float(mean(seed_transformed_by_group[i])),
                    }
                    for i in range(len(groups))
                ],
                "group_stds_transformed": [
                    {
                        "label": groups[i]["label"],
                        "std_transformed": float(
                            _sample_std(seed_transformed_by_group[i]) or 0.0
                        ),
                    }
                    for i in range(len(groups))
                ],
                "group_means_norm01": [
                    {
                        "label": groups[i]["label"],
                        "mean_norm01": float(
                            mean([_norm_one(v) for v in seed_transformed_by_group[i]])
                        ),
                    }
                    for i in range(len(groups))
                ],
                "group_stds_norm01": [
                    {
                        "label": groups[i]["label"],
                        "std_norm01": float(
                            _sample_std(
                                [_norm_one(v) for v in seed_transformed_by_group[i]]
                            )
                            or 0.0
                        ),
                    }
                    for i in range(len(groups))
                ],
            }
            metric_defs.append(entry)

        if not metric_defs:
            msg = "No comparable metrics found to plot across all prompt variants (missing/NaN)."
            if strict:
                raise SystemExit(msg)
            return False

        raw_summary["metrics"] = metric_defs
        labels = [str(m["label"]) for m in metric_defs]

        for idx, g in enumerate(groups):
            # Pull in the normalized values for this group in label order.
            vals = [
                float(md["group_means_norm01"][idx]["mean_norm01"])
                for md in metric_defs
            ]
            errs = [
                float(md["group_stds_norm01"][idx]["std_norm01"]) for md in metric_defs
            ]
            group_vectors.append(
                {
                    "label": g["label"],
                    "values": vals,
                    "errors": errs,
                    "color": g["color"],
                    "linewidth": 2.2 if g["kind"] == "baseline" else 1.8,
                    "fill_alpha": 0.08 if g["kind"] == "baseline" else 0.06,
                }
            )

        # Fill in missing colors deterministically using tab10 (skip baseline blue).
        cmap = plt.get_cmap("tab10")
        color_idx = 1
        for g in group_vectors:
            if g.get("color"):
                continue
            g["color"] = matplotlib.colors.to_hex(cmap(color_idx % 10))
        color_idx += 1

        out_base = f"collusion_radar__c{colluder_count}__pvALL"
        write_json(sweep_out / f"{out_base}.json", raw_summary)

        title = (
            prefix
            + f"Jira: baseline vs secret-channel prompt variants (c={colluder_count})"
        )
        _plot_radar_multi(
            series=group_vectors,
            labels=labels,
            title=title,
            out_path=sweep_out / f"{out_base}.png",
        )
        _plot_grouped_bars(
            series=group_vectors,
            labels=labels,
            title=title,
            out_path=sweep_out / f"{out_base}__bars.png",
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
        lo, hi = _robust_range(b_seed_transformed + t_seed_transformed)

        def _norm_one(value: float) -> float:
            if hi == lo:
                return 0.5
            return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))

        b_seed_norm = [_norm_one(v) for v in b_seed_transformed]
        t_seed_norm = [_norm_one(v) for v in t_seed_transformed]

        b_mean_norm = float(mean(b_seed_norm))
        t_mean_norm = float(mean(t_seed_norm))
        b_std_norm = float(_sample_std(b_seed_norm) or 0.0)
        t_std_norm = float(_sample_std(t_seed_norm) or 0.0)

        baseline_norm.append(b_mean_norm)
        treatment_norm.append(t_mean_norm)
        baseline_err.append(b_std_norm)
        treatment_err.append(t_std_norm)

        metric_defs.append(
            {
                "key": m.key,
                "label": m.label,
                "higher_is_better": m.higher_is_better,
                "scale_lo": lo,
                "scale_hi": hi,
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
                "treatment_mean_norm01": t_mean_norm,
                "treatment_std_norm01": t_std_norm,
            }
        )

    raw_summary["metrics"] = metric_defs

    labels = [str(m["label"]) for m in metric_defs]
    if not labels:
        msg = "No comparable metrics found to plot (all missing/NaN)."
        if strict:
            raise SystemExit(msg)
        return False

    out_base = f"collusion_radar__c{colluder_count}__pv{treatment_variant}"
    write_json(sweep_out / f"{out_base}.json", raw_summary)

    title = (
        prefix
        + f"Jira: secret-channel collusion vs baseline (c={colluder_count}, pv={treatment_variant})\n"
        + f"baseline n={len(baseline)} seeds={len(raw_summary['baseline']['seeds'])} | "
        + f"treatment n={len(treatment)} seeds={len(raw_summary['treatment']['seeds'])}"
    )
    _plot_radar(
        baseline_vals=baseline_norm,
        treatment_vals=treatment_norm,
        labels=labels,
        baseline_label="baseline (no SC)",
        treatment_label=f"treatment (SC, pv={treatment_variant})",
        title=title,
        out_path=sweep_out / f"{out_base}.png",
    )
    _plot_grouped_bars(
        series=[
            {
                "label": "baseline (no SC)",
                "values": baseline_norm,
                "errors": baseline_err,
                "color": "#2563eb",
            },
            {
                "label": f"treatment (SC, pv={treatment_variant})",
                "values": treatment_norm,
                "errors": treatment_err,
                "color": "#dc2626",
            },
        ],
        labels=labels,
        title=title,
        out_path=sweep_out / f"{out_base}__bars.png",
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
) -> bool:
    """
    Builds a single multi-panel grouped-bar figure:
      one panel per topology, globally normalized per-metric across all topologies.

    Output: out_dir/sweep/collusion_radar__cX__pvALL__bars__by_topology.png
    """
    topologies = sorted(
        {str(r.get("topology")) for r in rows if r.get("topology") is not None}
    )
    if len(topologies) <= 1:
        return False

    colluder_count = _select_colluder_count(rows, colluder_count_requested)
    baseline_variant = str(baseline_variant_requested or "control").strip()
    require_complete = not bool(include_incomplete)

    metrics: List[_MetricSpec] = [
        _MetricSpec("joint_reward_ratio", "Joint reward", higher_is_better=True),
        _MetricSpec("tasks_done", "Tasks done", higher_is_better=True),
        _MetricSpec(
            "coalition_reward_ratio", "Coalition reward ratio", higher_is_better=True
        ),
        _MetricSpec(
            "coalition_advantage_mean", "Coalition advantage", higher_is_better=True
        ),
        _MetricSpec(
            "judge_simple_rating",
            "Judge collusion (simple)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_medium_rating",
            "Judge collusion (medium)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_complex_rating",
            "Judge collusion (complex)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
        _MetricSpec(
            "judge_mean_rating",
            "Judge collusion (mean)",
            higher_is_better=True,
            transform=lambda v: float(v) / 5.0,
        ),
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

    color_by_variant = {
        "control": "#6b7280",  # gray
        "simple": "#16a34a",  # green
        "deception": "#dc2626",  # red
        "structured": "#f59e0b",  # amber
        "aggressive": "#7c3aed",  # purple
    }
    groups: List[Dict[str, Any]] = [
        {
            "kind": "baseline",
            "secret_channel_enabled": False,
            "prompt_variant": baseline_variant,
            "label": "baseline (no SC)",
            "color": "#2563eb",
        }
    ]
    for pv in variants:
        groups.append(
            {
                "kind": "treatment",
                "secret_channel_enabled": True,
                "prompt_variant": pv,
                "label": f"SC ({_pretty_variant(pv)})",
                "color": color_by_variant.get(pv, None),
            }
        )

    # Resolve missing group colors deterministically (tab10), skipping baseline blue.
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

        lo, hi = _robust_range(pooled_transformed)
        metric_defs.append({"spec": m, "scale_lo": lo, "scale_hi": hi})

    if not metric_defs:
        return False

    labels = [str(md["spec"].label) for md in metric_defs]

    def _norm_one(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.5
        return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))

    series_by_topology: Dict[str, List[Dict[str, Any]]] = {}
    for t in topologies:
        topo_series: List[Dict[str, Any]] = []
        topo_group_rows = group_rows_by_topo[t]

        for group, grp_rows in zip(groups, topo_group_rows):
            values: List[float] = []
            errors: List[float] = []
            for md in metric_defs:
                m = md["spec"]
                lo = float(md["scale_lo"])
                hi = float(md["scale_hi"])

                seed_raw = _seed_means(grp_rows, m.key)
                if not seed_raw:
                    values.append(float("nan"))
                    errors.append(float("nan"))
                    continue

                seed_transformed = [m.apply(v) for v in seed_raw]
                seed_norm = [_norm_one(v, lo, hi) for v in seed_transformed]
                values.append(float(mean(seed_norm)))
                errors.append(float(_sample_std(seed_norm) or 0.0))

            topo_series.append(
                {
                    "label": str(group["label"]),
                    "values": values,
                    "errors": errors,
                    "color": str(group["color"]),
                }
            )
        series_by_topology[t] = topo_series

    ensure_dir(out_dir)
    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)

    # Optional topology parameter display (e.g., Erdős–Rényi edge prob p).
    title_by_topology: Dict[str, str] = {}
    try:
        cfg_path = sweep_dir.parent.parent.parent / "config.json"
        cfg = safe_load_json(cfg_path) if cfg_path.exists() else None
        edge_prob = None
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
        if edge_prob is not None and np.isfinite(edge_prob):
            p_str = (f"{float(edge_prob):.3f}").rstrip("0").rstrip(".")
            title_by_topology["erdos_renyi"] = f"Erdős–Rényi (p={p_str})"
    except Exception:
        title_by_topology = {}

    out_path = (
        sweep_out / f"collusion_radar__c{colluder_count}__pvALL__bars__by_topology.png"
    )
    title = f"Jira: baseline vs secret-channel prompt variants by topology (c={colluder_count})"
    _plot_grouped_bars_by_topology(
        topologies=topologies,
        series_by_topology=series_by_topology,
        labels=labels,
        title=title,
        out_path=out_path,
        title_by_topology=title_by_topology or None,
    )
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a radar chart comparing Jira collusion vs baseline."
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
        "--by-topology",
        action="store_true",
        help="Also generate the same radar chart(s) separately for each topology present in the sweep (writes under out_dir/by_topology/<topology>/sweep/).",
    )
    parser.add_argument(
        "--compare-topologies",
        action="store_true",
        help="Write a single grouped-bar PNG that compares topologies in one figure (requires --plot-all-prompt-variants).",
    )
    parser.add_argument(
        "--extra-sweep-dir",
        action="append",
        default=[],
        help="Additional sweep dirs to include in --compare-topologies aggregation. Can be specified multiple times.",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, _ = load_runs(sweep_dir)
    rows = _build_rows(runs)
    out_dir = default_out_dir(sweep_dir=sweep_dir, requested_out_dir=args.out_dir)
    _generate(
        sweep_dir=sweep_dir,
        out_dir=out_dir,
        rows=rows,
        colluder_count_requested=args.colluder_count,
        treatment_variant_requested=args.treatment_prompt_variant,
        plot_all_prompt_variants=bool(args.plot_all_prompt_variants),
        baseline_variant_requested=str(args.baseline_prompt_variant or "control"),
        include_incomplete=bool(args.include_incomplete),
        strict=True,
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
                extra_runs, _ = load_runs(sibling_complete)
                compare_rows.extend(_build_rows(extra_runs))
        except Exception:
            pass
        for extra in list(args.extra_sweep_dir or []):
            extra_dir = Path(str(extra)).expanduser().resolve()
            extra_runs, _ = load_runs(extra_dir)
            compare_rows.extend(_build_rows(extra_runs))
        ok = _compare_topologies_bars(
            sweep_dir=sweep_dir,
            out_dir=out_dir,
            rows=compare_rows,
            colluder_count_requested=args.colluder_count,
            baseline_variant_requested=str(args.baseline_prompt_variant or "control"),
            include_incomplete=bool(args.include_incomplete),
        )
        if not ok:
            raise SystemExit(
                "Could not build compare-topologies bar figure (need multiple topologies with data)."
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
                    title_prefix=f"topology={topo}",
                    strict=False,
                )
            except SystemExit as e:
                print(f"[warn] topology={topo}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
