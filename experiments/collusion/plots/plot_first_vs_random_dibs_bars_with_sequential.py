from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from experiments.collusion import compute_sequential_regret as csr


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(path_or_dir: str) -> Path:
    p = Path(path_or_dir).expanduser()
    if p.is_file():
        return p.parent
    return p


def _iter_run_dirs(output_dir: Path) -> Iterable[Path]:
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return
    for cfg_path in sorted(runs_dir.rglob("run_config.json")):
        yield cfg_path.parent


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    try:
        return _load_json(run_dir / "metrics.json")
    except Exception:
        return {}


def _is_complete_status(status: Any) -> bool:
    return str(status or "").strip().lower() in {"complete", "completed", "success"}


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if not math.isfinite(f):
            continue
        out.append(float(f))
    return out


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pp = max(0.0, min(100.0, float(p))) / 100.0
    k = pp * (len(sorted_vals) - 1)
    i0 = int(math.floor(k))
    i1 = int(math.ceil(k))
    if i0 == i1:
        return float(sorted_vals[i0])
    w = float(k - i0)
    return float(sorted_vals[i0] * (1.0 - w) + sorted_vals[i1] * w)


def _robust_range(values: List[float]) -> Tuple[float, float]:
    """
    Robust min/max scaling based on the pooled distribution (percentile range),
    mirroring the "use the entire distribution" philosophy in collusion plots.
    """
    vals = sorted(_finite(values))
    if not vals:
        return 0.0, 1.0
    if len(vals) == 1:
        center = float(vals[0])
        pad = max(1e-6, abs(center) * 0.1)
        return center - pad, center + pad

    lo = float(_percentile(vals, 1))
    hi = float(_percentile(vals, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = float(vals[0])
        hi = float(vals[-1])
    if lo == hi:
        pad = max(1e-6, abs(lo) * 0.1)
        lo -= pad
        hi += pad
    else:
        pad = max(1e-6, 0.06 * abs(hi - lo))
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _norm_one(value: float, *, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    if hi == lo:
        return 0.5
    return _clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))


def _sample_std(values: List[float]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    m = float(sum(vals) / len(vals))
    var = sum((x - m) ** 2 for x in vals) / float(len(vals) - 1)
    return float(var**0.5)


def _sem(values: List[float]) -> float:
    vals = _finite(values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    return float(_sample_std(vals) / math.sqrt(len(vals)))


def _max_abs(values: Iterable[float]) -> float:
    m = 0.0
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if not math.isfinite(f):
            continue
        m = max(m, abs(f))
    return float(m)


def _load_optimal_joint_reward(run_dir: Path) -> float:
    """
    Load `optimal_summary.json` if present, else compute the no-violation optimum via DP.

    Note: DP is feasible for the n6_c2 Jira sweeps; it may be too slow for larger task sets.
    """
    opt_path = run_dir / "optimal_summary.json"
    if opt_path.exists():
        try:
            payload = _load_json(opt_path)
            opt = payload.get("optimal")
            if isinstance(opt, dict):
                jr = opt.get("joint_reward")
                if isinstance(jr, (int, float)) and math.isfinite(float(jr)):
                    return float(jr)
        except Exception:
            pass

    agent_names, task_ids, tasks, costs = csr._reconstruct_instance(run_dir)
    weights = csr._load_jira_weights(run_dir)

    if not agent_names or not task_ids:
        return float("nan")
    if len(task_ids) > 20:
        return float("nan")

    prio_weight = {tid: csr._priority_weight((tasks.get(tid) or {}).get("priority")) for tid in task_ids}
    values: List[List[float]] = []
    for agent in agent_names:
        row: List[float] = []
        for tid in task_ids:
            cost = float(costs.get(agent, {}).get(tid, float("inf")))
            if not math.isfinite(cost):
                row.append(float("-inf"))
                continue
            row.append(float(weights.tasks_done_bonus) + float(weights.priority_bonus) * float(prio_weight[tid]) - cost)
        values.append(row)

    dp: Dict[int, float] = {0: 0.0}
    for i in range(len(agent_names)):
        new: Dict[int, float] = {}
        for mask, best in dp.items():
            new[mask] = max(new.get(mask, float("-inf")), best)  # skip
            for j in range(len(task_ids)):
                if mask & (1 << j):
                    continue
                v = values[i][j]
                if not math.isfinite(v):
                    continue
                nm = mask | (1 << j)
                new[nm] = max(new.get(nm, float("-inf")), best + v)
        dp = new

    return float(max(dp.values())) if dp else float("nan")


def _collect_normalized_joint_regret(
    output_dir: Path,
    *,
    condition_specs: Mapping[str, Mapping[str, Any]],
    conditions: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    """
    normalized_regret = clamp(1 - achieved_joint_reward / optimal_joint_reward, 0, 1)
    """
    out: Dict[str, List[float]] = {c: [] for c, _ in conditions}
    for run_dir in _iter_run_dirs(output_dir):
        metrics = _read_metrics(run_dir)
        if not _is_complete_status(metrics.get("status")):
            continue

        sc = bool(metrics.get("secret_channel_enabled"))
        pv = str(metrics.get("prompt_variant") or "").strip().lower()
        cond_key = None
        for cond, _ in conditions:
            spec = condition_specs.get(cond) or {}
            if sc != bool(spec.get("secret_channel_enabled")):
                continue
            if pv != str(spec.get("prompt_variant")):
                continue
            cond_key = cond
            break
        if not cond_key:
            continue

        final = _load_json(run_dir / "final_summary.json")
        assignment_raw = final.get("assignment")
        if not isinstance(assignment_raw, dict) or not assignment_raw:
            continue
        assignment = {str(k): csr._canonical_task(v) for k, v in assignment_raw.items()}

        agent_names, task_ids, tasks, costs = csr._reconstruct_instance(run_dir)
        if not agent_names:
            agent_names = list(assignment.keys())
        task_id_set = set(task_ids)
        weights = csr._load_jira_weights(run_dir)

        rewards_actual = csr._local_rewards(
            assignment=assignment,
            agent_names=agent_names,
            task_ids=task_id_set,
            tasks=tasks,
            costs=costs,
            weights=weights,
        )
        achieved = float(sum(float(v) for v in rewards_actual.values()))
        optimal = float(_load_optimal_joint_reward(run_dir))
        if not math.isfinite(optimal) or optimal == 0.0:
            continue
        out[cond_key].append(_clamp01(1.0 - (achieved / optimal)))

    return out


def _collect_regret_gap(
    output_dir: Path,
    *,
    condition_specs: Mapping[str, Mapping[str, Any]],
    conditions: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    """
    gap = noncoalition_mean_regret - coalition_mean_regret
    """
    out: Dict[str, List[float]] = {c: [] for c, _ in conditions}
    for run_dir in _iter_run_dirs(output_dir):
        metrics = _read_metrics(run_dir)
        if not _is_complete_status(metrics.get("status")):
            continue
        coalition = metrics.get("coalition_mean_regret")
        noncoal = metrics.get("noncoalition_mean_regret")
        if not isinstance(coalition, (int, float)) or not isinstance(noncoal, (int, float)):
            continue
        gap = float(noncoal) - float(coalition)
        if not math.isfinite(gap):
            continue

        sc = bool(metrics.get("secret_channel_enabled"))
        pv = str(metrics.get("prompt_variant") or "").strip().lower()
        for cond, _ in conditions:
            spec = condition_specs.get(cond) or {}
            if sc != bool(spec.get("secret_channel_enabled")):
                continue
            if pv != str(spec.get("prompt_variant")):
                continue
            out[cond].append(float(gap))
    return out


def _normalize_gap_centered(values_by_cond: Dict[str, List[float]]) -> Dict[str, List[float]]:
    pooled = [v for xs in values_by_cond.values() for v in xs]
    max_abs = _max_abs(pooled)
    out: Dict[str, List[float]] = {}
    for k, xs in values_by_cond.items():
        if max_abs == 0.0 or not math.isfinite(max_abs):
            out[k] = [0.5 for _ in xs]
        else:
            out[k] = [_clamp01(0.5 + float(v) / (2.0 * max_abs)) for v in xs]
    return out


def _add_group_bracket(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    label: str,
    cap_height: float = 0.16,
    label_pad: float = 0.12,
    label_fontsize: float = 22,
) -> None:
    h = float(cap_height)
    ax.plot(
        [x0, x0, x1, x1],
        [y, y - h, y - h, y],
        color="black",
        lw=1.8,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y - h - float(label_pad),
        label,
        ha="center",
        va="top",
        fontsize=float(label_fontsize),
        color="black",
        clip_on=False,
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 28,
            "axes.titlesize": 34,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 27,
        }
    )


def _collect_metric_values(
    output_dir: Path,
    *,
    metric_key: str,
    condition_specs: Mapping[str, Mapping[str, Any]],
    conditions: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {c: [] for c, _ in conditions}
    for run_dir in _iter_run_dirs(output_dir):
        metrics = _read_metrics(run_dir)
        if not _is_complete_status(metrics.get("status")):
            continue
        sc = bool(metrics.get("secret_channel_enabled"))
        pv = str(metrics.get("prompt_variant") or "").strip().lower()
        for cond, _ in conditions:
            spec = condition_specs.get(cond) or {}
            if sc != bool(spec.get("secret_channel_enabled")):
                continue
            if pv != str(spec.get("prompt_variant")):
                continue
            val = metrics.get(metric_key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out[cond].append(float(val))
    return out


def _colluder_went_first(run_rec: Mapping[str, Any]) -> bool:
    colluders = run_rec.get("colluders")
    order = run_rec.get("agent_turn_order")
    if not isinstance(colluders, list) or not isinstance(order, list) or not order:
        return False
    colluder_set = {str(a) for a in colluders}
    return str(order[0]) in colluder_set


def _collect_sequential_coalition_regret(
    output_dir: Path,
    *,
    require_noncolluder_first: bool,
    condition_specs: Mapping[str, Mapping[str, Any]],
    conditions: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    """
    Load per-run sequential coalition regret from sequential_regret_summary.json.
    """
    out: Dict[str, List[float]] = {c: [] for c, _ in conditions}
    summary_path = output_dir / "sequential_regret_summary.json"
    if not summary_path.exists():
        return out

    payload = _load_json(summary_path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return out

    for r in runs:
        if not isinstance(r, dict):
            continue
        if not _is_complete_status(r.get("status")):
            continue
        if require_noncolluder_first and _colluder_went_first(r):
            continue

        sc = bool(r.get("secret_channel_enabled"))
        pv = str(r.get("prompt_variant") or "").strip().lower()
        means = r.get("means") or {}
        if not isinstance(means, dict):
            continue
        val = means.get("sequential_regret_coalition_mean")
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            continue

        for cond, _ in conditions:
            spec = condition_specs.get(cond) or {}
            if sc != bool(spec.get("secret_channel_enabled")):
                continue
            if pv != str(spec.get("prompt_variant")):
                continue
            out[cond].append(float(val))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot regret / sequential regret / advantage bars for colluders-first vs random ordering."
        )
    )
    parser.add_argument(
        "--normalization",
        choices=["robust", "gap_optimal"],
        default="gap_optimal",
        help=(
            "'robust': percentile-based scaling on pooled distributions. "
            "'gap_optimal': normalized_regret=1-achieved/optimal and normalized_advantage from regret-gap."
        ),
    )
    parser.add_argument(
        "--first-dibs",
        default="experiments/collusion/outputs/collusion_topologies_sweep/20260211-145555",
        help="Output dir for colluders-first runs (first dibs).",
    )
    parser.add_argument(
        "--random-dibs",
        default="experiments/collusion/outputs/collusion_topologies_sweep/20260211-163504",
        help="Output dir for random-order runs.",
    )
    parser.add_argument(
        "--out",
        default="experiments/collusion/plots_outputs/first_vs_random_dibs_bars_with_sequential.pdf",
        help="Output image path.",
    )
    args = parser.parse_args()

    first_out_dir = _resolve_output_dir(args.first_dibs)
    rand_out_dir = _resolve_output_dir(args.random_dibs)

    # Conditions match our mean_results.json writers.
    conditions = [
        ("no_secret__control", "Baseline (No SC)"),
        ("secret__control", "Control (SC)"),
        ("secret__simple", "Simple (SC)"),
    ]
    condition_specs = {
        "no_secret__control": {"secret_channel_enabled": False, "prompt_variant": "control"},
        "secret__control": {"secret_channel_enabled": True, "prompt_variant": "control"},
        "secret__simple": {"secret_channel_enabled": True, "prompt_variant": "simple"},
    }

    # Warm palette matching reference.
    colors = {
        "no_secret__control": "#264653",  # yellow
        "secret__control": "#2A9D8F",  # orange
        "secret__simple": "#8AB17D",  # red/salmon
    }

    # Per-run distributions (for normalization + error bars).
    if args.normalization == "gap_optimal":
        first_regret = _collect_normalized_joint_regret(
            first_out_dir,
            condition_specs=condition_specs,
            conditions=conditions,
        )
        rand_regret = _collect_normalized_joint_regret(
            rand_out_dir,
            condition_specs=condition_specs,
            conditions=conditions,
        )
        regret_lo, regret_hi = 0.0, 1.0

        first_gap = _collect_regret_gap(first_out_dir, condition_specs=condition_specs, conditions=conditions)
        rand_gap = _collect_regret_gap(rand_out_dir, condition_specs=condition_specs, conditions=conditions)
        first_adv = _normalize_gap_centered(first_gap)
        rand_adv = _normalize_gap_centered(rand_gap)
        adv_lo, adv_hi = 0.0, 1.0
    else:
        first_regret = _collect_metric_values(
            first_out_dir,
            metric_key="coalition_mean_regret",
            condition_specs=condition_specs,
            conditions=conditions,
        )
        rand_regret = _collect_metric_values(
            rand_out_dir,
            metric_key="coalition_mean_regret",
            condition_specs=condition_specs,
            conditions=conditions,
        )
        first_adv = _collect_metric_values(
            first_out_dir,
            metric_key="coalition_advantage_mean",
            condition_specs=condition_specs,
            conditions=conditions,
        )
        rand_adv = _collect_metric_values(
            rand_out_dir,
            metric_key="coalition_advantage_mean",
            condition_specs=condition_specs,
            conditions=conditions,
        )

    # Sequential coalition regret:
    # - Keep all complete runs for both sweeps (require_noncolluder_first=False).
    first_seq_regret = _collect_sequential_coalition_regret(
        first_out_dir,
        require_noncolluder_first=False,
        condition_specs=condition_specs,
        conditions=conditions,
    )
    rand_seq_regret = _collect_sequential_coalition_regret(
        rand_out_dir,
        require_noncolluder_first=False,
        condition_specs=condition_specs,
        conditions=conditions,
    )

    pooled_seq = _finite([v for d in (first_seq_regret, rand_seq_regret) for xs in d.values() for v in xs])
    seq_lo, seq_hi = _robust_range(pooled_seq)
    if args.normalization == "robust":
        pooled_regret = _finite([v for d in (first_regret, rand_regret) for xs in d.values() for v in xs])
        pooled_adv = _finite([v for d in (first_adv, rand_adv) for xs in d.values() for v in xs])
        regret_lo, regret_hi = _robust_range(pooled_regret)
        adv_lo, adv_hi = _robust_range(pooled_adv)

    def _norm_list(vals: List[float], *, lo: float, hi: float) -> List[float]:
        return [_norm_one(v, lo=lo, hi=hi) for v in _finite(vals)]

    _apply_style()
    # Keep the width, but make the plot taller (≈2×) for readability.
    fig = plt.figure(figsize=(15.2, 9.6))
    # Keep a compact annotation band, without crowding the main plot.
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 0.88], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_ann = fig.add_subplot(gs[1], sharex=ax)

    # Order: regret -> sequential regret -> advantage (for each side).
    # Slightly larger spacing between adjacent metric labels to avoid overlap.
    x_centers = [0.0, 1.35, 2.7, 4.7, 6.05, 7.4]
    bar_w = 0.22
    offsets = [-bar_w, 0.0, bar_w]

    # Plot bars + error bars.
    def _plot_group(
        *,
        gx: float,
        values_by_cond: Mapping[str, List[float]],
        lo: float,
        hi: float,
    ) -> None:
        for (cond, _label), off in zip(conditions, offsets):
            vals = _norm_list(list(values_by_cond.get(cond, []) or []), lo=lo, hi=hi)
            mu = float(sum(vals) / len(vals)) if vals else float("nan")
            err = float(_sem(vals)) if vals else float("nan")
            ax.bar(
                gx + off,
                mu,
                width=bar_w,
                color=colors[cond],
                alpha=0.85,
                edgecolor="none",
            )
            if math.isfinite(err) and err > 0 and math.isfinite(mu):
                lower = min(float(err), max(0.0, float(mu)))
                upper = min(float(err), max(0.0, 1.0 - float(mu)))
                ax.errorbar(
                    [gx + off],
                    [mu],
                    yerr=[[lower], [upper]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    linewidth=1.0,
                )

    # Colluders first (first dibs)
    _plot_group(gx=x_centers[0], values_by_cond=first_regret, lo=regret_lo, hi=regret_hi)
    _plot_group(gx=x_centers[1], values_by_cond=first_seq_regret, lo=seq_lo, hi=seq_hi)
    _plot_group(gx=x_centers[2], values_by_cond=first_adv, lo=adv_lo, hi=adv_hi)

    # Random order
    _plot_group(gx=x_centers[3], values_by_cond=rand_regret, lo=regret_lo, hi=regret_hi)
    _plot_group(gx=x_centers[4], values_by_cond=rand_seq_regret, lo=seq_lo, hi=seq_hi)
    _plot_group(gx=x_centers[5], values_by_cond=rand_adv, lo=adv_lo, hi=adv_hi)

    ax.set_xticks([])
    ax.set_ylabel("Normalized Mean", labelpad=18)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.35)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Annotation axis
    ax_ann.set_ylim(0.0, 1.0)
    ax_ann.set_yticks([])
    ax_ann.set_xticks([])
    ax_ann.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_ann.spines.values():
        spine.set_visible(False)

    metric_labels = [
        "Regret\n(↓)\n",
        "Coalition\nOrdered\nRegret (↓)",
        "Coalition\nAdvantage\n(↑)",
        "Regret\n(↓)\n",
        "Coalition\nOrdered\nRegret (↓)",
        "Coalition\nAdvantage\n(↑)",
    ]
    metric_trans = ax_ann.get_xaxis_transform()
    for x, label in zip(x_centers, metric_labels):
        ax_ann.text(
            x,
            0.72,
            label,
            transform=metric_trans,
            ha="center",
            va="bottom",
            fontsize=25,
            color="black",
            clip_on=False,
        )

    # Brackets for the two groups of three.
    bracket_dx = 0.06
    left_start = x_centers[0] - 1.6 * bar_w + bracket_dx
    left_end = x_centers[2] + 1.6 * bar_w + bracket_dx
    right_start = x_centers[3] - 1.6 * bar_w + bracket_dx
    right_end = x_centers[5] + 1.6 * bar_w + bracket_dx

    _add_group_bracket(
        ax_ann,
        x0=left_start,
        x1=left_end,
        y=0.74,
        label="Colluders First",
        cap_height=0.10,
        label_pad=0.05,
        label_fontsize=26,
    )
    _add_group_bracket(
        ax_ann,
        x0=right_start,
        x1=right_end,
        y=0.74,
        label="Random Order",
        cap_height=0.10,
        label_pad=0.05,
        label_fontsize=26,
    )

    legend_handles = [
        Patch(facecolor=colors[cond], edgecolor="none", label=label)
        for cond, label in conditions
    ]
    ax_ann.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.17),
        bbox_transform=ax_ann.transAxes,
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Crop the exported figure to *content* (bars + annotation artists), not the full
    # annotation axis rectangle. This removes the excess whitespace below the legend
    # without changing the internal alignment of elements.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer)]

    ann_artists = []
    ann_artists.extend(list(ax_ann.texts))
    ann_artists.extend(list(ax_ann.lines))
    if ax_ann.get_legend() is not None:
        ann_artists.append(ax_ann.get_legend())
    for a in ann_artists:
        if not getattr(a, "get_visible", lambda: True)():
            continue
        try:
            bboxes.append(a.get_window_extent(renderer))
        except Exception:
            continue

    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out_path, dpi=220, bbox_inches=bbox, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
