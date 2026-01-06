from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.plotting.io_utils import ensure_dir, groupby, mean, sem


def _style() -> None:
    # Avoid hard dependency on seaborn; matplotlib ships these styles.
    for style in ("seaborn-v0_8-whitegrid", "seaborn-v0_8", "ggplot"):
        try:
            plt.style.use(style)
            break
        except Exception:
            continue
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _finite(values: Iterable[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if np.isfinite(f):
            out.append(float(f))
    return out


def _sorted_unique(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    values = {r.get(key) for r in rows}
    values.discard(None)
    try:
        return sorted(values)
    except Exception:
        return sorted(values, key=lambda x: str(x))


def _compute_baseline_joint_reward_ratio(run_rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], float]:
    """
    Baseline is adversary_count == 0 within (topology, num_agents, seed).
    """
    baseline: Dict[Tuple[Any, ...], float] = {}
    for r in run_rows:
        if r.get("adversary_count") != 0:
            continue
        key = (r.get("topology"), r.get("num_agents"), r.get("seed"))
        val = r.get("joint_reward_ratio")
        if val is None:
            continue
        try:
            baseline[key] = float(val)
        except Exception:
            continue
    return baseline


def build_run_stats(
    *,
    run_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produces one row per run_id with:
      - joint_reward_ratio, delta vs baseline
      - propagation_rate_misinfo_non_adversary mean/max across targets
      - misinformation_messages / total_messages mean across targets
      - fraction of misinformation_posts attributable to non-adversaries (mean across targets)
    """
    by_run_targets = groupby(target_rows, ("run_id",))
    by_run_agents = groupby(agent_rows, ("run_id", "target_agent"))
    baseline = _compute_baseline_joint_reward_ratio(run_rows)

    out: List[Dict[str, Any]] = []
    for rr in run_rows:
        run_id = rr.get("run_id")
        if not run_id:
            continue
        key = (rr.get("topology"), rr.get("num_agents"), rr.get("seed"))
        base = baseline.get(key)
        jrr = rr.get("joint_reward_ratio")
        delta = None
        if base is not None and jrr is not None:
            try:
                delta = float(jrr) - float(base)
            except Exception:
                delta = None

        targets = by_run_targets.get((run_id,), [])
        prop_vals = _finite([t.get("propagation_rate_misinfo_non_adversary") for t in targets])
        msg_rate_vals: List[float] = []
        for t in targets:
            mm = t.get("misinformation_messages")
            tm = t.get("total_messages")
            try:
                if mm is None or tm is None:
                    continue
                tm_f = float(tm)
                if tm_f <= 0:
                    continue
                msg_rate_vals.append(float(mm) / tm_f)
            except Exception:
                continue

        non_adv_share_vals: List[float] = []
        # agent_rows are keyed by (run_id, target_agent)
        if targets:
            for t in targets:
                ta = t.get("target_agent") or t.get("target_agent")  # might already be present
                # Our flattened target rows include "target_agent" at top-level.
                ta = t.get("target_agent") if ta is None else ta
                if ta is None:
                    continue
                agents = by_run_agents.get((run_id, ta), [])
                if not agents:
                    continue
                adv_posts = 0
                non_adv_posts = 0
                for a in agents:
                    posts = a.get("misinformation_posts") or 0
                    try:
                        posts_i = int(posts)
                    except Exception:
                        continue
                    if str(a.get("role")) == "adversary":
                        adv_posts += posts_i
                    else:
                        non_adv_posts += posts_i
                denom = adv_posts + non_adv_posts
                if denom > 0:
                    non_adv_share_vals.append(non_adv_posts / denom)

        row = dict(rr)
        row.update(
            {
                "delta_joint_reward_ratio": delta,
                "prop_misinfo_mean": mean(prop_vals) if prop_vals else None,
                "prop_misinfo_max": max(prop_vals) if prop_vals else None,
                "misinfo_message_rate_mean": mean(msg_rate_vals) if msg_rate_vals else None,
                "misinfo_non_adv_post_share_mean": mean(non_adv_share_vals) if non_adv_share_vals else None,
            }
        )
        out.append(row)
    return out


def plot_sweep_summary(
    run_stats: List[Dict[str, Any]],
    *,
    out_path: Path,
    use_max_propagation: bool = False,
) -> None:
    _style()
    ensure_dir(out_path.parent)

    topologies = _sorted_unique(run_stats, "topology")
    if not topologies:
        return

    adv_counts = _sorted_unique(run_stats, "adversary_count")
    prop_key = "prop_misinfo_max" if use_max_propagation else "prop_misinfo_mean"

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(topologies),
        figsize=(4.2 * len(topologies), 6.0),
        sharex=True,
    )
    if len(topologies) == 1:
        axes = np.array(axes).reshape(2, 1)

    colors = plt.get_cmap("tab10")

    for col, topo in enumerate(topologies):
        subset = [r for r in run_stats if r.get("topology") == topo]
        by_adv = groupby(subset, ("adversary_count",))

        # Row 0: joint reward ratio
        ax0 = axes[0, col]
        for i, a in enumerate(adv_counts):
            rows = by_adv.get((a,), [])
            ys = _finite([r.get("joint_reward_ratio") for r in rows])
            if not ys:
                continue
            # scatter per seed
            for j, r in enumerate(rows):
                y = r.get("joint_reward_ratio")
                if y is None:
                    continue
                try:
                    yv = float(y)
                except Exception:
                    continue
                xj = float(a) + (j - (len(rows) - 1) / 2) * 0.06
                ax0.scatter(xj, yv, s=18, alpha=0.55, color=colors(i % 10))
            ax0.errorbar(
                [a],
                [mean(ys)],
                yerr=[sem(ys)],
                fmt="o",
                ms=6,
                color="black",
                capsize=3,
                zorder=5,
            )
        # mean line across adversary_count
        xs_line: List[float] = []
        ys_line: List[float] = []
        for a in adv_counts:
            ys = _finite([r.get("joint_reward_ratio") for r in by_adv.get((a,), [])])
            if ys:
                xs_line.append(float(a))
                ys_line.append(mean(ys))
        if xs_line:
            ax0.plot(xs_line, ys_line, color="black", linewidth=1.5, alpha=0.8)
        ax0.set_title(f"{topo}: Joint Utility")
        ax0.set_xlabel("Adversary Count")
        ax0.set_ylabel("Joint Reward Ratio")
        ax0.set_ylim(0.0, 1.05)

        # Row 1: propagation rate
        ax1 = axes[1, col]
        for i, a in enumerate(adv_counts):
            rows = by_adv.get((a,), [])
            ys = _finite([r.get(prop_key) for r in rows])
            if not ys:
                continue
            for j, r in enumerate(rows):
                y = r.get(prop_key)
                if y is None:
                    continue
                try:
                    yv = float(y)
                except Exception:
                    continue
                xj = float(a) + (j - (len(rows) - 1) / 2) * 0.06
                ax1.scatter(xj, yv, s=18, alpha=0.55, color=colors(i % 10))
            ax1.errorbar(
                [a],
                [mean(ys)],
                yerr=[sem(ys)],
                fmt="o",
                ms=6,
                color="black",
                capsize=3,
                zorder=5,
            )
        xs_line = []
        ys_line = []
        for a in adv_counts:
            ys = _finite([r.get(prop_key) for r in by_adv.get((a,), [])])
            if ys:
                xs_line.append(float(a))
                ys_line.append(mean(ys))
        if xs_line:
            ax1.plot(xs_line, ys_line, color="black", linewidth=1.5, alpha=0.8)
        ax1.set_title(f"{topo}: Misinformation Propagation")
        ax1.set_xlabel("Adversary Count")
        ax1.set_ylabel("Misinformation Propagation Rate (Non-adversaries)")
        ax1.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sweep_slopes(
    run_stats: List[Dict[str, Any]],
    *,
    out_path: Path,
    use_max_propagation: bool = False,
) -> None:
    _style()
    ensure_dir(out_path.parent)

    topologies = _sorted_unique(run_stats, "topology")
    if not topologies:
        return
    adv_counts = _sorted_unique(run_stats, "adversary_count")
    seeds = _sorted_unique(run_stats, "seed")
    prop_key = "prop_misinfo_max" if use_max_propagation else "prop_misinfo_mean"

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(topologies),
        figsize=(4.2 * len(topologies), 6.0),
        sharex=True,
    )
    if len(topologies) == 1:
        axes = np.array(axes).reshape(2, 1)

    cmap = plt.get_cmap("tab10")

    for col, topo in enumerate(topologies):
        subset = [r for r in run_stats if r.get("topology") == topo]
        by_seed = groupby(subset, ("seed",))

        ax0 = axes[0, col]
        for i, s in enumerate(seeds):
            rows = [r for r in by_seed.get((s,), []) if r.get("adversary_count") in adv_counts]
            rows = sorted(rows, key=lambda r: r.get("adversary_count") or -1)
            xs = [r.get("adversary_count") for r in rows]
            ys = [r.get("delta_joint_reward_ratio") for r in rows]
            xs_f: List[float] = []
            ys_f: List[float] = []
            for x, y in zip(xs, ys):
                if x is None or y is None:
                    continue
                try:
                    xs_f.append(float(x))
                    ys_f.append(float(y))
                except Exception:
                    continue
            if xs_f:
                ax0.plot(xs_f, ys_f, marker="o", linewidth=1.2, alpha=0.7, color=cmap(i % 10), label=f"seed {s}")
        ax0.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax0.set_title(f"{topo}: Δ Joint Reward Ratio vs a0")
        ax0.set_xlabel("Adversary Count")
        ax0.set_ylabel("Δ Joint Reward Ratio")

        ax1 = axes[1, col]
        for i, s in enumerate(seeds):
            rows = [r for r in by_seed.get((s,), []) if r.get("adversary_count") in adv_counts]
            rows = sorted(rows, key=lambda r: r.get("adversary_count") or -1)
            xs = [r.get("adversary_count") for r in rows]
            ys = [r.get(prop_key) for r in rows]
            xs_f = []
            ys_f = []
            for x, y in zip(xs, ys):
                if x is None or y is None:
                    continue
                try:
                    xs_f.append(float(x))
                    ys_f.append(float(y))
                except Exception:
                    continue
            if xs_f:
                ax1.plot(xs_f, ys_f, marker="o", linewidth=1.2, alpha=0.7, color=cmap(i % 10))
        ax1.set_title(f"{topo}: Propagation vs Adversary Count")
        ax1.set_xlabel("Adversary Count")
        ax1.set_ylabel("Misinformation Propagation Rate (Non-adversaries)")
        ax1.set_ylim(0.0, 1.05)

        if col == len(topologies) - 1 and seeds:
            ax0.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmaps(run_stats: List[Dict[str, Any]], *, out_dir: Path) -> None:
    _style()
    ensure_dir(out_dir)

    topologies = _sorted_unique(run_stats, "topology")
    adv_counts = _sorted_unique(run_stats, "adversary_count")
    if not topologies or not adv_counts:
        return

    def _matrix(value_key: str) -> np.ndarray:
        mat = np.full((len(topologies), len(adv_counts)), np.nan, dtype=float)
        for i, topo in enumerate(topologies):
            for j, a in enumerate(adv_counts):
                vals = _finite([r.get(value_key) for r in run_stats if r.get("topology") == topo and r.get("adversary_count") == a])
                if vals:
                    mat[i, j] = mean(vals)
        return mat

    matrices = [
        ("delta_joint_reward_ratio", "Δ Joint Reward Ratio vs a0", "heatmap_delta_joint_reward_ratio.png", "coolwarm"),
        ("prop_misinfo_mean", "Propagation rate (mean across targets)", "heatmap_propagation_rate.png", "viridis"),
        ("misinfo_message_rate_mean", "Misinformation message rate (mean)", "heatmap_misinfo_message_rate.png", "magma"),
    ]

    for key, title, fname, cmap in matrices:
        mat = _matrix(key)
        fig, ax = plt.subplots(figsize=(1.6 * len(adv_counts) + 2.5, 0.8 * len(topologies) + 1.8))
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(adv_counts)), [str(a) for a in adv_counts])
        ax.set_yticks(range(len(topologies)), [str(t) for t in topologies])
        ax.set_xlabel("Adversary Count")
        ax.set_ylabel("Topology")
        # annotate
        for i in range(len(topologies)):
            for j in range(len(adv_counts)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmap in {"magma"} else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)


def plot_tradeoff_scatter(run_stats: List[Dict[str, Any]], *, out_path: Path) -> None:
    _style()
    ensure_dir(out_path.parent)

    rows = [r for r in run_stats if r.get("delta_joint_reward_ratio") is not None and r.get("prop_misinfo_mean") is not None]
    if not rows:
        return

    topologies = _sorted_unique(rows, "topology")
    topo_to_color = {t: plt.get_cmap("tab10")(i % 10) for i, t in enumerate(topologies)}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for t in topologies:
        sub = [r for r in rows if r.get("topology") == t]
        xs = [float(r["prop_misinfo_mean"]) for r in sub if r.get("prop_misinfo_mean") is not None]
        ys = [float(r["delta_joint_reward_ratio"]) for r in sub if r.get("delta_joint_reward_ratio") is not None]
        ss = []
        for r in sub:
            a = r.get("adversary_count") or 0
            try:
                ss.append(40 + 35 * float(a))
            except Exception:
                ss.append(40)
        ax.scatter(xs, ys, s=ss[: len(xs)], alpha=0.75, label=str(t), color=topo_to_color.get(t))

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Tradeoff: Misinformation Propagation vs Δ Joint Reward Ratio")
    ax.set_xlabel("Misinformation Propagation Rate (Non-adversaries, Mean Across Targets)")
    ax.set_ylabel("Δ Joint Reward Ratio (vs a0 baseline)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_belief_composition(agent_rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    """
    Stacked bars of mean fraction of non-adversary agents who believe:
      - misinformation
      - truth
      - neither/unknown
    Aggregated across (run_id, target_agent) groups.
    """
    _style()
    ensure_dir(out_path.parent)

    # Group by run_id+target_agent to compute fractions.
    by_rt = groupby(agent_rows, ("run_id", "target_agent"))
    records: List[Dict[str, Any]] = []
    for (run_id, target_agent), agents in by_rt.items():
        if not run_id or not target_agent:
            continue
        any_row = agents[0]
        topo = any_row.get("topology")
        a = any_row.get("adversary_count")

        non_adv = [x for x in agents if str(x.get("role")) != "adversary"]
        if not non_adv:
            continue
        mis = sum(1 for x in non_adv if bool(x.get("believes_misinformation")))
        tru = sum(1 for x in non_adv if bool(x.get("believes_truth")))
        # Some judges might mark both/none; keep an "other" bucket.
        other = max(0, len(non_adv) - mis - tru)
        records.append(
            {
                "topology": topo,
                "adversary_count": a,
                "frac_mis": mis / len(non_adv),
                "frac_truth": tru / len(non_adv),
                "frac_other": other / len(non_adv),
            }
        )

    if not records:
        return

    topologies = _sorted_unique(records, "topology")
    adv_counts = _sorted_unique(records, "adversary_count")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(topologies),
        figsize=(4.2 * len(topologies), 3.6),
        sharey=True,
    )
    if len(topologies) == 1:
        axes = np.array([axes])

    colors = {"frac_mis": "#d62728", "frac_truth": "#2ca02c", "frac_other": "#7f7f7f"}

    for col, topo in enumerate(topologies):
        ax = axes[col]
        sub = [r for r in records if r.get("topology") == topo]
        by_adv = groupby(sub, ("adversary_count",))
        xs = [float(a) for a in adv_counts]
        mis_means = []
        tru_means = []
        oth_means = []
        for a in adv_counts:
            rows = by_adv.get((a,), [])
            mis_means.append(mean(_finite([x.get("frac_mis") for x in rows])) if rows else float("nan"))
            tru_means.append(mean(_finite([x.get("frac_truth") for x in rows])) if rows else float("nan"))
            oth_means.append(mean(_finite([x.get("frac_other") for x in rows])) if rows else float("nan"))
        mis = np.array(mis_means, dtype=float)
        tru = np.array(tru_means, dtype=float)
        oth = np.array(oth_means, dtype=float)
        ax.bar(xs, mis, color=colors["frac_mis"], label="misinfo", alpha=0.85)
        ax.bar(xs, tru, bottom=mis, color=colors["frac_truth"], label="truth", alpha=0.85)
        ax.bar(xs, oth, bottom=mis + tru, color=colors["frac_other"], label="other/unknown", alpha=0.85)
        ax.set_title(str(topo))
        ax.set_xlabel("Adversary Count")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xs, [str(a) for a in adv_counts])
        if col == 0:
            ax.set_ylabel("Fraction of Non-adversary Agents")
        if col == len(topologies) - 1:
            ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confidence_by_belief(agent_rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    """
    Boxplots of judge confidence for non-adversary agents, split by belief.
    One subplot per topology; x axis is adversary_count.
    """
    _style()
    ensure_dir(out_path.parent)

    non_adv_rows = [r for r in agent_rows if str(r.get("role")) != "adversary" and r.get("judge_confidence") is not None]
    if not non_adv_rows:
        return

    topologies = _sorted_unique(non_adv_rows, "topology")
    adv_counts = _sorted_unique(non_adv_rows, "adversary_count")
    if not topologies or not adv_counts:
        return

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(topologies),
        figsize=(4.2 * len(topologies), 6.2),
        sharex=True,
        sharey=True,
    )
    if len(topologies) == 1:
        axes = np.array(axes).reshape(2, 1)

    def _data(topo: Any, a: Any, which: str) -> List[float]:
        sub = [r for r in non_adv_rows if r.get("topology") == topo and r.get("adversary_count") == a]
        if which == "misinfo":
            sub = [r for r in sub if bool(r.get("believes_misinformation"))]
        elif which == "truth":
            sub = [r for r in sub if bool(r.get("believes_truth"))]
        return _finite([r.get("judge_confidence") for r in sub])

    for col, topo in enumerate(topologies):
        for row, which in enumerate(["misinfo", "truth"]):
            ax = axes[row, col]
            data = [_data(topo, a, which) for a in adv_counts]
            # Replace empty lists with [nan] so matplotlib keeps positions.
            data = [d if d else [float("nan")] for d in data]
            positions = [float(a) for a in adv_counts]
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            color = "#d62728" if which == "misinfo" else "#2ca02c"
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            which_label = "Misinformation" if which == "misinfo" else "Truth"
            ax.set_title(f"{topo} ({which_label})")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Adversary Count")
            if col == 0:
                ax.set_ylabel("Judge Confidence")
            ax.set_xticks(positions, [str(a) for a in adv_counts])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_belief_by_distance(agent_rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    """
    Belief fraction vs distance-to-nearest-adversary.
    Produces 2 rows (misinfo/truth) × columns(topology).
    Each adversary_count is a line.
    """
    _style()
    ensure_dir(out_path.parent)

    rows = [r for r in agent_rows if str(r.get("role")) != "adversary" and r.get("distance_to_nearest_adversary") is not None]
    if not rows:
        return

    topologies = _sorted_unique(rows, "topology")
    adv_counts = _sorted_unique(rows, "adversary_count")
    adv_counts = [a for a in adv_counts if a not in (None, 0)]
    if not topologies or not adv_counts:
        return

    # Aggregate counts by (topology, adversary_count, distance)
    counts: Dict[Tuple[Any, Any, int], Dict[str, int]] = {}
    for r in rows:
        topo = r.get("topology")
        a = r.get("adversary_count")
        if a in (None, 0):
            continue
        try:
            dist = int(r.get("distance_to_nearest_adversary"))
        except Exception:
            continue
        key = (topo, a, dist)
        bucket = counts.setdefault(key, {"n": 0, "mis": 0, "tru": 0})
        bucket["n"] += 1
        if bool(r.get("believes_misinformation")):
            bucket["mis"] += 1
        if bool(r.get("believes_truth")):
            bucket["tru"] += 1

    max_dist = 0
    for (_, _, d) in counts.keys():
        max_dist = max(max_dist, int(d))
    if max_dist <= 0:
        return

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(topologies),
        figsize=(4.2 * len(topologies), 6.2),
        sharex=True,
        sharey=True,
    )
    if len(topologies) == 1:
        axes = np.array(axes).reshape(2, 1)

    cmap = plt.get_cmap("viridis")
    adv_to_color = {a: cmap(i / max(1, len(adv_counts) - 1)) for i, a in enumerate(sorted(adv_counts))}

    xs = list(range(1, max_dist + 1))
    for col, topo in enumerate(topologies):
        for row, which in enumerate(["mis", "tru"]):
            ax = axes[row, col]
            for a in sorted(adv_counts):
                ys: List[float] = []
                for d in xs:
                    bucket = counts.get((topo, a, d))
                    if not bucket or bucket["n"] == 0:
                        ys.append(float("nan"))
                    else:
                        ys.append(bucket[which] / bucket["n"])
                ax.plot(xs, ys, marker="o", linewidth=1.3, alpha=0.85, color=adv_to_color[a], label=f"a={a}")
            which_label = "Misinformation" if which == "mis" else "Truth"
            ax.set_title(f"{topo} ({which_label})")
            ax.set_xlabel("Distance to Nearest Adversary")
            ax.set_ylim(0.0, 1.05)
            if col == 0:
                ax.set_ylabel("Fraction of Non-adversary Agents")
            if row == 0 and col == len(topologies) - 1:
                ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_misinfo_spread_indicators(agent_rows: List[Dict[str, Any]], *, out_dir: Path) -> None:
    """
    A couple sweep-level indicators derived from AgentMetrics:
      - non-adversary share of misinfo posts
      - exposures vs posts scatter
    """
    _style()
    ensure_dir(out_dir)

    # non-adversary share of misinfo posts per (run_id,target_agent)
    by_rt = groupby(agent_rows, ("run_id", "target_agent"))
    recs: List[Dict[str, Any]] = []
    for (run_id, target_agent), agents in by_rt.items():
        if not agents:
            continue
        any_row = agents[0]
        topo = any_row.get("topology")
        a = any_row.get("adversary_count")
        adv_posts = 0
        non_adv_posts = 0
        non_adv_exposures = 0
        for r in agents:
            try:
                posts = int(r.get("misinformation_posts") or 0)
                exposures = int(r.get("misinformation_exposures") or 0)
            except Exception:
                continue
            if str(r.get("role")) == "adversary":
                adv_posts += posts
            else:
                non_adv_posts += posts
                non_adv_exposures += exposures
        denom = adv_posts + non_adv_posts
        if denom <= 0:
            share = None
        else:
            share = non_adv_posts / denom
        recs.append(
            {
                "topology": topo,
                "adversary_count": a,
                "non_adv_post_share": share,
                "non_adv_exposures": non_adv_exposures,
                "non_adv_posts": non_adv_posts,
            }
        )

    if recs:
        topologies = _sorted_unique(recs, "topology")
        adv_counts = _sorted_unique(recs, "adversary_count")
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(topologies),
            figsize=(4.2 * len(topologies), 3.6),
            sharey=True,
        )
        if len(topologies) == 1:
            axes = np.array([axes])
        for col, topo in enumerate(topologies):
            ax = axes[col]
            sub = [r for r in recs if r.get("topology") == topo and r.get("non_adv_post_share") is not None]
            by_adv = groupby(sub, ("adversary_count",))
            xs = [float(a) for a in adv_counts]
            ys = []
            yerr = []
            for a in adv_counts:
                vals = _finite([x.get("non_adv_post_share") for x in by_adv.get((a,), [])])
                ys.append(mean(vals) if vals else float("nan"))
                yerr.append(sem(vals) if vals else float("nan"))
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=1.5)
            ax.set_title(str(topo))
            ax.set_xlabel("Adversary Count")
            ax.set_xticks(xs, [str(a) for a in adv_counts])
            ax.set_ylim(0.0, 1.05)
            if col == 0:
                ax.set_ylabel("Share of Misinformation Posts by Non-adversaries")
        fig.tight_layout()
        fig.savefig(out_dir / "non_adv_misinfo_post_share.png")
        plt.close(fig)

        # exposures vs posts scatter
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        topo_to_color = {t: plt.get_cmap("tab10")(i % 10) for i, t in enumerate(topologies)}
        for t in topologies:
            sub = [r for r in recs if r.get("topology") == t and r.get("non_adv_posts") is not None]
            xs = [float(r.get("non_adv_exposures") or 0) for r in sub]
            ys = [float(r.get("non_adv_posts") or 0) for r in sub]
            ax.scatter(xs, ys, alpha=0.75, s=45, label=str(t), color=topo_to_color.get(t))
        ax.set_title("Non-adversary Misinformation Exposures vs Posts")
        ax.set_xlabel("Non-adversary Misinformation Exposures (Sum)")
        ax.set_ylabel("Non-adversary Misinformation Posts (Sum)")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / "non_adv_exposures_vs_posts.png")
        plt.close(fig)
