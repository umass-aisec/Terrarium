from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.common.plotting.io_utils import (
    as_float,
    as_int,
    ensure_dir,
    finite,
    groupby,
    mean,
    safe_load_json,
    sem,
    sorted_unique,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style


logger = logging.getLogger(__name__)

# Match the palette used in collusion_hist__c2__pvALL__bars__se.png.
_PVALL_GROUP_PALETTE = [
    "#264653",  # Charcoal Blue
    "#2a9d8f",  # Verdigris
    "#8ab17d",  # Muted Olive
    "#e9c46a",  # Jasmine
    "#f4a261",  # Sandy Brown
    "#e76f51",  # Burnt Peach
]


def _format_param_float(x: float) -> str:
    s = (f"{float(x):.3f}").rstrip("0").rstrip(".")
    return s if s else "0"


def _finite_int(values: List[Any]) -> List[int]:
    out: List[int] = []
    for v in values:
        if v is None:
            continue
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _pretty_topology_label(
    topo: Any,
    rows: List[Dict[str, Any]],
    *,
    cn_params: Optional[Dict[str, Any]] = None,
    abbreviate_random: bool = False,
) -> str:
    s = str(topo or "").strip()
    if not s:
        return s

    key = s.lower().replace("-", "_").replace(" ", "_")
    alias_to_canonical = {"er": "erdos_renyi", "ws": "watts_strogatz", "ba": "barabasi_albert"}
    canonical = alias_to_canonical.get(key, key)

    full_names = {
        "erdos_renyi": "Erdős–Rényi",
        "barabasi_albert": "Barabási–Albert",
        "watts_strogatz": "Watts–Strogatz",
    }
    abbrev_names = {
        "erdos_renyi": "ER",
        "barabasi_albert": "BA",
        "watts_strogatz": "WS",
    }
    if abbreviate_random and canonical in abbrev_names:
        base = abbrev_names[canonical]
    else:
        base = full_names.get(canonical, s.replace("_", " ").title())

    if canonical == "erdos_renyi":
        ps = finite([r.get("cn_edge_prob") for r in rows])
        if ps:
            p_unique = sorted({round(float(p), 6) for p in ps})
            if len(p_unique) == 1:
                return f"{base} (p={_format_param_float(p_unique[0])})"
        if cn_params:
            p = as_float(cn_params.get("edge_prob"))
            if p is not None:
                return f"{base} (p={_format_param_float(p)})"
        return base

    if canonical == "watts_strogatz":
        ks = _finite_int([r.get("cn_k") for r in rows])
        ps = finite([r.get("cn_rewire_prob") for r in rows])
        parts: List[str] = []
        if ks:
            k_unique = sorted({int(k) for k in ks})
            if len(k_unique) == 1:
                parts.append(f"k={k_unique[0]}")
        if ps:
            p_unique = sorted({round(float(p), 6) for p in ps})
            if len(p_unique) == 1:
                parts.append(f"p={_format_param_float(p_unique[0])}")
        if cn_params:
            if not ks:
                k = as_int(cn_params.get("k"))
                if k is not None:
                    parts.append(f"k={int(k)}")
            if not ps:
                p = as_float(cn_params.get("rewire_prob"))
                if p is not None:
                    parts.append(f"p={_format_param_float(p)}")
        if parts:
            return f"{base} ({', '.join(parts)})"
        return base

    if canonical == "barabasi_albert":
        ms = _finite_int([r.get("cn_m") for r in rows])
        if ms:
            m_unique = sorted({int(m) for m in ms})
            if len(m_unique) == 1:
                return f"{base} (m={m_unique[0]})"
        if cn_params:
            m = as_int(cn_params.get("m"))
            if m is not None:
                return f"{base} (m={int(m)})"
        return base

    return base


def _ordered_topologies(topologies: List[Any]) -> List[Any]:
    preferred = [
        "path",
        "star",
        "complete",
        "watts_strogatz",
        "barabasi_albert",
        "erdos_renyi",
    ]
    preferred_idx = {name: i for i, name in enumerate(preferred)}

    def _key(t: Any) -> Tuple[int, str]:
        s = str(t or "")
        idx = preferred_idx.get(s, preferred_idx.get(s.lower(), 10_000))
        return (idx, s)

    return sorted(topologies, key=_key)


def _mean_sem_series(
    rows: List[Dict[str, Any]], *, adv_counts: List[Any], y_key: str
) -> Tuple[List[float], List[float], List[float]]:
    by_adv = groupby(rows, ("adversary_count",))
    xs: List[float] = []
    ys_mean: List[float] = []
    ys_sem: List[float] = []
    for a in adv_counts:
        ys = finite([r.get(y_key) for r in by_adv.get((a,), [])])
        if not ys:
            continue
        xs.append(float(a))
        ys_mean.append(mean(ys))
        ys_sem.append(sem(ys))
    return xs, ys_mean, ys_sem


def _plot_mean_with_sem_band(
    *,
    ax: Any,
    rows: List[Dict[str, Any]],
    adv_counts: List[Any],
    y_key: str,
    color: Any,
    linestyle: str = "-",
    marker: str = "o",
    band_alpha: float = 0.22,
    linewidth: float = 3.0,
    markersize: float = 8.8,
) -> None:
    xs, ys_mean, ys_sem = _mean_sem_series(rows, adv_counts=adv_counts, y_key=y_key)
    if not xs:
        return

    y = np.array(ys_mean, dtype=float)
    yerr = np.array(ys_sem, dtype=float)
    ax.fill_between(
        xs,
        (y - yerr).tolist(),
        (y + yerr).tolist(),
        color=color,
        alpha=band_alpha,
        linewidth=0.0,
    )
    ax.plot(
        xs,
        ys_mean,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        alpha=0.95,
    )


def _ensure_mid_tick(ax: Any) -> None:
    ticks = [float(t) for t in ax.get_yticks()]
    if all(abs(t - 0.5) > 1e-6 for t in ticks):
        ticks.append(0.5)
        ticks = sorted(set(ticks))
        ax.set_yticks(ticks)


def _apply_fixed_y_grid(ax: Any) -> None:
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, alpha=0.9)


def _load_sweep_communication_network_params(
    run_stats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Best-effort load of the sweep-level config.json to recover communication_network params.
    Needed because run_config.json files may omit these.
    """
    for r in run_stats:
        run_dir = r.get("run_dir")
        if not run_dir:
            continue
        try:
            sweep_dir = Path(str(run_dir)).expanduser().resolve().parent
        except Exception:
            sweep_dir = Path(str(run_dir)).parent

        # Search upwards for config.json (expects it at the timestamp folder).
        for parent in [sweep_dir, *list(sweep_dir.parents)[:8]]:
            cfg_path = parent / "config.json"
            if not cfg_path.exists():
                continue
            cfg = safe_load_json(cfg_path)
            if not isinstance(cfg, dict):
                continue
            cn = cfg.get("communication_network")
            if isinstance(cn, dict) and cn:
                return cn
    return {}


def _compute_baseline_joint_reward_ratio(
    run_rows: List[Dict[str, Any]],
) -> Dict[Tuple[Any, ...], float]:
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
      - misinformation propagation mean/max across targets (computed over non-adversaries, excluding the target agent)
      - misinformation_messages / total_messages mean across targets
      - fraction of misinformation_posts attributable to non-adversaries (mean across targets)
    """
    by_run_targets = groupby(target_rows, ("run_id",))
    by_run_agents = groupby(agent_rows, ("run_id", "target_agent"))
    baseline = _compute_baseline_joint_reward_ratio(run_rows)

    def _prop_rate_from_agents(
        agents: List[Dict[str, Any]],
        *,
        target_agent: str,
        belief_key: str,
    ) -> float:
        denom = 0
        num = 0
        for a in agents:
            if str(a.get("role")) == "adversary":
                continue
            if str(a.get("agent_name")) == str(target_agent):
                continue
            denom += 1
            if bool(a.get(belief_key, False)):
                num += 1
        return (num / denom) if denom else 0.0

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
        prop_vals: List[float] = []
        for t in targets:
            ta = t.get("target_agent")
            if not ta:
                continue
            agents = by_run_agents.get((run_id, ta), [])
            if agents:
                prop_vals.append(
                    _prop_rate_from_agents(
                        agents,
                        target_agent=str(ta),
                        belief_key="believes_misinformation",
                    )
                )
            else:
                # Fallback (should be rare): approximate using target-level counts,
                # assuming the target does not believe misinformation about themselves.
                total_agents = t.get("total_agents")
                adv_count = t.get("adversary_count")
                mis_believers = t.get("believes_misinformation_non_adversary")
                if (
                    total_agents is None
                    or adv_count is None
                    or mis_believers is None
                ):
                    continue
                try:
                    non_adv = int(total_agents) - int(adv_count)
                    denom = non_adv - 1
                    if denom <= 0:
                        prop_vals.append(0.0)
                    else:
                        rate = float(int(mis_believers)) / float(denom)
                        prop_vals.append(max(0.0, min(1.0, rate)))
                except Exception:
                    continue
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
                ta = t.get("target_agent") or t.get(
                    "target_agent"
                )  # might already be present
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
                "misinfo_message_rate_mean": mean(msg_rate_vals)
                if msg_rate_vals
                else None,
                "misinfo_non_adv_post_share_mean": mean(non_adv_share_vals)
                if non_adv_share_vals
                else None,
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
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    topologies_present = sorted_unique(run_stats, "topology")
    if not topologies_present:
        return

    adv_counts = sorted_unique(run_stats, "adversary_count")
    prop_key = "prop_misinfo_max" if use_max_propagation else "prop_misinfo_mean"

    cn_params = _load_sweep_communication_network_params(run_stats)
    topologies = _ordered_topologies(topologies_present)
    color_by_topo: Dict[Any, Any] = {}
    for i, topo in enumerate(topologies):
        color_by_topo[topo] = _PVALL_GROUP_PALETTE[i % len(_PVALL_GROUP_PALETTE)]

    deterministic_order = ["path", "star", "complete"]
    random_order = ["watts_strogatz", "barabasi_albert", "erdos_renyi"]
    top_row = [t for t in deterministic_order if t in topologies_present]
    bottom_row = [t for t in random_order if t in topologies_present]
    extras = [t for t in topologies if t not in top_row and t not in bottom_row]
    bottom_row.extend(extras)

    ncols = max(len(top_row), len(bottom_row))
    if ncols <= 0:
        return

    title_fs = 16
    label_fs = 14
    tick_fs = 13
    legend_fs = 13

    fig_width_scale = 0.7 if not use_max_propagation else 1.0
    fig_height_scale = 0.95 if not use_max_propagation else 1.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(4.2 * ncols * fig_width_scale, 6.2 * fig_height_scale),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(2, ncols)

    for row in range(2):
        row_topos = top_row if row == 0 else bottom_row
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(row_topos):
                ax.axis("off")
                continue
            topo = row_topos[col]
            subset = [r for r in run_stats if r.get("topology") == topo]
            color = color_by_topo.get(topo, "#4c72b0")

            # Solid: joint reward ratio
            _plot_mean_with_sem_band(
                ax=ax,
                rows=subset,
                adv_counts=adv_counts,
                y_key="joint_reward_ratio",
                color=color,
                linestyle="-",
                marker="o",
                band_alpha=0.22,
            )
            # Dashed: misinformation propagation rate
            _plot_mean_with_sem_band(
                ax=ax,
                rows=subset,
                adv_counts=adv_counts,
                y_key=prop_key,
                color=color,
                linestyle="--",
                marker="o",
                band_alpha=0.14,
            )

            ax.set_title(
                _pretty_topology_label(
                    topo,
                    subset,
                    cn_params=cn_params,
                    abbreviate_random=not use_max_propagation,
                ),
                fontsize=title_fs,
            )
            ax.set_ylim(0.0, 1.05)
            _apply_fixed_y_grid(ax)
            ax.tick_params(axis="both", labelsize=tick_fs)

            if adv_counts:
                ax.set_xticks(
                    [float(a) for a in adv_counts],
                    [str(a) for a in adv_counts],
                )
            if row == 1:
                ax.set_xlabel("Adversary Count", fontsize=label_fs)
            # Deliberately omit a shared y-axis label; the two plotted metrics have
            # different semantics but share a common [0, 1] range.

    metric_handles = [
        Line2D([0], [0], color="#111827", linewidth=2.2, linestyle="-"),
        Line2D([0], [0], color="#111827", linewidth=2.2, linestyle="--"),
    ]
    fig.legend(
        metric_handles,
        ["Joint Reward", "Misinformation Propagation Rate"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
        fontsize=legend_fs,
    )

    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.97))
    fig.subplots_adjust(wspace=0.10, hspace=0.30)
    fig.savefig(out_path, bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    # Also generate per-metric faceted PNGs (same deterministic/random layout).
    def _save_metric_facet(
        *,
        y_key: str,
        y_label: str,
        fname_suffix: str,
        linestyle: str,
        band_alpha: float,
    ) -> None:
        fig_s, axes_s = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(4.2 * ncols * fig_width_scale, 6.2 * fig_height_scale),
            sharex=True,
            sharey=True,
        )
        axes_s = np.array(axes_s).reshape(2, ncols)

        for row in range(2):
            row_topos = top_row if row == 0 else bottom_row
            for col in range(ncols):
                ax = axes_s[row, col]
                if col >= len(row_topos):
                    ax.axis("off")
                    continue
                topo = row_topos[col]
                subset = [r for r in run_stats if r.get("topology") == topo]
                color = color_by_topo.get(topo, "#4c72b0")

                _plot_mean_with_sem_band(
                    ax=ax,
                    rows=subset,
                    adv_counts=adv_counts,
                    y_key=y_key,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    band_alpha=band_alpha,
                )
                ax.set_title(
                    _pretty_topology_label(
                        topo,
                        subset,
                        cn_params=cn_params,
                        abbreviate_random=not use_max_propagation,
                    ),
                    fontsize=title_fs,
                )
                ax.set_ylim(0.0, 1.05)
                _apply_fixed_y_grid(ax)
                ax.tick_params(axis="both", labelsize=tick_fs)
                if adv_counts:
                    ax.set_xticks(
                        [float(a) for a in adv_counts],
                        [str(a) for a in adv_counts],
                    )
                if row == 1:
                    ax.set_xlabel("Adversary Count", fontsize=label_fs)
                if col == 0:
                    ax.set_ylabel(y_label, fontsize=label_fs)

        fig_s.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))
        fig_s.subplots_adjust(wspace=0.10, hspace=0.30)
        out_path_s = out_path.with_name(
            f"{out_path.stem}_{fname_suffix}{out_path.suffix}"
        )
        fig_s.savefig(out_path_s, bbox_inches="tight")
        log_saved_plot(out_path_s, logger=logger)
        plt.close(fig_s)

    _save_metric_facet(
        y_key="joint_reward_ratio",
        y_label="Joint Reward",
        fname_suffix="joint_reward_ratio",
        linestyle="-",
        band_alpha=0.22,
    )
    _save_metric_facet(
        y_key=prop_key,
        y_label="Misinformation Propagation Rate (excluding target)",
        fname_suffix="misinfo_propagation_rate",
        linestyle="--",
        band_alpha=0.14,
    )


def plot_sweep_slopes(
    run_stats: List[Dict[str, Any]],
    *,
    out_path: Path,
    use_max_propagation: bool = False,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    topologies = sorted_unique(run_stats, "topology")
    if not topologies:
        return
    adv_counts = sorted_unique(run_stats, "adversary_count")
    seeds = sorted_unique(run_stats, "seed")
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
            rows = [
                r
                for r in by_seed.get((s,), [])
                if r.get("adversary_count") in adv_counts
            ]
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
                ax0.plot(
                    xs_f,
                    ys_f,
                    marker="o",
                    linewidth=1.2,
                    alpha=0.7,
                    color=cmap(i % 10),
                    label=f"seed {s}",
                )
        ax0.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax0.set_title(f"{topo}: Δ Joint Reward vs a0")
        ax0.set_xlabel("Adversary Count")
        ax0.set_ylabel("Δ Joint Reward")

        ax1 = axes[1, col]
        for i, s in enumerate(seeds):
            rows = [
                r
                for r in by_seed.get((s,), [])
                if r.get("adversary_count") in adv_counts
            ]
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
                ax1.plot(
                    xs_f, ys_f, marker="o", linewidth=1.2, alpha=0.7, color=cmap(i % 10)
                )
        ax1.set_title(f"{topo}: Propagation vs Adversary Count")
        ax1.set_xlabel("Adversary Count")
        ax1.set_ylabel("Misinformation Propagation Rate (excluding target)")
        ax1.set_ylim(0.0, 1.05)

        if col == len(topologies) - 1 and seeds:
            ax0.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_heatmaps(
    run_stats: List[Dict[str, Any]],
    *,
    out_dir: Path,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_dir)

    topologies = sorted_unique(run_stats, "topology")
    adv_counts = sorted_unique(run_stats, "adversary_count")
    if not topologies or not adv_counts:
        return

    def _matrix(value_key: str) -> np.ndarray:
        mat = np.full((len(topologies), len(adv_counts)), np.nan, dtype=float)
        for i, topo in enumerate(topologies):
            for j, a in enumerate(adv_counts):
                vals = finite(
                    [
                        r.get(value_key)
                        for r in run_stats
                        if r.get("topology") == topo and r.get("adversary_count") == a
                    ]
                )
                if vals:
                    mat[i, j] = mean(vals)
        return mat

    prop_key = "prop_misinfo_mean"
    prop_title = "Propagation rate (mean across targets, excluding target)"

    matrices = [
        (
            "delta_joint_reward_ratio",
            "Δ Joint Reward vs a0",
            "heatmap_delta_joint_reward_ratio.png",
            "coolwarm",
        ),
        (
            prop_key,
            prop_title,
            "heatmap_propagation_rate.png",
            "viridis",
        ),
        (
            "misinfo_message_rate_mean",
            "Misinformation message rate (mean)",
            "heatmap_misinfo_message_rate.png",
            "magma",
        ),
    ]

    for key, title, fname, cmap in matrices:
        mat = _matrix(key)
        fig, ax = plt.subplots(
            figsize=(1.6 * len(adv_counts) + 2.5, 0.8 * len(topologies) + 1.8)
        )
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
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if cmap in {"magma"} else "black",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path)
        log_saved_plot(out_path, logger=logger)
        plt.close(fig)


def plot_tradeoff_scatter(
    run_stats: List[Dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    prop_key = "prop_misinfo_mean"
    rows = [
        r
        for r in run_stats
        if r.get("delta_joint_reward_ratio") is not None and r.get(prop_key) is not None
    ]
    if not rows:
        return

    topologies = sorted_unique(rows, "topology")
    topo_to_color = {t: plt.get_cmap("tab10")(i % 10) for i, t in enumerate(topologies)}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for t in topologies:
        sub = [r for r in rows if r.get("topology") == t]
        xs = [
            float(r[prop_key])
            for r in sub
            if r.get(prop_key) is not None
        ]
        ys = [
            float(r["delta_joint_reward_ratio"])
            for r in sub
            if r.get("delta_joint_reward_ratio") is not None
        ]
        ss = []
        for r in sub:
            a = r.get("adversary_count") or 0
            try:
                ss.append(40 + 35 * float(a))
            except Exception:
                ss.append(40)
        ax.scatter(
            xs,
            ys,
            s=ss[: len(xs)],
            alpha=0.75,
            label=str(t),
            color=topo_to_color.get(t),
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_title("Tradeoff: Misinformation Propagation vs Δ Joint Reward")
    ax.set_xlabel(
        "Misinformation Propagation Rate (mean across targets, excluding target)"
    )
    ax.set_ylabel("Δ Joint Reward (vs a0 baseline)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_belief_composition(
    agent_rows: List[Dict[str, Any]], *, out_path: Path
) -> None:
    """
    Stacked bars of mean fraction of non-adversary agents who believe:
      - misinformation
      - truth
      - neither/unknown
    Aggregated across (run_id, target_agent) groups.
    """
    apply_default_style(plt)
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

    topologies = sorted_unique(records, "topology")
    adv_counts = sorted_unique(records, "adversary_count")

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
            mis_means.append(
                mean(finite([x.get("frac_mis") for x in rows]))
                if rows
                else float("nan")
            )
            tru_means.append(
                mean(finite([x.get("frac_truth") for x in rows]))
                if rows
                else float("nan")
            )
            oth_means.append(
                mean(finite([x.get("frac_other") for x in rows]))
                if rows
                else float("nan")
            )
        mis = np.array(mis_means, dtype=float)
        tru = np.array(tru_means, dtype=float)
        oth = np.array(oth_means, dtype=float)
        ax.bar(xs, mis, color=colors["frac_mis"], label="misinfo", alpha=0.85)
        ax.bar(
            xs, tru, bottom=mis, color=colors["frac_truth"], label="truth", alpha=0.85
        )
        ax.bar(
            xs,
            oth,
            bottom=mis + tru,
            color=colors["frac_other"],
            label="other/unknown",
            alpha=0.85,
        )
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
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_confidence_by_belief(
    agent_rows: List[Dict[str, Any]], *, out_path: Path
) -> None:
    """
    Boxplots of judge confidence for non-adversary agents, split by belief.
    One subplot per topology; x axis is adversary_count.
    """
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    non_adv_rows = [
        r
        for r in agent_rows
        if str(r.get("role")) != "adversary" and r.get("judge_confidence") is not None
    ]
    if not non_adv_rows:
        return

    topologies = sorted_unique(non_adv_rows, "topology")
    adv_counts = sorted_unique(non_adv_rows, "adversary_count")
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
        sub = [
            r
            for r in non_adv_rows
            if r.get("topology") == topo and r.get("adversary_count") == a
        ]
        if which == "misinfo":
            sub = [r for r in sub if bool(r.get("believes_misinformation"))]
        elif which == "truth":
            sub = [r for r in sub if bool(r.get("believes_truth"))]
        return finite([r.get("judge_confidence") for r in sub])

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
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_belief_by_distance(
    agent_rows: List[Dict[str, Any]], *, out_path: Path
) -> None:
    """
    Belief fraction vs distance-to-nearest-adversary.
    Produces 2 rows (misinfo/truth) × columns(topology).
    Each adversary_count is a line.
    """
    apply_default_style(plt)
    ensure_dir(out_path.parent)

    rows = [
        r
        for r in agent_rows
        if str(r.get("role")) != "adversary"
        and r.get("distance_to_nearest_adversary") is not None
    ]
    if not rows:
        return

    topologies = sorted_unique(rows, "topology")
    adv_counts = sorted_unique(rows, "adversary_count")
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
    for _, _, d in counts.keys():
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
    adv_to_color = {
        a: cmap(i / max(1, len(adv_counts) - 1))
        for i, a in enumerate(sorted(adv_counts))
    }

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
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.3,
                    alpha=0.85,
                    color=adv_to_color[a],
                    label=f"a={a}",
                )
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
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)


def plot_misinfo_spread_indicators(
    agent_rows: List[Dict[str, Any]], *, out_dir: Path
) -> None:
    """
    A couple sweep-level indicators derived from AgentMetrics:
      - non-adversary share of misinfo posts
      - exposures vs posts scatter
    """
    apply_default_style(plt)
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
        topologies = sorted_unique(recs, "topology")
        adv_counts = sorted_unique(recs, "adversary_count")
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
            sub = [
                r
                for r in recs
                if r.get("topology") == topo and r.get("non_adv_post_share") is not None
            ]
            by_adv = groupby(sub, ("adversary_count",))
            xs = [float(a) for a in adv_counts]
            ys = []
            yerr = []
            for a in adv_counts:
                vals = finite(
                    [x.get("non_adv_post_share") for x in by_adv.get((a,), [])]
                )
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
        out_path = out_dir / "non_adv_misinfo_post_share.png"
        fig.savefig(out_path)
        log_saved_plot(out_path, logger=logger)
        plt.close(fig)

        # exposures vs posts scatter
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        topo_to_color = {
            t: plt.get_cmap("tab10")(i % 10) for i, t in enumerate(topologies)
        }
        for t in topologies:
            sub = [
                r
                for r in recs
                if r.get("topology") == t and r.get("non_adv_posts") is not None
            ]
            xs = [float(r.get("non_adv_exposures") or 0) for r in sub]
            ys = [float(r.get("non_adv_posts") or 0) for r in sub]
            ax.scatter(
                xs, ys, alpha=0.75, s=45, label=str(t), color=topo_to_color.get(t)
            )
        ax.set_title("Non-adversary Misinformation Exposures vs Posts")
        ax.set_xlabel("Non-adversary Misinformation Exposures (Sum)")
        ax.set_ylabel("Non-adversary Misinformation Posts (Sum)")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        out_path = out_dir / "non_adv_exposures_vs_posts.png"
        fig.savefig(out_path)
        log_saved_plot(out_path, logger=logger)
        plt.close(fig)
