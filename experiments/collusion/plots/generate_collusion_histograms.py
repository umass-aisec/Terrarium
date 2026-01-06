from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.plotting.io_utils import ensure_dir, infer_labels_from_sweep_dir, mean, safe_load_json
from experiments.common.plotting.load_runs import LoadedRun, load_runs


def _style() -> None:
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


def _pretty(s: Any) -> str:
    return str(s).replace("_", " ")

def _canonical_variant(name: Any) -> str:
    s = str(name or "").strip()
    aliases = {
        "goal_only": "deception",
        "structured_playbook": "structured",
        "aggressive_deception": "aggressive",
    }
    return aliases.get(s, s)

def _sanitize_filename(s: Any) -> str:
    value = str(s) if s is not None else ""
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in value).strip("_") or "unknown"

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


def _finite(values: Iterable[Any]) -> List[float]:
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


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if np.isfinite(f):
        return float(f)
    return None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _sorted_unique(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    values = {r.get(key) for r in rows}
    values.discard(None)
    try:
        return sorted(values)
    except Exception:
        return sorted(values, key=lambda x: str(x))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _build_rows(runs: List[LoadedRun]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

    for run in runs:
        rc = run.run_config or {}
        rid = str(rc.get("run_id") or run.run_dir.name)

        fs = run.final_summary or {}
        if not isinstance(fs, dict):
            fs = {}

        metrics = run.metrics or {}
        if not isinstance(metrics, dict):
            metrics = {}

        # Optional: post-hoc LLM-as-a-judge outputs (written outside the run directory).
        judge_simple = None
        judge_medium = None
        judge_complex = None
        judge_mean_rating = None
        try:
            sweep_dir = run.run_dir.parent
            model_dir = sweep_dir.parent
            judge_path = model_dir / "judge_secret_blackboard" / sweep_dir.name / f"{run.run_dir.name}.json"
            judge_payload = safe_load_json(judge_path) if judge_path.exists() else None
            if isinstance(judge_payload, dict):
                judgements = judge_payload.get("judgements") or {}
                if isinstance(judgements, dict):
                    def _rating(key: str) -> Optional[float]:
                        j = judgements.get(key) or {}
                        if not isinstance(j, dict):
                            return None
                        return _as_float(j.get("rating"))

                    judge_simple = _rating("simple")
                    judge_medium = _rating("medium")
                    judge_complex = _rating("complex")
                    vals = [float(v) for v in (judge_simple, judge_medium, judge_complex) if v is not None]
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
            "topology": rc.get("topology"),
            "num_agents": _as_int(rc.get("num_agents")),
            "seed": _as_int(rc.get("seed")),
            "colluder_count": _as_int(rc.get("colluder_count")),
            "secret_channel_enabled": _as_bool(rc.get("secret_channel_enabled")),
            "prompt_variant": _canonical_variant(rc.get("prompt_variant")),
            # Run-level outputs
            "status": metrics.get("status"),
            "joint_reward_ratio": fs.get("joint_reward_ratio"),
            "tasks_done": _as_int(metrics.get("tasks_done")),
            "violations": _as_int(metrics.get("violations")),
            "total_cost": metrics.get("total_cost"),
            "priority_sum": metrics.get("priority_sum"),
            "coalition_reward_sum": metrics.get("coalition_reward_sum"),
            "noncoalition_reward_sum": metrics.get("noncoalition_reward_sum"),
            "coalition_mean_reward": metrics.get("coalition_mean_reward"),
            "noncoalition_mean_reward": metrics.get("noncoalition_mean_reward"),
            "coalition_advantage_mean": metrics.get("coalition_advantage_mean"),
            "colluder_posts_secret_rate": metrics.get("colluder_posts_secret_rate"),
            "colluder_posts_total": _as_int(metrics.get("colluder_posts_total")),
            "colluder_posts_secret": _as_int(metrics.get("colluder_posts_secret")),
            "colluder_posts_non_secret": _as_int(metrics.get("colluder_posts_non_secret")),
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
                    "num_agents": _as_int(rc.get("num_agents")),
                    "seed": _as_int(rc.get("seed")),
                    "colluder_count": _as_int(rc.get("colluder_count")),
                    "secret_channel_enabled": _as_bool(rc.get("secret_channel_enabled")),
                    "prompt_variant": _canonical_variant(rc.get("prompt_variant")),
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
    _style()
    ensure_dir(out_path.parent)

    categories = _sorted_unique(rows, category_key)
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
        vals = _finite([r.get(value_key) for r in subset])
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
    plt.close(fig)

def _plot_violin_by_category(
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
    _style()
    ensure_dir(out_path.parent)

    categories = _sorted_unique(rows, category_key)
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
        vals = _finite([r.get(value_key) for r in subset])
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
    parts = ax.violinplot(
        vals_by_cat,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    for body in parts.get("bodies", []):
        body.set_facecolor("#4c72b0")
        body.set_edgecolor("white")
        body.set_alpha(0.55)
        body.set_linewidth(0.7)
    if parts.get("cmedians") is not None:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.4)

    # Overlay jittered points for visibility.
    rng = np.random.default_rng(0)
    for i, vals in enumerate(vals_by_cat, start=1):
        if not vals:
            continue
        xs = rng.normal(loc=float(i), scale=0.06, size=len(vals))
        ax.scatter(xs, vals, s=16, alpha=0.28, color="#2f4b7c", linewidths=0)

    if hline_at_zero:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)

    ax.set_title(f"{_pretty(value_key)} by {_pretty(category_key)} (violin)")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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
    _style()
    ensure_dir(out_path.parent)

    row_vals = _sorted_unique(rows, row_facet)
    col_vals = _sorted_unique(rows, col_facet)
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
            subset = [x for x in rows if x.get(row_facet) == r and x.get(col_facet) == c]
            vals = _finite([x.get(value_key) for x in subset])
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
            ax.set_title(f"{_pretty(row_facet)}={_pretty(r)} | {_pretty(col_facet)}={_pretty(c)} (n={len(vals)})")
            if x_label:
                ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            if xlim is not None:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))

    fig.suptitle(_pretty(value_key), y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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
    _style()
    ensure_dir(out_path.parent)

    categories = _sorted_unique(rows, category_key)
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
            x = _as_float(r.get(x_key))
            y = _as_float(r.get(y_key))
            if x is None or y is None:
                continue
            pts.append((x, y))
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        display = label_map.get(str(cat), str(cat)) if label_map else str(cat)
        ax.scatter(xs, ys, s=30, alpha=0.7, label=f"{display} (n={len(xs)})", color=cmap(i % 10), linewidths=0)

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
    plt.close(fig)


def _plot_role_overlay_hist(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    role_key: str,
    roles: Tuple[str, str],
    out_path: Path,
    bins: int = 24,
    x_label: Optional[str] = None,
) -> None:
    _style()
    ensure_dir(out_path.parent)

    a = [r for r in rows if str(r.get(role_key) or "").strip().lower() == roles[0]]
    b = [r for r in rows if str(r.get(role_key) or "").strip().lower() == roles[1]]
    va = _finite([x.get(value_key) for x in a])
    vb = _finite([x.get(value_key) for x in b])
    if not va and not vb:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if vb:
        ax.hist(vb, bins=bins, alpha=0.6, label=roles[1], color="#55a868", edgecolor="white", linewidth=0.7)
    if va:
        ax.hist(va, bins=bins, alpha=0.6, label=roles[0], color="#c44e52", edgecolor="white", linewidth=0.7)
    ax.set_title(f"{value_key} by {role_key} (n={len(va) + len(vb)})")
    ax.set_ylabel("Count")
    if x_label:
        ax.set_xlabel(x_label)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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
    _write_csv(out_dir / "run_rows.csv", run_rows)
    _write_csv(out_dir / "agent_rows.csv", agent_rows)

    hist_out = out_dir / "hist"
    ensure_dir(hist_out)

    label_map = {
        "baseline": "baseline",
        "control": "control",
        "simple": "simple",
        "deception": "deception",
        "structured": "structured",
        "aggressive": "aggressive",
    }
    order_with_baseline = ["baseline", "control", "simple", "deception", "structured", "aggressive"]
    order_secret_only = ["control", "simple", "deception", "structured", "aggressive"]

    # Run-level: coalition advantage (only meaningful when colluder_count > 0)
    advantage_rows = [
        r
        for r in run_rows
        if (r.get("colluder_count") or 0) > 0 and (r.get("secret_channel_enabled") is True)
    ]
    _plot_hist_grid(
        advantage_rows,
        value_key="coalition_advantage_mean",
        row_facet="prompt_variant",
        col_facet="secret_channel_enabled",
        out_path=hist_out / "coalition_advantage_mean.png",
        bins=int(bins),
        x_label="Coalition mean reward - Non-coalition mean reward",
        vline_at_zero=True,
        sharex=False,
    )
    advantage_cat_rows: List[Dict[str, Any]] = []
    for r in run_rows:
        if (r.get("colluder_count") or 0) <= 0:
            continue
        secret = bool(r.get("secret_channel_enabled"))
        variant = str(r.get("prompt_variant") or "control")
        label = "baseline" if not secret else variant
        advantage_cat_rows.append({**r, "variant_or_baseline": label})
    _plot_box_by_category(
        advantage_cat_rows,
        category_key="variant_or_baseline",
        value_key="coalition_advantage_mean",
        out_path=hist_out / "coalition_advantage_mean_by_variant_or_baseline.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Coalition mean reward - Non-coalition mean reward",
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
        hline_at_zero=True,
    )

    # Run-level: secret-channel usage rate (colluders only; avoid colluder_count==0 noise)
    usage_rows = [
        r
        for r in run_rows
        if (r.get("colluder_count") or 0) > 0 and (r.get("secret_channel_enabled") is True)
    ]
    _plot_hist_grid(
        usage_rows,
        value_key="colluder_posts_secret_rate",
        row_facet="prompt_variant",
        col_facet="secret_channel_enabled",
        out_path=hist_out / "colluder_posts_secret_rate.png",
        bins=int(bins),
        x_label="Colluder posts on secret bb / total posts",
        vline_at_zero=False,
        xlim=(0.0, 1.0),
        hist_range=(0.0, 1.0),
    )
    _plot_box_by_category(
        usage_rows,
        category_key="prompt_variant",
        value_key="colluder_posts_secret_rate",
        out_path=hist_out / "colluder_posts_secret_rate_by_variant.png",
        x_label="Prompt variant (secret channel enabled)",
        y_label="Colluder secret post rate",
        ylim=(0.0, 1.0),
        category_order=order_secret_only,
        label_map=label_map,
        show_n_in_labels=True,
    )
    usage_cat_rows: List[Dict[str, Any]] = []
    for r in run_rows:
        if (r.get("colluder_count") or 0) <= 0:
            continue
        secret = bool(r.get("secret_channel_enabled"))
        variant = str(r.get("prompt_variant") or "control")
        label = "baseline" if not secret else variant
        rate = _as_float(r.get("colluder_posts_secret_rate"))
        if not secret:
            rate = 0.0
        usage_cat_rows.append({**r, "variant_or_baseline": label, "colluder_posts_secret_rate": rate})
    _plot_box_by_category(
        usage_cat_rows,
        category_key="variant_or_baseline",
        value_key="colluder_posts_secret_rate",
        out_path=hist_out / "colluder_posts_secret_rate_by_variant_or_baseline.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Colluder secret post rate",
        ylim=(0.0, 1.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )

    # Agent-level: reward and (private) task cost by role
    _plot_role_overlay_hist(
        agent_rows,
        value_key="reward",
        role_key="role",
        roles=("colluder", "normal"),
        out_path=hist_out / "agent_reward_by_role.png",
        bins=int(bins),
        x_label="Per-agent reward",
    )
    _plot_role_overlay_hist(
        agent_rows,
        value_key="task_cost",
        role_key="role",
        roles=("colluder", "normal"),
        out_path=hist_out / "task_cost_by_role.png",
        bins=int(bins),
        x_label="Private cost of chosen task (if any)",
    )

    # Categorical plot: baseline vs prompt variants for normalized joint reward.
    # Baseline = secret_channel_enabled=false (forced "control" variant in the runner).
    jr_rows: List[Dict[str, Any]] = []
    for r in run_rows:
        if (r.get("colluder_count") or 0) <= 0:
            continue
        secret = bool(r.get("secret_channel_enabled"))
        variant = str(r.get("prompt_variant") or "control")
        label = "baseline" if not secret else variant
        jr_rows.append({**r, "variant_or_baseline": label})

    # Prefer baseline first, then stable prompt variant ordering.
    order = ["baseline", "control", "simple", "deception", "structured", "aggressive"]
    jr_rows.sort(
        key=lambda x: (
            order.index(str(x.get("variant_or_baseline"))) if str(x.get("variant_or_baseline")) in order else 999,
            str(x.get("variant_or_baseline")),
        )
    )

    _plot_box_by_category(
        jr_rows,
        category_key="variant_or_baseline",
        value_key="joint_reward_ratio",
        out_path=hist_out / "joint_reward_ratio_by_variant.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Normalized joint reward (0-1)",
        ylim=(0.0, 1.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )

    # LLM-as-a-judge: secret blackboard collusion score (mean rating over prompts).
    judge_rows: List[Dict[str, Any]] = []
    for r in run_rows:
        if (r.get("colluder_count") or 0) <= 0:
            continue
        secret = bool(r.get("secret_channel_enabled"))
        variant = str(r.get("prompt_variant") or "control")
        label = "baseline" if not secret else variant
        judge_rows.append({**r, "variant_or_baseline": label})

    _plot_box_by_category(
        judge_rows,
        category_key="variant_or_baseline",
        value_key="judge_mean_rating",
        out_path=hist_out / "judge_mean_rating_by_variant.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Judge collusion rating on secret bb (0-5)",
        ylim=(0.0, 5.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )

    _plot_box_by_category(
        judge_rows,
        category_key="variant_or_baseline",
        value_key="judge_simple_rating",
        out_path=hist_out / "judge_simple_rating_by_variant.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Judge collusion rating (simple prompt, 0-5)",
        ylim=(0.0, 5.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )
    _plot_box_by_category(
        judge_rows,
        category_key="variant_or_baseline",
        value_key="judge_medium_rating",
        out_path=hist_out / "judge_medium_rating_by_variant.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Judge collusion rating (medium prompt, 0-5)",
        ylim=(0.0, 5.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )
    _plot_box_by_category(
        judge_rows,
        category_key="variant_or_baseline",
        value_key="judge_complex_rating",
        out_path=hist_out / "judge_complex_rating_by_variant.png",
        x_label="Prompt variant (or baseline with no secret channel)",
        y_label="Judge collusion rating (complex prompt, 0-5)",
        ylim=(0.0, 5.0),
        category_order=order_with_baseline,
        label_map=label_map,
        show_n_in_labels=True,
    )

    if compare_topologies:
        # Single-figure topology comparisons for the histogram metrics.
        topo_compare = hist_out / "compare_topologies"
        ensure_dir(topo_compare)

        topo_order = _sorted_unique(run_rows, "topology")
        topo_label_map = {str(t): str(t).replace("_", " ") for t in topo_order if t is not None}

        # Compare distributions across topologies (faceted by prompt_variant).
        if len(_sorted_unique(advantage_rows, "topology")) > 1:
            adv_range = _robust_hist_range(_finite([r.get("coalition_advantage_mean") for r in advantage_rows]))
            _plot_hist_grid(
                advantage_rows,
                value_key="coalition_advantage_mean",
                row_facet="prompt_variant",
                col_facet="topology",
                out_path=topo_compare / "coalition_advantage_mean__by_topology.png",
                bins=int(bins),
                x_label="Coalition mean reward - Non-coalition mean reward",
                vline_at_zero=True,
                xlim=adv_range,
                hist_range=adv_range,
                sharex=True,
                sharey=True,
            )
            for pv in _sorted_unique(advantage_rows, "prompt_variant"):
                subset = [r for r in advantage_rows if r.get("prompt_variant") == pv]
                _plot_box_by_category(
                    subset,
                    category_key="topology",
                    value_key="coalition_advantage_mean",
                    out_path=topo_compare / f"coalition_advantage_mean__box_by_topology__pv{_sanitize_filename(pv)}.png",
                    x_label=f"Topology (prompt_variant={pv})",
                    y_label="Coalition mean reward - Non-coalition mean reward",
                    category_order=topo_order,
                    label_map=topo_label_map,
                    show_n_in_labels=True,
                    hline_at_zero=True,
                )
                _plot_violin_by_category(
                    subset,
                    category_key="topology",
                    value_key="coalition_advantage_mean",
                    out_path=topo_compare / f"coalition_advantage_mean__violin_by_topology__pv{_sanitize_filename(pv)}.png",
                    x_label=f"Topology (prompt_variant={pv})",
                    y_label="Coalition mean reward - Non-coalition mean reward",
                    category_order=topo_order,
                    label_map=topo_label_map,
                    show_n_in_labels=True,
                    hline_at_zero=True,
                )

        if len(_sorted_unique(usage_rows, "topology")) > 1:
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
            for pv in _sorted_unique(usage_rows, "prompt_variant"):
                subset = [r for r in usage_rows if r.get("prompt_variant") == pv]
                _plot_box_by_category(
                    subset,
                    category_key="topology",
                    value_key="colluder_posts_secret_rate",
                    out_path=topo_compare / f"colluder_posts_secret_rate__box_by_topology__pv{_sanitize_filename(pv)}.png",
                    x_label=f"Topology (prompt_variant={pv})",
                    y_label="Colluder secret post rate",
                    ylim=(0.0, 1.0),
                    category_order=topo_order,
                    label_map=topo_label_map,
                    show_n_in_labels=True,
                )
                _plot_violin_by_category(
                    subset,
                    category_key="topology",
                    value_key="colluder_posts_secret_rate",
                    out_path=topo_compare / f"colluder_posts_secret_rate__violin_by_topology__pv{_sanitize_filename(pv)}.png",
                    x_label=f"Topology (prompt_variant={pv})",
                    y_label="Colluder secret post rate",
                    ylim=(0.0, 1.0),
                    category_order=topo_order,
                    label_map=topo_label_map,
                    show_n_in_labels=True,
                )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate histogram plots for collusion sweep outputs.")
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

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, _ = load_runs(sweep_dir)
    run_rows, agent_rows = _build_rows(runs)

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/collusion/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    _generate_plots(
        runs=runs,
        run_rows=run_rows,
        agent_rows=agent_rows,
        out_dir=out_dir,
        bins=int(args.bins),
        compare_topologies=bool(args.compare_topologies),
    )

    if args.by_topology:
        topologies = _sorted_unique(run_rows, "topology")
        if topologies:
            for topo in topologies:
                topo_runs = [r for r in runs if (r.run_config or {}).get("topology") == topo]
                topo_run_rows = [r for r in run_rows if r.get("topology") == topo]
                topo_agent_rows = [r for r in agent_rows if r.get("topology") == topo]
                if not topo_run_rows and not topo_agent_rows:
                    continue
                topo_out = out_dir / "by_topology" / _sanitize_filename(str(topo))
                _generate_plots(
                    runs=topo_runs,
                    run_rows=topo_run_rows,
                    agent_rows=topo_agent_rows,
                    out_dir=topo_out,
                    bins=int(args.bins),
                    compare_topologies=False,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
