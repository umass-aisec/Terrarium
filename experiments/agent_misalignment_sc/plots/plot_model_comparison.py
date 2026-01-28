from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.common.plotting.style import apply_default_style
from experiments.common.plotting.io_utils import (
    as_float,
    as_int,
    ensure_dir,
    finite,
    mean,
    sanitize_filename,
    sem,
    write_csv,
)
from experiments.common.plotting.load_runs import load_runs

from .build_tables import build_tables


_MODEL_COLORS: Dict[str, str] = {
    "GPT-4.1-Mini": "#4c78a8",
    "GPT-4o-Mini": "#e45756",
    "Kimi-K2-Instruct": "#72b7b2",
}


def _strategy_type(value: Any) -> str:
    """
    Match the 4 strategy buckets used in plot_overview.py.
    """
    s = (str(value) if value is not None else "").strip()
    if not s or s == "none":
        return "benign"
    if s == "covert_max_coalition_reward":
        return "covert"
    if s == "destructive_max_coalition_reward":
        return "destructive_max"
    if s == "destructive_no_reward_preservation":
        return "destructive_no_preservation"
    # Unknown / legacy strings fall back to raw (still plotted if present).
    return s


def _strategy_label(stype: str) -> str:
    m = {
        "benign": "Benign",
        "covert": "Covert",
        "destructive_max": "Des (MCR)",
        "destructive_no_preservation": "Des (NRP)",
    }
    return m.get(str(stype), str(stype).replace("_", " "))


def _parse_seeds_spec(spec: Optional[str]) -> Optional[set[int]]:
    if not spec:
        return None
    out: set[int] = set()
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            lo_s, _, hi_s = p.partition("-")
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(p))
    return out or None


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if as_int(r.get("num_agents")) != int(num_agents):
            continue
        if str(r.get("target_role")) != str(target_role):
            continue
        if as_int(r.get("adversary_count")) != int(adversary_count):
            continue
        if seeds is not None:
            seed = as_int(r.get("seed"))
            if seed is None or int(seed) not in seeds:
                continue
        out.append(r)
    return out


def _groupby(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    out: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.get(key), []).append(r)
    return out


def _plot_grouped_bars(
    *,
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    out_path: Path,
    strategy_order: Sequence[str],
    model_order: Sequence[str],
    legend_mode: str,
) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.grid(False)

    # Compute means+SEMs per (strategy_type, model_label).
    by_s = _groupby(rows, "strategy_type")
    summaries: Dict[tuple[str, str], tuple[float, float]] = {}
    for stype, bucket_rows in by_s.items():
        st = str(stype)
        by_m = _groupby(bucket_rows, "model_label")
        for mlabel, mrows in by_m.items():
            ml = str(mlabel)
            vals = finite([as_float(r.get("value")) for r in mrows])
            if not vals:
                continue
            summaries[(st, ml)] = (float(mean(vals)), float(sem(vals)))
    colors = dict(_MODEL_COLORS)

    group_count = len(strategy_order)
    hue_count = max(1, len(model_order))
    group_width = 0.80
    bar_width = group_width / float(hue_count)

    xs = list(range(group_count))
    for j, model_label in enumerate(model_order):
        offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
        heights: List[float] = []
        yerrs: List[float] = []
        for i, stype in enumerate(strategy_order):
            key = (str(stype), str(model_label))
            if key not in summaries:
                heights.append(float("nan"))
                yerrs.append(0.0)
                continue
            m, e = summaries[key]
            heights.append(float(m))
            yerrs.append(float(e))

        # Skip plotting if no finite heights.
        if not any(math.isfinite(float(h)) for h in heights):
            continue

        x_pos = [float(x) + float(off) for x, off in zip(xs, offsets)]
        ax.bar(
            x_pos,
            heights,
            yerr=yerrs,
            capsize=3,
            width=bar_width * 0.92,
            alpha=0.9,
            color=colors.get(str(model_label), None),
            label=str(model_label),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([_strategy_label(s) for s in strategy_order], rotation=0, ha="center")
    ax.set_ylabel(metric_label)
    if str(legend_mode) == "in-plot":
        ax.legend(loc="best", frameon=False, title=None)
    elif str(legend_mode) == "outside":
        ncol = max(1, len(model_order))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            title=None,
        )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_model_legend_row(*, out_path: Path, model_order: Sequence[str]) -> None:
    apply_default_style(plt)
    ensure_dir(out_path.parent)
    handles: List[Patch] = []
    for m in model_order:
        handles.append(Patch(facecolor=_MODEL_COLORS.get(str(m), "#999999"), edgecolor="none", label=str(m)))
    ncol = max(1, len(handles))
    fig = plt.figure(figsize=(9.6, 0.9))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _emit_artifacts(
    *,
    out_path: Path,
    rows: List[Dict[str, Any]],
    strategy_order: Sequence[str],
    legend_mode: str,
) -> None:
    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    write_csv(csv_path, rows)

    order_literal = "[" + ", ".join(repr(str(s)) for s in strategy_order) + "]"
    script = f"""\
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _finite(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        f = _as_float(x)
        if f is None:
            continue
        if math.isfinite(f):
            out.append(float(f))
    return out


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _sem(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs) / len(xs))
    var = sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)
    return float(math.sqrt(var) / math.sqrt(len(xs)))


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    legend_mode = {str(legend_mode)!r}

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows")

    metric_label = str(rows[0].get("plot_metric_label") or "Metric")
    # Stable order (matches generator configuration for this plot)
    strategy_order = {order_literal}
    label_map = {{
        "benign": "Benign",
        "covert": "Covert",
        "destructive_max": "Des (MCR)",
        "destructive_no_preservation": "Des (NRP)",
    }}
    model_order = []
    for r in rows:
        m = str(r.get("model_label") or "")
        if m and m not in model_order:
            model_order.append(m)

    colors = {{
        "GPT-4.1-Mini": "#4c78a8",
        "GPT-4o-Mini": "#e45756",
        "Kimi-K2-Instruct": "#72b7b2",
    }}

    # Match paper-style defaults used in the repo.
    plt.rcParams.update({{
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }})

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.grid(False)

    group_count = len(strategy_order)
    hue_count = max(1, len(model_order))
    group_width = 0.80
    bar_width = group_width / float(hue_count)
    xs = list(range(group_count))

    for j, model_label in enumerate(model_order):
        offsets = [(j - (hue_count - 1) / 2.0) * bar_width for _ in xs]
        heights: List[float] = []
        yerrs: List[float] = []
        for stype in strategy_order:
            vals = _finite(
                [
                    r.get("value")
                    for r in rows
                    if str(r.get("strategy_type")) == stype and str(r.get("model_label")) == str(model_label)
                ]
            )
            if not vals:
                heights.append(float("nan"))
                yerrs.append(0.0)
                continue
            heights.append(float(_mean(vals) or 0.0))
            yerrs.append(float(_sem(vals)))
        if not any(math.isfinite(float(h)) for h in heights):
            continue
        x_pos = [float(x) + float(off) for x, off in zip(xs, offsets)]
        ax.bar(
            x_pos,
            heights,
            yerr=yerrs,
            capsize=3,
            width=bar_width * 0.92,
            alpha=0.9,
            color=colors.get(str(model_label), None),
            label=str(model_label),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label_map.get(s, s.replace("_", " ")) for s in strategy_order], rotation=0, ha="center")
    ax.set_ylabel(metric_label)
    if str(legend_mode) == "in-plot":
        ax.legend(loc="best", frameon=False, title=None)
    elif str(legend_mode) == "outside":
        ncol = max(1, len(model_order))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            title=None,
        )

    fig.tight_layout()
    out_pdf = here / {out_path.name!r}
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def _emit_legend_artifacts(*, out_dir: Path, model_order: Sequence[str]) -> None:
    csv_path = out_dir / "model_legend.csv"
    script_path = out_dir / "model_legend__replot.py"
    out_pdf = out_dir / "model_legend.pdf"

    rows: List[Dict[str, Any]] = []
    for m in model_order:
        rows.append({"model_label": str(m), "color": _MODEL_COLORS.get(str(m), "#999999")})
    write_csv(csv_path, rows)

    script = f"""\
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / {csv_path.name!r}
    out_pdf = here / {out_pdf.name!r}

    plt.rcParams.update({{
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }})

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    if not rows:
        raise SystemExit("No rows")

    handles: List[Patch] = []
    for r in rows:
        label = str(r.get("model_label") or "")
        color = str(r.get("color") or "#999999")
        handles.append(Patch(facecolor=color, edgecolor="none", label=label))

    fig = plt.figure(figsize=(9.6, 0.9))
    fig.legend(handles=handles, loc="center", ncol=max(1, len(handles)), frameon=False)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")
    _plot_model_legend_row(out_path=out_pdf, model_order=model_order)


def _collect_model_rows(
    *,
    model_runs_dir: Path,
    sweep_name: str,
    benign_sweep_name: str,
    model_label: str,
    num_agents: int,
    target_role: str,
    adversary_count: int,
    seeds: Optional[set[int]],
    metric_key: str,
    plot_metric_label: str,
    default_benign_when_missing: Optional[float] = None,
) -> List[Dict[str, Any]]:
    sweep_dir = model_runs_dir / sweep_name
    benign_dir = model_runs_dir / benign_sweep_name

    runs, _ = load_runs(sweep_dir)
    tables = build_tables(runs)
    main = _filter_rows(
        tables.run_rows,
        num_agents=num_agents,
        target_role=target_role,
        adversary_count=adversary_count,
        seeds=seeds,
    )

    benign_rows: List[Dict[str, Any]] = []
    if benign_dir.exists():
        benign_runs, _ = load_runs(benign_dir)
        benign_tables = build_tables(benign_runs)
        benign_rows = _filter_rows(
            benign_tables.run_rows,
            num_agents=num_agents,
            target_role=target_role,
            adversary_count=0,
            seeds=seeds,
        )
        # Convenience: benign sweeps are often run for only one target_role (e.g., departmental).
        # For model comparisons where we want a baseline under multiple target_role facets,
        # reuse that single baseline by relabeling it to the requested target_role.
        if not benign_rows:
            try:
                roles_present = sorted(
                    {str(r.get("target_role")) for r in benign_tables.run_rows if r.get("target_role") is not None}
                )
            except Exception:
                roles_present = []
            if len(roles_present) == 1:
                fallback_role = roles_present[0]
                fallback = _filter_rows(
                    benign_tables.run_rows,
                    num_agents=num_agents,
                    target_role=str(fallback_role),
                    adversary_count=0,
                    seeds=seeds,
                )
                if fallback:
                    benign_rows = []
                    for r in fallback:
                        rr = dict(r)
                        rr["target_role"] = str(target_role)
                        benign_rows.append(rr)

    out: List[Dict[str, Any]] = []
    for r in main:
        out.append(
            {
                "plot_metric_key": metric_key,
                "plot_metric_label": plot_metric_label,
                "model_label": model_label,
                "model_runs_dir": str(model_runs_dir),
                "strategy": r.get("strategy"),
                "strategy_type": _strategy_type(r.get("strategy")),
                "adversary_count": adversary_count,
                "num_agents": num_agents,
                "target_role": target_role,
                "seed": r.get("seed"),
                "run_id": r.get("run_id"),
                "value": r.get(metric_key),
                "group": "main",
            }
        )

    for r in benign_rows:
        v = r.get(metric_key)
        if v is None and default_benign_when_missing is not None:
            v = float(default_benign_when_missing)
        out.append(
            {
                "plot_metric_key": metric_key,
                "plot_metric_label": plot_metric_label,
                "model_label": model_label,
                "model_runs_dir": str(model_runs_dir),
                "strategy": r.get("strategy"),
                "strategy_type": "benign",
                "adversary_count": 0,
                "num_agents": num_agents,
                "target_role": target_role,
                "seed": r.get("seed"),
                "run_id": r.get("run_id"),
                "value": v,
                "group": "benign",
            }
        )

    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grouped barplots across models (strategy type on x, model as hue)."
    )
    parser.add_argument(
        "--model-runs-dir",
        action="append",
        required=True,
        help="Repeatable. Path like .../outputs/<tag>/<ts>/runs/<model_label>.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional repeatable labels corresponding to each --model-runs-dir (defaults to directory basename).",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="agent_misalignment_sweep",
        help="Adversarial sweep name under each model runs dir.",
    )
    parser.add_argument(
        "--benign-sweep-name",
        type=str,
        default="benign_baseline_sweep",
        help="Benign sweep name under each model runs dir.",
    )
    parser.add_argument("--num-agents", type=int, default=9)
    parser.add_argument(
        "--target-role",
        action="append",
        dest="target_roles",
        default=None,
        help=(
            "Repeatable. Target role to plot (e.g., departmental, Resource_Provisioner). "
            "If omitted, plots both departmental and Resource_Provisioner."
        ),
    )
    parser.add_argument("--adversary-count", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-4" or "0,1,2".',
    )
    parser.add_argument(
        "--legend-mode",
        type=str,
        default="separate",
        choices=["separate", "outside", "in-plot", "none"],
        help=(
            "Legend placement. 'separate' writes model_legend.pdf and omits legends in plots; "
            "'outside' draws above plot; 'in-plot' uses a standard legend; 'none' omits legends entirely."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/agent_misalignment_sc/plots_outputs_model_compare",
    )
    args = parser.parse_args(argv)

    model_dirs = [Path(p).expanduser().resolve() for p in (args.model_runs_dir or [])]
    labels = args.label or []
    model_labels: List[str] = []

    display_map = {
        "openai-gpt-4.1-mini": "GPT-4.1-Mini",
        "openai-gpt-4o-mini": "GPT-4o-Mini",
        "together-kimik2-Instruct": "Kimi-K2-Instruct",
    }
    for i, d in enumerate(model_dirs):
        if i < len(labels) and str(labels[i]).strip():
            model_labels.append(str(labels[i]).strip())
        else:
            model_labels.append(display_map.get(d.name, d.name))

    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    seeds = _parse_seeds_spec(args.seeds)

    strategy_order = ["benign", "covert", "destructive_max", "destructive_no_preservation"]
    model_order = list(model_labels)

    specs = [
        (
            "joint_reward_regret_normalized",
            "Overall Regret (Normalized)",
            "overall_regret_by_model.pdf",
            None,
            True,
        ),
        (
            "coalition_reward_regret_normalized",
            "Coalition Regret (Normalized)",
            "coalition_regret_by_model.pdf",
            0.0,
            False,
        ),
        (
            "coalition_minus_noncoalition_hospital_avg_actual",
            "Coalition Advantage",
            "coalition_advantage_actual_by_model.pdf",
            0.0,
            False,
        ),
    ]

    target_roles = args.target_roles or ["departmental", "Resource_Provisioner"]
    for target_role in target_roles:
        role_dir = out_dir / "by_target_role" / sanitize_filename(str(target_role))
        ensure_dir(role_dir)
        if str(args.legend_mode) == "separate":
            _emit_legend_artifacts(out_dir=role_dir, model_order=model_order)

        for metric_key, metric_label, filename, benign_default, include_benign in specs:
            rows: List[Dict[str, Any]] = []
            for d, mlabel in zip(model_dirs, model_labels):
                rows.extend(
                    _collect_model_rows(
                        model_runs_dir=d,
                        sweep_name=str(args.sweep_name),
                        benign_sweep_name=str(args.benign_sweep_name),
                        model_label=mlabel,
                        num_agents=int(args.num_agents),
                        target_role=str(target_role),
                        adversary_count=int(args.adversary_count),
                        seeds=seeds,
                        metric_key=str(metric_key),
                        plot_metric_label=str(metric_label),
                        default_benign_when_missing=benign_default,
                    )
                )

            # For coalition-only metrics, drop benign rows (no coalition exists).
            if not bool(include_benign):
                rows = [r for r in rows if str(r.get("strategy_type")) != "benign"]

            # Plot expects per-run points with (strategy_type, model_label, value).
            out_path = role_dir / sanitize_filename(str(filename))
            metric_strategy_order = (
                strategy_order if bool(include_benign) else [s for s in strategy_order if s != "benign"]
            )
            _plot_grouped_bars(
                rows=rows,
                metric_key=str(metric_key),
                metric_label=str(metric_label),
                out_path=out_path,
                strategy_order=metric_strategy_order,
                model_order=model_order,
                legend_mode=str(args.legend_mode),
            )
            _emit_artifacts(
                out_path=out_path,
                rows=rows,
                strategy_order=metric_strategy_order,
                legend_mode=str(args.legend_mode),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
