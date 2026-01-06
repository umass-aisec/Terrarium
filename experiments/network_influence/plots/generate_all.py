from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.common.plotting.io_utils import ensure_dir, infer_labels_from_sweep_dir
from experiments.common.plotting.load_runs import LoadedRun, load_runs

from .build_tables import Tables, build_tables
from .plot_deep_dive import build_graph_from_blackboards, plot_belief_network, plot_run_timeline
from .plot_sweep import (
    build_run_stats,
    plot_belief_by_distance,
    plot_belief_composition,
    plot_confidence_by_belief,
    plot_heatmaps,
    plot_misinfo_spread_indicators,
    plot_sweep_slopes,
    plot_sweep_summary,
    plot_tradeoff_scatter,
)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable column ordering: union of keys.
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _sanitize_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s).strip("_") or "unknown"


def _select_deep_dive_targets(
    *,
    target_rows: List[Dict[str, Any]],
    mode: str,
    max_per_topology: int = 1,
    max_overall: int = 6,
) -> List[Tuple[str, str]]:
    """
    Returns list of (run_id, target_agent).
    """
    if mode == "none":
        return []

    # Prefer runs with adversaries and measurable propagation.
    candidates = [
        r
        for r in target_rows
        if (r.get("adversary_count") not in (None, 0))
        and (r.get("propagation_rate_misinfo_non_adversary") is not None)
        and (r.get("run_id") is not None)
        and (r.get("target_agent") is not None)
    ]
    if mode == "all":
        return [(str(r["run_id"]), str(r["target_agent"])) for r in candidates]

    # sample: pick top per topology + top overall
    candidates = sorted(
        candidates,
        key=lambda r: float(r.get("propagation_rate_misinfo_non_adversary") or 0.0),
        reverse=True,
    )

    selected: List[Tuple[str, str]] = []
    seen = set()

    # Top per topology
    by_topo: Dict[str, int] = {}
    for r in candidates:
        topo = str(r.get("topology") or "unknown")
        if by_topo.get(topo, 0) >= max_per_topology:
            continue
        key = (str(r["run_id"]), str(r["target_agent"]))
        if key in seen:
            continue
        selected.append(key)
        seen.add(key)
        by_topo[topo] = by_topo.get(topo, 0) + 1

    # Top overall
    for r in candidates:
        if len(selected) >= max_overall:
            break
        key = (str(r["run_id"]), str(r["target_agent"]))
        if key in seen:
            continue
        selected.append(key)
        seen.add(key)

    return selected


def _find_code_for_target(target_rows: List[Dict[str, Any]], run_id: str, target_agent: str) -> Optional[str]:
    for r in target_rows:
        if str(r.get("run_id")) == run_id and str(r.get("target_agent")) == target_agent:
            code = r.get("code")
            return str(code) if code else None
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate plots for an experiments sweep directory.")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/network_influence/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/network_influence/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    parser.add_argument(
        "--deep-dive",
        type=str,
        choices=["none", "sample", "all"],
        default="sample",
        help="Generate deep-dive per-run plots for a subset or all targets.",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir)
    tables: Tables = build_tables(runs)
    run_stats = build_run_stats(run_rows=tables.run_rows, target_rows=tables.target_rows, agent_rows=tables.agent_rows)

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/network_influence/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    ensure_dir(out_dir)

    # Persist data tables for reproducibility/debugging.
    _write_json(out_dir / "meta.json", {"meta": meta, "counts": {"runs": len(runs), "run_rows": len(tables.run_rows), "target_rows": len(tables.target_rows), "agent_rows": len(tables.agent_rows)}})
    _write_csv(out_dir / "run_rows.csv", tables.run_rows)
    _write_csv(out_dir / "target_rows.csv", tables.target_rows)
    _write_csv(out_dir / "run_stats.csv", run_stats)

    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)
    plot_sweep_summary(run_stats, out_path=sweep_out / "summary_mean.png", use_max_propagation=False)
    plot_sweep_summary(run_stats, out_path=sweep_out / "summary_max.png", use_max_propagation=True)
    plot_sweep_slopes(run_stats, out_path=sweep_out / "slopes_mean.png", use_max_propagation=False)
    plot_sweep_slopes(run_stats, out_path=sweep_out / "slopes_max.png", use_max_propagation=True)
    plot_heatmaps(run_stats, out_dir=sweep_out / "heatmaps")
    plot_tradeoff_scatter(run_stats, out_path=sweep_out / "tradeoff_scatter.png")
    plot_belief_composition(tables.agent_rows, out_path=sweep_out / "belief_composition.png")
    plot_confidence_by_belief(tables.agent_rows, out_path=sweep_out / "confidence_by_belief.png")
    plot_belief_by_distance(tables.agent_rows, out_path=sweep_out / "belief_by_distance.png")
    plot_misinfo_spread_indicators(tables.agent_rows, out_dir=sweep_out / "spread_indicators")

    # Deep dive
    deep_out = out_dir / "deep_dive"
    ensure_dir(deep_out)
    runs_by_id: Dict[str, LoadedRun] = {}
    for r in runs:
        rid = str((r.run_config or {}).get("run_id") or r.run_dir.name)
        runs_by_id[rid] = r

    selections = _select_deep_dive_targets(target_rows=tables.target_rows, mode=args.deep_dive)
    for run_id, target_agent in selections:
        run = runs_by_id.get(run_id)
        if not run:
            continue
        rc = run.run_config or {}
        topo = rc.get("topology")
        a = rc.get("adversary_count")
        seed = rc.get("seed")

        agent_rows = [r for r in tables.agent_rows if str(r.get("run_id")) == run_id and str(r.get("target_agent")) == target_agent]
        if not agent_rows:
            continue

        code = _find_code_for_target(tables.target_rows, run_id, target_agent)
        graph = build_graph_from_blackboards(run.blackboards)

        title = f"{topo} a={a} seed={seed} target={target_agent}"
        subdir = deep_out / _sanitize_filename(str(topo)) / f"a{_sanitize_filename(str(a))}" / f"seed{_sanitize_filename(str(seed))}" / _sanitize_filename(str(target_agent))
        ensure_dir(subdir)
        plot_belief_network(
            graph=graph,
            topology=str(topo) if topo is not None else None,
            agent_rows=agent_rows,
            run_title=title,
            out_path=subdir / "belief_network.png",
            target_agent=target_agent,
        )
        plot_run_timeline(
            run_title=title,
            tool_events=run.tool_events,
            agent_rows=agent_rows,
            code=code,
            out_path=subdir / "timeline.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
