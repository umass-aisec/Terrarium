from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

from experiments.common.plotting.io_utils import (
    ensure_dir,
    infer_labels_from_sweep_dir,
    sanitize_filename,
    write_csv,
    write_json,
)
from experiments.common.plotting.logging_utils import configure_basic_logging
from experiments.common.plotting.load_runs import load_runs

from .build_tables import build_tables
from .plot_sweep import plot_sweep_metrics


def _parse_seeds_spec(spec: str) -> Set[int]:
    """
    Accepts comma-separated ints and inclusive ranges like:
      - "0-9"
      - "0-9,11,15-16"
      - "1,2,3"
    """
    out: Set[int] = set()
    s = (spec or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
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
    return out


def _filter_rows_by_seeds(rows: List[dict], seeds: Optional[Set[int]]) -> List[dict]:
    if not seeds:
        return rows
    out = []
    for r in rows:
        try:
            seed = int(r.get("seed")) if r.get("seed") is not None else None
        except Exception:
            seed = None
        if seed is None:
            continue
        if seed in seeds:
            out.append(r)
    return out


def _filter_rows_by_int(rows: List[dict], key: str, value: int) -> List[dict]:
    out = []
    for r in rows:
        raw = r.get(key)
        if raw is None:
            continue
        try:
            if int(raw) == int(value):
                out.append(r)
        except Exception:
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for an agent_misalignment sweep directory."
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path like experiments/agent_misalignment/outputs/<tag>/<ts>/runs/<model>/<sweep_name>",
    )
    parser.add_argument(
        "--benign-sweep-dir",
        "--baseline-sweep-dir",
        dest="benign_sweep_dir",
        type=str,
        default=None,
        help=(
            "Optional sweep dir for a separately-run benign baseline (same sweep layout). "
            "When provided, baseline points are overlaid in the sweep plots."
        ),
    )
    parser.add_argument(
        "--benign-label",
        type=str,
        default="Benign baseline",
        help="Legend label for the benign/baseline overlay.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-9" or "0-9,11,15-16".',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/agent_misalignment/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    runs, meta = load_runs(sweep_dir)
    tables = build_tables(runs)

    seed_filter = _parse_seeds_spec(args.seeds) if args.seeds else None
    if seed_filter:
        tables = type(tables)(
            run_rows=_filter_rows_by_seeds(tables.run_rows, seed_filter),
            target_rows=tables.target_rows,
            agent_rows=_filter_rows_by_seeds(tables.agent_rows, seed_filter),
        )

    benign_tables = None
    benign_meta = None
    if args.benign_sweep_dir:
        benign_sweep_dir = Path(args.benign_sweep_dir).expanduser().resolve()
        benign_runs, benign_meta = load_runs(benign_sweep_dir)
        benign_tables = build_tables(benign_runs)
        if seed_filter:
            benign_tables = type(benign_tables)(
                run_rows=_filter_rows_by_seeds(benign_tables.run_rows, seed_filter),
                target_rows=benign_tables.target_rows,
                agent_rows=_filter_rows_by_seeds(benign_tables.agent_rows, seed_filter),
            )

    labels = infer_labels_from_sweep_dir(sweep_dir)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("experiments/agent_misalignment/plots_outputs")
        / labels.experiment_tag
        / labels.timestamp
        / labels.model_label
        / labels.sweep_name
    )
    ensure_dir(out_dir)

    # Diagnostics: detect mismatches between requested sweep params (from run_id)
    # and realized values (from run_config).
    mismatch_counts = {"num_agents": 0, "adversary_count": 0}
    for r in tables.run_rows:
        try:
            req_n = int(r.get("num_agents")) if r.get("num_agents") is not None else None
            act_n = int(r.get("actual_num_agents")) if r.get("actual_num_agents") is not None else None
        except Exception:
            req_n, act_n = None, None
        if req_n is not None and act_n is not None and req_n != act_n:
            mismatch_counts["num_agents"] += 1
        try:
            req_a = int(r.get("adversary_count")) if r.get("adversary_count") is not None else None
            act_a = (
                int(r.get("actual_adversary_count"))
                if r.get("actual_adversary_count") is not None
                else None
            )
        except Exception:
            req_a, act_a = None, None
        if req_a is not None and act_a is not None and req_a != act_a:
            mismatch_counts["adversary_count"] += 1

    if mismatch_counts["num_agents"] or mismatch_counts["adversary_count"]:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Found requested-vs-actual mismatches: %s", mismatch_counts)

    write_json(
        out_dir / "meta.json",
        {
            "meta": meta,
            "benign_meta": benign_meta,
            "mismatches": mismatch_counts,
            "counts": {
                "runs": len(runs),
                "run_rows": len(tables.run_rows),
                "agent_rows": len(tables.agent_rows),
                "benign_runs": len(benign_tables.run_rows) if benign_tables else 0,
            },
        },
        sort_keys=True,
    )
    write_csv(out_dir / "run_rows.csv", tables.run_rows)
    write_csv(out_dir / "agent_rows.csv", tables.agent_rows)
    if benign_tables:
        write_csv(out_dir / "benign_run_rows.csv", benign_tables.run_rows)
        write_csv(out_dir / "benign_agent_rows.csv", benign_tables.agent_rows)

    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)
    plot_sweep_metrics(
        run_rows=tables.run_rows,
        agent_rows=tables.agent_rows,
        out_dir=sweep_out,
        baseline_run_rows=benign_tables.run_rows if benign_tables else None,
        baseline_agent_rows=benign_tables.agent_rows if benign_tables else None,
        baseline_label=str(args.benign_label),
    )

    # Stratified: separate plot sets per number of agents.
    agent_counts = sorted(
        {
            int(r.get("num_agents"))
            for r in tables.run_rows
            if r.get("num_agents") is not None
        }
    )
    if agent_counts:
        by_agents_root = sweep_out / "by_num_agents"
        ensure_dir(by_agents_root)
        for n_agents in agent_counts:
            n_run_rows = _filter_rows_by_int(tables.run_rows, "num_agents", n_agents)
            n_agent_rows = _filter_rows_by_int(tables.agent_rows, "num_agents", n_agents)
            if not n_run_rows:
                continue
            n_out = by_agents_root / f"n{n_agents}"
            ensure_dir(n_out)
            plot_sweep_metrics(
                run_rows=n_run_rows,
                agent_rows=n_agent_rows,
                out_dir=n_out,
                baseline_run_rows=_filter_rows_by_int(benign_tables.run_rows, "num_agents", n_agents)
                if benign_tables
                else None,
                baseline_agent_rows=_filter_rows_by_int(benign_tables.agent_rows, "num_agents", n_agents)
                if benign_tables
                else None,
                baseline_label=str(args.benign_label),
            )

    # Stratified: separate plot sets per adversarial strategy.
    strategies = sorted(
        {
            str(r.get("strategy"))
            for r in tables.run_rows
            if r.get("strategy") is not None and str(r.get("strategy")).strip()
        }
    )
    if strategies:
        by_strat_root = sweep_out / "by_strategy"
        ensure_dir(by_strat_root)
        for strat in strategies:
            strat_run_rows = [r for r in tables.run_rows if str(r.get("strategy")) == strat]
            strat_agent_rows = [r for r in tables.agent_rows if str(r.get("strategy")) == strat]
            if not strat_run_rows:
                continue
            strat_out = by_strat_root / sanitize_filename(strat)
            ensure_dir(strat_out)
            plot_sweep_metrics(
                run_rows=strat_run_rows,
                agent_rows=strat_agent_rows,
                out_dir=strat_out,
                baseline_run_rows=benign_tables.run_rows if benign_tables else None,
                baseline_agent_rows=benign_tables.agent_rows if benign_tables else None,
                baseline_label=str(args.benign_label),
            )

            strat_agent_counts = sorted(
                {
                    int(r.get("num_agents"))
                    for r in strat_run_rows
                    if r.get("num_agents") is not None
                }
            )
            if strat_agent_counts:
                strat_by_agents_root = strat_out / "by_num_agents"
                ensure_dir(strat_by_agents_root)
                for n_agents in strat_agent_counts:
                    n_run_rows = _filter_rows_by_int(strat_run_rows, "num_agents", n_agents)
                    n_agent_rows = _filter_rows_by_int(strat_agent_rows, "num_agents", n_agents)
                    if not n_run_rows:
                        continue
                    n_out = strat_by_agents_root / f"n{n_agents}"
                    ensure_dir(n_out)
                    plot_sweep_metrics(
                        run_rows=n_run_rows,
                        agent_rows=n_agent_rows,
                        out_dir=n_out,
                        baseline_run_rows=_filter_rows_by_int(
                            benign_tables.run_rows, "num_agents", n_agents
                        )
                        if benign_tables
                        else None,
                        baseline_agent_rows=_filter_rows_by_int(
                            benign_tables.agent_rows, "num_agents", n_agents
                        )
                        if benign_tables
                        else None,
                        baseline_label=str(args.benign_label),
                    )

    return 0


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
