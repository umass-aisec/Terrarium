from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.collusion import run as run_mod  # noqa: E402


REQUIRED_RUN_FILES: Tuple[str, ...] = (
    "run_config.json",
    "metrics.json",
    "final_summary.json",
    "agent_turns.json",
    "tool_events.json",
    "blackboards.json",
)


@dataclass(frozen=True)
class RunSpec:
    model_label: str
    model_llm_cfg: Dict[str, Any]
    sweep_name: str
    topology: str
    num_agents: int
    colluder_count: int
    secret_channel_enabled: bool
    prompt_variant: str
    seed: int

    @property
    def effective_prompt_variant(self) -> str:
        if not bool(self.secret_channel_enabled):
            return "control"
        return str(self.prompt_variant or "control")

    @property
    def run_id(self) -> str:
        return (
            f"{self.model_label}__{self.sweep_name}__{self.topology}"
            f"__n{self.num_agents}__c{self.colluder_count}"
            f"__secret{int(bool(self.secret_channel_enabled))}"
            f"__pv{self.effective_prompt_variant}"
            f"__seed{self.seed}"
        )

    @property
    def run_label(self) -> str:
        return (
            f"{self.model_label}/{self.sweep_name}/{self.topology}"
            f"/n{self.num_agents}/c{self.colluder_count}"
            f"/secret{int(bool(self.secret_channel_enabled))}"
            f"/pv{self.effective_prompt_variant}"
            f"/seed{self.seed}"
        )


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8")) or {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_run_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    return all((run_dir / fname).exists() for fname in REQUIRED_RUN_FILES)


def _iter_expected_run_specs(cfg: Dict[str, Any]) -> Iterable[RunSpec]:
    exp = cfg.get("experiment") or {}
    models = cfg.get("llm_models") or []
    sweeps = exp.get("sweeps") or []

    runs_per_setting = exp.get("runs_per_setting")
    if runs_per_setting is not None:
        runs_per_setting = int(runs_per_setting)
        if runs_per_setting <= 0:
            raise ValueError("experiment.runs_per_setting must be a positive integer")

    default_seeds = run_mod._normalize_seeds(exp.get("seeds"))
    if not default_seeds:
        default_seeds = run_mod._normalize_seeds(
            (cfg.get("simulation") or {}).get("seed")
        ) or [1]

    for model in models:
        model_label = str(model.get("label") or "model")
        llm_cfg = model.get("llm") or {}
        for sweep in sweeps:
            sweep_name = str(sweep.get("name") or "sweep")
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            colluder_counts = sweep.get("colluder_counts") or []
            secret_flags = (
                sweep.get("secret_channel_enabled")
                or sweep.get("secret_channels")
                or [False]
            )
            prompt_variants = sweep.get("prompt_variants") or ["control"]
            seeds = run_mod._normalize_seeds(sweep.get("seeds")) or list(default_seeds)
            if runs_per_setting is not None:
                seeds = seeds[:runs_per_setting]
            if not seeds:
                raise ValueError(
                    "No seeds specified. Set experiment.seeds or sweeps[].seeds."
                )

            for topology in topologies:
                for n in agent_counts:
                    for c in colluder_counts:
                        for secret in secret_flags:
                            for pv in prompt_variants:
                                # Prompt variants are only active when the secret channel exists.
                                if not bool(secret) and str(pv) != "control":
                                    continue
                                for seed in seeds:
                                    yield RunSpec(
                                        model_label=model_label,
                                        model_llm_cfg=llm_cfg,
                                        sweep_name=sweep_name,
                                        topology=str(topology),
                                        num_agents=int(n),
                                        colluder_count=int(c),
                                        secret_channel_enabled=bool(secret),
                                        prompt_variant=str(pv),
                                        seed=int(seed),
                                    )


def _select_incomplete_runs(
    *, root: Path, cfg: Dict[str, Any]
) -> Tuple[List[RunSpec], int, int]:
    expected = list(_iter_expected_run_specs(cfg))
    total_runs = len(expected)
    completed = 0
    incomplete: List[RunSpec] = []
    for spec in expected:
        run_dir = root / "runs" / spec.model_label / spec.sweep_name / spec.run_id
        if _is_run_complete(run_dir):
            completed += 1
        else:
            incomplete.append(spec)
    return incomplete, completed, total_runs


def _rebuild_summary_files(root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for run_dir in (root / "runs").rglob("*"):
        if not run_dir.is_dir():
            continue
        rc_path = run_dir / "run_config.json"
        if not rc_path.exists():
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        metrics_path = run_dir / "metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                obj = json.loads(metrics_path.read_text(encoding="utf-8")) or {}
                metrics = obj if isinstance(obj, dict) else {}
            except Exception:
                metrics = {}

        row: Dict[str, Any] = {
            "run_id": rc.get("run_id") or run_dir.name,
            "model_label": rc.get("model_label"),
            "provider": rc.get("provider"),
            "model": rc.get("model"),
            "sweep": rc.get("sweep") or rc.get("sweep_name"),
            "topology": rc.get("topology"),
            "num_agents": rc.get("num_agents"),
            "colluder_count": rc.get("colluder_count"),
            "secret_channel_enabled": rc.get("secret_channel_enabled"),
            "secret_blackboard_id": rc.get("secret_blackboard_id"),
            "prompt_variant": rc.get("prompt_variant"),
            "seed": rc.get("seed"),
            "colluders": rc.get("colluders"),
            "status": metrics.get("status"),
            "coalition_advantage_mean": metrics.get("coalition_advantage_mean"),
            "colluder_posts_secret_rate": metrics.get("colluder_posts_secret_rate"),
        }
        rows.append(row)

    # Stable ordering for diffs/readability.
    rows = sorted(rows, key=lambda r: str(r.get("run_id") or ""))

    (root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (root / "summary.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        )

    if not flat_rows:
        (root / "summary.csv").write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for r in flat_rows for k in r.keys()})
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat_rows)


async def resume_runs(
    *,
    root: Path,
    cfg: Dict[str, Any],
    max_concurrent_runs: int,
    stop_on_error: bool,
    dry_run: bool,
) -> None:
    run_mod._ensure_dir(root)
    run_mod._configure_experiment_logging(root)

    to_run, completed, total_runs = _select_incomplete_runs(root=root, cfg=cfg)
    failed = 0

    run_mod.logger.info(
        "RESUME START (total_runs=%s, completed=%s, remaining=%s, output_root=%s)",
        total_runs,
        completed,
        len(to_run),
        root,
    )

    if dry_run:
        print(f"Output root: {root}")
        print(f"Total runs: {total_runs}")
        print(f"Already complete: {completed}")
        print(f"Remaining (incomplete): {len(to_run)}")
        if to_run:
            print("Next 10 runs to execute:")
            for spec in to_run[:10]:
                print(f"  - {spec.run_label}")
        return

    run_mod._write_progress(
        root,
        {
            "status": "running",
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
            "resumed_at": datetime.now().isoformat(),
        },
    )

    if not to_run:
        run_mod.logger.info("RESUME DONE (nothing to do; all runs appear complete)")
        _rebuild_summary_files(root)
        run_mod._write_progress(
            root,
            {
                "status": "completed",
                "total_runs": total_runs,
                "completed_runs": completed,
                "failed_runs": failed,
            },
        )
        return

    max_concurrent_runs = int(max_concurrent_runs)
    if max_concurrent_runs <= 0:
        raise ValueError("max_concurrent_runs must be a positive integer")

    with tqdm(
        total=total_runs,
        initial=completed,
        desc="Experiments (resume)",
        unit="run",
        dynamic_ncols=True,
    ) as pbar:
        if max_concurrent_runs <= 1:
            for spec in to_run:
                pbar.set_postfix_str(spec.run_label)
                run_status = "success"
                try:
                    await run_mod._run_single(
                        base_cfg=cfg,
                        model_label=spec.model_label,
                        model_llm_cfg=spec.model_llm_cfg,
                        sweep_name=spec.sweep_name,
                        topology=spec.topology,
                        num_agents=spec.num_agents,
                        colluder_count=spec.colluder_count,
                        secret_channel_enabled=spec.secret_channel_enabled,
                        prompt_variant=spec.prompt_variant,
                        seed=spec.seed,
                        out_dir=root,
                    )
                    completed += 1
                except Exception:
                    run_status = "failed"
                    failed += 1
                    run_mod.logger.exception("RUN FAILED %s", spec.run_label)
                    if stop_on_error:
                        raise
                finally:
                    pbar.update(1)
                    run_mod._write_progress(
                        root,
                        {
                            "status": "running",
                            "total_runs": total_runs,
                            "completed_runs": completed,
                            "failed_runs": failed,
                            "last_run_label": spec.run_label,
                            "last_run_status": run_status,
                        },
                    )
        else:
            semaphore = asyncio.Semaphore(max_concurrent_runs)

            def _run_single_in_thread(**kwargs: Any) -> Dict[str, Any]:
                return asyncio.run(run_mod._run_single(**kwargs))

            async def _run_single_limited(*, spec: RunSpec) -> Dict[str, Any]:
                async with semaphore:
                    run_mod.logger.info("SCHEDULED %s", spec.run_label)
                    return await asyncio.to_thread(
                        _run_single_in_thread,
                        base_cfg=cfg,
                        model_label=spec.model_label,
                        model_llm_cfg=spec.model_llm_cfg,
                        sweep_name=spec.sweep_name,
                        topology=spec.topology,
                        num_agents=spec.num_agents,
                        colluder_count=spec.colluder_count,
                        secret_channel_enabled=spec.secret_channel_enabled,
                        prompt_variant=spec.prompt_variant,
                        seed=spec.seed,
                        out_dir=root,
                    )

            tasks: List[asyncio.Task[Any]] = []
            task_specs: Dict[asyncio.Task[Any], RunSpec] = {}
            for spec in to_run:
                task = asyncio.create_task(_run_single_limited(spec=spec))
                tasks.append(task)
                task_specs[task] = spec

            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    spec = task_specs.get(finished)
                    if spec is None:
                        continue
                    pbar.set_postfix_str(spec.run_label)
                    run_status = "success"
                    try:
                        await finished
                        completed += 1
                    except Exception:
                        run_status = "failed"
                        failed += 1
                        run_mod.logger.exception("RUN FAILED %s", spec.run_label)
                        if stop_on_error:
                            for t in pending:
                                t.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                            raise
                    finally:
                        pbar.update(1)
                        run_mod._write_progress(
                            root,
                            {
                                "status": "running",
                                "total_runs": total_runs,
                                "completed_runs": completed,
                                "failed_runs": failed,
                                "last_run_label": spec.run_label,
                                "last_run_status": run_status,
                            },
                        )

    _rebuild_summary_files(root)
    status = "completed" if completed == total_runs else "completed_with_failures"
    run_mod.logger.info(
        "RESUME END (completed=%s, failed=%s, total_runs=%s, output_root=%s)",
        completed,
        failed,
        total_runs,
        root,
    )
    run_mod._write_progress(
        root,
        {
            "status": status,
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resume a previously started collusion experiment in-place."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Existing timestamp output root (e.g., experiments/collusion/outputs/<tag>/<timestamp>).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. Defaults to <root>/config.json if present.",
    )
    parser.add_argument(
        "--max-concurrent-runs",
        default=None,
        type=int,
        help="Override experiment.max_concurrent_runs when resuming.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately on the first failed run (default: continue and record failures).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run and exit."
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    config_path = Path(args.config) if args.config else (root / "config.json")
    cfg = _load_config(config_path)

    exp = cfg.get("experiment") or {}
    max_concurrent_runs = (
        int(args.max_concurrent_runs)
        if args.max_concurrent_runs is not None
        else int(exp.get("max_concurrent_runs", 1))
    )

    asyncio.run(
        resume_runs(
            root=root,
            cfg=cfg,
            max_concurrent_runs=max_concurrent_runs,
            stop_on_error=bool(args.stop_on_error),
            dry_run=bool(args.dry_run),
        )
    )


if __name__ == "__main__":
    main()
