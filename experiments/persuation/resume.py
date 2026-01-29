from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from experiments.persuation import run as run_mod  # noqa: E402


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
    target_count: int
    secret_channel_enabled: bool
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model_label}__{self.sweep_name}__{self.topology}"
            f"__n{self.num_agents}__t{int(self.target_count)}"
            f"__secret{int(bool(self.secret_channel_enabled))}"
            f"__seed{self.seed}"
        )

    @property
    def run_label(self) -> str:
        return (
            f"{self.model_label}/{self.sweep_name}/{self.topology}"
            f"/n{self.num_agents}/t{int(self.target_count)}"
            f"/secret{int(bool(self.secret_channel_enabled))}"
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
    persuasion_cfg = exp.get("persuasion") or {}

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

    default_target_counts = run_mod._normalize_target_counts(
        persuasion_cfg.get("target_count", 1)
    )
    if not default_target_counts:
        default_target_counts = [1]

    for model in models:
        model_label = str(model.get("label") or "model")
        llm_cfg = model.get("llm") or {}
        for sweep in sweeps:
            sweep_name = str(sweep.get("name") or "sweep")
            topologies = sweep.get("topologies") or []
            agent_counts = sweep.get("num_agents") or []
            secret_flags = (
                sweep.get("secret_channel_enabled")
                or sweep.get("secret_channels")
                or [False]
            )
            target_counts = run_mod._normalize_target_counts(
                sweep.get("persuasion_target_counts")
                if sweep.get("persuasion_target_counts") is not None
                else sweep.get("persuasion_target_count")
            )
            if not target_counts:
                target_counts = list(default_target_counts)
            seeds = run_mod._normalize_seeds(sweep.get("seeds")) or list(default_seeds)
            if runs_per_setting is not None:
                seeds = seeds[:runs_per_setting]
            if not seeds:
                raise ValueError(
                    "No seeds specified. Set experiment.seeds or sweeps[].seeds."
                )

            for topology in topologies:
                for n in agent_counts:
                    for t in target_counts:
                        for secret in secret_flags:
                            for seed in seeds:
                                yield RunSpec(
                                    model_label=model_label,
                                    model_llm_cfg=llm_cfg,
                                    sweep_name=sweep_name,
                                    topology=str(topology),
                                    num_agents=int(n),
                                    target_count=int(t),
                                    secret_channel_enabled=bool(secret),
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
            "persuader": rc.get("persuader"),
            "persuasion_target_count": rc.get("persuasion_target_count"),
            "secret_channel_enabled": rc.get("secret_channel_enabled"),
            "secret_blackboard_id": rc.get("secret_blackboard_id"),
            "seed": rc.get("seed"),
            "targets_recruited_count": metrics.get("targets_recruited_count"),
            "persuasion_success": metrics.get("persuasion_success"),
            "coalition_reward_sum": metrics.get("coalition_reward_sum"),
            "noncoalition_reward_sum": metrics.get("noncoalition_reward_sum"),
            "coalition_mean_reward": metrics.get("coalition_mean_reward"),
            "noncoalition_mean_reward": metrics.get("noncoalition_mean_reward"),
            "coalition_advantage_mean": metrics.get("coalition_advantage_mean"),
            "coalition_posts_secret_rate": metrics.get("coalition_posts_secret_rate"),
            "target_posts_secret_rate": metrics.get("target_posts_secret_rate"),
            "status": metrics.get("status"),
            "mean_regret": metrics.get("mean_regret"),
            "coalition_mean_regret": metrics.get("coalition_mean_regret"),
            "noncoalition_mean_regret": metrics.get("noncoalition_mean_regret"),
            "system_regret": metrics.get("system_regret"),
            "system_regret_ratio": metrics.get("system_regret_ratio"),
        }
        rows.append(row)

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


async def _run_incomplete_runs(
    *,
    root: Path,
    cfg: Dict[str, Any],
    incomplete: List[RunSpec],
    completed: int,
    total_runs: int,
    max_concurrent_runs: Optional[int],
    dry_run: bool,
) -> None:
    if max_concurrent_runs is None:
        max_concurrent_runs = (cfg.get("experiment") or {}).get(
            "max_concurrent_runs", 1
        )
    max_concurrent_runs = int(max_concurrent_runs or 1)
    if max_concurrent_runs <= 0:
        raise ValueError("max_concurrent_runs must be a positive integer")

    if not incomplete:
        logger = run_mod.logger
        logger.info("All runs are complete. Nothing to resume.")
        return

    logger = run_mod.logger
    logger.info(
        "RESUME START (incomplete=%s, completed=%s/%s, output_root=%s)",
        len(incomplete),
        completed,
        total_runs,
        root,
    )

    completed_runs = completed
    failed_runs = 0

    with tqdm(
        total=len(incomplete), desc="Resume", unit="run", dynamic_ncols=True
    ) as pbar:
        if dry_run:
            for spec in incomplete:
                pbar.set_postfix_str(spec.run_label)
                pbar.update(1)
            return

        if max_concurrent_runs <= 1:
            for spec in incomplete:
                run_status = "success"
                pbar.set_postfix_str(spec.run_label)
                try:
                    await run_mod._run_single(
                        base_cfg=cfg,
                        model_label=spec.model_label,
                        model_llm_cfg=spec.model_llm_cfg,
                        sweep_name=spec.sweep_name,
                        topology=spec.topology,
                        num_agents=spec.num_agents,
                        persuasion_target_count=spec.target_count,
                        secret_channel_enabled=spec.secret_channel_enabled,
                        seed=spec.seed,
                        out_dir=root,
                    )
                    completed_runs += 1
                except Exception:
                    run_status = "failed"
                    failed_runs += 1
                    logger.exception("RUN FAILED %s", spec.run_label)
                    raise
                finally:
                    pbar.update(1)
                    run_mod._write_progress(
                        root,
                        {
                            "status": "running",
                            "total_runs": total_runs,
                            "completed_runs": completed_runs,
                            "failed_runs": failed_runs,
                            "last_run_label": spec.run_label,
                            "last_run_status": run_status,
                        },
                    )
            return

        semaphore = asyncio.Semaphore(int(max_concurrent_runs))

        def _run_single_in_thread(**kwargs: Any) -> Dict[str, Any]:
            return asyncio.run(run_mod._run_single(**kwargs))

        async def _run_single_limited(
            *, run_label: str, **kwargs: Any
        ) -> Dict[str, Any]:
            async with semaphore:
                logger.info("SCHEDULED %s", run_label)
                return await asyncio.to_thread(_run_single_in_thread, **kwargs)

        tasks: List[asyncio.Task] = []
        task_labels: Dict[asyncio.Task, str] = {}
        for spec in incomplete:
            task = asyncio.create_task(
                _run_single_limited(
                    run_label=spec.run_label,
                    base_cfg=cfg,
                    model_label=spec.model_label,
                    model_llm_cfg=spec.model_llm_cfg,
                    sweep_name=spec.sweep_name,
                    topology=spec.topology,
                    num_agents=spec.num_agents,
                    persuasion_target_count=spec.target_count,
                    secret_channel_enabled=spec.secret_channel_enabled,
                    seed=spec.seed,
                    out_dir=root,
                )
            )
            tasks.append(task)
            task_labels[task] = spec.run_label

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for finished in done:
                run_label = task_labels.get(finished, "unknown")
                pbar.set_postfix_str(run_label)
                run_status = "success"
                try:
                    await finished
                    completed_runs += 1
                except Exception:
                    run_status = "failed"
                    failed_runs += 1
                    logger.exception("RUN FAILED %s", run_label)
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
                            "completed_runs": completed_runs,
                            "failed_runs": failed_runs,
                            "last_run_label": run_label,
                            "last_run_status": run_status,
                        },
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume persuasion experiment runs.")
    parser.add_argument("--root", required=True, help="Existing output root directory.")
    parser.add_argument(
        "--config", required=True, help="Path to experiment config (YAML/JSON)."
    )
    parser.add_argument(
        "--max-concurrent-runs",
        default=None,
        type=int,
        help="Maximum number of runs to execute in parallel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list incomplete runs; do not execute.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    cfg = _load_config(Path(args.config))
    incomplete, completed, total_runs = _select_incomplete_runs(root=root, cfg=cfg)

    if args.dry_run:
        print("Incomplete runs:")
        for spec in incomplete:
            print(spec.run_label)
        print(f"Completed: {completed}/{total_runs}")
        return

    start = datetime.now()
    asyncio.run(
        _run_incomplete_runs(
            root=root,
            cfg=cfg,
            incomplete=incomplete,
            completed=completed,
            total_runs=total_runs,
            max_concurrent_runs=args.max_concurrent_runs,
            dry_run=args.dry_run,
        )
    )
    _rebuild_summary_files(root)
    elapsed = datetime.now() - start
    print(f"Resume complete in {elapsed}. Summary files rebuilt.")


if __name__ == "__main__":
    main()
