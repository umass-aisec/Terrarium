from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_utils import infer_labels_from_sweep_dir, iter_run_dirs, safe_load_json


@dataclass(frozen=True)
class LoadedRun:
    run_dir: Path
    run_config: Dict[str, Any]
    final_summary: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    judge_results: Optional[Dict[str, Any]]
    survey_responses: Optional[Dict[str, Any]]
    tool_events: Optional[List[Any]]
    agent_turns: Optional[List[Any]]
    blackboards: Optional[List[Any]]


def load_run(run_dir: Path) -> LoadedRun:
    run_config = safe_load_json(run_dir / "run_config.json") or {}
    return LoadedRun(
        run_dir=run_dir,
        run_config=run_config,
        final_summary=safe_load_json(run_dir / "final_summary.json"),
        metrics=safe_load_json(run_dir / "metrics.json"),
        judge_results=safe_load_json(run_dir / "judge_results.json"),
        survey_responses=safe_load_json(run_dir / "survey_responses.json"),
        tool_events=safe_load_json(run_dir / "tool_events.json"),
        agent_turns=safe_load_json(run_dir / "agent_turns.json"),
        blackboards=safe_load_json(run_dir / "blackboards.json"),
    )


def load_runs(sweep_dir: Path) -> Tuple[List[LoadedRun], Dict[str, str]]:
    labels = infer_labels_from_sweep_dir(sweep_dir)
    meta = {
        "experiment_tag": labels.experiment_tag,
        "timestamp": labels.timestamp,
        "model_label": labels.model_label,
        "sweep_name": labels.sweep_name,
        "sweep_dir": str(sweep_dir),
    }
    runs = [load_run(d) for d in iter_run_dirs(sweep_dir)]
    return runs, meta
