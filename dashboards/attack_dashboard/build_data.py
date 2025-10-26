"""Generate attack dashboard data as a static JSON bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not config_path:
        return None
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def summarize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not config:
        return {}
    return {
        "environment": config.get("environment", {}),
        "simulation": config.get("simulation", {}),
        "llm": config.get("llm", {}),
        "attacks": config.get("attacks", []),
    }


def load_attack_runs(log_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not log_root.exists():
        return runs

    for env_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for tag_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            timestamp_dirs = sorted(
                p for p in tag_dir.iterdir()
                if p.is_dir() and not p.name.startswith("seed_")
            )
            for timestamp_dir in timestamp_dirs:
                run_timestamp = timestamp_dir.name
                seed_dirs = sorted(p for p in timestamp_dir.glob("seed_*") if p.is_dir())
                for seed_dir in seed_dirs:
                    summary_path = seed_dir / "attack_summary.json"
                    if not summary_path.exists():
                        continue
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue

                    events: List[Dict[str, Any]] = []
                    events_path = seed_dir / "attack_events.jsonl"
                    if events_path.exists():
                        with events_path.open("r", encoding="utf-8") as handle:
                            for line in handle:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    events.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue

                    scores = []
                    for score_file in sorted(seed_dir.glob("scores_iteration_*.json")):
                        try:
                            score_data = json.loads(score_file.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            continue
                        scores.append({
                            "iteration": score_data.get("iteration"),
                            "global_score": score_data.get("global_score"),
                            "timestamp": score_data.get("timestamp"),
                        })
                    scores.sort(key=lambda s: (s.get("iteration") is None, s.get("iteration")))

                    runs.append({
                        "environment": summary.get("environment", env_dir.name),
                        "tag_model": tag_dir.name,
                        "seed": summary.get("seed", seed_dir.name.replace("seed_", "")),
                        "run_timestamp": summary.get("run_timestamp", run_timestamp),
                        "attack_counts": summary.get("attack_counts", {}),
                        "events": events,
                        "log_dir": str(seed_dir),
                        "scores": scores,
                    })
    return runs


def aggregate_counts(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    aggregate: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for run in runs:
        for attack_type, counts in run.get("attack_counts", {}).items():
            aggregate[attack_type]["success"] += counts.get("success", 0)
            aggregate[attack_type]["failure"] += counts.get("failure", 0)
    return aggregate


def compute_chart_payload(counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    labels, success, failure = [], [], []
    for attack_type, metrics in sorted(counts.items()):
        labels.append(attack_type)
        success.append(metrics.get("success", 0))
        failure.append(metrics.get("failure", 0))
    return {"labels": labels, "success": success, "failure": failure}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static data for the attack dashboard")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config to embed")
    parser.add_argument("--output", type=Path, default=Path("dashboards/attack_dashboard/public/attack_data.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_attack_runs(args.logs_root)
    config = load_config(args.config) if args.config else None
    aggregate = aggregate_counts(runs)
    chart_payload = compute_chart_payload(aggregate)

    data_bundle = {
        "config": summarize_config(config),
        "runs": runs,
        "aggregate_counts": aggregate,
        "chart_data": chart_payload,
        "logs_root": str(args.logs_root.resolve()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data_bundle, handle, indent=2)
    print(f"Wrote dashboard data to {args.output}")


if __name__ == "__main__":
    main()
