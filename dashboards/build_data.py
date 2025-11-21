"""Generate dashboard data as a static JSON bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def normalize_tags(raw: Optional[Any]) -> List[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        normalized: List[str] = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    return []


def load_tags_from_yaml(path: Path) -> List[str]:
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []
    simulation = data.get("simulation")
    if not isinstance(simulation, dict):
        return []
    return normalize_tags(simulation.get("tags"))


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
        "scenarios": config.get("scenarios", config.get("attacks", [])),
    }


def load_runs(log_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not log_root.exists():
        return runs

    def process_seed_dir(*, seed_dir: Path, env_dir: Path, tag_model_label: str, run_timestamp: Optional[str]) -> None:
        summary: Dict[str, Any] = {}
        summary_path = seed_dir / "attack_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}

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
                "model_info": score_data.get("model_info"),
                "metadata": score_data.get("metadata"),
            })
        scores.sort(key=lambda s: (s.get("iteration") is None, s.get("iteration")))

        logs_bundle = {
            "blackboards": {},
            "tool_calls": None,
            "agent_prompts_json": None,
            "agent_prompts_markdown": None,
            "agent_trajectories": None,
        }

        for blackboard_file in sorted(seed_dir.glob("blackboard_*.txt")):
            try:
                logs_bundle["blackboards"][blackboard_file.name] = blackboard_file.read_text(encoding="utf-8")
            except OSError:
                continue

        note_text: Optional[str] = summary.get("note")
        tags: List[str] = normalize_tags(summary.get("tags")) or normalize_tags(summary.get("tag"))
        if not tags:
            for candidate in sorted(seed_dir.glob("*.yaml")):
                config_tags = load_tags_from_yaml(candidate)
                if config_tags:
                    tags = config_tags
                    break

        note_path = seed_dir / "experiment_note.txt"
        if note_path.exists():
            try:
                file_note = note_path.read_text(encoding="utf-8").strip()
                if file_note:
                    note_text = file_note
            except OSError:
                pass

        tool_calls_path = seed_dir / "tool_calls.json"
        if tool_calls_path.exists():
            try:
                logs_bundle["tool_calls"] = tool_calls_path.read_text(encoding="utf-8")
            except OSError:
                pass

        prompts_json_path = seed_dir / "agent_prompts.json"
        if prompts_json_path.exists():
            try:
                logs_bundle["agent_prompts_json"] = prompts_json_path.read_text(encoding="utf-8")
            except OSError:
                pass

        prompts_md_path = seed_dir / "agent_prompts.md"
        if prompts_md_path.exists():
            try:
                logs_bundle["agent_prompts_markdown"] = prompts_md_path.read_text(encoding="utf-8")
            except OSError:
                pass

        trajectories_path = seed_dir / "agent_trajectories.json"
        if trajectories_path.exists():
            try:
                logs_bundle["agent_trajectories"] = trajectories_path.read_text(encoding="utf-8")
            except OSError:
                pass

        completion_summary: Optional[Dict[str, Any]] = None
        success_rate: Optional[float] = None
        environment_name = summary.get("environment", env_dir.name)
        completion_specs = {
            "MeetingScheduling": ("total_meetings_scheduled", "total_meetings", "Meeting completion"),
            "PersonalAssistant": ("total_outfits_selected", "total_agents", "Outfit completion"),
            "SmartGrid": ("total_tasks_scheduled", "total_tasks", "Task completion"),
        }
        spec = completion_specs.get(environment_name)
        if spec and scores:
            completed_key, total_key, label = spec
            for score_entry in reversed(scores):
                metadata = score_entry.get("metadata") or {}
                completed_raw = metadata.get(completed_key)
                total_raw = metadata.get(total_key)
                if total_raw is None:
                    continue
                try:
                    completed_val = int(completed_raw) if completed_raw is not None else 0
                    total_val = int(total_raw)
                except (TypeError, ValueError):
                    continue
                if total_val <= 0:
                    continue
                success_rate = (completed_val / total_val) * 100
                completion_summary = {
                    "label": label,
                    "completed": completed_val,
                    "total": total_val,
                    "rate": success_rate,
                }
                break

        runs.append({
            "environment": environment_name,
            "tag_model": tag_model_label,
            "model_info": next((score.get("model_info") for score in reversed(scores) if score.get("model_info")), None),
            "seed": summary.get("seed", seed_dir.name.replace("seed_", "")),
            "run_timestamp": summary.get("run_timestamp", run_timestamp),
            "event_counts": summary.get("attack_counts", {}),
            "events": events,
            "log_dir": str(seed_dir),
            "scores": scores,
            "action_success": completion_summary,
            "success_rate": success_rate,
            "note": note_text,
            "tags": tags,
            "logs": logs_bundle,
        })

    for env_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        seed_dirs = sorted(p for p in env_dir.rglob("seed_*") if p.is_dir())
        for seed_dir in seed_dirs:
            relative_parts = seed_dir.relative_to(env_dir).parts
            if len(relative_parts) < 2:
                continue

            parent_component = relative_parts[-2]
            has_timestamp = not parent_component.startswith("seed_")
            run_timestamp = parent_component if has_timestamp else None

            tag_parts = list(relative_parts)
            tag_parts.pop()  # remove seed component
            if has_timestamp and tag_parts:
                tag_parts.pop()
            if not tag_parts:
                tag_parts = [env_dir.name]
            tag_model_label = "/".join(tag_parts)

            process_seed_dir(
                seed_dir=seed_dir,
                env_dir=env_dir,
                tag_model_label=tag_model_label,
                run_timestamp=run_timestamp,
            )
    return runs



def aggregate_event_counts(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    aggregate: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for run in runs:
        for category, counts in run.get("event_counts", {}).items():
            aggregate[category]["success"] += counts.get("success", 0)
            aggregate[category]["failure"] += counts.get("failure", 0)
    return aggregate


def compute_chart_payload(counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    labels, success, failure = [], [], []
    for category, metrics in sorted(counts.items()):
        labels.append(category)
        success.append(metrics.get("success", 0))
        failure.append(metrics.get("failure", 0))
    return {"labels": labels, "success": success, "failure": failure}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static data for the dashboard")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config to embed")
    parser.add_argument("--output", type=Path, default=Path("dashboards/public/dashboard_data.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.logs_root)
    config = load_config(args.config) if args.config else None
    aggregate = aggregate_event_counts(runs)
    chart_payload = compute_chart_payload(aggregate)

    data_bundle = {
        "config": summarize_config(config),
        "runs": runs,
        "event_totals": aggregate,
        "aggregate_counts": aggregate,  # legacy key for compatibility
        "chart_data": chart_payload,
        "logs_root": str(args.logs_root.resolve()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(data_bundle, handle, indent=2)
    print(f"Wrote dashboard data to {args.output}")


if __name__ == "__main__":
    main()
