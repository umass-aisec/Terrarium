from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_hospital(agent_name: str) -> Optional[str]:
    if agent_name == "Resource_Provisioner":
        return None
    if agent_name.startswith("General_Hospital_"):
        return "General_Hospital"
    if agent_name.startswith("St_Marys_Center_"):
        return "St_Marys_Center"
    # fallback: best-effort prefix split
    if "_" in agent_name:
        return agent_name.split("_", 1)[0]
    return None


@dataclass(frozen=True)
class _ScheduleStats:
    schedule_calls: int
    unique_patient_steps: int
    duplicated_agent_patient_steps: int
    max_dup_per_agent_patient_step: int
    duplicated_patient_steps_global: int


def _schedule_stats(tool_events: List[Dict[str, Any]]) -> _ScheduleStats:
    sched_events = [e for e in tool_events if e.get("tool_name") == "schedule_patient"]
    keys_agent: Counter[Tuple[str, str, int]] = Counter()
    keys_global: Counter[Tuple[str, int]] = Counter()

    for e in sched_events:
        agent = str(e.get("agent_name") or "")
        args = e.get("arguments") or {}
        pid = args.get("patient_id")
        step = args.get("step_index")
        if pid is None or step is None:
            continue
        try:
            step_i = int(step)
        except Exception:
            continue
        keys_agent[(agent, str(pid), step_i)] += 1
        keys_global[(str(pid), step_i)] += 1

    duplicated_agent = sum(1 for v in keys_agent.values() if v > 1)
    duplicated_global = sum(1 for v in keys_global.values() if v > 1)

    return _ScheduleStats(
        schedule_calls=len(sched_events),
        unique_patient_steps=len(keys_global),
        duplicated_agent_patient_steps=duplicated_agent,
        max_dup_per_agent_patient_step=(max(keys_agent.values()) if keys_agent else 0),
        duplicated_patient_steps_global=duplicated_global,
    )


@dataclass(frozen=True)
class _TransferStats:
    total_transfers: int
    by_resource_amount: Dict[str, int]
    by_hospital_amount: Dict[str, int]
    rp_transfers: int


def _transfer_stats(tool_events: List[Dict[str, Any]]) -> _TransferStats:
    transfers = [e for e in tool_events if e.get("tool_name") == "transfer_resources"]
    by_res: Counter[str] = Counter()
    by_hosp: Counter[str] = Counter()
    rp = 0

    for e in transfers:
        agent = str(e.get("agent_name") or "")
        if agent == "Resource_Provisioner":
            rp += 1
        args = e.get("arguments") or {}
        res = args.get("resource_type")
        to_h = args.get("to_hospital")
        amt = args.get("amount")
        try:
            amt_i = int(amt)
        except Exception:
            amt_i = 0
        if res is not None:
            by_res[str(res)] += amt_i
        if to_h is not None:
            by_hosp[str(to_h)] += amt_i

    return _TransferStats(
        total_transfers=len(transfers),
        by_resource_amount=dict(by_res),
        by_hospital_amount=dict(by_hosp),
        rp_transfers=rp,
    )


@dataclass(frozen=True)
class _FailureStats:
    num_step_failures: int
    by_agent_missing: Dict[str, Dict[str, int]]
    by_hospital_missing: Dict[str, Dict[str, int]]


def _failure_stats(final_summary: Dict[str, Any]) -> _FailureStats:
    step_failures = final_summary.get("step_resource_failures") or []
    by_agent: Dict[str, Counter[str]] = defaultdict(Counter)
    by_hosp: Dict[str, Counter[str]] = defaultdict(Counter)
    if not isinstance(step_failures, list):
        step_failures = []

    for f in step_failures:
        if not isinstance(f, dict):
            continue
        agent = str(f.get("agent") or "")
        missing = f.get("missing") or []
        if not isinstance(missing, list):
            continue
        hosp = _infer_hospital(agent) or "Unknown"
        for r in missing:
            by_agent[agent][str(r)] += 1
            by_hosp[hosp][str(r)] += 1

    return _FailureStats(
        num_step_failures=len(step_failures),
        by_agent_missing={k: dict(v) for k, v in by_agent.items()},
        by_hospital_missing={k: dict(v) for k, v in by_hosp.items()},
    )


def _schedule_fill_stats(final_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use final_summary['schedule'] to estimate schedule utilization:
    schedule[agent] is a dict(start_time -> list(patients)).
    """
    sched = final_summary.get("schedule")
    if not isinstance(sched, dict):
        return {}

    by_agent: Dict[str, Dict[str, float]] = {}
    by_hosp_slots: Counter[str] = Counter()

    for agent, entries in sched.items():
        if not isinstance(entries, dict):
            continue
        slots_per_time: List[int] = []
        for t, patients in entries.items():
            if isinstance(patients, list):
                slots_per_time.append(len(patients))
        if not slots_per_time:
            continue
        cap = max(slots_per_time) if slots_per_time else 0
        total_slots = sum(slots_per_time)
        hours = len(slots_per_time)
        util = (total_slots / float(cap * hours)) if cap > 0 and hours > 0 else 0.0
        by_agent[str(agent)] = {
            "capacity_est": float(cap),
            "scheduled_hours": float(hours),
            "total_slots": float(total_slots),
            "utilization": float(util),
        }
        hosp = _infer_hospital(str(agent))
        if hosp:
            by_hosp_slots[hosp] += total_slots

    return {
        "by_agent": by_agent,
        "by_hospital_total_slots": dict(by_hosp_slots),
    }


def _iter_run_dirs(outputs_dir: Path) -> Iterable[Path]:
    for metrics_path in outputs_dir.rglob("metrics.json"):
        run_dir = metrics_path.parent
        if (run_dir / "run_config.json").exists() and (run_dir / "final_summary.json").exists() and (run_dir / "tool_events.json").exists():
            yield run_dir


def _short(d: Dict[str, Any], max_items: int = 4) -> str:
    items = list(d.items())
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    shown = items[:max_items]
    rest = len(items) - len(shown)
    s = ", ".join(f"{k}:{v}" for k, v in shown)
    if rest > 0:
        s += f", +{rest} more"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", type=Path, required=True)
    ap.add_argument("--only-run-id", type=str, default=None)
    args = ap.parse_args()

    outputs_dir: Path = args.outputs_dir
    if not outputs_dir.exists():
        raise SystemExit(f"Not found: {outputs_dir}")

    run_dirs = sorted(_iter_run_dirs(outputs_dir))
    if args.only_run_id:
        run_dirs = [r for r in run_dirs if args.only_run_id in r.name]

    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        run_config = _read_json(run_dir / "run_config.json")
        final_summary = _read_json(run_dir / "final_summary.json")
        tool_events = _read_json(run_dir / "tool_events.json")

        attack_type = metrics.get("attack_type") or metrics.get("strategy") or "unknown"
        joint = (metrics.get("meeting_outcome") or {}).get("joint_reward") or metrics.get("joint_reward") or final_summary.get("joint_reward")
        conv = (metrics.get("meeting_outcome") or {}).get("converged_patients")
        total_pat = (metrics.get("meeting_outcome") or {}).get("total_patients")

        sched_stats = _schedule_stats(tool_events if isinstance(tool_events, list) else [])
        transf_stats = _transfer_stats(tool_events if isinstance(tool_events, list) else [])
        fail_stats = _failure_stats(final_summary if isinstance(final_summary, dict) else {})
        fill_stats = _schedule_fill_stats(final_summary if isinstance(final_summary, dict) else {})

        adversaries = run_config.get("adversaries") if isinstance(run_config, dict) else None

        print(f"\n{run_dir.name}")
        print(f"  attack_type: {attack_type}")
        print(f"  joint_reward: {joint}  converged: {conv}/{total_pat}")
        print(f"  adversaries: {adversaries}")
        print(
            "  schedule:",
            f"calls={sched_stats.schedule_calls}",
            f"unique_patient_steps={sched_stats.unique_patient_steps}",
            f"dup_agent_patient_steps={sched_stats.duplicated_agent_patient_steps}",
            f"max_dup={sched_stats.max_dup_per_agent_patient_step}",
            f"dup_patient_steps_global={sched_stats.duplicated_patient_steps_global}",
        )
        print(
            "  transfers:",
            f"total={transf_stats.total_transfers}",
            f"rp_transfers={transf_stats.rp_transfers}",
            f"by_resource_amount=({_short(transf_stats.by_resource_amount)})",
            f"by_hospital_amount=({_short(transf_stats.by_hospital_amount)})",
        )
        print(
            "  step_resource_failures:",
            f"count={fail_stats.num_step_failures}",
            f"by_hospital=({_short({h: sum(v.values()) for h, v in fail_stats.by_hospital_missing.items()})})",
        )
        if fail_stats.by_agent_missing:
            top_agents = {a: sum(v.values()) for a, v in fail_stats.by_agent_missing.items()}
            print(f"    by_agent=({_short(top_agents)})")
        if fill_stats:
            print(f"  schedule_fill: by_hospital_total_slots={fill_stats.get('by_hospital_total_slots')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

