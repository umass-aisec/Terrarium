from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from terrarium.utils import build_log_dir, get_run_timestamp, get_tag_model_subdir


class CompactionLogger:
    """
    Persists every compaction decision (skipped or triggered) to disk, per run.

    This exists so paired-trajectory comparisons (oracle vs. compacted run) can
    diff the *exact* prompt text an agent saw, rather than reconstructing it —
    needed for failure-driven compaction-prompt tuning (ACON-style iteration).
    """

    def __init__(self, config: Dict[str, Any], run_timestamp: Optional[str] = None):
        self.config = config
        self.environment_name = config.get("environment", {}).get("name", "Unknown")
        self.seed = config.get("simulation", {}).get("seed", 0)
        self.tag_model = get_tag_model_subdir(config or {})
        self.run_timestamp = run_timestamp or get_run_timestamp(config)
        self.log_dir = build_log_dir(
            self.environment_name, self.tag_model, self.seed, self.run_timestamp
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.log_dir / "compaction_events.jsonl"
        self.md_path = self.log_dir / "compaction_events.md"

    def reset_log(self) -> None:
        """Clear compaction logs. Called once at the start of each simulation run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self.md_path.write_text(
            f"# Compaction Log - {self.environment_name} (Seed: {self.seed})\n\n",
            encoding="utf-8",
        )

    def log_compaction(
        self,
        *,
        agent_name: str,
        blackboard_id: Any,
        phase: Optional[str],
        iteration: Optional[int],
        technique: str,
        triggered: bool,
        token_count: int,
        token_threshold: int,
        pre_text: str,
        post_text: str,
        pinned_text: str = "",
        summary_text: str = "",
    ) -> None:
        """Append one compaction decision. Safe to call on every agent turn."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "blackboard_id": blackboard_id,
            "phase": phase,
            "iteration": iteration,
            "technique": technique,
            "triggered": triggered,
            "token_count": token_count,
            "token_threshold": token_threshold,
            "pre_text": pre_text,
            "post_text": post_text,
            "pinned_text": pinned_text,
            "summary_text": summary_text,
        }
        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

        try:
            status = "TRIGGERED" if triggered else "skipped"
            header = (
                f"## {agent_name} — blackboard {blackboard_id} — {status} "
                f"({token_count}/{token_threshold} tok, technique={technique})\n"
                f"**Phase:** {phase} | **Iteration:** {iteration} | "
                f"**Timestamp:** {entry['timestamp']}\n"
            )
            body_parts = [header]
            if pinned_text:
                body_parts.append(f"\n### Pinned (never compacted)\n```\n{pinned_text}\n```\n")
            if triggered:
                body_parts.append(f"\n### Pre-compaction\n```\n{pre_text}\n```\n")
                body_parts.append(f"\n### Summary\n```\n{summary_text}\n```\n")
            body_parts.append(f"\n### Post-compaction (what the agent saw)\n```\n{post_text}\n```\n\n---\n")
            with self.md_path.open("a", encoding="utf-8") as f:
                f.write("".join(body_parts))
        except OSError:
            pass
