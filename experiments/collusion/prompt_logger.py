from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExperimentPromptLogger:
    """Write agent prompts into an experiment run directory.

    This mirrors `src.logger.PromptLogger` output (JSON + Markdown) but keeps
    artifacts co-located under `experiments/<project>/outputs/.../<run>/`.
    """

    def __init__(self, *, log_root: Path, environment_name: str, seed: int) -> None:
        self.environment_name = str(environment_name or "Unknown")
        self.seed = int(seed)
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)

        self.log_file_path = self.log_root / "agent_prompts.json"
        self.md_log_path = self.log_root / "agent_prompts.md"

        if not self.log_file_path.exists():
            self.log_file_path.write_text("[]", encoding="utf-8")
        if not self.md_log_path.exists():
            self.md_log_path.write_text(
                f"# Agent Prompts Log - {self.environment_name} (Seed: {self.seed})\n\n",
                encoding="utf-8",
            )

    def log_prompts(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        phase: str,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> None:
        timestamp = datetime.now().isoformat()

        entry = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": round_num,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        # JSON
        try:
            existing = json.loads(
                self.log_file_path.read_text(encoding="utf-8") or "[]"
            )
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        existing.append(entry)
        self.log_file_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Markdown
        meta_parts = [f"**Phase:** {phase}"]
        if iteration is not None:
            meta_parts.append(f"**Iteration:** {iteration}")
        if round_num is not None:
            meta_parts.append(f"**Round:** {round_num}")
        metadata = " | ".join(meta_parts)

        md_entry = (
            f"## {agent_name} - {metadata}\n"
            f"**Timestamp:** {timestamp}\n\n"
            "### System Prompt\n"
            "```\n"
            f"{system_prompt}\n"
            "```\n\n"
            "### User Prompt\n"
            "```\n"
            f"{user_prompt}\n"
            "```\n\n"
            "---\n\n"
        )
        with self.md_log_path.open("a", encoding="utf-8") as f:
            f.write(md_entry)

    def reset_log(self) -> None:
        self.log_file_path.write_text("[]", encoding="utf-8")
        self.md_log_path.write_text(
            f"# Agent Prompts Log - {self.environment_name} (Seed: {self.seed})\n\n",
            encoding="utf-8",
        )
