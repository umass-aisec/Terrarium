#!/usr/bin/env python3
"""
Batch runner for the MeetingScheduling domain across multiple Hugging Face
checkpoints. This script is the *large model* preset: it reuses the shared
batch harness from `run_meeting_models.py` but defaults to a list of 30B–235B
checkpoints that typically require multiple GPUs.

Usage:
    python examples/run_meeting_models_large.py \
        --config examples/configs/meeting_scheduling.yaml

Notes:
- Ensure the MCP server is running first:
      python src/server.py
- vLLM auto-start is expected to be enabled in the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Allow importing the shared small-model harness that lives next to this file.
sys.path.append(str(Path(__file__).resolve().parent))
from examples.run_meeting_models_small import load_yaml, run_single


LARGE_MODELS: List[str] = [
    "Salesforce/Llama-xLAM-2-70b-fc-r",
    "Salesforce/xLAM-2-32b-fc-r",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.2-Exp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch test MeetingScheduling on large vLLM models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/meeting_scheduling.yaml"),
        help="Path to base YAML config (will be copied per model).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of model ids to test (space separated). Defaults to curated large-model list.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/model_batch_tests_large"),
        help="Directory to store stdout/stderr per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    models = args.models or LARGE_MODELS
    main_py = Path("examples/base/main.py")

    print(f"Running MeetingScheduling on {len(models)} models...")
    results = []
    for model in models:
        print(f"\n=== {model} ===")
        try:
            res = run_single(model, base_cfg, main_py, args.log_dir)
            status = "OK" if res["success"] else f"FAIL ({res['error']})"
            print(f"{status}  | {res['duration_sec']}s  | logs: {res['log_prefix']}.stdout/.stderr")
            results.append(res)
        except Exception as exc:  # Catch config or unexpected issues and continue
            err = f"{type(exc).__name__}: {exc}"
            print(f"FAIL (setup) | {err}")
            results.append(
                {
                    "model": model,
                    "success": False,
                    "returncode": -1,
                    "duration_sec": 0,
                    "error": err,
                    "log_prefix": "",
                }
            )

    # Summary
    print("\nSummary:")
    for res in results:
        status = "OK" if res["success"] else "FAIL"
        err = f" - {res['error']}" if res["error"] else ""
        print(f"- {status}: {res['model']} ({res['duration_sec']}s){err}")


if __name__ == "__main__":
    main()
