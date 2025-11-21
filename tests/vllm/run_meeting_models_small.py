#!/usr/bin/env python3
"""
Batch runner for the MeetingScheduling domain across multiple Hugging Face
checkpoints. This script is the *small/medium* model preset: it swaps the vLLM
model in the provided config, runs the simulation, and logs per-model results
without stopping on errors (e.g., OOM).

Usage:
    python examples/run_meeting_models.py \
        --config examples/configs/meeting_scheduling.yaml

Notes:
- Ensure the MCP server is running first:
      python src/server.py
- vLLM auto-start is expected to be enabled in the config.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import yaml


# Curated set of small/medium models that typically fit on a single 48GB GPU
# for MeetingScheduling. (Large models have their own runner:
# `examples/run_meeting_models_large.py`.)
SMALL_MODELS: List[str] = [
    # Mistral / Hermes family
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-2-Mistral-7B",
    "teknium/OpenHermes-2.5-Mistral-7B",
    # Qwen 2.5 / 3
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B-Instruct",
    # InternLM
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm2_5-20b-chat",
    "internlm/internlm2_5-7b",
    # Granite
    "ibm-granite/granite-20b-code-instruct",
    "ibm-granite/granite-7b-base",
    "ibm-granite/granite-7b-instruct",
    # GLM
    "THUDM/glm-4-9b-chat",
    "THUDM/glm-4-9b",
    # Jamba
    "ai21labs/Jamba-v0.1",
    "ai21labs/Jamba-Instruct",
]

# Backwards-compat: keep the original name for callers that may still import it
DEFAULT_MODELS = SMALL_MODELS


def load_yaml(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def slugify(model: str) -> str:
    """Make a filesystem-friendly slug from a model id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def sanitize_model_id(value: str) -> str:
    """Mirror llm_server.vllm.runtime's _sanitize_model_id for log lookup."""
    cleaned = value.lower().replace("/", "_").replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "model"


def derive_max_len(model: str, default: int) -> int:
    """
    Some models have tighter max_position_embeddings than our baseline config.
    Return an override when we know it, otherwise fall back to the config default.
    """
    m = model.lower()
    # Llama 3 family (Hermes-2-Pro-Llama-3-8B etc.) caps at 8192.
    if "llama-3" in m or "llama3" in m or "llama-3.1" in m:
        return 8192
    # Qwen3 4B/8B/14B typically 8192; keep default unless higher.
    if "qwen3" in m:
        return min(default, 8192) if default else 8192
    # Jamba often runs better with shorter contexts on 48GB; use 8192.
    if "jamba" in m:
        return 8192
    # Granite 7B/20B ship with 8192.
    if "granite" in m:
        return 8192
    # GLM-4-9B ships with 8k context.
    if "glm-4-9b" in m or "glm4-9b" in m:
        return 8192
    return default


# Per-model chat template overrides (vLLM expects a path or packaged template name)
CHAT_TEMPLATE_OVERRIDES = {
    # Mistral tool-calling template packaged with vLLM
    "mistralai/mistral-7b-instruct-v0.3": "tool_chat_template_mistral.jinja",
    "mistralai/mistral-7b-v0.3": "tool_chat_template_mistral.jinja",
    # Llama-3 / Hermes on Llama-3 use the official llama3.1 JSON template
    "nousresearch/hermes-2-pro-llama-3-8b": "tool_chat_template_llama3.1_json.jinja",
    # GLM 4.5/4-9B template bundled with vLLM
    "thudm/glm-4-9b": "tool_chat_template_glm4.jinja",
    "thudm/glm-4-9b-chat": "tool_chat_template_glm4.jinja",
    "zai-org/glm-4.5": "tool_chat_template_glm4.jinja",
    "zai-org/glm-4.5-air": "tool_chat_template_glm4.jinja",
    # Granite tool template
    "ibm-granite/granite-7b-instruct": "tool_chat_template_granite.jinja",
    "ibm-granite/granite-20b-code-instruct": "tool_chat_template_granite_20b_fc.jinja",
}


def write_temp_config(base_cfg: Dict, model: str) -> Path:
    cfg = copy.deepcopy(base_cfg)

    vllm_cfg = cfg.get("llm", {}).get("vllm")
    if not vllm_cfg or not vllm_cfg.get("models"):
        raise ValueError("Config must include llm.vllm.models[0].")

    # Swap the checkpoint and set a served_model_name for clarity
    vllm_cfg["models"][0]["checkpoint"] = model
    vllm_cfg["models"][0].setdefault("served_model_name", model.split("/")[-1])

    # Adjust max_model_len per model when known limits differ from baseline config.
    base_max_len = vllm_cfg["models"][0].get("max_model_len")
    vllm_cfg["models"][0]["max_model_len"] = derive_max_len(model, base_max_len)

    # Apply chat template override when known
    template = CHAT_TEMPLATE_OVERRIDES.get(model.lower())
    if template:
        vllm_cfg["models"][0]["chat_template"] = template

    # Add a run note for easier log inspection
    cfg.setdefault("simulation", {})["note"] = f"batch_test {model}"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    return tmp_path


def run_single(model: str, base_cfg: Dict, main_py: Path, log_dir: Path) -> Dict:
    temp_cfg = write_temp_config(base_cfg, model)
    cmd = [sys.executable, str(main_py), "--config", str(temp_cfg)]

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - start

    slug = slugify(model)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{slug}.stdout").write_text(proc.stdout)
    stderr_path = log_dir / f"{slug}.stderr"
    stderr_path.write_text(proc.stderr)

    success = proc.returncode == 0
    err_msg = None
    if not success:
        lower_err = proc.stderr.lower() if proc.stderr else ""
        if "out of memory" in lower_err or "cuda out of memory" in lower_err:
            err_type = "OOM"
        else:
            err_type = "runtime_error"
        tail = proc.stderr.splitlines()[-1] if proc.stderr else ""
        err_msg = f"{err_type}: {tail}"

        # Append recent vLLM server logs (if present) to help debug 404/launch issues
        model_id = sanitize_model_id(model)
        vllm_log = Path("logs") / "vllm" / f"{model_id}.log"
        if vllm_log.exists():
            try:
                blob = vllm_log.read_text()
                tail_lines = "\n".join(blob.splitlines()[-200:])  # last 200 lines
                with stderr_path.open("a") as f:
                    f.write("\n\n=== vLLM log tail ===\n")
                    f.write(tail_lines)
                    f.write("\n=== end vLLM log tail ===\n")
            except Exception:
                pass

    temp_cfg.unlink(missing_ok=True)
    return {
        "model": model,
        "success": success,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 1),
        "error": err_msg,
        "log_prefix": str(log_dir / slug),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test MeetingScheduling on multiple vLLM models.")
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
        help="Optional list of model ids to test (space separated). Defaults to curated small/medium list.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/model_batch_tests"),
        help="Directory to store stdout/stderr per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    models = args.models or DEFAULT_MODELS
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
