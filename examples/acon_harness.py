"""
ACON-style paired-trajectory harness (scaffolding).

Runs the same scenario twice per seed — once with compaction effectively
disabled (the "oracle"), once with a named compaction technique active — and
scores whether the two runs diverge on ground-truth outcome (joint reward,
settled state). Divergent pairs are the input a diagnose-and-rewrite step
would need: compare both transcripts to see what the summary dropped.

Techniques are a mechanism × pinning × retrieval factorial, matching
compaction-techniques-survey.md — pinning and retrieval are modifiers applied
on top of whichever mechanism is active, not peer techniques of their own.
Pinning changes what compact_events() is allowed to summarize; retrieval
doesn't touch compact_events() at all — it's a recall() tool that lets the
compacted agent search the full, never-pruned archive for something the
summary dropped. See terrarium/compaction/compactor.py:MECHANISMS for the
mechanism list (baseline, anchored, extractive, eviction, query_conditioned,
structured).

The oracle config never varies by mechanism/pin/retrieval — it only ever sets
token_threshold to effectively infinity — so running a fresh oracle for every
technique wastes a paired simulation on a config that's identical every time.
Each (run_tag, seed)'s oracle result is cached to disk on first success and
reused by every later `--mechanism`/`--pin`/`--retrieval` combo sharing that
run_tag, so a 24-combo × 3-seed sweep costs 3 oracle + 72 compacted runs
instead of 72 + 72. Pass --fresh-oracle to bypass the cache for one call.

Usage:
    uv run python examples/acon_harness.py \
        --config examples/configs/is_this_seat_taken.yaml \
        --seeds 7,21,42 \
        --mechanism anchored --pin --retrieval \
        --run-tag anchored_pinned_retrieval_vs_oracle

    # A later call reusing the same run-tag + seeds skips re-running the oracle:
    uv run python examples/acon_harness.py \
        --config examples/configs/is_this_seat_taken.yaml \
        --seeds 7,21,42 \
        --mechanism extractive --pin \
        --run-tag anchored_pinned_retrieval_vs_oracle
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from terrarium.utils import load_config, configure_logging
from terrarium.compaction.compactor import MECHANISMS

import base_main  # examples/base_main.py — single-simulation driver, reused as-is


ORACLE_TOKEN_THRESHOLD = 10**9  # high enough that compact_events() never triggers

def _technique_label(mechanism: str, pin_context_events: bool, retrieval_enabled: bool = False) -> str:
    label = mechanism
    if pin_context_events:
        label += "+pinned_context"
    if retrieval_enabled:
        label += "+retrieval"
    return label


def _set_compaction_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    llm_config = config.setdefault("llm", {})
    compaction_config = llm_config.setdefault("compaction", {})
    compaction_config.update(overrides)


def build_oracle_config(base_config_path: str, seed: int, run_tag: str) -> Dict[str, Any]:
    """Build the oracle (compaction-disabled) config for one seed. Never
    varies by mechanism/pin/retrieval — see the module docstring on why the
    oracle is cached instead of rebuilt per technique."""
    oracle_config = load_config(base_config_path)
    oracle_config.setdefault("simulation", {})["seed"] = seed
    oracle_config["simulation"]["run_timestamp"] = f"acon_{run_tag}_seed{seed}_oracle"
    _set_compaction_overrides(oracle_config, {"token_threshold": ORACLE_TOKEN_THRESHOLD})
    return oracle_config


def build_compacted_config(
    base_config_path: str,
    seed: int,
    mechanism: str,
    pin_context_events: bool,
    retrieval_enabled: bool,
    run_tag: str,
) -> Dict[str, Any]:
    """Build the compacted (technique-under-test) config for one seed."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Known: {MECHANISMS}")

    compacted_config = load_config(base_config_path)
    compacted_config.setdefault("simulation", {})["seed"] = seed
    label = _technique_label(mechanism, pin_context_events, retrieval_enabled)
    compacted_config["simulation"]["run_timestamp"] = f"acon_{run_tag}_seed{seed}_{label}"
    _set_compaction_overrides(
        compacted_config,
        {
            "mechanism": mechanism,
            "pin_context_events": pin_context_events,
            "retrieval_enabled": retrieval_enabled,
        },
    )
    return compacted_config


def _oracle_cache_path(run_tag: str, seed: int, results_dir: str) -> Path:
    return Path(results_dir) / "_oracle_cache" / f"{run_tag}_seed{seed}.json"


def _load_cached_oracle(run_tag: str, seed: int, results_dir: str) -> Optional[Dict[str, Any]]:
    path = _oracle_cache_path(run_tag, seed, results_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_oracle_cache(run_tag: str, seed: int, result: Dict[str, Any], results_dir: str) -> None:
    path = _oracle_cache_path(run_tag, seed, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


async def run_pair(
    base_config_path: str,
    seed: int,
    mechanism: str,
    pin_context_events: bool,
    retrieval_enabled: bool,
    run_tag: str,
    reuse_oracle: bool = True,
    results_dir: str = "logs/acon_trials",
) -> Dict[str, Any]:
    """
    Run the oracle and compacted variants for one seed.

    With reuse_oracle=True (the default), a successful oracle result for this
    (run_tag, seed) is cached to disk and reused across every later call that
    shares the same run_tag and seed — regardless of which mechanism/pin/
    retrieval combo those later calls test — since the oracle config never
    depends on any of those. A failed cached run is not reused; it's retried.
    """
    oracle_result = _load_cached_oracle(run_tag, seed, results_dir) if reuse_oracle else None
    if oracle_result is not None and oracle_result.get("success"):
        print(f"[seed {seed}] reusing cached oracle from {oracle_result.get('log_dir')}")
    else:
        oracle_config = build_oracle_config(base_config_path, seed, run_tag)
        oracle_result = await base_main.run_simulation(oracle_config)
        if reuse_oracle and oracle_result.get("success"):
            _save_oracle_cache(run_tag, seed, oracle_result, results_dir)

    compacted_config = build_compacted_config(
        base_config_path, seed, mechanism, pin_context_events, retrieval_enabled, run_tag
    )
    compacted_result = await base_main.run_simulation(compacted_config)

    return {
        "seed": seed,
        "technique": _technique_label(mechanism, pin_context_events, retrieval_enabled),
        "mechanism": mechanism,
        "pin_context_events": pin_context_events,
        "retrieval_enabled": retrieval_enabled,
        "oracle": oracle_result,
        "compacted": compacted_result,
    }


def score_divergence(
    pair_result: Dict[str, Any],
    reward_delta_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Flag whether the compacted run diverged from the oracle on ground truth.

    Reads IsThisSeatTakenEnvironment.get_final_summary() fields (joint_reward,
    settled). Swap the extraction below if scoring a different environment —
    this is the one place environment-specific logic lives.
    """
    oracle = pair_result["oracle"]
    compacted = pair_result["compacted"]

    if not oracle.get("success") or not compacted.get("success"):
        return {
            "seed": pair_result["seed"],
            "technique": pair_result["technique"],
            "diverged": None,
            "reason": "one or both runs failed",
            "oracle_error": oracle.get("error"),
            "compacted_error": compacted.get("error"),
        }

    oracle_summary = oracle.get("final_summary") or {}
    compacted_summary = compacted.get("final_summary") or {}

    oracle_reward = float(oracle_summary.get("joint_reward", 0.0))
    compacted_reward = float(compacted_summary.get("joint_reward", 0.0))
    reward_delta = compacted_reward - oracle_reward

    oracle_settled = bool(oracle_summary.get("settled", False))
    compacted_settled = bool(compacted_summary.get("settled", False))

    settled_diverged = oracle_settled and not compacted_settled
    reward_diverged = abs(reward_delta) > reward_delta_threshold

    return {
        "seed": pair_result["seed"],
        "technique": pair_result["technique"],
        "diverged": settled_diverged or reward_diverged,
        "oracle_reward": oracle_reward,
        "compacted_reward": compacted_reward,
        "reward_delta": reward_delta,
        "oracle_settled": oracle_settled,
        "compacted_settled": compacted_settled,
        "settled_diverged": settled_diverged,
        "reward_diverged": reward_diverged,
        "oracle_log_dir": oracle.get("log_dir"),
        "compacted_log_dir": compacted.get("log_dir"),
    }


async def run_trial_suite(
    base_config_path: str,
    seeds: List[int],
    mechanism: str,
    pin_context_events: bool,
    retrieval_enabled: bool,
    run_tag: str,
    reward_delta_threshold: float = 1.0,
    results_dir: str = "logs/acon_trials",
    reuse_oracle: bool = True,
) -> List[Dict[str, Any]]:
    """Run paired trials across seeds and persist a scored manifest to disk."""
    label = _technique_label(mechanism, pin_context_events, retrieval_enabled)
    records = []
    for seed in seeds:
        pair_result = await run_pair(
            base_config_path,
            seed,
            mechanism,
            pin_context_events,
            retrieval_enabled,
            run_tag,
            reuse_oracle=reuse_oracle,
            results_dir=results_dir,
        )
        record = score_divergence(pair_result, reward_delta_threshold=reward_delta_threshold)
        records.append(record)
        status = "DIVERGED" if record["diverged"] else ("skipped" if record["diverged"] is None else "matched")
        print(f"[seed {seed}] {label}: {status} (reward_delta={record.get('reward_delta')})")

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{run_tag}_{label}.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} trial records to {manifest_path}")

    return records


if __name__ == "__main__":
    configure_logging()
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run paired oracle-vs-compacted trials for ACON-style comparison."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--seeds", type=str, required=True, help="Comma-separated seeds, e.g. 7,21,42"
    )
    parser.add_argument(
        "--mechanism", type=str, default="baseline", choices=list(MECHANISMS)
    )
    parser.add_argument(
        "--pin", action="store_true", help="Apply the context-event pinning modifier."
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Give the compacted agent a recall() tool over the full uncompacted archive.",
    )
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--reward-delta-threshold", type=float, default=1.0)
    parser.add_argument(
        "--fresh-oracle",
        action="store_true",
        help="Ignore any cached oracle for this run-tag/seed and run a new one.",
    )

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    asyncio.run(
        run_trial_suite(
            args.config,
            seeds,
            args.mechanism,
            args.pin,
            args.retrieval,
            args.run_tag,
            reward_delta_threshold=args.reward_delta_threshold,
            reuse_oracle=not args.fresh_oracle,
        )
    )
