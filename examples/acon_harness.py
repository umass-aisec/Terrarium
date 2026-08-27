"""
ACON-style paired-trajectory harness (scaffolding).

Runs the same scenario twice per seed — once with compaction effectively
disabled (the "oracle"), once with a named compaction technique active — and
scores whether the two runs diverge on ground-truth outcome (joint reward,
settled state). Divergent pairs are exactly what the not-yet-built
diagnose-and-rewrite step needs: feed both transcripts to a strong model, ask
what the compacted run's summary lost, use the answer to edit the compaction
prompt. See `diagnose_divergent_pair` below for that stub.

Techniques are a mechanism × pinning × retrieval factorial, matching
compaction-techniques-survey.md — pinning and retrieval are modifiers applied
on top of whichever mechanism is active, not peer techniques of their own.
Pinning changes what compact_events() is allowed to summarize; retrieval
doesn't touch compact_events() at all — it's a recall() tool that lets the
compacted agent search the full, never-pruned archive for something the
summary dropped. See terrarium/compaction/compactor.py:MECHANISMS for the
mechanism list (baseline, anchored, extractive, eviction, query_conditioned,
structured).

Usage:
    uv run python examples/acon_harness.py \
        --config examples/configs/is_this_seat_taken_seed7.yaml \
        --seeds 7,21,42 \
        --mechanism anchored --pin --retrieval \
        --run-tag anchored_pinned_retrieval_vs_oracle
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def build_paired_configs(
    base_config_path: str,
    seed: int,
    mechanism: str,
    pin_context_events: bool,
    retrieval_enabled: bool,
    run_tag: str,
) -> Dict[str, Dict[str, Any]]:
    """Build the (oracle, compacted) config pair for one seed."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Known: {MECHANISMS}")

    # Load twice rather than deep-copying once — load_config() reparses the
    # YAML fresh each call, so there's no risk of the two configs sharing
    # nested dict references.
    oracle_config = load_config(base_config_path)
    compacted_config = load_config(base_config_path)

    for cfg in (oracle_config, compacted_config):
        cfg.setdefault("simulation", {})["seed"] = seed

    label = _technique_label(mechanism, pin_context_events, retrieval_enabled)
    oracle_config["simulation"]["run_timestamp"] = f"acon_{run_tag}_seed{seed}_oracle"
    compacted_config["simulation"]["run_timestamp"] = f"acon_{run_tag}_seed{seed}_{label}"

    # retrieval doesn't touch compact_events() at all — it's a tool
    # (recall()) the agent calls, not a compaction parameter. The oracle
    # never compacts, so there's nothing for it to recall; only the
    # compacted config needs the flag.
    _set_compaction_overrides(oracle_config, {"token_threshold": ORACLE_TOKEN_THRESHOLD})
    _set_compaction_overrides(
        compacted_config,
        {
            "mechanism": mechanism,
            "pin_context_events": pin_context_events,
            "retrieval_enabled": retrieval_enabled,
        },
    )

    return {"oracle": oracle_config, "compacted": compacted_config}


async def run_pair(
    base_config_path: str,
    seed: int,
    mechanism: str,
    pin_context_events: bool,
    retrieval_enabled: bool,
    run_tag: str,
) -> Dict[str, Any]:
    """Run the oracle and compacted variants for one seed, sequentially."""
    configs = build_paired_configs(
        base_config_path, seed, mechanism, pin_context_events, retrieval_enabled, run_tag
    )

    oracle_result = await base_main.run_simulation(configs["oracle"])
    compacted_result = await base_main.run_simulation(configs["compacted"])

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
) -> List[Dict[str, Any]]:
    """Run paired trials across seeds and persist a scored manifest to disk."""
    label = _technique_label(mechanism, pin_context_events, retrieval_enabled)
    records = []
    for seed in seeds:
        pair_result = await run_pair(
            base_config_path, seed, mechanism, pin_context_events, retrieval_enabled, run_tag
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


def diagnose_divergent_pair(record: Dict[str, Any]) -> str:
    """
    TODO — not yet implemented. This is the next piece of the ACON loop.

    For a divergent record, this should:
      1. Read `record['compacted_log_dir']/compaction_events.jsonl` for the
         exact pre/post-compaction text the compacted agent saw.
      2. Read the oracle's full (uncompacted) prompt logs from
         `record['oracle_log_dir']/agent_prompts.md` for the same turns.
      3. Feed both to a strong model: "the compacted agent failed where the
         full-context agent succeeded — what specific fact, commitment, or
         instruction did the summary drop or distort that explains this?"
      4. Return the diagnosis so it can be turned into a compaction-prompt or
         compaction-logic edit (e.g. a new pinned event kind, a rewritten
         `_summarize()` instruction).
    """
    raise NotImplementedError(
        "Diagnose-and-rewrite step not yet built — see docstring for the plan."
    )


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
        )
    )
