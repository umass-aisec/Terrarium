from __future__ import annotations
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from terrarium.core.blackboard import format_blackboard_events_for_prompt
import logging
logger = logging.getLogger(__name__)

MECHANISMS: Tuple[str, ...] = (
    "baseline",
    "anchored",
    "extractive",
    "eviction",
    "query_conditioned",
    "structured",
)

# Event kinds the "eviction" mechanism drops outright before summarizing.
# Physical actions (move/settle/stand) are already visible to every agent
# directly through their own environment state each turn (see "WHAT YOU
# KNOW" in the system prompt) — the blackboard's action_executed log is
# redundant for negotiation history, so it's the one class of event that's
# safe to delete rather than paraphrase.
DEFAULT_EVICT_KINDS: FrozenSet[str] = frozenset({"action_executed"})

# Keyword markers the "extractive" mechanism uses to flag a communication
# event as high-signal (a number or a commitment word) and therefore worth
# keeping verbatim instead of folding into the abstractive summary.
_EXTRACTIVE_MARKERS: Tuple[str, ...] = (
    "agree", "deal", "promise", "commit", "accept", "confirm", "will ",
)


def _count_tokens(text: str) -> int:
    return len(text) // 4


def _is_high_signal(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return False
    content = str(payload.get("content") or "")
    if not content:
        return False
    lowered = content.lower()
    return any(ch.isdigit() for ch in content) or any(m in lowered for m in _EXTRACTIVE_MARKERS)


def _split_pinned(events: List[Any], pin_context_events: bool) -> Tuple[List[Any], List[Any]]:
    if not pin_context_events:
        return [], events
    pinned = [e for e in events if isinstance(e, dict) and e.get("kind") == "context"]
    compactable = [e for e in events if not (isinstance(e, dict) and e.get("kind") == "context")]
    return pinned, compactable


def compact_events(
    events,
    llm_client=None,
    model_name=None,
    token_threshold: int = 3000,
    keep_recent: int = 3,
    mechanism: str = "baseline",
    pin_context_events: bool = False,
    cache: Optional[Dict[str, Any]] = None,
    evict_kinds: Optional[Set[str]] = None,
    extract_limit: int = 5,
    compaction_logger=None,
    agent_name: Optional[str] = None,
    blackboard_id: Any = None,
    phase: Optional[str] = None,
    iteration: Optional[int] = None,
) -> str:
    """
    Compact a blackboard's event history into prompt text.

    `mechanism` selects how the aged-out portion of history is compressed
    once the token budget is exceeded (see MECHANISMS). `pin_context_events`
    is an orthogonal modifier, not a mechanism of its own: it holds every
    `kind == "context"` event (channel-purpose / standing-rule messages
    posted once when a blackboard is created) out of whichever mechanism is
    active, so it can never be paraphrased or evicted away once it ages past
    `keep_recent`. mechanism="baseline", pin_context_events=False reproduces
    the original behavior exactly (control group).

    `cache`, if provided, is a mutable dict the caller owns (one per
    blackboard) used by mechanism="anchored" to carry its running summary
    across calls instead of re-summarizing the whole history every turn.
    Ignored by every other mechanism.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown compaction mechanism '{mechanism}'. Known: {MECHANISMS}")

    technique = mechanism + ("+pinned_context" if pin_context_events else "")
    events = events if isinstance(events, list) else []

    pinned, compactable = _split_pinned(events, pin_context_events)
    pinned_text = format_blackboard_events_for_prompt(pinned) if pinned else ""
    formatted = format_blackboard_events_for_prompt(events)
    compactable_formatted = format_blackboard_events_for_prompt(compactable)
    token_count = _count_tokens(compactable_formatted)

    if llm_client is None or model_name is None or token_count <= token_threshold:
        logger.info(f"Compaction skipped: {token_count} tokens (threshold={token_threshold})")
        logger.debug("[NON-COMPACTED PROMPT]\n%s", formatted)
        # Non-compacted path already contains pinned events in natural order,
        # so no separate section is needed — mechanism/pinning only change
        # behavior once compaction actually triggers.
        _log(
            compaction_logger, agent_name=agent_name, blackboard_id=blackboard_id,
            phase=phase, iteration=iteration, technique=technique, triggered=False,
            token_count=token_count, token_threshold=token_threshold,
            pre_text=formatted, post_text=formatted, pinned_text=pinned_text,
        )
        return formatted

    logger.info(f"Compaction triggered: {token_count} tokens exceeds {token_threshold}, summarizing {len(compactable) - keep_recent} events (mechanism={mechanism})")
    logger.debug("[PRE-COMPACTION PROMPT]\n%s", formatted)

    old = compactable[:-keep_recent] if keep_recent else compactable
    if not old:
        _log(
            compaction_logger, agent_name=agent_name, blackboard_id=blackboard_id,
            phase=phase, iteration=iteration, technique=technique, triggered=False,
            token_count=token_count, token_threshold=token_threshold,
            pre_text=formatted, post_text=formatted, pinned_text=pinned_text,
        )
        return formatted

    recent = compactable[-keep_recent:] if keep_recent else []
    recent_text = format_blackboard_events_for_prompt(recent)

    extractive_text = ""
    if mechanism == "baseline":
        summary = _summarize(format_blackboard_events_for_prompt(old), llm_client, model_name)
    elif mechanism == "anchored":
        summary = _summarize_anchored(old, cache, llm_client, model_name)
    elif mechanism == "eviction":
        kinds = evict_kinds if evict_kinds is not None else DEFAULT_EVICT_KINDS
        kept = [e for e in old if not (isinstance(e, dict) and e.get("kind") in kinds)]
        summary = _summarize(format_blackboard_events_for_prompt(kept), llm_client, model_name)
    elif mechanism == "extractive":
        summary, extractive_text = _summarize_extractive(old, llm_client, model_name, extract_limit)
    elif mechanism == "query_conditioned":
        summary = _summarize_query_conditioned(
            format_blackboard_events_for_prompt(old), llm_client, model_name, agent_name, phase
        )
    elif mechanism == "structured":
        summary = _summarize_structured(format_blackboard_events_for_prompt(old), llm_client, model_name)

    sections = []
    if pinned_text:
        sections.append(f"[Standing context]\n{pinned_text}")
    sections.append(f"[Summary of earlier conversation]\n{summary}")
    if extractive_text:
        sections.append(f"[Notable events kept verbatim]\n{extractive_text}")
    sections.append(f"[Recent messages]\n{recent_text}")
    result = "\n\n".join(sections)

    logger.debug("[COMPACTED PROMPT]\n%s", result)
    _log(
        compaction_logger, agent_name=agent_name, blackboard_id=blackboard_id,
        phase=phase, iteration=iteration, technique=technique, triggered=True,
        token_count=token_count, token_threshold=token_threshold,
        pre_text=formatted, post_text=result, pinned_text=pinned_text,
        summary_text=summary,
    )
    return result


def _log(compaction_logger, **kwargs) -> None:
    if compaction_logger is None:
        return
    try:
        compaction_logger.log_compaction(**kwargs)
    except Exception:
        logger.debug("Compaction logging failed", exc_info=True)


_MAX_SUMMARY_TOKENS = 500

# Shared by the bullet-style mechanisms so the instruction can't drift between them.
_BULLET_RULES = (
    "Summarize this agent conversation in 3-5 bullet points. "
    "Be extremely concise — one short sentence per bullet. "
    "No headers, no bold, no markdown formatting, no caveats about missing data. "
    "Only include actual decisions, time slots, and commitments. "
    "If nothing was decided, write only: 'No decisions made.'\n\n"
)


def _ask(llm_client, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Single place every mechanism goes through to call the summarizer model."""
    context = llm_client.init_context(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    _, summary = llm_client.generate_response(
        input=context,
        params={"max_tokens": _MAX_SUMMARY_TOKENS, "model": model_name},
    )
    return summary


def _summarize(text: str, llm_client, model_name: str) -> str:
    return _ask(
        llm_client, model_name,
        "You are a helpful assistant that summarizes agent conversations concisely.",
        f"{_BULLET_RULES}{text}",
    )


def _summarize_incremental(previous_summary: str, new_text: str, llm_client, model_name: str) -> str:
    """Fold new events into an existing summary instead of re-summarizing everything."""
    prompt = (
        "You maintain a running summary of an ongoing agent conversation. "
        "Update the EXISTING SUMMARY by folding in the NEW MESSAGES below it. "
        "Keep the result 3-5 bullet points, one short sentence each. "
        "No headers, no bold, no markdown formatting. "
        "Only decisions, time slots, and commitments — drop anything that's now stale or superseded. "
        "Don't restate facts from the existing summary that haven't changed; just carry them forward silently.\n\n"
        f"EXISTING SUMMARY:\n{previous_summary}\n\nNEW MESSAGES:\n{new_text}"
    )
    return _ask(
        llm_client, model_name,
        "You are a helpful assistant that maintains a running summary of an agent conversation.",
        prompt,
    )


def _summarize_anchored(old_events: List[Any], cache: Optional[Dict[str, Any]], llm_client, model_name: str) -> str:
    """
    Incremental/anchored summarization: reuse the cached summary and fold in
    only events that arrived since the last compaction call for this
    blackboard, instead of re-summarizing the whole history every turn.
    Falls back to a fresh summarize if no cache was supplied (e.g. a caller
    that doesn't maintain per-blackboard state), or on the first call.
    """
    if cache is None:
        return _summarize(format_blackboard_events_for_prompt(old_events), llm_client, model_name)

    previous_summary = cache.get("summary")
    summarized_count = int(cache.get("summarized_count", 0))

    if previous_summary and len(old_events) >= summarized_count:
        new_events = old_events[summarized_count:]
        if not new_events:
            return previous_summary
        new_text = format_blackboard_events_for_prompt(new_events)
        summary = _summarize_incremental(previous_summary, new_text, llm_client, model_name)
    else:
        summary = _summarize(format_blackboard_events_for_prompt(old_events), llm_client, model_name)

    cache["summary"] = summary
    cache["summarized_count"] = len(old_events)
    return summary


def _summarize_extractive(
    old_events: List[Any], llm_client, model_name: str, extract_limit: int
) -> Tuple[str, str]:
    """
    Extractive-then-abstractive: keep events that look like commitments or
    contain numbers verbatim (up to extract_limit, oldest-first), summarize
    only the residue abstractly. Sidesteps the numeric/temporal fidelity loss
    that pure summarization causes, at the cost of only being as good as the
    keyword heuristic in _is_high_signal.
    """
    high_signal = [e for e in old_events if _is_high_signal(e)][:extract_limit]
    high_signal_ids = {id(e) for e in high_signal}
    residue = [e for e in old_events if id(e) not in high_signal_ids]

    extractive_text = format_blackboard_events_for_prompt(high_signal) if high_signal else ""
    if residue:
        summary = _summarize(format_blackboard_events_for_prompt(residue), llm_client, model_name)
    else:
        summary = "No decisions made."
    return summary, extractive_text


def _summarize_query_conditioned(
    text: str, llm_client, model_name: str, agent_name: Optional[str], phase: Optional[str]
) -> str:
    """Condition the summary on who's about to read it, so the compressor
    isn't blind to what the summary will actually be used for."""
    reader_note = ""
    if agent_name:
        reader_note = f"This summary will be read by {agent_name}"
        if phase:
            reader_note += f", who is about to act during the {phase} phase"
        reader_note += (
            ". Prioritize facts most relevant to their next decision — their own "
            "commitments, requests directed at them, and unresolved asks — over "
            "facts that only concern other agents.\n\n"
        )
    return _ask(
        llm_client, model_name,
        "You are a helpful assistant that summarizes agent conversations concisely for a specific reader.",
        f"{reader_note}{_BULLET_RULES}{text}",
    )


def _summarize_structured(text: str, llm_client, model_name: str) -> str:
    """Force the summary into labeled slots instead of free-text bullets, so
    numeric/temporal facts (identifiers, time steps) have a dedicated place to
    survive rather than getting paraphrased away."""
    prompt = (
        "Summarize this agent conversation into these labeled sections. One short "
        "line per item. Omit a section entirely if it has nothing in it. No other "
        "text, no markdown formatting beyond the labels below.\n\n"
        "DECISIONS: finalized agreements\n"
        "COMMITMENTS: promises made — who, to whom, and any condition attached\n"
        "OPEN REQUESTS: asks that haven't been resolved yet\n"
        "STATE FACTS: specific identifiers, positions, quantities, or counts mentioned\n\n"
        f"{text}"
    )
    return _ask(
        llm_client, model_name,
        "You are a helpful assistant that summarizes agent conversations into a fixed structured format.",
        prompt,
    )
