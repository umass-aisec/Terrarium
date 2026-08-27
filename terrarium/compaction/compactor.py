from __future__ import annotations
from typing import Any, Dict, List, Optional
from terrarium.core.blackboard import format_blackboard_events_for_prompt
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _count_tokens(text: str) -> int:
    return len(text) // 4

def compact_events(
    events,
    llm_client=None,
    model_name=None,
    token_threshold: int = 3000,
    keep_recent: int = 3,
    pin_context_events: bool = False,
    compaction_logger=None,
    agent_name: Optional[str] = None,
    blackboard_id: Any = None,
    phase: Optional[str] = None,
    iteration: Optional[int] = None,
) -> str:
    """
    Compact a blackboard's event history into prompt text.

    pin_context_events=False reproduces the original baseline behavior exactly
    (control group). pin_context_events=True holds every `kind == "context"`
    event (channel-purpose / standing-rule messages posted once when a
    blackboard is created) out of the summarizer entirely, so it can never be
    paraphrased away once it ages past `keep_recent` — this is the
    "constraint pinning" technique.
    """
    technique = "pinned_context" if pin_context_events else "baseline"
    events = events if isinstance(events, list) else []

    if pin_context_events:
        pinned = [e for e in events if isinstance(e, dict) and e.get("kind") == "context"]
        compactable = [e for e in events if not (isinstance(e, dict) and e.get("kind") == "context")]
    else:
        pinned = []
        compactable = events

    pinned_text = format_blackboard_events_for_prompt(pinned) if pinned else ""
    formatted = format_blackboard_events_for_prompt(events)
    compactable_formatted = format_blackboard_events_for_prompt(compactable)
    token_count = _count_tokens(compactable_formatted)

    if llm_client is None or model_name is None or token_count <= token_threshold:
        logger.info(f"Compaction skipped: {token_count} tokens (threshold={token_threshold})")
        logger.debug(f"[NON-COMPACTED PROMPT]\n{formatted}")
        # Non-compacted path already contains pinned events in natural order,
        # so no separate section is needed — pinning only changes behavior
        # once compaction actually triggers.
        result = formatted
        _log(
            compaction_logger, agent_name=agent_name, blackboard_id=blackboard_id,
            phase=phase, iteration=iteration, technique=technique, triggered=False,
            token_count=token_count, token_threshold=token_threshold,
            pre_text=formatted, post_text=result, pinned_text=pinned_text,
        )
        return result

    logger.info(f"Compaction triggered: {token_count} tokens exceeds {token_threshold}, summarizing {len(compactable) - keep_recent} events")
    logger.debug(f"[PRE-COMPACTION PROMPT]\n{formatted}")

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
    old_text = format_blackboard_events_for_prompt(old)
    summary = _summarize(old_text, llm_client, model_name)
    recent_text = format_blackboard_events_for_prompt(recent)

    sections = []
    if pinned_text:
        sections.append(f"[Standing context]\n{pinned_text}")
    sections.append(f"[Summary of earlier conversation]\n{summary}")
    sections.append(f"[Recent messages]\n{recent_text}")
    result = "\n\n".join(sections)

    logger.debug(f"[COMPACTED PROMPT]\n{result}")
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


def _summarize(text: str, llm_client, model_name: str) -> str:
    prompt = (
        "Summarize this agent conversation in 3-5 bullet points. "
        "Be extremely concise — one short sentence per bullet. "
        "No headers, no bold, no markdown formatting, no caveats about missing data. "
        "Only include actual decisions, time slots, and commitments. "
        "If nothing was decided, write only: 'No decisions made.'\n\n"
        f"{text}"
    )
    context = llm_client.init_context(
        system_prompt="You are a helpful assistant that summarizes agent conversations concisely.",
        user_prompt=prompt,
    )
    _, summary = llm_client.generate_response(
        input=context,
        params={"max_tokens": 500, "model": model_name},
    )
    return summary
