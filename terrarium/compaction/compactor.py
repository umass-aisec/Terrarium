from __future__ import annotations
from terrarium.core.blackboard import format_blackboard_events_for_prompt
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _count_tokens(text: str) -> int:
    # Cheap heuristic instead of using a full tokenizer
    return len(text) // 4

def compact_events(
    events,
    llm_client=None,
    model_name=None,
    token_threshold: int = 10,
    keep_recent: int = 1,
) -> str:
    formatted = format_blackboard_events_for_prompt(events)
    token_count = _count_tokens(formatted)

    if llm_client is None or model_name is None or token_count <= token_threshold:
        logger.info(f"Compaction skipped: {token_count} tokens (threshold={token_threshold})")
        logger.debug(f"[NON-COMPACTED PROMPT]\n{formatted}")
        return formatted

    logger.info(f"Compaction triggered: {token_count} tokens exceeds {token_threshold}, summarizing {len(events) - keep_recent} events")
    logger.debug(f"[PRE-COMPACTION PROMPT]\n{formatted}")  

    old = events[:-keep_recent]
    recent = events[-keep_recent:]
    old_text = format_blackboard_events_for_prompt(old)
    summary = _summarize(old_text, llm_client, model_name)
    recent_text = format_blackboard_events_for_prompt(recent)
    result = f"[Summary of earlier conversation]\n{summary}\n\n[Recent messages]\n{recent_text}"
    
    logger.debug(f"[COMPACTED PROMPT]\n{result}")
    return result


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