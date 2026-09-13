# Blackboard Compaction

Implementation:
- Compactor: `terrarium/compaction/compactor.py`
- Logger: `terrarium/compaction/logger.py`
- Call site: `terrarium/communication_protocols/sequential.py`
- Retrieval tool: `terrarium/core/blackboard.py` (`Megaboard.recall`), registered in `terrarium/tools/discovery.py`

Agents read their channel history as text injected into the prompt each turn.
That history grows without bound, so long runs eventually exceed the context
window or spend most of the budget re-reading old messages.

Compaction compresses that history when the prompt is assembled. The underlying
event log is **never pruned** — only the prompt text built from it is lossy.
That is what allows the `recall` tool (section 5) to recover dropped detail.

## 1) Opt-in

Compaction is **off unless `llm.compaction` is present** in the run config.
Without that block, agents receive the raw transcript and no compaction logger
is created.

This matters because `token_threshold` defaults to 3000. If compaction were on
by default, every environment would begin receiving summarized prompts as soon
as a channel exceeded that size.

## 2) Entry point

```python
from terrarium.compaction.compactor import compact_events, MECHANISMS

text = compact_events(
    events,                   # raw blackboard events for one channel
    llm_client=client,
    model_name="gpt-5.4-nano",
    token_threshold=3000,
    keep_recent=3,
    mechanism="baseline",
    pin_context_events=False,
    cache=per_blackboard_dict,
)
```

`SequentialCommunicationProtocol` calls this once per agent, per channel, per
turn. Passing `llm_client=None` or `model_name=None` disables compaction for
that call.

## 3) Pipeline

1. If `pin_context_events` is set, events with `kind == "context"` are held
   aside as pinned; everything else is compactable.
2. Token count is estimated as `len(text) // 4` over the compactable text.
3. If the count is at or below `token_threshold`, the raw transcript is
   returned unchanged and a `triggered=false` record is logged.
4. Otherwise the compactable events are split into `old` (everything but the
   last `keep_recent`) and `recent`.
5. `old` is compressed according to `mechanism`.
6. The result is assembled in this order, omitting empty sections:

```
[Standing context]                 pinned events, if any
[Summary of earlier conversation]  mechanism output
[Notable events kept verbatim]     extractive mechanism only
[Recent messages]                  the last keep_recent events, verbatim
```

The threshold is measured against the uncompacted history, which only grows
(see section 8).

## 4) Mechanisms

`mechanism` selects how `old` is compressed. The list is exported as
`MECHANISMS`.

- `baseline`: one abstractive summary of `old` into 3-5 bullets. Numbers and
  identifiers can be lost to paraphrase.
- `anchored`: keeps a running summary in `cache` and folds in only the events
  that arrived since the last call. Avoids re-summarizing the whole history,
  but errors carry forward across folds.
- `eviction`: drops event kinds listed in `evict_kinds` before summarizing the
  rest. Physical actions are already visible to agents through their own
  environment state, so dropping them is cheap; a wrong `evict_kinds` silently
  removes signal.
- `extractive`: keeps up to `extract_limit` high-signal events verbatim and
  summarizes only the residue. Protects numbers and commitments, but is only as
  good as the keyword heuristic.
- `query_conditioned`: tells the summarizer which agent will read it and in
  which phase, so the summary prioritizes that agent's own commitments and open
  requests. The result is not reusable across agents.
- `structured`: forces labeled sections (`DECISIONS`, `COMMITMENTS`,
  `OPEN REQUESTS`, `STATE FACTS`) instead of free-text bullets, giving
  identifiers a dedicated slot. Rigid when a conversation does not fit them.

`eviction` defaults to dropping `action_executed`. `extractive` treats an event
as high-signal when its `payload.content` contains a digit or one of `agree`,
`deal`, `promise`, `commit`, `accept`, `confirm`, `will `; it reads only that
field, so events without message content are never kept verbatim.

All mechanisms issue their model call through `_ask()`, so `max_tokens` (500)
and the shared bullet instructions are defined in one place.

## 5) Modifiers

These compose with any mechanism rather than replacing it.

### pin_context_events

Holds `kind == "context"` events — the channel-purpose and standing-rule
messages posted once when a channel is created — out of whichever mechanism is
active, so they cannot be paraphrased or evicted once they age past
`keep_recent`. They are re-emitted verbatim under `[Standing context]`.

### retrieval_enabled

Registers a `recall(query, blackboard_id?)` tool that searches the full,
uncompacted log through `Megaboard.recall()` using a case-insensitive substring
match, most recent first. It does not change what compaction produces; it gives
an agent a way to recover a detail the summary dropped.

`mechanism="baseline"` with both modifiers off reproduces the pre-compaction
behavior exactly.

## 6) Logging and outputs

When compaction is enabled, `CompactionLogger` writes to the run's log
directory:

- `compaction_events.jsonl`: one record per call, triggered or not, with
  `technique`, `token_count`, `token_threshold`, `pre_text`, `post_text`,
  `pinned_text`, `summary_text`, and the agent, channel, phase and iteration.
- `compaction_events.md`: the same records, human-readable.

`pre_text` and `post_text` make it possible to reconstruct exactly what an
agent would have seen without compaction and what it saw with it, for any turn.

## 7) Configuration reference (llm.compaction section)

```yaml
llm:
  provider: foundry
  foundry:
    model: gpt-5.5
    params: { max_tokens: 1500 }

  compaction:                 # presence of this block enables compaction
    provider: foundry
    foundry:
      model: gpt-5.4-nano     # a cheaper tier than the agents use
      params: { max_tokens: 128, temperature: 0.0 }
    token_threshold: 3000
    keep_recent: 3
    mechanism: baseline
    pin_context_events: false
    retrieval_enabled: false
    extract_limit: 5
    evict_kinds: [action_executed]
```

Key fields in `llm.compaction:` (defaults shown where relevant):

- `token_threshold` (default `3000`): compact only above this estimated token count
- `keep_recent` (default `3`): trailing events always kept verbatim
- `mechanism` (default `baseline`): one of `MECHANISMS`
- `pin_context_events` (default `false`): protect `kind == "context"` events
- `retrieval_enabled` (default `false`): register the `recall` tool
- `extract_limit` (default `5`): maximum verbatim events for `extractive`
- `evict_kinds` (default `["action_executed"]`): event kinds `eviction` drops
- `provider` and its provider block: the summarizer model; falls back to
  `llm.provider` when omitted

The compaction client is resolved once and cached. If it cannot be constructed,
a warning is logged and compaction falls back to the agent's own client, so a
misconfigured compaction block affects cost rather than correctness.

## 8) Notes / limitations

- The trigger compares total history size against `token_threshold`, not the
  amount of new content. Once a channel crosses the threshold it does not fall
  back under, so every later turn on that channel issues a fresh summarization.
- Summaries are not memoized. When no new events arrive between turns, the same
  input is summarized again. For `baseline`, `eviction`, `extractive` and
  `structured` the output is a pure function of the input, so these calls are
  avoidable; `anchored` already short-circuits.
- Skipped compactions are logged in full. `pre_text` and `post_text` are
  identical on a skip, and both are written, so log volume grows quickly on runs
  where compaction rarely triggers.
- Token counting is `len(text) // 4`, not a real tokenizer. It is a relative
  trigger and is not comparable to provider token counts.
- `recall` is registered for the planning phase only, while compaction runs in
  both phases, so it is unavailable to an agent acting during execution.
- The mechanisms have no unit tests; they are currently exercised only through
  full runs.
