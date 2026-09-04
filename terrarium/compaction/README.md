# Blackboard compaction

Agents in Terrarium read their channel history as text injected into the prompt
each turn. That history grows without bound, so long runs eventually exceed the
context window or spend most of their budget re-reading old chatter.

This package compresses that history at prompt-assembly time. The underlying
event log is **never pruned** — only the prompt string built from it is lossy.
That distinction is what makes retrieval (§5) possible and what the paired
oracle-vs-compacted harness (§7) relies on.

The research question is which compression strategy loses the *least
decision-relevant* information, so the package is structured as a set of
interchangeable mechanisms behind one entry point.

---

## 1) Opt-in

**Compaction is off unless `llm.compaction` is present in the run config.**
With no such block, agents see the raw transcript exactly as they did before
this package existed, and no compaction logger is created.

This matters because the default `token_threshold` is 3000: were compaction
on by default, every environment would silently start receiving summarized
prompts once a channel passed that size.

---

## 2) Entry point

```python
from terrarium.compaction.compactor import compact_events, MECHANISMS

text = compact_events(
    events,                    # raw blackboard events for one channel
    llm_client=client,
    model_name="gpt-5.4-nano",
    token_threshold=3000,
    keep_recent=3,
    mechanism="baseline",
    pin_context_events=False,
    cache=per_blackboard_dict,
    ...
)
```

`SequentialCommunicationProtocol` calls this once per agent, per channel, per
turn. Passing `llm_client=None` or `model_name=None` disables compaction for
that call.

---

## 3) Pipeline

1. **Split.** If `pin_context_events` is set, events with `kind == "context"`
   are held aside as *pinned*; everything else is compactable.
2. **Measure.** Token count is estimated as `len(text) // 4` over the
   compactable text.
3. **Gate.** If the count is `<= token_threshold`, return the raw formatted
   transcript unchanged and log a `triggered=false` entry.
4. **Partition.** Otherwise split the compactable events into `old` (everything
   but the last `keep_recent`) and `recent`.
5. **Compress** `old` according to `mechanism` (§4).
6. **Assemble**, in this order, omitting empty sections:

```
[Standing context]                 ← pinned events, if any
[Summary of earlier conversation]  ← mechanism output
[Notable events kept verbatim]     ← extractive mechanism only
[Recent messages]                  ← the last keep_recent events, verbatim
```

The threshold is measured against the **uncompacted** history, which only
grows — see §8.

---

## 4) Mechanisms

Selected by `mechanism`; the list lives in `MECHANISMS`.

| Mechanism | What it does | Trade-off |
|---|---|---|
| `baseline` | One abstractive summary of `old` into 3–5 bullets | Control condition. Loses numbers and identifiers to paraphrase |
| `anchored` | Keeps a running summary in `cache` and folds in only events since the last call | Avoids re-summarizing history each turn; errors compound across folds |
| `eviction` | Drops event kinds in `evict_kinds` outright, then summarizes the rest | Physical actions are already visible via environment state, so dropping them is cheap. Wrong `evict_kinds` silently deletes signal |
| `extractive` | Keeps up to `extract_limit` high-signal events verbatim, summarizes only the residue | Protects numbers and commitments. Only as good as the keyword heuristic |
| `query_conditioned` | Tells the summarizer who will read it and in which phase | Prioritizes the reader's own commitments and open asks. Summary is no longer reusable across agents |
| `structured` | Forces labeled slots: `DECISIONS`, `COMMITMENTS`, `OPEN REQUESTS`, `STATE FACTS` | Gives identifiers a dedicated place to survive. Rigid when a conversation does not fit the slots |

`eviction` defaults to dropping `action_executed` events. `extractive` flags an
event as high-signal if its `payload.content` contains a digit or one of
`agree`, `deal`, `promise`, `commit`, `accept`, `confirm`, `will ` — it reads
only that field, so events without message content can never be kept verbatim.

All six route their model call through a single `_ask()` helper, so
`max_tokens` (500) and the shared bullet instructions are defined once.

---

## 5) Modifiers

These are orthogonal to `mechanism` — they compose with any of the six, rather
than being alternatives to them.

### `pin_context_events`

Holds `kind == "context"` events (channel-purpose and standing-rule messages
posted once at channel creation) out of whichever mechanism is active, so they
can never be paraphrased or evicted once they age past `keep_recent`. They are
re-emitted verbatim under `[Standing context]`.

### `retrieval_enabled`

Gives agents a `recall(query, blackboard_id=None)` tool that searches the
**full, uncompacted** log via `Megaboard.recall()` — a case-insensitive
substring match, most recent first. This does not change what compaction
produces; it gives the agent a way to recover something the summary dropped.

Note the tool is registered for the **planning phase only**, while compaction
runs in both phases — so an agent acting on a compacted context during
execution cannot call `recall()`.

`mechanism="baseline"` with both modifiers off reproduces the original
pre-compaction behaviour exactly, which is the control condition.

---

## 6) Configuration

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

| Key | Default | Meaning |
|---|---|---|
| `token_threshold` | `3000` | Compact only above this estimated token count |
| `keep_recent` | `3` | Trailing events always kept verbatim |
| `mechanism` | `baseline` | One of `MECHANISMS` |
| `pin_context_events` | `false` | Protect `kind == "context"` events |
| `retrieval_enabled` | `false` | Register the `recall()` tool |
| `extract_limit` | `5` | Max verbatim events for `extractive` |
| `evict_kinds` | `["action_executed"]` | Event kinds `eviction` drops |
| `provider` + provider block | falls back to `llm.provider` | Summarizer model |

The compaction tier is resolved once and cached. If it cannot be constructed,
a warning is logged and compaction falls back to the **agent's own client** —
so a typo in the compaction block degrades cost, not correctness.

---

## 7) Logging

When enabled, `CompactionLogger` writes to the run's log directory:

- **`compaction_events.jsonl`** — one record per call, triggered or not, with
  `technique`, `token_count`, `token_threshold`, `pre_text`, `post_text`,
  `pinned_text`, `summary_text`, plus agent/channel/phase/iteration.
- **`compaction_events.md`** — the same, human-readable.

`pre_text`/`post_text` are what makes a paired comparison possible: for any
turn you can diff exactly what the agent would have seen against what it did
see.

`examples/acon_harness.py` uses this to run each scenario twice per seed —
once with compaction effectively disabled (the *oracle*,
`token_threshold=10**9`) and once with a technique active — and score whether
the two runs diverge on joint reward and settled state. Oracle runs are cached
per `(run_tag, seed)` since they do not vary by technique.

```bash
python examples/acon_harness.py \
    --config examples/configs/is_this_seat_taken.yaml \
    --seeds 7,21,42 \
    --mechanism anchored --pin --retrieval \
    --run-tag anchored_pinned_retrieval_vs_oracle
```

---

## 8) Known limitations

Measured across 28 recorded runs; all are cost or fidelity issues, not
correctness bugs.

- **The trigger tests total size, not new content.** Once a channel first
  crosses `token_threshold` it can never fall back under, so every subsequent
  turn issues a fresh summarization. In the recorded runs this was 669 of 669
  post-threshold calls. Gating on events-since-last-summary would cut this
  substantially.
- **Identical inputs are not memoized.** 26% of summarizer calls (151 of 592)
  received byte-identical input — typically an agent that posted nothing,
  causing the next agent to re-summarize the same text. For `baseline`,
  `eviction`, `extractive` and `structured` the output is a pure function of
  the input, so this is safely cacheable. `anchored` already short-circuits.
- **Skipped compactions are logged in full.** `pre_text` and `post_text` are
  identical on a skip but both are written, so an oracle run produced ~6.4 MB
  of logs describing compactions that never happened.
- **Token counting is `len // 4`,** not a real tokenizer. Fine as a relative
  trigger, not comparable to provider token counts.
- **`recall()` is planning-phase only** (§5).
- **No unit tests.** The six mechanisms are exercised only through full runs;
  `compact_events` has no direct test coverage.
