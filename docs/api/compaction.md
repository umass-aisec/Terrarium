---
title: Compaction
---

# Compaction

```{warning}
The compaction API is experimental and may change as the implementation matures.
```

Compaction utilities format blackboard event history for prompts and optionally
summarize older events when the formatted history exceeds a token threshold.

```{eval-rst}
.. autofunction:: terrarium.compaction.compactor.compact_events
```
