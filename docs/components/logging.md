---
title: Logging
---

# Logging

Logs are the evidence trail for a Terrarium run. The framework records what the agents saw, what tools they called, what messages appeared on each blackboard, and how the environment summarized the final outcome. This matters because the interesting failure is often not the final score alone, but the coordination path that produced it.

The base runner creates tool and trajectory loggers at the start of a simulation.

```python
from terrarium.core.logger import AgentTrajectoryLogger, ToolCallLogger

tool_logger = ToolCallLogger(
    environment_name,
    seed,
    config,
    run_timestamp=run_timestamp,
)

trajectory_logger = AgentTrajectoryLogger(
    environment_name,
    seed,
    config,
    run_timestamp=run_timestamp,
)
```

The communication protocol creates its own `BlackboardLogger`, and attack experiments can use `AttackLogger`.

```python
from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.core.logger import AttackLogger

communication_protocol = SequentialCommunicationProtocol(
    config,
    tool_logger,
    run_timestamp=run_timestamp,
)

attack_logger = AttackLogger(
    environment_name,
    seed,
    config,
    run_timestamp=run_timestamp,
)
```

## Log Layout

Run artifacts are written below:

```text
logs/<environment>/<tag_model>/<run_timestamp>/seed_<seed>/
```

Typical files include:

| File | What It Captures |
| --- | --- |
| `tool_calls.json` | Every tool call, parameters, result, phase, iteration, and timing. |
| `trajectories.json` | Agent reasoning steps, tool calls, and model metadata. |
| `blackboard_<id>.txt` | Full visible event history for each blackboard. |
| `communication_network.png` | A plot of the topology used to create channels. |
| `attack_events.jsonl` | Attack events and metadata when an attack logger is used. |
| `experiment_note.txt` | Optional run note from config or CLI. |

## Reading a Run

A useful debugging order is:

1. Start with `tool_calls.json` to see whether agents committed valid actions.
2. Open the relevant `blackboard_<id>.txt` file to inspect what each channel contained before execution.
3. Check `trajectories.json` when an agent did not call a tool or called the wrong one.
4. Compare the final environment summary with the best possible reward.

The logs are intentionally redundant. Tool calls show what changed state, blackboards show what information was available, and trajectories show how the model moved from prompt to action.

See the [Core API](../api/core.md) for generated logger reference details.
