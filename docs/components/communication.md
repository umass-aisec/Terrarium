---
title: Communication
---

# Communication

Communication is the part of Terrarium that turns a set of independent LLM calls into a multi-agent simulation. The protocol decides when agents plan, when they act, which blackboard messages are visible, and how tool calls are routed. The blackboard stores the messages themselves.

In the default runner, the protocol is created before the environment and then bound to it.

```python
from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol

communication_protocol = SequentialCommunicationProtocol(
    config,
    tool_logger,
    run_timestamp=run_timestamp,
)

environment = create_environment(
    communication_protocol,
    config["environment"]["name"],
    config,
    tool_logger,
)

communication_protocol.bind_environment(environment)
```

## Protocol

`SequentialCommunicationProtocol` runs a config-driven phase sequence: by default all agents plan (for `max_planning_rounds` rounds), then all agents execute, once per iteration. During planning, agents can write messages to blackboards. During execution, agents use environment tools to commit task actions. The sequence is driven by `simulation.phases` (see [Quick Start](../quickstart.md#simulation)) and every turn goes through the generic `agent_turn`:

```python
phases = resolve_phases(config["simulation"])
# default: [{"name": "planning", "rounds": max_planning_rounds or 1},
#           {"name": "execution", "rounds": 1}]

for phase in phases:
    for round_num in range(1, phase["rounds"] + 1):
        for agent in environment.agents:
            agent_context = environment.build_agent_context(
                agent.name,
                phase=phase["name"],
                iteration=iteration,
                planning_round=round_num,
            )
            await communication_protocol.agent_turn(
                agent,
                agent.name,
                agent_context,
                environment,
                iteration,
                phase=phase["name"],
                round_num=round_num,
            )
```

`agent_planning_turn`/`agent_execution_turn` remain available as thin wrappers around `agent_turn` with the fixed `"planning"`/`"execution"` phase. Custom phase names get sensible defaults: the planning (blackboard-only) toolset and the execution tool instructions.

The important boundary is that the protocol does not score actions. It decides turn order and routes tools. The environment decides whether an action was useful.

## Blackboards

Terrarium uses blackboards as append-only communication channels. A blackboard has a participant list, and only those participants can read or write normal messages. The protocol creates these channels from the communication network during environment initialization.

```python
blackboard_id = await communication_protocol.generate_comm_network(
    participants=["agent_0", "agent_1"],
    context="Coordinate on task A. Only these agents can see this channel.",
)

await communication_protocol.blackboard_handle_tool_call(
    "post_message",
    "agent_0",
    {
        "blackboard_id": blackboard_id,
        "message": "I can take the morning slot if you cover the afternoon.",
    },
    phase="planning",
    iteration=1,
)

events = communication_protocol.megaboard.get(blackboard_id, "agent_1")
```

System messages can seed or modify blackboards without pretending to be an agent. That is useful for environment context and for protocol-level attack experiments.

```python
await communication_protocol.post_system_message(
    blackboard_id,
    "context",
    {"message": "The channel contains scheduling constraints for task A."},
)
```

## Secret Channels

Environments can designate hidden channels from the run config under `environment.secret_channels`. Each entry names its members and an optional seed context; member names must match the environment's agents. At initialization, the environment creates one blackboard per entry with template `{"private_channel": true, "visibility": "secret"}`. Such a channel is listed only for its members, is rendered as a `PRIVATE` channel in their prompts, is unreadable to non-members, and never receives action-trace logs.

```yaml
environment:
  secret_channels:
    - members: ["agent_0", "agent_1"]
      context: "Optional seed text for the hidden channel."
```

The environment records the created boards in `environment.secret_blackboard_ids` (frozenset of members to blackboard id) and `environment.secret_channel_by_agent` (agent name to blackboard id).

Separately, `environment.allow_agent_channels: true` opts the environment into agent-created private channels at runtime (the `create_channel` tool), without setting `supports_private_channels` on the environment's Tools class.

<!-- TODO: add runnable example configs and a link to the channel-labeling details. -->

## When to Change Communication

Change the communication protocol when the timing or routing of interaction changes: simultaneous planning, extra critique rounds, hidden channels, or injected system events. Change the network when only the topology changes. Change the environment when the task itself changes.

See the [Communication API](../api/communication.md) for generated reference details.
