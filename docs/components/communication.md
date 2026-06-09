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

`SequentialCommunicationProtocol` currently runs simple protocol: all agents plan, then all agents execute for one iteration of the simulation. During planning, agents can write messages to blackboards. During execution, agents use environment tools to commit task actions.

```python
for planning_round in range(1, max_planning_rounds + 1):
    for agent in environment.agents:
        agent_context = environment.build_agent_context(
            agent.name,
            phase="planning",
            iteration=iteration,
            planning_round=planning_round,
        )
        await communication_protocol.agent_planning_turn(
            agent,
            agent.name,
            agent_context,
            environment,
            iteration,
            planning_round,
        )

for agent in environment.agents:
    agent_context = environment.build_agent_context(
        agent.name,
        phase="execution",
        iteration=iteration,
    )
    await communication_protocol.agent_execution_turn(
        agent,
        agent.name,
        agent_context,
        environment,
        iteration,
    )
```

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

## When to Change Communication

Change the communication protocol when the timing or routing of interaction changes: simultaneous planning, extra critique rounds, hidden channels, or injected system events. Change the network when only the topology changes. Change the environment when the task itself changes.

See the [Communication API](../api/communication.md) for generated reference details.
