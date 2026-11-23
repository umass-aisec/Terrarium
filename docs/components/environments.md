---
title: Environments
---

# Environments

The environment is the world the agents are trying to act in. It decides who the agents are, what private state each one sees, which actions are legal, how rewards are computed, and when the simulation is finished. Terrarium's communication layer can move messages around, but the environment gives those messages meaning.

A run usually creates the environment first, attaches a communication network, and then asks the environment to initialize blackboards from that network.

```python
from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.networks import build_communication_network
from terrarium.utils import create_environment

communication_protocol = SequentialCommunicationProtocol(config, tool_logger)

environment = create_environment(
    communication_protocol,
    config["environment"]["name"],
    config,
    tool_logger,
)

communication_protocol.bind_environment(environment)

agent_names = environment.get_agent_names()
communication_network = build_communication_network(agent_names, config)
environment.set_communication_network(communication_network)

await environment.async_init()
```

## Implemented Environments

Terrarium currently ships with single-step DCOP-style environments. Each one gives agents partial information and asks them to coordinate toward a shared objective.

| Environment | What The Agents Coordinate |
| --- | --- |
| `MeetingSchedulingEnvironment` | Meeting attendance over fixed time windows with strict and soft meeting constraints. |
| `PersonalAssistantEnvironment` | Outfit or item selection under preferences, social norms, and availability limits. |
| `SmartGridEnvironment` | Appliance or machine assignment to shared energy resources while avoiding overload. |
| `HospitalEnvironment` | Patient workflow and scarce resource decisions across hospital-inspired departments. |
| `JiraTicketEnvironment` | Issue microtask assignment using skills, priorities, availability, and workload constraints. |

## The Environment Contract

Every environment inherits from `AbstractEnvironment`. The protocol expects the environment to answer the same core questions on every run:

- `get_agent_names()`: who participates?
- `build_agent_context(...)`: what does this agent see right now?
- `done(iteration)`: should the run stop?
- `joint_reward(actions)`: how good is the combined outcome?
- `agent_reward(agent_name, action)`: how should individual credit be computed?
- `get_final_summary()`: what should be written at the end?

That contract keeps the protocol reusable. A meeting scheduler and a hospital workflow can share the same turn structure because the protocol does not need to know what a meeting or patient task means.

```python
from terrarium.environments.abstract_environment import AbstractEnvironment


class WarehouseEnvironment(AbstractEnvironment):
    def __init__(self, communication_protocol, config, tool_logger):
        self.communication_protocol = communication_protocol
        self.config = config
        self.env_config = config["environment"]
        self.tool_logger = tool_logger
        self.agent_names = ["agent_0", "agent_1"]
        self.assignments = {}

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        return {
            "agent_name": agent_name,
            "phase": phase,
            "known_assignments": dict(self.assignments),
        }

    def done(self, iteration):
        return len(self.assignments) == len(self.agent_names)

    def compute_max_joint_reward(self):
        return 1.0

    def joint_reward(self, actions):
        return sum(self.agent_reward(agent, action) for agent, action in actions.items())

    def agent_reward(self, agent_name, action):
        return 1.0 if action else 0.0
```

## Prompts and Tools

An environment normally has two companion modules:

- A `Prompts` class that turns environment state and blackboard messages into system/user prompts.
- A `Tools` class that exposes domain actions, such as assigning a ticket or attending a meeting.

Those files let the environment stay focused on state and scoring while the tool layer handles action schemas and execution.

See the [Environments API](../api/environments.md) for generated reference details.

## Hospital Collusion Configuration (skeleton)

`HospitalEnvironment` supports a native, config-driven collusion block under `environment` (TODO: link the full collusion-via-tools guide once written):

```yaml
environment:
  name: HospitalEnvironment
  collusion:
    enabled: true
    members: ["General_Hospital_Surgery", "Resource_Provisioner"]  # or {indices: [0, 2]} resolved against env.agent_names
    strategy: "covert"                    # free-form label, echoed into context/prompts
    instructions_file: "examples/configs/hospital_collusion_instructions.txt"  # text rendered to adversaries only; relative paths resolve against the config file's directory
  prompt_wrapper: "my_package.my_prompts:MisalignmentPrompts"  # optional "module:Class" prompt decorator
```

- TODO: document the per-agent context keys added by `build_agent_context` — `role` (`adversary`/`cooperative`), `coalition_members`, `collusion_strategy`, `collusion_instructions`, `secret_blackboard_id`.
- TODO: document `HospitalPrompts.get_coalition_instructions` — renders the instructions-file text only for adversaries, appended to the end of the user prompt; prompts are unchanged when collusion is disabled/absent.
- TODO: document `environment.prompt_wrapper` — `wrapper_cls(env, full_config, base_prompts=...)` replaces `env.prompts` natively (signature-compatible with Colosseum's `MisalignmentPrompts`).
