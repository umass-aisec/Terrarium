---
title: Quick Start
---

# Quick Start

Run a simulation by passing one of the example configuration files to the base runner:

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

For details on the configuration file, view the following sections.

**Terrarium run configs are YAML files with four required top-level sections**:

```yaml
simulation:
  max_iterations: 1
  max_planning_rounds: 1
  max_conversation_steps: 3
  seed: 42

environment:
  name: SomeEnvironment

communication_network:
  topology: complete
  num_agents: 2

llm:
  provider: openai
```

## Simulation

The `simulation` section controls run length, reproducibility, and optional run metadata.

```yaml
simulation:
  max_iterations: 1
  max_planning_rounds: 1
  max_conversation_steps: 3
  seed: 42
  tags:
    - baseline
  note: "Short description for this run"
```

- `max_iterations`: maximum environment iterations/steps where a *step constitutes as all agents taking an action in the environment*.
- `max_planning_rounds`: number of planning rounds before execution round (dependent on the communication protocol).
- `max_conversation_steps`: maximum LLM/tool loop steps per agent turn.
- `seed`: random seed for reproducible environment and network generation.
- `tags` and `note`: optional metadata used for experiment tracking.

## Environment

The `environment` section selects the task and passes task-specific settings. Use the implemented environment class name as `environment.name`.

```yaml
environment:
  name: MeetingSchedulingEnvironment
  assignment_filling: false
  # Below is specific to MeetingScheduling
  num_meetings: 2
  timeline_length: 12
  min_participants: 2
  max_participants: 4
  soft_meeting_ratio: 0.6
```

Implemented environment names include `MeetingSchedulingEnvironment`, `PersonalAssistantEnvironment`, `SmartGridEnvironment`, `HospitalEnvironment`, and `JiraTicketEnvironment`. Each will have varying task-specific settings, so view the example task specifications in the example configs for more details at `examples/configs`.

## Communication Network

The `communication_network` section controls who can communicate with whom. Terrarium builds blackboard channels from this network.

```yaml
communication_network:
  topology: erdos_renyi
  num_agents: 6
  edge_prob: 0.7
  consolidate_channels: true
```

- `topology`: one of `complete`, `path`, `star`, `erdos_renyi`, `watts_strogatz`, or `barabasi_albert`.
- `num_agents`: number of agents in the simulation.
- `consolidate_channels`: when true, Terrarium can combine dense network regions into shared multi-agent channels using cliques. Useful for preventing many unnecessary communication channels.
- Random topologies need extra parameters, such as `edge_prob` for `erdos_renyi`, `k` and `rewire_prob` for `watts_strogatz`, or `m` for `barabasi_albert`.

## Inference

### API Inference
If using API-based inference, provider keys are loaded from environment variables. Copy the example file and set only the keys for the providers you use:

```bash
cp -n .env.example .env
```

The selected provider and model are configured in YAML:

```yaml
llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
    params:
      max_tokens: 1024
      temperature: 0.7
```

### vLLM Inference

If using local vLLM inference, no provider API key is required. Install the vLLM extra and make sure CUDA is available:

```bash
pip install "terrarium-agents[vllm]"
```

Then set the provider and model server in your YAML config:

```yaml
llm:
  provider: vllm
  vllm:
    auto_start_server: true
    persistent_server: false
    models:
      - checkpoint: Qwen/Qwen2.5-7B-Instruct
        served_model_name: Qwen2.5-7B-Instruct
        host: 127.0.0.1
        port: 8001
        tensor_parallel_size: 1
```

## External Environments

For CoLLAB environments, clone CoLLAB and point Terrarium at it:

```bash
git clone https://github.com/Saad-Mahmud/CoLLAB_SEA.git /path/to/CoLLAB
export TERRARIUM_COLLAB_PATH=/path/to/CoLLAB
```

This is required for DCOP environments that use CoLLAB instance generation, including `MeetingSchedulingEnvironment`, `PersonalAssistantEnvironment`, and `SmartGridEnvironment`.


## External MCP Servers

```{warning}
This is experimental and may not work for your particular configuration.
```

External MCP servers can be attached per LLM client:

```yaml
llm:
  provider: openai
  external_mcp_servers:
    - name: filesystem
      url: http://127.0.0.1:9000/mcp
      enabled: true
      tool_prefix: fs_
```
