<p align="center">
  <img src="dev/terrarium_logo_rounded.png" alt="Terrarium logo" width="620">
</p>

<!-- <h1 align="center">Terrarium</h1> -->

<p align="center">
  A configurable laboratory for studying decentralized LLM-based multi-agent systems.
</p>

<p align="center">
  <a href="docs/index.md">Docs</a>
  ·
  <a href="https://arxiv.org/pdf/2510.14312v1">Paper</a>
  ·
  <a href="docs/quickstart.md">Quick start</a>
  ·
  <a href="examples/configs">Example configs</a>
  ·
  <a href="docs/api/index.md">API reference</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/terrarium-agents/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/terrarium-agents.svg">
  </a>
  <a href="https://pypi.org/project/terrarium-agents/">
    <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  </a>
  <a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
  <a href="https://arxiv.org/pdf/2510.14312v1">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.14312-b31b1b.svg">
  </a>
</p>

Terrarium is an open-source framework for building controlled multi-agent environments and testing how communication, topology, private information, tools, model providers, and adversarial conditions affect coordination.

It is not primarily a production agent orchestrator or a fixed benchmark. Terrarium is built for researchers, engineers, and students who want to run repeatable multi-agent experiments where the environment, communication structure, and attack surface are explicit.

![Terrarium framework diagram](dev/framework_rounded.png)

## Why Terrarium?

Modern LLM agents increasingly act through tools, APIs, shared memory, and other agents. That makes multi-agent behavior hard to evaluate with only single-agent prompts or fixed task suites. Terrarium gives you a small set of swappable components so you can ask sharper questions:

- What changes when agents communicate through a complete graph instead of a path?
- Does a shared blackboard improve reward, or does it spread bad information faster?
- Which agents need private state for coordination to work?
- How does an attack change the final utility, tool calls, and message trajectory?
- Do different model providers behave differently under the same protocol?

Terrarium makes these experiments available to researchers and practitioners instead of hiding them inside deeply convoluted or less modular frameworks.

## Install

Install the common provider, science, and plotting extras:

```bash
pip install "terrarium-agents[providers,science,plots]"
```

For local development:

```bash
git clone https://github.com/umass-aisec/Terrarium.git
cd Terrarium
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
```

Some DCOP environments use CoLLAB instance generation. If CoLLAB is not available through the repository submodule, clone it and point Terrarium at it:

```bash
git clone https://github.com/Saad-Mahmud/CoLLAB_SEA.git /path/to/CoLLAB
export TERRARIUM_COLLAB_PATH=/path/to/CoLLAB
```

## First Run

Copy the environment template and set only the provider keys you need:

```bash
cp -n .env.example .env
```

Then run a meeting-scheduling simulation:

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Terrarium writes prompts, tool calls, blackboard state, communication network plots, and agent trajectories below:

```text
logs/<environment>/<tag_model>/<run_timestamp>/seed_<seed>/
```

## Minimal Config

Terrarium runs are configured with YAML. The four required sections are `simulation`, `environment`, `communication_network`, and `llm`.

```yaml
simulation:
  max_iterations: 3
  max_planning_rounds: 1
  max_conversation_steps: 3
  seed: 42
  tags:
    - baseline

environment:
  name: MeetingSchedulingEnvironment
  assignment_filling: false
  num_meetings: 2
  timeline_length: 12
  min_participants: 2
  max_participants: 4

communication_network:
  topology: erdos_renyi
  num_agents: 2
  edge_prob: 0.7
  consolidate_channels: true

llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
    params:
      max_tokens: 1024
      temperature: 0.7
```

See [Quick Start](docs/quickstart.md) for the full configuration guide.

## Core Ideas

Terrarium runs are built from components. Swap one component while holding the others fixed to isolate the effect of an environment, protocol, topology, model, or attack.

| Component | Base surface | What it controls |
| --- | --- | --- |
| Agents | `BaseAgent` | LLM/tool loop for each participant. |
| Environments | `AbstractEnvironment` | Task state, private observations, legal actions, rewards, and stopping conditions. |
| Communication protocols | `BaseCommunicationProtocol` | Turn order, phases, and tool routing. |
| Communication Networks | `CommunicationNetwork` | Which agents can share communication channels. |
| Communication Channels | `Blackboard` / `Megaboard` | Append-only message state visible to channel participants. |
| Tools | `ToolsetDiscovery` and environment `Tools` classes | Blackboard communication, environment actions, and optional external MCP tools. |
| LLM clients | `AbstractClient` | Provider-specific generation and tool-call parsing. |

Component guides live in [docs/components](docs/components).


## Repository Layout

```text
terrarium/
  agents/                  # Base agent and factory
  attacks/                 # Reference adversarial components
  communication_protocols/ # Planning/execution protocols
  core/                    # Blackboards, loggers, async helpers
  environments/            # Implemented environments and task tools
  llm/                     # Provider clients and vLLM runtime
  networks/                # Communication topology builders
  tools/                   # Tool discovery and prompt helpers

examples/
  base_main.py             # Standard simulation runner
  attack_main.py           # Attack simulation runner
  configs/                 # Example YAML configs

docs/                      # Documentation source
dashboards/                # Static dashboard utilities
```

## Contributing

Contributions are welcome, especially focused fixes, new environments, new tools, documentation improvements, and reproducible attack or evaluation workflows.

For larger changes, open an issue first so the scope is clear. A good pull request should include:

- A focused description of the change and why it is needed.
- Updated docs or examples when behavior changes.
- Relevant checks, or a short note explaining why a check was not run.
- A small config or reproduction when the change affects simulation behavior.

## Citation

If you use Terrarium in your research, please cite us using the following:

```bibtex
@article{nakamura2025terrarium,
  title={Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies},
  author={Nakamura, Mason and Kumar, Abhinav and Mahmud, Saaduddin and Abdelnabi, Sahar and Zilberstein, Shlomo and Bagdasarian, Eugene},
  journal={arXiv preprint arXiv:2510.14312},
  year={2025}
}
```

## License

Terrarium is released under the MIT License. See [LICENSE](LICENSE).
