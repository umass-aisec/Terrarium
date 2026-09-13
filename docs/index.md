---
title: Terrarium
---

# Terrarium

Terrarium is a hackable, modular, and configurable framework for studying decentralized LLM-based multi-agent systems in isolated environments.

```{image} ../dev/framework_rounded.png
:alt: Terrarium framework diagram
```

```{toctree}
:hidden:
:caption: Introduction
:maxdepth: 2

basic_usage
comparison_to_other_frameworks
quickstart
```

```{toctree}
:hidden:
:caption: Components
:maxdepth: 2

components/agents
components/environments
components/communication
components/networks
components/tools
components/llm
components/attacks
components/logging
```

```{toctree}
:hidden:
:caption: Tutorial
:maxdepth: 2

attacks
```

```{toctree}
:hidden:
:caption: API
:maxdepth: 2

api/index
```

```{toctree}
:hidden:
:caption: Development
:maxdepth: 1

release_notes
```

## Installation

Install Terrarium with the common provider, science, and plotting extras:

```bash
pip install "terrarium-agents[providers,science,plots]"
```

For a source checkout:

```bash
git clone https://github.com/umass-aisec/Terrarium.git
cd Terrarium
uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
```

## First Run

Run one of the included simulations with an example configuration:

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Terrarium writes logs below `logs/<environment>/<tag_model>/` for prompts, tool calls, trajectories, and blackboard state.
