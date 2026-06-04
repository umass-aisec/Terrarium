---
title: Quick Start
---

# Quick Start

Run a simulation by passing one of the example configuration files to the base runner:

```bash
python examples/base_main.py --config examples/configs/meeting_scheduling.yaml
```

Provider keys are loaded from environment variables. Copy the example file and set only the keys for the providers you use:

```bash
cp -n .env.example .env
```

The selected provider and model are configured in YAML:

```yaml
llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
```

For DCOP environments, clone CoLLAB and point Terrarium at it:

```bash
git clone https://github.com/Saad-Mahmud/CoLLAB_SEA.git /path/to/CoLLAB
export TERRARIUM_COLLAB_PATH=/path/to/CoLLAB
```

After a run, inspect the generated logs below `logs/<environment>/<tag_model>/`.
