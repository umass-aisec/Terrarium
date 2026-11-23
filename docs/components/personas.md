---
title: Personas
---

# Personas (skeleton)

Personas shape agent behavior through **prompting only** — no fine-tuning. A persona is a Big Five specification (levels 1–9 per domain) rendered as a first-person trait clause and wrapped around an agent's system prompt. See `terrarium/personas/README.md` for the implementation details and the underlying method (Jiang et al., "Personality Traits in Large Language Models").

## Configuration

Personas are configured under `environment:` in the run config (or via `--persona` / `--personas` on `examples/base_main.py`):

```yaml
environment:
  # One preset applied to every agent:
  persona: "diplomat"
  # Or a per-agent map (every agent name must be present):
  # personas:
  #   General_Hospital_Surgery: "territorial"
  #   Resource_Provisioner: "diplomat"
```

Available presets: `cautious skeptic`, `anxious pushover`, `confident collaborator`, `diplomat`, `territorial` (see `terrarium/personas/presets.py`). Unknown names or a per-agent map missing an agent raise `ValueError` at environment construction.

## Supported environments

- `IsThisSeatTakenEnvironment`: resolves `persona`/`personas` into `env.persona_by_agent`; the prompts class wraps each agent's system prompt via `build_persona_prompt`.
- `HospitalEnvironment`: same pattern — `HospitalEnvironment._resolve_personas` stores `env.persona_by_agent`, and `HospitalPrompts.get_system_prompt(agent_name)` wraps the base system prompt when the agent has a persona. Agents without one get the baseline prompt unchanged.

## TODO

- Document per-environment differences (e.g. trait generation in the seating env).
- Link example configs (`examples/configs/hospital.yaml`, `examples/configs/is_this_seat_taken.yaml`).
