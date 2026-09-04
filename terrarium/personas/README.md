# Personas

Big Five personality shaping for Terrarium agents, applied **purely through
prompting** — no fine-tuning, no model changes.

The method follows Jiang et al., *"Personality Traits in Large Language
Models"* ([paper](https://www.researchgate.net/publication/372074802)). Each
personality dimension is set to a level from 1–9, rendered as qualified
bipolar adjectives, and wrapped around the agent's task as a first-person
self-description.

The package is a pure library: no framework imports, no I/O, no LLM calls. It
turns a trait specification into a string.

---

## 1) Quick start

```python
from terrarium.personas import PRESETS, Persona, build_persona_prompt

prompt = build_persona_prompt(PRESETS["diplomat"], task="You are seated in row 2.")
```

produces:

```
For the following task, respond in a way that matches this description: "A
diplomat. I'm extremely extraverted, extremely talkative, extremely bold,
extremely adventurous, extremely trustful, extremely moral, extremely kind,
extremely cooperative, and extremely sympathetic."

You are seated in row 2.
```

In a run config, set a persona on the environment instead — see §6.

---

## 2) Trait levels

Every trait takes an integer level from **1 to 9**, where **5 is neutral**.
Below 5 selects the adjective marking the low end of the pair, above 5 the high
end; the distance from neutral picks the qualifier:

| Level | Rendering | | Level | Rendering |
|---|---|---|---|---|
| 1 | `extremely <low>` | | 6 | `a bit <high>` |
| 2 | `very <low>` | | 7 | `<high>` (bare) |
| 3 | `<low>` (bare) | | 8 | `very <high>` |
| 4 | `a bit <low>` | | 9 | `extremely <high>` |
| 5 | — contributes nothing | | | |

A persona whose traits are all 5 renders an empty clause, and
`build_persona_prompt` then returns the task unchanged.

---

## 3) Domains and facets

Traits are keyed by either a **domain** code or a **facet**.

The five domains:

| Code | Domain |
|---|---|
| `EXT` | Extraversion |
| `AGR` | Agreeableness |
| `CON` | Conscientiousness |
| `NEU` | Neuroticism |
| `OPE` | Openness |

A domain key shapes every facet under that domain at once. A facet key — either
its code (`"A1"`) or its name (`"Trust"`) — shapes just that one:

```python
Persona(traits={"AGR": 9})     # every Agreeableness facet
Persona(traits={"Trust": 9})   # -> "I'm extremely trustful."
```

The adjective database in `adjectives.py` is transcribed from Table 12 of the
paper, itself adapted from Goldberg's bipolar trait markers. Each row pairs a
facet with its low-end and high-end adjective.

---

## 4) Verbosity

`verbosity` controls how many adjectives a **domain-level** trait expands into.
It has no effect on facet-level traits, which are already a single row.

| Value | Behaviour |
|---|---|
| `"compact"` (default) | A fixed, hand-picked 4–5 marker subset per domain |
| `"full"` | Every Table 12 facet marker for that domain |

The same trait at `EXT=9`:

```
compact: I'm extremely extraverted, extremely talkative, extremely bold, and
         extremely adventurous.
full:    I'm extremely friendly, extremely extraverted, extremely talkative,
         extremely bold, extremely assertive, extremely active, extremely
         energetic, extremely adventurous and daring, and extremely cheerful.
```

The compact subset is hardcoded rather than sampled, so the same `Persona`
always renders the same prompt. Its selection rationale — including which
facets the paper's own worked example uses — is documented inline above
`COMPACT_TRAITS` in `adjectives.py`.

---

## 5) Presets

Five presets ship in `presets.py`, available as `PRESETS[name]` and as module
constants. Unspecified domains stay neutral.

| Name | EXT | AGR | CON | NEU | OPE | Constant |
|---|---|---|---|---|---|---|
| `cautious skeptic` | 3 | 2 | 8 | – | – | `CAUTIOUS_SKEPTIC` |
| `anxious pushover` | – | 8 | – | 8 | – | `ANXIOUS_PUSHOVER` |
| `confident collaborator` | 7 | 7 | 5 | – | – | `CONFIDENT_COLLABORATOR` |
| `diplomat` | 9 | 9 | – | – | – | `DIPLOMAT` |
| `territorial` | 1 | 1 | – | – | – | `TERRITORIAL` |

`diplomat` and `territorial` are deliberate polar opposites on the two
dimensions that matter most for negotiation, which is what makes them a usable
A/B pair.

Custom personas need no preset:

```python
Persona.from_big_five(ext=2, agr=8, con=6, name="quiet helper",
                      description="A quiet helper.")
```

---

## 6) Using personas in a run

Set either key under `environment:` — they are mutually exclusive in practice,
and `personas` wins if both are present.

```yaml
environment:
  persona: diplomat          # one preset for every agent
```

```yaml
environment:
  personas:                  # per-agent; must cover every agent
    agent_0: territorial
    agent_1: territorial
    agent_2: diplomat
    agent_3: diplomat
    agent_4: diplomat
```

An unknown name raises `ValueError` listing the available presets; a `personas`
map missing any agent raises listing the missing ones. Neither fails silently.

From the CLI:

```bash
python examples/base_main.py --config examples/configs/is_this_seat_taken.yaml --persona diplomat
python examples/base_main.py --config examples/configs/is_this_seat_taken.yaml \
    --personas agent_0=territorial,agent_1=diplomat
```

### Effects beyond the prompt

In `IsThisSeatTakenEnvironment`, a persona also biases the agent's *own*
generated public traits: Extraversion shifts `loudness` and `talkativeness` by
`((level − 5) / 4) × 0.3`, clamped to `[0, 1]`. The random draw is shifted, not
replaced, so a persona makes a tendency more likely without determining it.

`scent` is deliberately left unbiased — there is no personality-research basis
for tying body odour to a trait, and inventing one risks an unpleasant
association.

---

## 7) API

| Function | Purpose |
|---|---|
| `Persona(name, description, traits, verbosity)` | The dataclass. Validates `verbosity` on construction |
| `Persona.from_big_five(ext, agr, con, neu, ope, ...)` | Build from the standard 5-tuple; unspecified domains default to neutral |
| `build_trait_clause(persona)` | `"I'm <adjectives>."`, or `""` if everything is neutral |
| `build_persona_description(persona)` | Description + trait clause |
| `build_persona_prompt(persona, task)` | Wraps `task` with the shaping instruction |
| `qualify(level, low, high)` | Render one adjective pair; `None` at neutral |

Constants: `MIN_LEVEL` (1), `MAX_LEVEL` (9), `NEUTRAL_LEVEL` (5),
`VERBOSITY_COMPACT`, `VERBOSITY_FULL`, `DOMAINS`, `TABLE_12`, `COMPACT_TRAITS`.

---

## 8) Notes and limitations

- **Prompt-level only.** Nothing enforces that a model actually behaves in
  character; this shapes the instruction, not the policy. Whether a given model
  complies is an empirical question per model.
- **Levels are not calibrated.** 9 is "extremely" in wording, not a
  psychometrically validated intensity.
- **Trait clauses get long.** At `verbosity="full"` a single domain can add nine
  adjectives; five domains at full verbosity is a substantial prompt prefix and
  competes with task instructions for attention.
- **`qualify()` raises outside 1–9**, but `Persona` does not validate levels at
  construction — an out-of-range level fails later, when the clause is built.
- Tests live in `terrarium/tests/test_personas.py`.
