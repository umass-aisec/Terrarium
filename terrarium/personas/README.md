# Personas

Implementation:
- Persona type and rendering: `terrarium/personas/persona.py`
- Prompt wrapper: `terrarium/personas/prompt.py`
- Trait adjective database: `terrarium/personas/adjectives.py`
- Presets: `terrarium/personas/presets.py`
- Tests: `terrarium/tests/test_personas.py`

Personas shape agent behavior through **prompting only** — no fine-tuning and
no model changes. Each Big Five dimension is set to a level from 1 to 9,
rendered as qualified bipolar adjectives, and wrapped around the agent's task
as a first-person self-description.

The method follows Jiang et al., "Personality Traits in Large Language Models"
(https://www.researchgate.net/publication/372074802). The package is a pure
library: no framework imports, no I/O, no LLM calls.

## 1) Quick start

```python
from terrarium.personas import PRESETS, build_persona_prompt

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

To use personas in a run, set them on the environment instead (see section 5).

## 2) Trait levels

Every trait takes an integer level from 1 to 9, where 5 is neutral. Levels
below 5 select the adjective marking the low end of the pair, levels above 5
the high end, and the distance from neutral selects the qualifier:

- `1`: `extremely <low>`
- `2`: `very <low>`
- `3`: `<low>`
- `4`: `a bit <low>`
- `5`: contributes nothing
- `6`: `a bit <high>`
- `7`: `<high>`
- `8`: `very <high>`
- `9`: `extremely <high>`

A persona whose traits are all neutral renders an empty clause, and
`build_persona_prompt` returns the task unchanged.

## 3) Domains and facets

Traits are keyed by either a domain code or a facet.

The five domains are `EXT` (Extraversion), `AGR` (Agreeableness), `CON`
(Conscientiousness), `NEU` (Neuroticism) and `OPE` (Openness).

A domain key shapes every facet under that domain at once. A facet key, given
either as its code (`"A1"`) or its name (`"Trust"`), shapes only that facet:

```python
Persona(traits={"AGR": 9})     # every Agreeableness facet
Persona(traits={"Trust": 9})   # -> "I'm extremely trustful."
```

The adjective database in `adjectives.py` is transcribed from Table 12 of the
paper, itself adapted from Goldberg's bipolar trait markers. Each row pairs a
facet with its low-end and high-end adjective.

## 4) Verbosity

`verbosity` controls how many adjectives a domain-level trait expands into. It
has no effect on facet-level traits, which are already a single row.

- `"compact"` (default): a fixed 4-5 marker subset per domain
- `"full"`: every Table 12 facet marker for that domain

The same trait at `EXT=9`:

```
compact: I'm extremely extraverted, extremely talkative, extremely bold, and
         extremely adventurous.
full:    I'm extremely friendly, extremely extraverted, extremely talkative,
         extremely bold, extremely assertive, extremely active, extremely
         energetic, extremely adventurous and daring, and extremely cheerful.
```

The compact subset is hardcoded rather than sampled, so a given `Persona`
always renders the same prompt. The rationale for which facets it keeps is
documented inline above `COMPACT_TRAITS` in `adjectives.py`.

## 5) Presets

Five presets ship in `presets.py`, reachable as `PRESETS[name]` and as module
constants. Domains not listed stay neutral.

- `cautious skeptic` (`CAUTIOUS_SKEPTIC`): EXT 3, AGR 2, CON 8
- `anxious pushover` (`ANXIOUS_PUSHOVER`): AGR 8, NEU 8
- `confident collaborator` (`CONFIDENT_COLLABORATOR`): EXT 7, AGR 7, CON 5
- `diplomat` (`DIPLOMAT`): EXT 9, AGR 9
- `territorial` (`TERRITORIAL`): EXT 1, AGR 1

`diplomat` and `territorial` are opposites on the two dimensions that most
affect negotiation, which makes them a usable contrast pair.

A persona does not have to be a preset:

```python
Persona.from_big_five(ext=2, agr=8, con=6, name="quiet helper",
                      description="A quiet helper.")
```

## 6) Using personas in a run

Set one of two keys under `environment:`. If both are present, `personas` takes
precedence.

```yaml
environment:
  persona: diplomat          # one preset for every agent
```

```yaml
environment:
  personas:                  # per-agent; must cover every agent
    agent_0: territorial
    agent_1: diplomat
```

An unknown preset name raises `ValueError` listing the available presets, and a
`personas` map that misses an agent raises listing the missing agents.

From the command line:

```bash
python examples/base_main.py --config examples/configs/is_this_seat_taken.yaml \
    --persona diplomat
python examples/base_main.py --config examples/configs/is_this_seat_taken.yaml \
    --personas agent_0=territorial,agent_1=diplomat
```

An environment may also use the persona for more than prompting.
`IsThisSeatTakenEnvironment` biases the agent's own generated `loudness` and
`talkativeness` by `((level - 5) / 4) * 0.3` from Extraversion, clamped to
`[0, 1]`; the random draw is shifted, not replaced. `scent` is not biased.

## 7) API reference

- `Persona(name, description, traits, verbosity)`: the dataclass; validates
  `verbosity` on construction
- `Persona.from_big_five(ext, agr, con, neu, ope, name, description, verbosity)`:
  build from the five domain levels, each defaulting to neutral
- `build_trait_clause(persona)`: renders `"I'm <adjectives>."`, or `""` when
  every trait is neutral
- `build_persona_description(persona)`: description plus trait clause
- `build_persona_prompt(persona, task)`: wraps `task` with the shaping
  instruction
- `qualify(level, low, high)`: renders one adjective pair; returns `None` at
  neutral

Constants: `MIN_LEVEL`, `MAX_LEVEL`, `NEUTRAL_LEVEL`, `VERBOSITY_COMPACT`,
`VERBOSITY_FULL`, `DOMAINS`, `TABLE_12`, `COMPACT_TRAITS`.

## 8) Notes / limitations

- Shaping is prompt-level only. Nothing enforces that a model stays in
  character; whether a given model complies is model-dependent.
- Levels are wording intensities, not calibrated psychometric values.
- At `verbosity="full"` a single domain can add nine adjectives, so a fully
  specified persona is a substantial prompt prefix competing with task
  instructions.
- `qualify()` rejects levels outside 1-9, but `Persona` does not validate
  levels on construction, so an out-of-range level fails when the clause is
  built rather than when the persona is created.
