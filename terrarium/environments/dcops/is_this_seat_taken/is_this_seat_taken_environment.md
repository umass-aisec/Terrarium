# IsThisSeatTakenEnvironment (social-coordination benchmark)

`IsThisSeatTakenEnvironment` is a multi-agent seating environment: agents occupy
seats in a shared space, each holds a *private* preference profile, and the only
way to learn about anyone else's preferences is to talk to them on the
blackboard. Agents negotiate over a scarce resource (good seats) while applying
and absorbing social pressure.

It differs from the other DCOP environments in this package in three ways:

- **No CoLLAB dependency.** Instances are generated in-process from the config
  and the seed; there is no external solver or instance file.
- **Preferences are private and only partly observable.** An agent sees its
  neighbours' *public traits* (loudness, scent, talkativeness) but never their
  preference profiles.
- **Social pressure is a first-class channel.** Agents can pressure each other
  during planning, and pressure degrades the target's tolerance — but never
  forces a move. The target decides how to react.

---

## 1) What problem does this environment model?

Each agent wants a seat that matches its own preferences (a seat *role* it
likes, neighbours it likes, or isolation), while avoiding neighbours whose
public traits exceed its tolerance. Preferences conflict: two agents may want
the same seat, or one may want to sit next to someone who wants to be alone.

Because preferences are private, the joint optimum is only reachable through
communication. An agent that never talks can only optimise its own seat role;
it cannot discover that a neighbour would happily swap.

Moving is not free (`move_cost`), pressuring is not free
(`social_action_cost`), and the joint score is penalised per move
(`group_move_penalty`), so churn is punished. The intended behaviour is
negotiate → move once → settle.

---

## 2) Entities and state

### Seats

Built by `_build_layout()` as a `rows × cols` grid. Each `Seat` carries:

| Field | Meaning |
|---|---|
| `seat_id` | `seat_{row+1}_{col+1}`, e.g. `seat_2_3` (1-indexed in the id, 0-indexed internally) |
| `row`, `col` | 0-indexed grid position |
| `role` | Scenario-dependent label — see §3 |
| `neighbors` | Seat ids orthogonally adjacent (up/down/left/right; **no diagonals**) |
| `occupied_by` | Agent name, or `None` |

Seat count is `num_agents + empty_seat_buffer`, so there is always slack to move
into. `empty_seat_buffer` defaults to `max(2, ceil(num_agents * 0.35))`.

### Agents

`num_agents` comes from `communication_network.num_agents`, **not** from the
`environment` block. Per-agent state (`_build_agent_state`):

| Field | Meaning |
|---|---|
| `preference_profile` | The private goal — see §4 |
| `current_seat` | Seat id, or `None` while standing |
| `satisfaction_score` | Per-agent reward — see §6 |
| `last_instant_reward` | Previous turn's seat reward, used to compute deltas |
| `base_tolerance` | `uniform(1.0, 2.5)`; threshold above which a neighbour's traits hurt |
| `settled` | Agent has declared itself done |
| `social_pressure` | Accumulated pressure from others |
| `pressure_requests`, `pressure_complaints` | Counts, used to degrade tolerance |
| `pending_reaction` | Set when complained at |
| `public_traits` | `loudness`, `scent`, `talkativeness` in `[0, 1]` — visible to neighbours |

`public_traits` are the only agent attribute other agents can observe.
Extraversion biases `loudness` and `talkativeness` when a persona is set (§8);
`scent` is always an unbiased draw.

---

## 3) Scenario layouts and seat roles

`scenario_type` (alias: `layout_type`) picks the default column count and the
role-assignment rule. Default columns:

| `scenario_type` | cols | Roles produced |
|---|---|---|
| `bus` | 2 | `window` (col 0), `aisle` (last col), `middle` |
| `cinema` | 5 | `window` (both ends), `aisle` (adjacent to the ends when cols > 2), `front`/`back` (first/last row), `middle` |
| `wedding` | 4 | `edge` (perimeter), `table_center` |
| `taxi` | 3 | `front` (row 0), `back` (last row), `middle` |
| `airplane` | 6 | `window` (both ends), `aisle` (either side of the centre line), `middle` |
| anything else | 4 | `window` (both ends), `middle` |

Worked examples (verified against `_seat_role`):

```
bus,      cols=6 → window, middle, middle, middle, middle, aisle
airplane, cols=6 → window, middle, aisle,  aisle,  middle, window
airplane, cols=3 → window, aisle,  window
```

`cols` and `rows` can be set explicitly to override the defaults.

---

## 4) Private preference generation

Each agent draws a profile from the seeded RNG:

| Component | Rule |
|---|---|
| `preferred_roles` | 1–2 roles sampled from the roles present in the layout |
| `avoided_roles` | 1 role from those not preferred |
| `preferred_neighbors` | one other agent, with probability 0.45 |
| `avoided_neighbors` | one other agent (not the preferred one), with probability 0.35 |
| `prefer_isolation` | `True` with probability 0.30 |
| `hard_required_role` | the first preferred role, with probability 0.20 |
| `hard_required_neighbor` | the preferred neighbour, with probability 0.15 |
| `hard_avoid_neighbor` | the avoided neighbour, with probability 0.20 |

The `hard_*` constraints are *not* enforced by the environment — they are extra
penalty terms, so they behave as strong soft constraints.

---

## 5) Actions, tools, and phases

`IsThisSeatTakenTools` sets `supports_private_channels = True`, so agents may
also call `create_channel()` to open private side channels for negotiation.

### Planning phase — social pressure

| Tool | Effect |
|---|---|
| `request_move(agent_id, message=None)` | Adds `request_pressure` to the target; increments `pressure_requests` |
| `complain(agent_id, message=None)` | Adds `complaint_pressure`; increments `pressure_complaints`; sets the target's `pending_reaction` |

Both require the target to be a current **neighbour** and cost the caller
`social_action_cost`. Calling either clears the caller's own `settled` flag.

Neither forces the target to move. The target is shown a qualitative pressure
*level*, never the raw number, and decides for itself:

| `social_pressure / base_tolerance` | Level shown |
|---|---|
| `< 0.3` | `none` |
| `< 0.7` | `mild` |
| `< 1.1` | `building` |
| `≥ 1.1` | `high` |

### Execution phase — physical actions

| Tool | Effect |
|---|---|
| `move(seat_id)` | Move to an empty seat (or stand if `seat_id` is omitted); costs `move_cost`, increments `total_moves` |
| `settle()` | Declare done. Rejected while standing |
| `stand()` | Leave the current seat; costs `move_cost` |

Social tools are rejected during execution, and vice versa.

### Between iterations

`log_iteration()` calls `_decay_social_pressure(factor=0.7)`, which multiplies
`social_pressure`, `pressure_complaints`, and `pressure_requests` by 0.7. Without
this, pressure only ever ratcheted upward and agents never converged.

---

## 6) Reward model

### Per-turn seat reward (`_compute_instant_reward`)

Starts at 0; every term is ±1. An agent standing (no seat) scores 0.

| +1 for each | −1 for each |
|---|---|
| seat role in `preferred_roles` | seat role in `avoided_roles` |
| `prefer_isolation` and no neighbours | `prefer_isolation` and any neighbour |
| a `preferred_neighbor` adjacent | an `avoided_neighbor` adjacent |
| | `hard_required_role` not satisfied |
| | `hard_required_neighbor` not adjacent |
| | `hard_avoid_neighbor` adjacent |
| | each neighbour trait (`loudness`, `scent`, `talkativeness`) above effective tolerance |

Note the last row is per-trait per-neighbour, so one loud, smelly, talkative
neighbour costs −3.

### Effective tolerance

```
effective = base_tolerance
          − social_pressure     × 0.15
          − pressure_complaints × 0.05
          − pressure_requests   × 0.02
          (floored at 0)
```

Pressure therefore makes an agent *less* able to put up with its neighbours —
being complained at makes the seat genuinely worse, not just annoying.

### `satisfaction_score` — read this carefully

`_refresh_rewards()` adds `instant − last_instant` each turn. Those deltas
telescope, so the seat-quality part collapses to the **current** instant reward
rather than a running total. Action costs, added directly, do persist:

```
satisfaction_score = instant_reward(current seat)
                   + Σ move_cost      (every move/stand)
                   + Σ social_action_cost (every request/complain)
```

So it reads as "how good is my seat right now, minus what I spent getting
here" — not a cumulative sum of seat quality over time.

### Joint reward

```
joint_reward = Σ satisfaction_score − group_move_penalty × total_moves
```

Moves are penalised twice on purpose: once privately via `move_cost`, once
jointly via `group_move_penalty`.

---

## 7) Termination

`done(iteration)` returns `True` when either:

1. `iteration > max_iterations`, or
2. **all** agents are `settled` **and** either
   - the last `convergence_window` joint rewards span ≤ `reward_convergence_threshold`, or
   - the current joint reward is ≥ 90% of `compute_max_joint_reward()`.

`compute_max_joint_reward()` is a deliberately loose optimistic bound
(`3 + |preferred_roles| + |preferred_neighbors| + isolation + 5` per agent, plus
`2` per agent). It is **not** a solved optimum — treat it as a normalisation
constant, not a target.

---

## 8) Personas

If `environment.persona` (uniform) or `environment.personas` (per-agent map) is
set, each name is resolved against `terrarium.personas.PRESETS`. An unknown name
raises; a `personas` map missing any agent raises.

Personas do two things here:

1. **Prompt shaping** — the trait clause is wrapped around the system prompt.
2. **Trait biasing** — Extraversion nudges the agent's own `loudness` and
   `talkativeness` by `((level − 5) / 4) × 0.3`, clamped to `[0, 1]`. The draw is
   shifted, not replaced. `scent` is deliberately left unbiased.

See [`terrarium/personas/README.md`](../../../personas/README.md).

---

## 9) Configuration reference

Under `environment:` in the run config.

| Key | Default | Meaning |
|---|---|---|
| `name` | — | Must be `IsThisSeatTakenEnvironment` |
| `scenario_type` | `cinema` | Layout family; alias `layout_type` |
| `cols` | per scenario (§3) | Grid columns |
| `rows` | `ceil(seat_count / cols)` | Grid rows |
| `empty_seat_buffer` | `max(2, ceil(num_agents × 0.35))` | Spare seats beyond `num_agents` |
| `move_cost` | `-0.5` | Private cost per move/stand |
| `social_action_cost` | `-0.3` | Private cost per request/complain |
| `group_move_penalty` | `0.25` | Joint penalty per move |
| `request_pressure` | `0.4` | Pressure added by `request_move` |
| `complaint_pressure` | `0.9` | Pressure added by `complain` |
| `reward_convergence_threshold` | `0.05` | Convergence band (§7) |
| `convergence_window` | `3` | Iterations inspected for convergence |
| `persona` | unset | One preset applied to every agent |
| `personas` | unset | Per-agent map; must cover every agent |

Seeding and iteration count come from `simulation.seed` and
`simulation.max_iterations`; agent count from
`communication_network.num_agents`.

### Running

The environment ships one base config plus a fast smoke-test variant. Seed,
persona, and model are CLI flags rather than separate config files:

```bash
python examples/base_main.py \
    --config examples/configs/is_this_seat_taken.yaml \
    --seed 7 --persona diplomat --model gpt-5.5
```

---

## 10) Logging and outputs

Standard framework logs (`blackboard_*.txt`, `tool_calls.json`,
`agent_prompts.json`) plus, when compaction is configured,
`compaction_events.jsonl` / `.md`.

`get_final_summary()` returns `scenario_type`, `current_time_step`,
`seat_count`, `total_moves`, `agents_still_unsettled`, `joint_reward`,
`settled`, a per-agent `agent_summaries` block (seat, settled, satisfaction),
and the full final `layout`.

A live viewer is available:

```bash
python terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py \
    --log logs/IsThisSeatTakenEnvironment/<tag>/seed_<n>/blackboard_0.txt
```

---

## 11) Notes and limitations

- **Nothing bounds time steps or standing.** `time_step` increments per action
  and is reported in the final summary, but it is telemetry only — no episode
  bound is derived from it, and an agent may stand indefinitely. (Config keys
  `max_time_steps` and `standing_limit` used to be shipped for this and were
  removed, as no code ever read them.)
- **`compute_max_joint_reward()` is an optimistic bound, not an optimum.** The
  90%-of-max termination branch is therefore heuristic.
- **`satisfaction_score` is not cumulative in the seat-quality term** (§6). Any
  analysis treating it as a running total of seat quality will be wrong.
- **Neighbour adjacency is 4-way.** Diagonal seats are not neighbours, which
  matters for `airplane` and `cinema` layouts where a diagonal is intuitively
  "next to" you.
- **Hard constraints are soft.** `hard_required_role` and friends only add
  penalties; nothing prevents violating them.
- **Tolerance is one-directional.** Pressure lowers tolerance and decay restores
  it, but there is no mechanism by which a pleasant neighbour raises it.
