# IsThisSeatTakenEnvironment (social coordination benchmark)

Implementation:
- Environment: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_env.py`
- Prompts: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_prompts.py`
- Tools: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_tools.py`
- GUI: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py`
- Example config: `examples/configs/is_this_seat_taken.yaml`

Unlike the other DCOP environments in this package, instances are generated
in-process from the config and seed. There is no CoLLAB dependency and no
instance file.

## 1) What problem does this environment model?

This environment models **social seat selection**: agents occupy seats in a
shared space and negotiate over which seat each one ends up in.

- There are `m` agents and `m + empty_seat_buffer` seats.
- Each agent has a **private** preference profile (seat roles it wants, agents
  it wants to sit near or away from).
- Each agent can see its neighbours' **public traits** (loudness, scent,
  talkativeness) but never their preferences.
- Agents apply social pressure to each other during planning, and move, settle
  or stand during execution.

Because preferences are private, a good joint outcome requires communication:
an agent acting alone can only optimize its own seat role, and cannot discover
that a neighbour would be happy to swap.

Moving and pressuring both cost the acting agent, and the joint score is
penalized per move, so repeated churn is discouraged. The intended trajectory
is negotiate, move once, settle.

## 2) Entities and state

### Seats

`_build_layout()` builds a `rows x cols` grid. Each `Seat` has:

- `seat_id`: `seat_{row+1}_{col+1}`, e.g. `seat_2_3` (ids are 1-indexed, `row`
  and `col` are 0-indexed)
- `row`, `col`: grid position
- `role`: scenario-dependent label (see section 3)
- `neighbors`: orthogonally adjacent seat ids (up/down/left/right, no diagonals)
- `occupied_by`: agent name, or `None`

Seat count is `num_agents + empty_seat_buffer`, so there is always somewhere to
move to.

### Agents

Agent count comes from `communication_network.num_agents`, not from the
`environment` block. Each agent has **private state**:

- `preference_profile`: the private goal (see section 4)
- `base_tolerance`: drawn from `uniform(1.0, 2.5)`; the threshold above which a
  neighbour's traits become a penalty
- `satisfaction_score`: per-agent reward (see section 6)
- `last_instant_reward`: previous turn's seat reward, used to compute deltas
- `social_pressure`, `pressure_requests`, `pressure_complaints`: accumulated
  pressure and its counts
- `current_seat`, `settled`, `pending_reaction`

and **public state**:

- `public_traits`: `loudness`, `scent`, `talkativeness`, each in `[0, 1]`

`public_traits` is the only agent attribute other agents can observe.

## 3) Scenario layouts and seat roles

`scenario_type` (alias `layout_type`) selects the default column count and the
role assignment rule:

- `bus` (2 cols): `window` at col 0, `aisle` at the last col, `middle` between
- `cinema` (5 cols): `window` at both ends, `aisle` adjacent to the ends when
  `cols > 2`, `front`/`back` on the first/last row, `middle` otherwise
- `wedding` (4 cols): `edge` on the perimeter, `table_center` inside
- `taxi` (3 cols): `front` on row 0, `back` on the last row, `middle` between
- `airplane` (6 cols): `window` at both ends, `aisle` either side of the centre
  line, `middle` otherwise
- anything else (4 cols): `window` at both ends, `middle` otherwise

Examples:

```
bus,      cols=6 -> window, middle, middle, middle, middle, aisle
airplane, cols=6 -> window, middle, aisle,  aisle,  middle, window
airplane, cols=3 -> window, aisle,  window
```

`rows` and `cols` override the defaults.

## 4) Private preference generation

Each agent's profile is drawn from the seeded RNG:

- `preferred_roles`: 1-2 roles sampled from the roles present in the layout
- `avoided_roles`: 1 role from those not preferred
- `preferred_neighbors`: one other agent, with probability 0.45
- `avoided_neighbors`: one other agent, with probability 0.35
- `prefer_isolation`: true with probability 0.30
- `hard_required_role`: the first preferred role, with probability 0.20
- `hard_required_neighbor`: the preferred neighbour, with probability 0.15
- `hard_avoid_neighbor`: the avoided neighbour, with probability 0.20

The `hard_*` fields are scored as penalties rather than enforced as
constraints, so they behave as strong soft constraints.

## 5) Actions, tools, and phases

`IsThisSeatTakenTools` sets `supports_private_channels = True`, so agents can
also call `create_channel` to negotiate in a private channel.

### Planning phase

- `request_move(agent_id, message?)`: adds `request_pressure` to the target and
  increments its `pressure_requests`
- `complain(agent_id, message?)`: adds `complaint_pressure`, increments
  `pressure_complaints`, and sets the target's `pending_reaction`

Both require the target to currently be a neighbour, cost the caller
`social_action_cost`, and clear the caller's own `settled` flag.

Neither forces the target to move. The target is shown a qualitative pressure
level derived from `social_pressure / base_tolerance`, never the raw number:

- `< 0.3`: `none`
- `< 0.7`: `mild`
- `< 1.1`: `building`
- `>= 1.1`: `high`

### Execution phase

- `move(seat_id)`: move to an empty seat, or stand if `seat_id` is omitted;
  costs `move_cost` and increments `total_moves`
- `settle()`: declare done; rejected while standing
- `stand()`: leave the current seat; costs `move_cost`

Planning tools are rejected during execution and vice versa.

### Between iterations

`log_iteration()` calls `_decay_social_pressure(factor=0.7)`, multiplying
`social_pressure`, `pressure_complaints` and `pressure_requests` by 0.7.
Without decay, pressure only ever increases and agents do not converge.

## 6) Reward model

### Per-turn seat reward

`_compute_instant_reward()` starts at 0 and applies `+1`/`-1` terms. A standing
agent scores 0.

Rewards:
- seat role is in `preferred_roles`
- `prefer_isolation` is set and the agent has no neighbours
- a `preferred_neighbor` is adjacent

Penalties:
- seat role is in `avoided_roles`
- `prefer_isolation` is set and the agent has any neighbour
- an `avoided_neighbor` is adjacent
- `hard_required_role` is set and not satisfied
- `hard_required_neighbor` is set and not adjacent
- `hard_avoid_neighbor` is set and adjacent
- each neighbour trait above effective tolerance

The last penalty is applied per trait per neighbour, so a single neighbour can
contribute up to `-3`.

### Effective tolerance

```
effective = base_tolerance
          - social_pressure     * 0.15
          - pressure_complaints * 0.05
          - pressure_requests   * 0.02
```

floored at 0. Pressure therefore reduces an agent's ability to tolerate its
neighbours, so being pressured makes the current seat score worse.

### satisfaction_score

`_refresh_rewards()` adds `instant - last_instant` each turn. Those deltas
telescope, so the seat term resolves to the **current** instant reward rather
than a running total. Action costs are added directly and do accumulate:

```
satisfaction_score = instant_reward(current seat)
                   + sum of move_cost for every move/stand
                   + sum of social_action_cost for every request/complain
```

### Joint reward

```
joint_reward = sum(satisfaction_score) - group_move_penalty * total_moves
```

Moves are charged twice by design: once to the acting agent through
`move_cost`, and once to the group through `group_move_penalty`.

## 7) Termination

`done(iteration)` returns true when either:

1. `iteration > max_iterations`, or
2. all agents are `settled` **and** either the last `convergence_window` joint
   rewards span no more than `reward_convergence_threshold`, or the current
   joint reward is at least 90% of `compute_max_joint_reward()`.

## 8) Personas

Setting `environment.persona` (one preset for all agents) or
`environment.personas` (a per-agent map) resolves each name against
`terrarium.personas.PRESETS`. An unknown name raises, and a `personas` map that
does not cover every agent raises.

Personas affect the run in two places:

1. The persona's trait clause is wrapped around the agent's system prompt.
2. Extraversion biases that agent's own generated `loudness` and
   `talkativeness` by `((level - 5) / 4) * 0.3`, clamped to `[0, 1]`. The draw
   is shifted, not replaced. `scent` is not biased.

See `terrarium/personas/README.md`.

## 9) Logging and outputs

Standard framework logs apply (`blackboard_*.txt`, `tool_calls.json`,
`agent_prompts.json`), plus `compaction_events.jsonl` and `.md` when compaction
is configured.

`get_final_summary()` returns `scenario_type`, `current_time_step`,
`seat_count`, `total_moves`, `agents_still_unsettled`, `joint_reward`,
`settled`, an `agent_summaries` map (seat, settled, satisfaction per agent),
and the final `layout`.

To watch a run live:

```bash
python terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py \
    --log logs/IsThisSeatTakenEnvironment/<tag_model>/seed_<seed>/blackboard_0.txt
```

## 10) Configuration reference (environment section)

Key fields in `environment:` (defaults shown where relevant):

- `name`: must be `IsThisSeatTakenEnvironment`
- `scenario_type` (default `cinema`): layout family; alias `layout_type`
- `cols` (default per scenario, see section 3): grid columns
- `rows` (default `ceil(seat_count / cols)`): grid rows
- `empty_seat_buffer` (default `max(2, ceil(num_agents * 0.35))`): spare seats
  beyond `num_agents`
- `move_cost` (default `-0.5`): charged to an agent that moves or stands
- `social_action_cost` (default `-0.3`): charged to an agent that requests or
  complains
- `group_move_penalty` (default `0.25`): joint reward penalty per move
- `request_pressure` (default `0.4`): pressure added by `request_move`
- `complaint_pressure` (default `0.9`): pressure added by `complain`
- `reward_convergence_threshold` (default `0.05`): convergence band width
- `convergence_window` (default `3`): iterations inspected for convergence
- `persona`: one preset name applied to every agent
- `personas`: per-agent map of agent name to preset name

Seed and iteration count come from `simulation.seed` and
`simulation.max_iterations`; agent count from
`communication_network.num_agents`.

Seed, persona and model can be set per run without a separate config file:

```bash
python examples/base_main.py \
    --config examples/configs/is_this_seat_taken.yaml \
    --seed 7 --persona diplomat --model gpt-5.5
```

## 11) Notes / limitations

- The `hard_*` preference fields are penalties, not enforced constraints.
- `compute_max_joint_reward()` is an **upper bound**, not a solved optimum, by
  design (no solver dependency). The 90%-of-maximum termination branch is
  therefore heuristic.
- `satisfaction_score` accumulates action costs but not seat quality; its seat
  term is the current instant reward (see section 6).
- Seat adjacency is orthogonal only. Diagonal seats are not neighbours, which
  matters most for the `airplane` and `cinema` layouts.
- `time_step` increments per action and is reported, but nothing is derived from
  it: there is no time-step bound and no limit on how long an agent may stand.
- Tolerance moves in one direction only. Pressure lowers it and decay restores
  it; no interaction raises it above `base_tolerance`.
