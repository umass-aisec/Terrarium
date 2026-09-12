# IsThisSeatTakenEnvironment (social coordination benchmark)

Implementation:
- Environment: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_env.py`
- Prompts: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_prompts.py`
- Tools: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_tools.py`
- GUI: `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py`
- Example config: `examples/configs/is_this_seat_taken.yaml`

Unlike the other DCOP environments, instances are generated
in-process from the config and seed. There is no CoLLAB dependency and no
instance file.

## 1) What problem does this environment model?

This environment models **social seat selection**: agents occupy seats in a
shared space and negotiate over which seat each one ends up in.

- There are `m` agents and `m + empty_seat_buffer` seats.
- Each agent has a **private** preference profile (seat roles it wants, agents
  it wants to sit near or away from).
- Each agent sees noisy estimates of its neighbours' **public traits**
  (loudness, scent, talkativeness) but never their preferences.
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

Agent count comes from `communication_network.num_agents`. Each agent has **private state**:

- `preference_profile`: the private goal (see section 4)
- `base_tolerance`: drawn from `uniform(1.0, 2.5)`; the threshold above which a
  neighbour's traits become a penalty
- `satisfaction_score`: per-agent reward (see section 7)
- `last_instant_reward`: previous turn's seat reward, used to compute deltas
- `social_pressure`, `pressure_requests`, `pressure_complaints`: accumulated
  pressure and its counts
- `current_seat`, `settled`, `pending_reaction`

and **public state**:

- `public_traits`: `loudness`, `scent`, `talkativeness`, each in `[0, 1]`

`public_traits` is the only agent attribute other agents can observe, and only
as a noisy estimate. What else each agent is told is covered in section 6.

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

- `move(seat_id)`: move to an empty seat; costs `move_cost` and increments
  `total_moves`. `seat_id` is required. A move into an occupied seat fails but
  still costs `move_cost`, without incrementing `total_moves`. A missing or
  unknown seat, or the agent's current seat, is rejected at no cost.
- `settle()`: declare done; rejected while standing. No cost.
- `stand()`: leave the current seat. No cost, and does not increment
  `total_moves`.

Any execution action other than `settle` clears the agent's `settled` flag
before it is validated, so even a rejected action un-settles the agent.

Planning tools are rejected during execution and vice versa.

### Between iterations

`log_iteration()` calls `_decay_social_pressure(factor=0.7)`, multiplying
`social_pressure`, `pressure_complaints` and `pressure_requests` by 0.7.
Without decay, pressure only ever increases and agents do not converge.

## 6) What agents observe

Each turn an agent receives a system prompt and a user prompt, both rebuilt from
scratch. There is no memory across turns beyond what these two prompts contain.

### System prompt

Built by `IsThisSeatTakenPrompts.get_system_prompt(agent_name)`. It is the same
every turn for a given agent and contains:

- a scenario introduction (specific text for `airplane` and `cinema`, generic
  otherwise)
- a `WHAT YOU KNOW` list and the available tools for each phase; `recall` is
  listed only when `llm.compaction.retrieval_enabled` is true
- that pressure is information, not a command, and that the agent's own traits
  are fixed and cannot be changed by promising to behave differently
- a `WHAT AFFECTS HOW YOU FEEL` list describing the reward factors
  qualitatively and by relative size, without the formula or weights, and
  stating that the agent gets no credit for anyone else's comfort
- goal and tone guidance, and how to address a private channel by
  `blackboard_id`
- when a persona is set, the whole prompt wrapped in the persona instruction

### User prompt

Built each turn by `_get_user_prompt_impl()` from `build_agent_context()`. It
contains, in order:

- **Your situation**: current seat, or standing
- **How you feel**: a comfort label, a noisy score, and a tolerance description
- **What matters to you**: preferred and avoided seat roles, preferred and
  avoided neighbours, and isolation preference
- **Time pressure**: `low`, `medium` or `high`
- **Social pressure on you**: the pressure level, shown only when it is not
  `none` or the agent was complained at
- **Your immediate neighbors**: each adjacent agent's id, seat and perceived
  traits
- **Nearby empty seats**: empty seats in view
- **Progress**: current iteration out of `max_iterations`
- **What people are saying in the chat**: the channel transcripts, possibly
  compacted
- **What to do now**: phase-specific instructions

Example (planning phase):

```
**Your situation:** You're sitting at seat_1_2.
**How you feel:** you feel **uncomfortable** right now (rough score: ~-1.17; your threshold is moderate).
**What matters to you:** you prefer window seats • you dislike middle seats • you'd rather sit near agent_3.

**Time pressure:** low.

**Social pressure on you:** mild. Someone complained about you and is waiting for you to react.

**Your immediate neighbors:**
  • agent_1 at seat_1_1 — loudness: 0.82, talkativeness: 0.4, scent: 0.11

**Nearby empty seats (use exact ID when calling move):** seat_1_3, seat_2_2.
```

### Derived signals

The environment converts internal state into coarse or noisy signals before it
reaches the prompt:

- `felt_satisfaction`: `satisfaction_score` plus `uniform(-0.3, 0.3)` noise,
  clamped to `[-5, 10]`
- `comfort_signal.label`: `great`, `comfortable`, `uncomfortable` or
  `miserable`, from the gap between satisfaction and effective tolerance and
  from the current instant reward
- `comfort_signal.tolerance_description`: from effective tolerance —
  `< 1.2` is `low (you settle easily)`, `< 1.8` is `moderate`, otherwise
  `high (you're picky)`
- `comfort_signal.time_pressure`: from `iteration / max_iterations` —
  `>= 0.8` is `high`, `>= 0.5` is `medium`, otherwise `low`
- `social_pressure_level`: the four levels in section 5
- `perceived_traits`: each neighbour trait plus `gauss(0, 0.15)` noise, clamped
  to `[0, 1]`, resampled every turn
- `visible_seats`: seats within Manhattan distance 1 of the agent's seat, or
  every seat when the agent is standing

### What agents learn from the chat

Channels are created from the communication network. Under the shipped config
(`topology: complete`, `consolidate_channels: true`) all agents share a single
channel, so every agent sees the following. On sparser topologies an agent sees
them only for agents it shares a channel with.

- Each channel opens with a `context` message from `get_network_context()`
  naming the scenario and listing the channel's participants.
- `async_init()` then posts `Initial seating: agent_0→seat_1_1, ...` as a
  `context` message, so every agent starts with the full initial seating map.
- `request_move` and `complain` post the requester's message (or a default
  message) as ordinary chat, to non-private channels only.
- `move`, `settle` and `stand` are posted as `action_executed` events with the
  full result, including seat ids and failed attempts, to every channel the
  acting agent belongs to, private channels included:

```
[2] [action_executed] agent_2 payload={"action_params":{"action":"move","seat_id":"seat_1_3"},"action_type":"move","details":{"result":{"action":"move","agent":"agent_2","from_seat":"seat_1_2","to_seat":"seat_1_3"},"status":"success"},"result_status":"success"}
```

Agents can therefore track global occupancy from the chat even though
`visible_seats` is local.

### Information agents do not receive

- other agents' preference profiles, satisfaction or tolerance
- neighbours' true trait values (only noisy estimates)
- their own exact `satisfaction_score`, `base_tolerance`, effective tolerance
  or `social_pressure` (only the noisy score and the labels above)
- the reward formula and its weights, the joint reward, or the termination rule
- their own `hard_required_role`, `hard_required_neighbor` and
  `hard_avoid_neighbor` (see section 12)
- the result of their own `request_move` or `complain`: a successful
  environment tool call ends the agent's turn, so the returned
  `target_pressure_level` is written to `tool_calls.json` but never shown to
  the model
- private channels they are not a member of

## 7) Reward model

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
                   + sum of move_cost for every move, including failed moves
                     into occupied seats
                   + sum of social_action_cost for every request/complain
```

### Joint reward

```
joint_reward = sum(satisfaction_score) - group_move_penalty * total_moves
```

Moves are charged twice by design: once to the acting agent through
`move_cost`, and once to the group through `group_move_penalty`.

## 8) Termination

`done(iteration)` returns true when either:

1. `iteration > max_iterations`, or
2. all agents are `settled` **and** either the last `convergence_window` joint
   rewards span no more than `reward_convergence_threshold`, or the current
   joint reward is at least 90% of `compute_max_joint_reward()`.

## 9) Personas

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

## 10) Logging and outputs

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

## 11) Configuration reference (environment section)

Key fields in `environment:` (defaults shown where relevant):

- `name`: must be `IsThisSeatTakenEnvironment`
- `scenario_type` (default `cinema`): layout family; alias `layout_type`
- `cols` (default per scenario, see section 3): grid columns
- `rows` (default `ceil(seat_count / cols)`): grid rows
- `empty_seat_buffer` (default `max(2, ceil(num_agents * 0.35))`): spare seats
  beyond `num_agents`
- `move_cost` (default `-0.5`): charged to an agent for each move, including a
  failed move into an occupied seat; `stand` is free
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

## 12) Notes / limitations

- The `hard_*` preference fields are penalties, not enforced constraints, and
  they are never shown to the agent. The system prompt says hard requirements
  cost more than ordinary preferences, but the user prompt renders only the soft
  preference fields, so an agent is penalized for constraints it cannot see.
- The user prompt and the `settle` tool description both tell agents that the
  simulation ends when every agent has settled. Termination also requires reward convergence or
  near-maximum reward, and happens regardless once `max_iterations` is exceeded
  (see section 8).
- The system prompt lists "What seats are empty" under `WHAT YOU KNOW`, but a
  seated agent is shown only empty seats within distance 1.
- Perception noise is resampled every turn, so the same neighbour's perceived
  traits change between turns without the neighbour changing.
- `compute_max_joint_reward()` is an **upper bound**, not a solved optimum, by
  design (no solver dependency). The 90%-of-maximum termination branch is
  therefore heuristic.
- `satisfaction_score` accumulates action costs but not seat quality; its seat
  term is the current instant reward (see section 7).
- Seat adjacency is orthogonal only. Diagonal seats are not neighbours, which
  matters most for the `airplane` and `cinema` layouts.
- `time_step` increments per action and is reported, but nothing is derived from
  it: there is no time-step bound and no limit on how long an agent may stand.
- Tolerance moves in one direction only. Pressure lowers it and decay restores
  it; no interaction raises it above `base_tolerance`.
