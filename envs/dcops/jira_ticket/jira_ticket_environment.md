# JiraTicketEnvironment (DCOP benchmark)

Implementation:
- Environment: `envs/dcops/jira_ticket/jira_ticket_env.py`
- Prompts: `envs/dcops/jira_ticket/jira_ticket_prompts.py`
- Tools: `envs/dcops/jira_ticket/jira_ticket_tools.py`
- Example config: `examples/configs/jira_ticket.yaml`

## 1) What problem does this environment model?

This environment models a **one-shot sprint task allocation** problem inspired by JIRA-style work:

- There are `m` agents (engineers).
- There are `n` tasks (microtasks derived from synthetic “issues”).
- Each agent chooses **at most one** task (or uses `skip`).
- Each task should be chosen by **at most one** agent (duplicates are heavily penalized).
- Some tasks may exceed an agent’s clearance; selecting them incurs an extra clearance cost penalty.

The intended objective is **lexicographic**:
1. Maximize number of tasks completed.
2. Subject to that, prefer higher total priority.
3. Subject to that, minimize total cost of the chosen assignments.

The environment implements this using a single scalar score with a **Big‑M** construction:

```
score = big_m * tasks_done + priority_bonus * priority_sum - total_cost - violation_penalty * violations
```

This makes “doing one more task” dominate reasonable cost differences.

## 2) Entities and state

### Agents

Agents are given synthetic first names (e.g. `Alice`, `Ben`, …) using the `names` package.
If `names` is not installed in the active Python environment, initialization will fail.

Each agent has **private state**:
- `skills`: a tag -> skill score map (0.0–1.0)
- `availability`: a sprint bandwidth number (hours)
- `clearance`: an integer permission level

These are generated synthetically (see section 4).

### Tasks

Tasks are generated from synthetic issues and then expanded into microtasks. Each task has **public metadata**:
- `id`: e.g. `ISSUE-0001::implement`
- `title`: e.g. `"Build backend [implement]"`
- `tags`: 1–2 skill tags (e.g. `["backend"]` or `["backend", "api-development"]`)
- `effort`: positive float (work estimate)
- `clearence_threshold`: integer permission level required
- `priority`: a label used to weight reward (higher priority => more reward)
- `work_type`: microtask type (e.g. `"implement"`, `"review"`)

### Assignment state

The environment stores the partial joint assignment as:

```
self.assignment: Dict[agent_name, task_id_or_None]
```

Agents write into it via the `assign_task` tool (see section 6).

## 3) Microtasks: how issues become tasks

Synthetic “issues” are expanded into multiple microtasks using a fixed microtask list:
- `implement`, `review`, `test`, `docs`, `triage`

Example:
- Issue `ISSUE-0001` becomes:
  - `ISSUE-0001::implement`
  - `ISSUE-0001::review`
  - `ISSUE-0001::test`
  - `ISSUE-0001::docs`
  - `ISSUE-0001::triage`

Microtask effort is scaled via fixed multipliers (defaults):
- `implement`: `1.0 * issue_effort`
- `review`: `0.5 * issue_effort`
- `test`: `0.7 * issue_effort`
- `docs`: `0.5 * issue_effort`
- `triage`: `0.4 * issue_effort`
- unknown microtask type: `0.6 * issue_effort`

## 4) Synthetic instance generation

### 4.1 Synthetic issues

The number of base issues is:

```
issue_count = ceil(max_tasks / len(microtask_types))
```

Each issue is sampled with:
- `tags`: 1–2 tags from a fixed tag pool (see `DEFAULT_SKILL_TAGS` in code)
- `priority`: sampled from fixed labels (`low`, `medium`, `high`, `critical`) (affects reward)
- `effort`: integer in `[2, 8]`
- `clearence_threshold`: integer in `[0, environment.max_clearence_threshold]`

### 4.2 Synthetic agents

For each agent:
- Choose 1–2 “primary” tags from the global tag pool.
- Set `skills[tag] ~ Uniform(0.6, 1.0)` for those tags only (all other tags are treated as skill 0.0).
- Sample:
  - `availability` from `environment.availability_range` (e.g. `[4, 10]`)
  - `clearance` from `[0, environment.max_clearence_threshold]`

## 5) Feasibility and private cost model

### Clearance penalty

If an agent’s clearance is below a task’s clearence_threshold, the assignment is still allowed but incurs an extra penalty:

```
if clearance < clearence_threshold:
  cost += clearence_cost
```

### Private cost function (skills + availability)

For a feasible (agent, task) pair, the private cost is:

```
match = average(skills.get(tag, 0.0) for tag in task.tags)   # 0..1
skill_adjusted = effort / max(eps, match + eps)
overload = load_cost * max(0, effort - availability)

cost = skill_adjusted + overload + clearance_penalty
```

Where:
- `eps = environment.skill_eps` (default `0.1`)
- `load_cost = environment.cost_weights.load` (default `1.0`)

Costs are computed once at initialization and stored as:

```
self.costs[agent][task_id] -> float
```

## 6) Actions, tools, and phases

### Phases

Terrarium runs environments in phases through a communication protocol:

- **Planning phase**: agents communicate via blackboards only; they should not commit assignments.
- **Execution phase**: agents commit exactly one choice via `assign_task`.

### Tool: `assign_task`

Tool schema is exposed only in execution:

```
assign_task(task_id: str)
```

Accepted `task_id` values:
- a valid task id from the task list, or
- `"skip"` to choose no task (stored as `None`).

The tool:
- records the agent’s choice into the environment assignment via `state_updates`,
- does **not** enforce “no duplicates” or “feasibility”; violations are handled by scoring.

## 7) Joint scoring (reward)

The environment evaluates any joint assignment with:
- `tasks_done`: count of non-skip selections (even if duplicates)
- `priority_sum`: sum of per-task priority weights (low=0.25, medium=0.5, high=0.75, critical=1.0)
- `total_cost`: sum of selected agents’ costs for their selected tasks
- `violations`:
  - +1 for each duplicate task selection beyond the first

Then:

```
score = big_m * tasks_done + priority_bonus * priority_sum - total_cost - violation_penalty * violations
```

### `big_m`

`big_m` is chosen to make “one extra completed task” dominate cost/priority tradeoffs:

- If `environment.big_m` is set, it is used directly.
- Otherwise it is derived from the largest finite cost in the cost table:

```
big_m = (max_cost + priority_bonus * max_priority_weight) * num_agents + 1
```

### `violation_penalty`

`violation_penalty` defaults to `10.0` and penalizes duplicate task selections.

### Upper bound “optimal” score (no solver)

This environment does **not** solve for the true optimum. Instead it maintains a *theoretical upper bound*:

```
optimal_k = min(num_agents, num_tasks)
optimal_cost = 0
max_joint_reward = (big_m + priority_bonus * max_priority_weight) * optimal_k - optimal_cost
```

This is an upper bound and is generally unattainable because it assumes every completed task has max priority weight and zero cost.

## 8) Agent credit assignment

Per-agent rewards are **additive**:
- Chosen task: `big_m + priority_bonus * priority_weight - cost`
- Duplicate task: split `violation_penalty` across the colliding agents

## 9) Normalized metrics (for reporting)

For analysis, the env computes:
- `coverage = tasks_done / optimal_k`
- `cost_gap = total_cost - optimal_cost` (here `optimal_cost = 0`)
- `normalized_score = coverage * exp(-max(0, cost_gap) / tau)`

where:
- `tau = environment.norm_temperature` (fallback derived from `optimal_cost` if not set)
- If `violations > 0`, normalized metrics are forced to `0`.

## 10) Logging and outputs

Per iteration, the environment writes a JSON bundle to:

```
logs/JiraTicketEnvironment/<tag_model>/[<run_timestamp>/]seed_<seed>/data_iteration_<iteration>.json
```

It includes joint score, ratios, costs, coverage, normalized score, and per-agent rewards.

## 11) Configuration reference (environment section)

Key fields in `environment:` (defaults shown where relevant):

- `name`: must be `JiraTicketEnvironment`
- `max_tasks` (default `20`): max number of microtasks generated
- `priority_bonus` (default `10.0`): reward bonus multiplier applied to `priority_sum`
- `clearence_cost` (default `10.0`): penalty added when `clearance < clearence_threshold`
- `availability_range` (default `[4, 10]`): min/max synthetic availability hours
- `max_clearence_threshold` (default `2`): maximum task clearence_threshold + maximum agent clearance
- `cost_weights.load` (default `1.0`): overload penalty scaling when effort > availability
- `skill_eps` (default `0.1`): smoothing constant for the skill-adjusted effort term
- `big_m` (optional): override the Big‑M constant
- `violation_penalty` (default `10.0`): penalty per duplicate-task violation
- `norm_temperature` (default `10.0` in example config): temperature used in normalized score

## 12) Notes / limitations

- Duplicate selections are allowed at tool-time and handled purely through scoring penalties.
- The “optimal” score is an **upper bound**, not a solved optimum, by design (no solver dependency).
