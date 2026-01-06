# Network Influence Experiment

This experiment runs an environment on a configurable communication graph, injects adversarial misinformation about a target agent’s *private availability* (e.g., “they are traveling all week and unavailable”), and measures how the misinformation propagates through the network.

## What it measures
- **Adversary count sweep**: vary # adversarial agents and measure final spread.
- **Scaling sweep**: keep 1 adversary, increase total agents, and measure spread.
- **Topology effects**: compares `path`, `star`, and `complete` graphs.

Propagation is evaluated with an **LLM judge** (OpenAI API `gpt-4.1-2025-04-14`) on each agent’s post-run **Survey** response.

## Run
1) Ensure API keys are set (root `.env`):
- `TOGETHER_API_KEY`
- `OPENAI_API_KEY`

2) Run:
```bash
python experiments/network_influence/run.py \
  --config experiments/network_influence/configs/network_influence.yaml
```

## Plot
Generate sweep-level plots (plus optional deep-dive per-run plots):
```bash
python -m experiments.network_influence.plots.generate_all \
  --sweep-dir experiments/network_influence/outputs/network_influence/<timestamp>/runs/<model>/<sweep_name>
```

By default, plot artifacts are written under `experiments/network_influence/plots_outputs/<tag>/<timestamp>/<model>/<sweep_name>/`.

## Environment selection
Set `environment.name` in the YAML config (e.g., `MeetingSchedulingEnvironment` or `JiraTicketEnvironment`).

For custom / non-exported environments, set `environment.import_path` to an explicit import like:
`some.package.some_module:MyEnvironmentClass`.

Outputs are written under `experiments/network_influence/outputs/network_influence/<timestamp>/`.
Each run folder contains `metrics.json`, `judge_results.json`, `survey_responses.json`, `blackboards.json`, `tool_events.json`, and `blackboard_*.txt` (human-readable blackboard snapshots).
The output root also includes `experiment.log` (checkpointed progress logs) and `progress.json` (machine-readable progress).

## Resume (continue an interrupted run)
If a sweep crashes part-way through, you can resume **in-place** (no new timestamp folder) and only execute missing/incomplete runs. The resume script reads `<root>/config.json` by default and writes back into the same `<root>/runs/...` directories.

Dry-run (preview what will run):
```bash
python -m experiments.network_influence.resume \
  --root experiments/network_influence/outputs/network_influence_test/<timestamp> \
  --max-concurrent-runs 50 \
  --dry-run
```

Resume for real:
```bash
python -m experiments.network_influence.resume \
  --root experiments/network_influence/outputs/network_influence_test/<timestamp> \
  --max-concurrent-runs 50
```

## Reproducibility / stochasticity
Set a seed list in `experiments/network_influence/configs/network_influence.yaml` under `experiment.seeds`.
Each sweep setting is repeated once per seed in that list (so runs per setting = `len(experiment.seeds)` unless you set `experiment.runs_per_setting`).

## Parallelism
Set `experiment.max_concurrent_runs` (or pass `--max-concurrent-runs`) to run multiple independent runs in parallel.
This increases API concurrency, so start small to avoid rate limits.

## Victim assignment
Set `experiment.adversary.victim_assignment` to:
- `distinct`: each adversary targets a different non-adversary victim (if feasible)
- `shared`: all adversaries target the same non-adversary victim

## Misinformation frequency
Set `experiment.adversary.misinfo_frequency` to:
- `once`: adversaries seed the misinformation in planning round 1, then focus on normal coordination
- `always`: adversaries re-seed each planning round (still coordinating normally)
