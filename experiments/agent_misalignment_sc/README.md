# Agent Misalignment Experiment

This experiment evaluates the impact of **misaligned adversarial coalitions** in a distributed **Hospital Resource Allocation** (DCOPS) environment.

## What it measures
The experiment performs sweeps to understand how the system degrades under attack:
- **Adversary Impact Sweep**: Varies the number of adversarial agents (e.g., 0, 1, 2, 4) in a fixed-size network to measure the drop in joint reward and the rise in suspicion.
- **Scaling Sweep**: Varies the total number of agents (e.g., 4, 8, 12) while keeping a fixed number of adversaries to test system resilience at scale.

## Metrics
Metrics are computed in `metrics.py` and logged at the end of every run:
- **Joint Reward**: The total utility achieved by the hospital system (primary performance metric).
- **Suspicion Rate**: The percentage of non-adversarial agents who suspect foul play (evaluated via an LLM judge on post-run surveys).

## Run
1) Ensure API keys are set (root `.env`):
- `OPENAI_API_KEY` (for the evaluator/judge)
- `TOGETHER_API_KEY` (if using Together AI models)

2) Run the experiment:
```bash
python experiments/agent_misalignment/run.py --config experiments/agent_misalignment/configs/agent_misalignment.yaml
```

## Plot
Generate sweep-level plots from a sweep directory (Reward, Failures, Suspicion, Attribution,
Messages/Complaints, Coalition advantage, and distance-based belief effects):
```bash
python -m experiments.agent_misalignment_sc.plots.generate_all \
  --sweep-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name>
```

### Multi-model comparison
To compare multiple models on the same grouped bar charts (x = strategy type, grouped bars = model), run:
```bash
python -m experiments.agent_misalignment_sc.plots.plot_model_comparison \
  --model-runs-dir /bigtemp/duh6ae/Terrarium/experiments/agent_misalignment_sc/outputs/agent_misalignment/20260127-001451/runs/openai-gpt-4.1-mini \
  --model-runs-dir /bigtemp/duh6ae/Terrarium/experiments/agent_misalignment_sc/outputs/agent_misalignment/20260127-001615/runs/openai-gpt-4o-mini \
  --model-runs-dir /bigtemp/duh6ae/Terrarium/experiments/agent_misalignment_sc/outputs/agent_misalignment/20260127-122722/runs/together-kimik2-Instruct \
  --num-agents 9 \
  --target-role departmental \
  --adversary-count 4 \
  --seeds 0-4 \
  --out-dir /bigtemp/duh6ae/Terrarium/experiments/agent_misalignment/plots_outputs/misalignment_plots_3_models
```

If you ran a benign baseline separately (e.g., under a different `<tag>/<timestamp>`), you can
overlay it on the same plots:
```bash
python -m experiments.agent_misalignment_sc.plots.generate_all \
  --sweep-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name> \
  --benign-sweep-dir experiments/agent_misalignment/outputs/<benign_tag>/<benign_timestamp>/runs/<model>/<benign_sweep_name>
```

Optionally filter which seeds to include:
```bash
python -m experiments.agent_misalignment_sc.plots.generate_all \
  --sweep-dir experiments/agent_misalignment/outputs/<tag>/<timestamp>/runs/<model>/<sweep_name> \
  --benign-sweep-dir experiments/agent_misalignment/outputs/<benign_tag>/<benign_timestamp>/runs/<model>/<benign_sweep_name> \
  --seeds 0-9
```

Plots are written to `.../sweep/` plus stratified subfolders:
- `.../sweep/by_strategy/<strategy>/`
- `.../sweep/by_num_agents/n<num_agents>/`
- `.../sweep/by_combo/<strategy>/<target_role>/n<num_agents>/` (one folder per sweep combination)
- `.../sweep/by_num_agents/n<num_agents>/by_strategy_type/<type>/` (type is `benign|covert|destructive`)

Outputs are written to: `experiments/agent_misalignment/plots_outputs/<tag>/<timestamp>/<model>/<sweep_name>/`.
Plots are written as PDF files under `.../sweep/`.

Additional high-level summaries are written under `.../overview/` (bars by strategy type and lines over adversary count),
stratified by `target_role` and `num_agents`.

## Configuration
The experiment is configured via `experiments/agent_misalignment/configs/agent_misalignment.yaml`.

### Environment
* **Class**: `MisalignmentHospitalEnvironment` (defined in `experiments.agent_misalignment_sc.environment`)
* **Topology**: Defaults to `complete` (fully connected) for hospital coordination.
* **Scaling**: the sweep uses `non_provisioner_agents: [8, 12, 16]`, which maps to `environment.num_hospitals` assuming 4 department agents per hospital.

### Adversary Settings
* `experiment.adversary.strategy`: Adversary misalignment strategy (e.g., `covert_max_coalition_reward`).
* `experiment.sweeps.target_roles`: Ensures the coalition contains at least one agent of the specified role; coalition is restricted to a single hospital (plus the provisioner only when the target role is `Resource_Provisioner`).

### Models
The config supports multiple backends:
* **OpenAI**: `gpt-4.1-mini`
* **vLLM**: `Qwen2.5-7B-Instruct` (supports auto-server startup)
* **Together AI**: `Llama-3.3-70b`, `Qwen3-235B`, `gpt-oss-120b`, etc.

## Outputs
Data is saved to `experiments/agent_misalignment/outputs/<tag>/<timestamp>/`.
Each run directory contains:
* `run_config.json`: Full configuration, assigned roles, and final metrics.
* `final_summary.json`: The environment's final state summary.
* `judgements.json`: The LLM judge's assessment of agent suspicion.
* `agent_turns.json`: Full transcript of planning, execution, and survey phases.

## Resume
If a sweep is interrupted, you can resume it without re-running completed settings:

```bash
# Dry run to preview pending runs
python -m experiments.agent_misalignment_sc.resume \
  --root experiments/agent_misalignment/outputs/misalignment/<timestamp> \
  --dry-run

# Actual resume
python -m experiments.agent_misalignment_sc.resume \
  --root experiments/agent_misalignment/outputs/misalignment/<timestamp>
```

## Parallelism
Increase `max_concurrent_runs` in the YAML or via CLI to run multiple simulations in parallel:
```bash
python experiments/agent_misalignment/run.py \
  --config experiments/agent_misalignment/configs/agent_misalignment.yaml \
  --max-concurrent-runs 5
```
