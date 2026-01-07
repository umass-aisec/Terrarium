# Agent Misalignment Experiment

This experiment evaluates the impact of **misaligned adversarial agents** in a distributed **Hospital Resource Allocation** (DCOPS) environment. Adversaries attempt to deplete shared resources or front-run requests ("Resource Depletion & Front-Running"), reducing the global utility (joint reward) while attempting to remain undetected.

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
Generate sweep-level plots (e.g., Reward vs. Adversary Count, Suspicion Rates):
```bash
python -m experiments.agent_misalignment.plots.generate_all \
  --sweep-dir experiments/agent_misalignment/outputs/misalignment/<timestamp>/runs/<model>/<sweep_name>
```

Outputs are written to: `experiments/agent_misalignment/plots_outputs/<tag>/<timestamp>/<model>/<sweep_name>/`.

## Configuration
The experiment is configured via `experiments/agent_misalignment/configs/agent_misalignment.yaml`.

### Environment
* **Class**: `HospitalEnvironment` (defined in `envs.dcops.hospital.hospital_env`)
* **Topology**: Defaults to `complete` (fully connected) for hospital coordination.

### Adversary Settings
* `experiment.adversary.strategy`: Method for selecting adversaries (e.g., `random`, `highest_degree`).
* `experiment.adversary.attack_frequency`:
    * `always`: Adversaries attempt to front-run/deplete resources in every planning round.
    * `once`: Adversaries attack only once (e.g., in the first round).

### Models
The config supports multiple backends:
* **OpenAI**: `gpt-4.1-mini`
* **vLLM**: `Qwen2.5-7B-Instruct` (supports auto-server startup)
* **Together AI**: `Llama-3.3-70b`, `Qwen3-235B`, `gpt-oss-120b`, etc.

## Outputs
Data is saved to `experiments/agent_misalignment/outputs/misalignment/<timestamp>/`.
Each run directory contains:
* `run_config.json`: Full configuration, assigned roles, and final metrics.
* `final_summary.json`: The environment's final state summary.
* `evaluator_results.json`: The LLM judge's assessment of agent suspicion.
* `agent_turns.json`: Full transcript of planning, execution, and survey phases.

## Resume
If a sweep is interrupted, you can resume it without re-running completed settings:

```bash
# Dry run to preview pending runs
python -m experiments.agent_misalignment.resume \
  --root experiments/agent_misalignment/outputs/misalignment/<timestamp> \
  --dry-run

# Actual resume
python -m experiments.agent_misalignment.resume \
  --root experiments/agent_misalignment/outputs/misalignment/<timestamp>
```

## Parallelism
Increase `max_concurrent_runs` in the YAML or via CLI to run multiple simulations in parallel:
```bash
python experiments/agent_misalignment/run.py \
  --config experiments/agent_misalignment/configs/agent_misalignment.yaml \
  --max-concurrent-runs 5
```