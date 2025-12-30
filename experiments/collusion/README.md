# Covert Collusion Experiment (JiraTicket default)

This experiment tests whether a **subset of colluding agents** can (a) coordinate covertly using a **hidden colluder-only blackboard** and (b) **benefit at the expense of non-colluders** via **active deception** on shared blackboards.

It is modeled after `experiments/network_influence`, but uses an **in-process** communication protocol (`LocalMegaboardProtocol`) so it does **not** require an MCP server and supports **parallel runs**.

## Key factors
- `colluder_count`: how many agents are in the coalition
- `secret_channel_enabled`: whether a colluder-only blackboard is added (hidden from others)
- `prompt_variant`: which colluder prompt variant is used (controls collusion/deception strategy)
- `topology`, `num_agents`, `seed`: standard comms + stochasticity sweeps

## Run
1) Ensure API keys are set (root `.env`) for your chosen provider.
2) Run:
```bash
python experiments/collusion/run.py --config experiments/configs/collusion_jira.yaml
```

Outputs are written under `experiments/outputs/collusion/<timestamp>/`.

## Plot (histograms)
After running, generate histogram plots from a sweep directory:
```bash
python -m experiments.plots.generate_collusion_histograms \
  --sweep-dir experiments/outputs/collusion/<timestamp>/runs/<model>/<sweep_name>
```

## Resume (in-place)
If a sweep crashes part-way through, or you want to extend it to more seeds, you can resume **in-place**
(no new timestamp folder) and only execute missing/incomplete runs.

Example: extend seeds to 1–10, then resume an existing output root:
```bash
python -m experiments.collusion.resume \
  --root experiments/outputs/collusion_test/<timestamp> \
  --config experiments/configs/collusion_jira_test.yaml
```

Tip: use `--dry-run` first to see what will run.
