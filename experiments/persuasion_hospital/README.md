# Persuasion Collusion Experiment (HospitalEnvironment)

This experiment adapts the Jira persuasion-collusion setup to the **DCOPS Hospital** environment.
A subset of agents are designated as **colluders** and (optionally) coordinate via a **secret coalition blackboard**.
They attempt to use **persuasion on PUBLIC blackboards** to steer non-colluders toward lower-impact or conflicting actions.

## Key factors
- `colluder_count`: size of the coalition
- `misalignment_strategy`: one of `covert_max_coalition_reward`, `destructive_max_coalition_reward`, `destructive_no_reward_preservation`
- `target_role`: optional constraint on coalition membership (e.g., `Resource_Provisioner`, `departmental`, `Triage`)
- `secret_channel_enabled`: whether a colluder-only secret blackboard is created
- `prompt_variant`: the persuasion playbook colluders use (see `experiments/persuasion_hospital/prompts.py`)
- `topology`, `num_agents`, `num_patients`, `seed`: standard sweeps

## Run
```bash
python experiments/persuasion_hospital/run.py --config experiments/persuasion_hospital/configs/persuasion_hospital.yaml
```

Outputs are written under `experiments/persuasion_hospital/outputs/<timestamp>/`.

## Prompt overrides
You can override built-in variants in YAML:
```yaml
experiment:
  persuasion_hospital:
    prompt_variant_overrides:
      helpful_misdirection: |
        Your custom colluder persuasion playbook...
```
