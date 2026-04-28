# Microsoft Foundry Quick Start

Run these commands from the repo root.

Replace these placeholders before running:

- `<resource>`
- `<project>`
- `<deployment-or-model-name>`

## 1. Install everything

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

uv venv --python 3.11 .venv
source .venv/bin/activate
uv sync
./.venv/bin/pip install azure-identity

az login --use-device-code
```

## 2. Configure Foundry auth

```bash
cp -n .env.example .env

cat >> .env <<'EOF'
AI_FOUNDRY_AUTH_MODE=entra
AI_FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
EOF
```

If one model needs its own Foundry key, add another env var:

```bash
cat >> .env <<'EOF'
AI_FOUNDRY_RBR_EAST_US_2_PROJECT_ENDPOINT=https://rbr-east-us-2-resource.services.ai.azure.com/api/projects/rbr-east-us-2
AI_FOUNDRY_RBR_EAST_US_2_API_KEY=<your-rbr-east-us-2-project-key>
EOF
```

## 3. Create a one-run smoke test

```bash
cat > /tmp/foundry_smoke.yaml <<'EOF'
simulation:
  max_iterations: 1
  max_planning_rounds: 1
  max_conversation_steps: 1
  seed: 1

environment:
  name: JiraTicketEnvironment
  assignment_filling: true
  max_tasks: 8

communication_network:
  topology: complete
  num_agents: 6
  consolidate_channels: true

llm_models:
  - label: foundry-smoke
    llm:
      provider: foundry
      foundry:
        model: <deployment-or-model-name>
        params:
          max_tokens: 1500

experiment:
  tag: collusion_foundry_smoke
  output_dir: /tmp/terrarium-collusion-smoke
  max_concurrent_runs: 1
  seeds: [1]
  runs_per_seed: 1
  collusion:
    colluder_selection: random
  sweeps:
    - name: complete_n6_c2_smoke
      topologies: [complete]
      num_agents: [6]
      colluder_counts: [2]
      secret_channel_enabled: [false]
      prompt_variants: [control]
EOF
```

If one model needs a separate key from `.env`, use:

```yaml
llm:
  provider: foundry
  foundry:
    auth_mode: api_key
    project_endpoint_env_var: AI_FOUNDRY_RBR_EAST_US_2_PROJECT_ENDPOINT
    api_key_env_var: AI_FOUNDRY_RBR_EAST_US_2_API_KEY
    model: claude-opus-4-6
    request_timeout: 120
    params:
      max_tokens: 1500
```

For Foundry deployments that only advertise chat completions, add:

```yaml
llm:
  provider: foundry
  foundry:
    project_endpoint_env_var: AI_FOUNDRY_PROJECT_ENDPOINT
    api_key_env_var: AI_FOUNDRY_API_KEY
    api_style: chat_completions
    model: grok-4-20-reasoning
    request_timeout: 120
    params:
      max_tokens: 1500
```

## 4. Run the smoke test

```bash
env PYTHONPATH=.:build/lib ./.venv/bin/python -m experiments.collusion.run \
  --config /tmp/foundry_smoke.yaml \
  --out-dir /tmp/terrarium-collusion-smoke \
  --max-concurrent-runs 1
```

## 5. Check whether it worked

```bash
find /tmp/terrarium-collusion-smoke -maxdepth 2 \( -name progress.json -o -name experiment.log \) | sort
```

If the run succeeds, move on to your real config.
