# Terrarium

![alt text](dev/terrarium_logo_rounded.png)

## Overview :herb:

Terrarium is a hackable, modular, and configurable open-source framework for studying and evaluating decentralized LLM-based multi-agent systems (MAS). As the capabilities of agents progress (e.g., tool calling) and their state space expands (e.g., the internet), multi-agent systems will naturally arise in unique and unexpected scenarios. This repo aims to provide researchers, engineers, and students the ability to study this new agentic paradigm in an isolated playground for studying agent behavior, vulnerabilities, and safety. It enables full customization of the communication protocol, communication proxy, environment, tool usage, and agents. View the paper at [https://arxiv.org/pdf/2510.14312v1](https://arxiv.org/pdf/2510.14312v1).

This repo is under active development :gear:, so please raise an issue for new features, bugs, or suggestions. If you find this repo useful or interesting please :star: it!

![Framework Diagram](dev/framework_rounded.png)

## Features

- **Blackboards (Communication Proxies)**: Append-only event/communication log which acts as a component of the agent's observation and communication with other agents.
- **Two-Phase Communication Protocol**: The implemented communication protocol containes two phases, a (1) *planing phase* and an (2) *execution phase*. The planning phase enables communcation between agents to faciliate better action selection during the executation phase. During the executation phase, the agents take **actions** that affect their environment. This is done in a predefined sequential order to avoid environment simulation clashes.
- **MCP Servers**: We use MCP servers to provide easy integration with varying LLM client APIs while enabling easier configuration of environment and external tools.
- **DCOP Environments**: DCOPs (Distributed Constraint Optimization Problems) have a **ground-truth solution** and a well-grounded evalution function, evaluating the actions taken by a set of agents. We implement DCOP environments from the [CoLLAB](https://openreview.net/pdf?id=372FjQy1cF) benchmark.
  - SmartGrid - A home agent's objecitve is to schedule appliance usage throughout the day without overworking the powergrid (Uses real-world home-meter data)
  - MeetingScheduling - A calendar agent is tasked with assigning meetings with other agents, trying to satisfy preferences and constraints with respect to other agents schedules (Uses real-world locations)
  - PersonalAssistant - An assistant agent chooses outfits for a human while meeting social norm preferences, the preferences of the human, and constrained outfit selection (Uses fully synthetic data)
<!-- - **One Stochastic Game Environment (Trading)**: A simple trading environment where agents trade and buy items to maximize their personal cumulative inventory utility. Agents trade items (e.g., TV, phone, banana) and negotiate with each other given limited resources. This environment allows multi-step simulation with multiple evaluation steps. -->

## Documentation 

Use the following [documentation](https://aisec.cs.umass.edu/projects/terrarium/docs) for detailed instructions about on how to use the framework. 

Follow the quick guide provided below for basic testing.

## Quick Start

**Requirements**: Python 3.11+

1. Clone the repository:
```bash
git clone <repository-url>
cd terrarium
```

2. Start up a new conda environment and activate it
```bash
conda create --name terrarium
conda activate terrarium
```

3. Install dependencies (recommended to use Conda env):
Use uv for fast dependency installation
```bash
uv pip install -e .
```
or just 
```bash
pip install -e .
```


<!-- 3a. (If using vLLM for servicing) Start the vLLM server (Read `server/docs/USAGE.md` before using vLLM):
```bash
python server/utils/start_vllm_server.py
```
and set the `"use_openai_api": false` in `configs/config.json` -->
4. Create a .env file at the root directory --> terrarium/.env
5. Set your provider keys in the .env file (vLLM integration coming soon :gear:):
```bash
# In .env file
OPENAI_API_KEY=<your_key>
GOOGLE_API_KEY=<your_key>
ANTHROPIC_API_KEY=<your_key>
```

### Running
1. Start up the MCP server
```bash
python src/server.py
```
2. Run the base simulation using a config file:
```bash
python3 examples/base/main.py --config <yaml_config_path>
```

## Attack Scenarios

Terrarium ships three reference attacks that exercise different points in the stack. Implementations live in `attack_module/attack_modules.py` and can be mixed into any simulation via the provided runners.

| Attack | What it targets | Entry point | Payload config |
| --- | --- | --- | --- |
| Agent poisoning | Replaces every `post_message` payload from the compromised agent before it reaches the blackboard. | `examples/attacks/main.py --attack_type agent_poisoning` | `examples/configs/attack_config.yaml` (`poisoning_string`) |
| Context overflow | Appends a large filler block to agent messages to force downstream context truncation. | `examples/attacks/main.py --attack_type context_overflow` | `examples/configs/attack_config.yaml` (`header`, `filler_token`, `repeat`, `max_chars`) |
| Communication protocol poisoning | Injects malicious system messages into every blackboard via the MCP layer. | `examples/attacks/main.py --communication_protocol_poisoning` | `examples/configs/attack_config.yaml` (`poisoning_string`) |

### Running agent-side attacks

Use the unified driver to launch both the standard run and the selected attack:

```bash
# Agent poisoning example
python examples/attacks/main.py \
  --config examples/configs/meeting_scheduling.yaml \
  --poison_payload examples/configs/attack_config.yaml \
  --attack_type agent_poisoning

# Context overflow example
python examples/attacks/main.py \
  --config examples/configs/meeting_scheduling.yaml \
  --poison_payload examples/configs/attack_config.yaml \
  --attack_type context_overflow
```

## Quick Tips
- When working with Terrarium, use sublass definitions (e.g., A2ACommunicationProtocol, EvilAgent) of the base module classes (e.g., CommunicationProtocol, Agent) rather than directly changing the base module classes.
- When creating new environments, ensure they inherit the AbstractEnvironment class and all methods are properly defined.
- Keep in mind some models (e.g., gpt-4.1-nano) are not capable enough of utilizing tools to take actions in the environment, so track the completion rate such as `Meeting completion: 15/15 (100.0%)` for MeetingScheduling.

## vLLM Provider (Open-Source Models)
1. Install vLLM (`pip install vllm`) and make sure CUDA is available.
2. Set `llm.provider: "vllm"` in your config and describe the single server under `llm.vllm`.
3. All agents share the one configured vLLM model; advanced routing is disabled in this setup.

Best *small* model for successful tool use tested so far: Qwen/Qwen2.5-7B-Instruct. We have not tested on large >70B open-source models, but use use the [Berkeley Function-Calling Leaderboard - BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) as a reference.

Minimal example:

```yaml
llm:
  provider: "vllm"
  vllm:
    auto_start_server: true
    startup_timeout: 180
    models:
      - checkpoint: "/data/models/Qwen2-7B-Instruct"
        served_model_name: "Qwen2-7B-Instruct"
        host: "127.0.0.1"
        port: 8001
        tensor_parallel_size: 1
        trust_remote_code: true
        additional_args:
          - "--max-model-len"
          - "65536"
```

If `auto_start_server` is true and the configured endpoint is unreachable, Terrarium launches `python -m vllm.entrypoints.openai.api_server` with the supplied checkpoint and writes stdout/stderr to `logs/vllm/<model_id>.log`. Processes are cleaned up automatically after each run.

## Dashboard

Consolidates runs and logs into a static dashboard for easier navigation:

1. Export the data bundle (runs + config):

   ```bash
   python dashboards/build_data.py \
     --logs-root logs \
     --config examples/configs/meeting_scheduling.yaml \
     --output dashboards/public/dashboard_data.json
   ```

2. Serve the static front-end (or simply open the file via your browser if it allows `file://` fetches – a local server is recommended):

   ```bash
   python -m http.server 5050 --directory dashboards/public
   ```

3. Navigate to <http://127.0.0.1:5050> to inspect the raw event logs parsed directly from `dashboard_data.json` in the browser (no backend required).

4. New runs? Simply repeat step (1.) and refresh the website (No need to restart the server)

## Tooling (MCP Servers)

To standardize tool usage among different model providers, we employ an MCP server using FastMCP. Each environment has their own set of MCP tools that are readily available to the agent with the functionality of permitting certain tools by the communication protocol. Some examples of environment tools are MeetingScheduling -> schedule_meeting(.), PersonalAssistant -> choose_outfit(.), and SmartGrid -> schedule_task(.).


## Logging

Terrarium incorporates a set of loggers for prompts, tool usage, agent trajectories, and blackboards. All loggers are defined in `src/logger.py`, conisting of
- BlackboardLogger -- Logs events for all existing blackboards in human-readable format (Useful for tracking conversations between agents and tool calls)
- ToolCallLogger -- Tracks the tool called, success, and duration for each agent (Useful for debugging tool implementations)
- PromptLogger -- Shows exact system and user prompts used (Useful for debugging F-string formatted prompts)
- AgentTrajectoryLogger -- Logs the multi-step conversation of each agent showing their pseudo-reasoning traces (Useful for approximately evaluating the internal reasoning of agents and their associated tool calls)

All logs are saved to `logs/<environment>/<tag_model>/<run_timestamp>/seed_<seed>/`, including a snapshot of the config used for that run.

## TODOs
- [x] !! Get parallelized simulations working on multiple seeds
- [x] !! Implement vLLM client
- [ ] !! Optimize parallel inference for fast large agent experiments
- [ ] ! Add multi-step negotiation environments (e.g., Trading)
- [ ] ! Update the CoLLAB environments
- [x] Improve the Dashboard UI
- [ ] Add tests directory

## Paper Citation
```bibtex
@article{nakamura2025terrarium,
  title={Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies},
  author={Nakamura, Mason and Kumar, Abhinav and Mahmud, Saaduddin and Abdelnabi, Sahar and Zilberstein, Shlomo and Bagdasarian, Eugene},
  journal={arXiv preprint arXiv:2510.14312},
  year={2025}
}
```

## License

MIT

## Contributing

We welcome pull requests and issues that improve Terrarium’s tooling, environments, docs, or general ecosystem. Before opening a PR, start a brief issue or discussion outlining the change so we can coordinate scope and avoid overlap. If you are unsure whether an idea fits, just ask.
