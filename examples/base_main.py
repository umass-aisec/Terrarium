"""
Main script to run a base/vanilla simulation
"""
# Add project root to path so we can import modules
import logging
import sys
from pathlib import Path

# Add project root to sys.path BEFORE importing local modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import our modules
import argparse
from typing import Any, Dict
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from datetime import datetime
import traceback

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol
from terrarium.agents.agent_factory import build_agents
from terrarium.networks import build_communication_network
from terrarium.utils import (
    configure_logging,
    load_config,
    create_environment,
    get_model_name,
    build_vllm_runtime,
    get_generation_params,
    prepare_simulation_config,
)
import asyncio
from terrarium.core.logger import ToolCallLogger, AgentTrajectoryLogger
from dotenv import load_dotenv

async def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    vllm_runtime = None
    try:
        seed = config["simulation"]["seed"]
        run_timestamp = config.get("simulation", {}).get("run_timestamp")
        if not run_timestamp:
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config.setdefault("simulation", {})["run_timestamp"] = run_timestamp

        # Initialize environment
        environment_name = config["environment"]["name"]

        # Initialize loggers
        tool_logger = ToolCallLogger(environment_name, seed, config, run_timestamp=run_timestamp)
        trajectory_logger = AgentTrajectoryLogger(environment_name, seed, config, run_timestamp=run_timestamp)

        communication_protocol = SequentialCommunicationProtocol(
            config, tool_logger, run_timestamp=run_timestamp
        )
        environment = create_environment(communication_protocol, environment_name, config, tool_logger)
        communication_protocol.bind_environment(environment)

        agent_names = environment.get_agent_names()
        communication_network = build_communication_network(agent_names, config)
        environment.set_communication_network(communication_network)

        # Reset tool call log for new simulation
        environment.tool_logger.reset_log()
        await environment.async_init()

        # Get provider and model name
        llm_config = config["llm"]
        provider_label = llm_config.get("provider", "unknown")
        provider = provider_label.lower()
        log_path = None
        if provider == "vllm":
            vllm_runtime = build_vllm_runtime(llm_config)
            model_name = vllm_runtime.describe_default_model()
            log_path = vllm_runtime.describe_log_path()
        else:
            model_name = get_model_name(provider, llm_config)
        log_suffix = f" (server logs: {log_path})" if log_path else ""
        logging.info(f"Using provider: {provider_label}, model: {model_name}{log_suffix}")
        generation_params = get_generation_params(llm_config)

        max_conversation_steps = config["simulation"].get("max_conversation_steps", 3)

        agents = build_agents(
            agent_names,
            provider=provider,
            provider_label=provider_label,
            llm_config=llm_config,
            model_name=model_name,
            max_conversation_steps=max_conversation_steps,
            tool_logger=tool_logger,
            trajectory_logger=trajectory_logger,
            environment=environment,
            generation_params=generation_params,
            vllm_runtime=vllm_runtime if provider == "vllm" else None,
        )
        environment.set_agent_clients(agents)

        max_iterations = config["simulation"].get("max_iterations", 1)
        max_planning_rounds = config["simulation"].get("max_planning_rounds", 1)
        try:
            with logging_redirect_tqdm():
                # Main iteration
                for iteration in tqdm(range(1, max_iterations + 1), desc="Iterations", position=0, leave=True, ncols=80):
                    current_iteration = iteration
                    if environment.done(current_iteration):
                        logging.info(f"Environment requested simulation stop at iteration {current_iteration}")
                        break
                    # Planning Phase
                    for planning_round in tqdm(range(1, max_planning_rounds + 1), desc="  Planning", position=1, leave=False, ncols=80):
                        # Use consistent agent order for this iteration
                        for agent in tqdm(environment.agents, desc="       Agents", position=2, leave=False, ncols=80):
                            agent_context = environment.build_agent_context(agent.name, phase="planning", iteration=iteration, planning_round=planning_round)
                            await communication_protocol.agent_planning_turn(agent, agent.name, agent_context, environment, iteration, planning_round)

                    # Execution Phase
                    with tqdm(total=1, desc="  Execution", position=1, leave=False, ncols=80) as pbar:
                        for agent in tqdm(environment.agents, desc="       Agents", position=2, leave=False, ncols=80):
                            agent_context = environment.build_agent_context(agent.name, phase="execution", iteration=iteration)
                            await communication_protocol.agent_execution_turn(agent, agent.name, agent_context, environment, iteration)
                        pbar.update(1)

                    environment.log_iteration_summary(current_iteration)
                final_summary = environment.generate_final_summary()
        finally:
            if provider == "vllm" and vllm_runtime:
                vllm_runtime.shutdown()

        return {
            "success": True,
            "final_summary": final_summary,
            "log_dir": str(tool_logger.log_dir),
            "run_timestamp": run_timestamp,
        }

    except Exception as e:
        print(f"Simulation failed: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "final_summary": {},
            "log_dir": None,
            "run_timestamp": None,
            "error": str(e),
        }

# Agent-tier model presets. The compaction tier is deliberately left alone: it
# stays on a cheap model whatever the agents run on.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "gpt-5.4-nano": {"max_tokens": 256, "temperature": 0.2},
    # Reasoning-tier: spends part of the budget on hidden reasoning before
    # emitting the tool call, so 256 silently truncates. Also rejects a
    # non-default temperature -- see restricted_models in
    # terrarium/llm/clients/openai_client.py -- hence no temperature key.
    "gpt-5.5": {"max_tokens": 1500},
}


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Layer --seed/--persona/--personas/--model onto a loaded config."""
    if args.seed is not None:
        config = prepare_simulation_config(config, args.seed)
    if args.note:
        config.setdefault("simulation", {})["note"] = args.note

    env = config.setdefault("environment", {})
    if args.persona:
        env["persona"] = args.persona
        env.pop("personas", None)
    elif args.personas:
        env["personas"] = {
            k.strip(): v.strip()
            for k, v in (p.split("=", 1) for p in args.personas.split(",") if p.strip())
        }
        env.pop("persona", None)

    if args.model:
        provider = str(config.get("llm", {}).get("provider", "foundry"))
        block = config.setdefault("llm", {}).setdefault(provider, {})
        block["model"] = args.model
        # Replace params wholesale so a restricted model does not inherit a
        # temperature the previous model allowed.
        if args.model in MODEL_PRESETS:
            block["params"] = dict(MODEL_PRESETS[args.model])

    # The compaction tier follows --model unless pinned separately.
    compaction_model = args.compaction_model or args.model
    if compaction_model:
        compaction = config.setdefault("llm", {}).setdefault("compaction", {})
        provider = str(compaction.get("provider", "foundry"))
        block = compaction.setdefault(provider, {})
        block["model"] = compaction_model
        if compaction_model in MODEL_PRESETS:
            params = dict(block.get("params") or {})
            # Restricted models reject an explicit temperature.
            if "temperature" not in MODEL_PRESETS[compaction_model]:
                params.pop("temperature", None)
            block["params"] = params

    # tags[0] names the log subdirectory (see get_tag_model_subdir), so derive
    # them deterministically whenever an arm is selected on the command line.
    if args.persona or args.personas or args.model:
        persona_tag = "persona_mixed" if args.personas else f"persona_{args.persona or 'none'}"
        model_slug = (args.model or "").replace(".", "").replace("-", "")
        config.setdefault("simulation", {})["tags"] = (
            [persona_tag] + ([f"model_{model_slug}"] if args.model else [])
        )
    return config


if __name__ == "__main__":
    configure_logging()
    # Load API keys and other environment variables from .env file
    load_dotenv()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a multi-agent simulation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--note", type=str, default=None,
                        help="Optional experiment note to record alongside logs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override simulation.seed")
    parser.add_argument("--persona", type=str, default=None,
                        help="Apply one persona to every agent, e.g. diplomat")
    parser.add_argument("--personas", type=str, default=None,
                        help="Per-agent personas, e.g. agent_0=territorial,agent_1=diplomat")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Override the agent model. Presets: {', '.join(MODEL_PRESETS)}")
    parser.add_argument("--compaction-model", type=str, default=None,
                        help="Pin the compaction model (defaults to following --model)")

    args = parser.parse_args()
    if args.persona and args.personas:
        parser.error("--persona and --personas are mutually exclusive")
    config = apply_cli_overrides(load_config(args.config), args)
    # For running a single simulation
    asyncio.run(run_simulation(config))
