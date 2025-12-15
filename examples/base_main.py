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

from src.communication_protocols.sequential import SequentialCommunicationProtocol
from src.agents.agent_factory import build_agents
from src.networks import build_communication_network
from src.utils import (
    configure_logging,
    load_config,
    create_environment,
    get_model_name,
    build_vllm_runtime,
    handle_mcp_connection_error,
    get_generation_params,
)
import asyncio
from fastmcp import Client
from requests.exceptions import ConnectionError
from src.logger import ToolCallLogger, AgentTrajectoryLogger
from dotenv import load_dotenv

# Run src.server.py to initialzie MCP server before running main.py
try:
    mcp_client = Client("http://localhost:8000/mcp")
except ConnectionError as exc:
    raise RuntimeError(
        "MCP server is not running. Start it with `python src/server.py` before retrying."
    ) from exc

async def run_simulation(config: Dict[str, Any]) -> bool:
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
            config, tool_logger, mcp_client, run_timestamp=run_timestamp
        )
        environment = create_environment(communication_protocol, environment_name, config, tool_logger)

        # Initialize environment-specific tools on the MCP server
        async with mcp_client as client:
            env_class_name = environment.__class__.__name__
            result = await client.call_tool("initialize_environment_tools", {"environment_name": env_class_name})
            logging.info(f"MCP server environment tools initialization: {result.data}")

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
                environment.generate_final_summary()
        finally:
            if provider == "vllm" and vllm_runtime:
                vllm_runtime.shutdown()

        return True

    except Exception as e:
        if handle_mcp_connection_error(e):
            return False

        print(f"Simulation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    configure_logging()
    # Load API keys and other environment variables from .env file
    load_dotenv()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a multi-agent simulation")
    parser.add_argument("--config", type=str)
    parser.add_argument("--note", type=str, default=None,
                        help="Optional experiment note to record alongside logs")

    args = parser.parse_args()
    config = load_config(args.config)
    if args.note:
        config.setdefault("simulation", {})["note"] = args.note
    # For running a single simulation
    asyncio.run(run_simulation(config))
