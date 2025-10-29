"""
Main script to run a base agent
"""
# Add project root to path so we can import modules
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
import argparse
import random
from typing import Any, Dict
from tqdm import tqdm
from datetime import datetime

from src.agent import Agent
from src.communication_protocol import CommunicationProtocol
from src.logger import ToolCallLogger, AgentTrajectoryLogger, AttackLogger
from src.utils import load_config, get_client_instance, create_environment, get_model_name
import asyncio
from fastmcp import Client
from requests.exceptions import ConnectionError
from attack_module.framework import AttackManager

try:
    mcp_client = Client("http://localhost:8000/mcp")
except ConnectionError as exc:
    raise RuntimeError(
        "MCP server is not running. Start it with `python src/server.py` before retrying."
    ) from exc


async def run_simulation(config: Dict[str, Any]) -> bool:
    """
    Run a single simulation.

    Args:
        config: Configuration for the simulation

    Returns:
        True if simulation succeeded, False otherwise
    """
    try:
        seed = config["environment"]["rng_seed"]
        run_timestamp = config.get("simulation", {}).get("run_timestamp")
        if not run_timestamp:
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config.setdefault("simulation", {})["run_timestamp"] = run_timestamp
        print(f"\n{'='*60}")
        print(f"SIMULATION - SEED: {seed}")
        print(f"{'='*60}")

        # Initialize environment
        environment_name = config["environment"]["name"]

        # Initialize loggers
        tool_logger = ToolCallLogger(environment_name, seed, config, run_timestamp=run_timestamp)
        trajectory_logger = AgentTrajectoryLogger(environment_name, seed, config, run_timestamp=run_timestamp)
        attack_logger = AttackLogger(environment_name, seed, config, run_timestamp=run_timestamp)

        communication_protocol = CommunicationProtocol(config, tool_logger, mcp_client, run_timestamp=run_timestamp)
        environment = create_environment(communication_protocol, environment_name, config, tool_logger)

        # Initialize environment-specific tools on the MCP server
        async with mcp_client as client:
            result = await client.call_tool("initialize_environment_tools", {"environment_name": environment_name})
            print(f"MCP server environment tools: {result.data}")

        # Reset tool call log for new simulation
        environment.tool_logger.reset_log()
        await environment.async_init()

        attack_manager = AttackManager(config.get("attacks"), attack_logger=attack_logger)

        # Initialize agents
        agent_names = environment.get_agent_names()

        # Get provider and model name
        llm_config = config["llm"]
        provider = llm_config.get("provider", None)
        model_name = get_model_name(provider, llm_config)
        print(f"Using provider: {provider}, model: {model_name}")

        max_conversation_steps = config["simulation"].get("max_conversation_steps", 3)

        # Create agents with appropriate client for each
        agents = []
        for name in agent_names:
            client = get_client_instance(llm_config)
            print(f"Initializing Agent: {name} with {provider} - {model_name}")
            agent = attack_manager.create_agent(
                base_class=Agent,
                client=client,
                name=name,
                model_name=model_name,
                max_conversation_steps=max_conversation_steps,
                tool_logger=tool_logger,
                trajectory_logger=trajectory_logger,
                environment_name=environment_name,
            )
            agents.append(agent)
        # Shuffle initial agent order, and maintain order through simulation
        random.shuffle(agents)

        environment.set_agent_clients(agents)

        max_iterations = config["simulation"].get("max_iterations", 1)
        max_planning_rounds = config["simulation"].get("max_planning_rounds", 1)
        try:
            # Main iteration progress bar
            for iteration in tqdm(range(1, max_iterations + 1), desc="Iterations", position=0, leave=True, ncols=80):
                current_iteration = iteration
                if not environment.should_continue_simulation(current_iteration):
                    print(f"Environment requested simulation stop at iteration {current_iteration}")
                    break
                ## Protocol-level attacks (e.g., poisoning blackboards) before the planning turn
                await attack_manager.run_protocol_hooks(
                    "pre_planning",
                    communication_protocol,
                    iteration=iteration,
                    phase="planning",
                )
                # Planning Phase with progress bar
                for planning_round in tqdm(range(1, max_planning_rounds + 1), desc="  Planning", position=1, leave=False, ncols=80):
                    # Use consistent agent order for this iteration
                    for agent in tqdm(environment.agents, desc="       Agents", position=2, leave=False, ncols=80):
                        agent_context = environment.build_agent_context(agent.name, phase="planning", iteration=iteration, planning_round=planning_round)
                        await communication_protocol.agent_planning_turn(agent, agent.name, agent_context, environment, iteration, planning_round)

                # Execution Phase with single progress indicator
                with tqdm(total=1, desc="  Execution", position=1, leave=False, ncols=80) as pbar:
                    for agent in tqdm(environment.agents, desc="       Agents", position=2, leave=False, ncols=80):
                        agent_context = environment.build_agent_context(agent.name, phase="execution", iteration=iteration)
                        await communication_protocol.agent_execution_turn(agent, agent.name, agent_context, environment, iteration)
                    pbar.update(1)

                environment._log_iteration_summary(current_iteration)
            environment._generate_final_summary()
        finally:
            environment.cleanup()

        print(f"\nSimulation completed successfully")
        return True

    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a multi-agent simulation")
    parser.add_argument("--config", type=str)

    args = parser.parse_args()
    config = load_config(args.config)

    # For running a single simulation
    asyncio.run(run_simulation(config))
