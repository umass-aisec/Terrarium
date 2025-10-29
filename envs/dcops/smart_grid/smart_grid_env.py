"""
SmartGrid Environment Adaptor

Adaptor to integrate the SmartGrid domain (power grid task scheduling)
with the black_boards_v5 communication protocol framework.

The SmartGrid environment involves home agents coordinating to schedule
power-consuming tasks while minimizing main grid draw and respecting
sustainable capacity constraints.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from src.agent import Agent

# Import SmartGrid modules with proper relative imports
from envs.dcops.CoLLAB.SmartGrid.data_structure import NeighborhoodPowerLiteEnv
from envs.dcops.CoLLAB.SmartGrid.generate import generate_instance_from_catalog

# Import abstract environment interface and logger
from envs.abstract_environment import AbstractEnvironment
from envs.dcops.plotter import ScorePlotter
from src.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
    build_plots_dir,
)

# Import SmartGrid tools and prompts
from .smartgrid_prompts import SmartGridPrompts


class SmartGridEnvironment(AbstractEnvironment):
    """
    SmartGrid environment adaptor for power grid task scheduling.

    Home agents coordinate to schedule power-consuming tasks while minimizing
    main grid draw and respecting sustainable capacity constraints.
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """
        Initialize the SmartGrid environment.

        Args:
            communication_protocol: CommunicationProtocol instance for blackboard management
            config: Full configuration dictionary containing all simulation settings
            tool_logger: ToolCallLogger instance for logging tool calls
        """
        # Store configuration
        self.full_config = config
        self.config = config["environment"]  # Extract environment-specific config
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.tool_logger = tool_logger

        # Get the correct seed from environment config
        self.current_seed = self.config.get("rng_seed", 42)

        # Simulation state tracking
        self.current_iteration = 0
        self.simulation_config = config["simulation"]
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)

        # Action types to log to blackboards
        self.action_logging_config = ["schedule_task"]

        # Instance management
        # Use tuple keys internally; convert to/from strings only at serialization boundaries
        self.task_schedules: Dict[Tuple[str, str], int] = {}  # (home_id, task_id) -> start_time
        self.agents_list: List[str] = []  # Home IDs
        self.agents: List['Agent'] = []  # Agent instances

        # Data paths
        self.data_root = Path(__file__).parent.parent / "CoLLAB" / "SmartGrid" / "data"
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Score tracking (main grid draw history)
        self.grid_draw_history: List[float] = []
        self.home_scores_history: Dict[str, List[float]] = {}
        self.score_plotter: Optional[ScorePlotter] = None

        # Generate SmartGrid instance - resolve catalog path relative to project root
        catalog_path = str((self.project_root / self.config.get("catalog_path", self.data_root / "devices.json")).resolve())
        generation_config = {
            "T": self.config.get("T", 24),
            "n_homes": self.config.get("n_homes", 5),
            "tasks_per_home": self.config.get("tasks_per_home", (2, 4)),
            "allowed_window_len": self.config.get("allowed_window_len", (2, 6)),
            "seed": self.config.get("rng_seed", 42)
        }

        self.instance = generate_instance_from_catalog(generation_config, catalog_path)
        self.environment = NeighborhoodPowerLiteEnv(self.instance)

        # Extract home IDs as agents
        self.agents_list = [home.id for home in self.instance.homes]

        # Clear seed directories FIRST to ensure clean state for this run
        clear_seed_directories("SmartGrid", self.current_seed, self.full_config)

        # Initialize prompts (must be after instance creation)
        # Note: tools are now in MCP server, not in environment
        self.prompts = SmartGridPrompts(self, self.full_config)

        # Initialize score tracking
        self.home_scores_history = {home_id: [] for home_id in self.agents_list}

        # Get tag_model subdirectory for plots
        tag_model = get_tag_model_subdir(self.full_config)
        plots_dir = build_plots_dir("SmartGrid", tag_model, self.current_seed, self.run_timestamp)
        self.score_plotter = ScorePlotter(save_dir=str(plots_dir))

        print(f"SmartGrid environment initialized with {len(self.agents_list)} homes")
        print(f"Homes: {', '.join(self.agents_list)}")
        print(f"Total tasks: {sum(len(home.tasks) for home in self.instance.homes)}")
        print(f"Time horizon: {self.instance.T} slots")

    async def async_init(self):
        """Async initialization - create blackboards from factors."""
        await self._create_blackboards_from_factors()

    def set_agent_clients(self, agents: List['Agent']):
        """Set the agents for the environment."""
        self.agents = agents


    async def _create_blackboards_from_factors(self):
        """Create blackboards for SmartGrid power coordination. This happens only during initialization"""

        # Create a global blackboard for all homes to coordinate power usage
        context = f"Power Grid Coordination: All homes must coordinate to minimize main grid draw. " \
                 f"Sustainable capacity varies over {self.instance.T} time slots. " \
                 f"Share your scheduling intentions and work together to avoid peak consumption."

        # Create a pseudo-factor for the global coordination blackboard
        class GlobalFactor:
            def __init__(self, agent_scope):
                self.agent_scope = agent_scope

        global_factor = GlobalFactor(self.agents_list)
        blackboard_id = await self.communication_protocol.generate_blackboard_network_from_factor(global_factor, context)
        print(f"Created Global Power Grid Blackboard {blackboard_id}: {', '.join(self.agents_list)}")

    def get_agent_names(self) -> List[str]:
        """Get list of agent names."""
        return self.agents_list.copy()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int,
                          blackboard_contexts: Optional[Dict[str, str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            blackboard_contexts: Blackboard contexts from the protocol
            **kwargs: Additional context

        Returns:
            Dictionary with agent context
        """
        if not self.instance:
            return {"error": "Environment not initialized"}

        # Clear task schedules at the start of each new iteration's planning phase
        # to allow homes to make new scheduling choices
        if phase == "planning" and iteration > 1 and self.task_schedules:
            print(f"SmartGrid: Clearing task schedules for iteration {iteration}")
            self.task_schedules = {}

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "task_schedules": self.task_schedules.copy(),
            "total_homes": len(self.agents_list),
            "tasks_scheduled": len(self.task_schedules),
            "tasks_remaining": sum(len(home.tasks) for home in self.instance.homes) - len(self.task_schedules)
        }

        # Add configuration info (consistent with trading environment)
        assert self.config is not None, "Config not available"
        context["max_iterations"] = self.config.get("max_iterations", 10)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            context[key] = value

        return context

    def should_continue_simulation(self, iteration: int) -> bool:
        """
        Check if the simulation should continue running.

        Args:
            iteration: Current iteration number

        Returns:
            True if simulation should continue, False to stop
        """
        # Check max iterations first (consistent with trading environment)
        assert self.config is not None, "Config not available"
        max_iterations = self.config.get("max_iterations", 10)
        if iteration > max_iterations:
            print(f"Reached max iterations ({max_iterations}) - stopping simulation")
            return False

        # Stop early if all tasks have been scheduled
        total_tasks = sum(len(home.tasks) for home in self.instance.homes)
        if len(self.task_schedules) == total_tasks and self.environment:
            result = self.environment.evaluate(self.task_schedules)
            if result["ok"]:
                main_grid_draw = result["MainGridEnergy"]
                print(f"All tasks scheduled with main grid draw: {main_grid_draw:.2f} kWh - simulation complete")
                return False
            else:
                print(f"All tasks scheduled but evaluation failed - continuing")

        return True


    def log_state(self, iteration: int, phase: str) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
            phase: Current phase
        """
        print(f"=== SmartGrid State - Iteration {iteration}, Phase {phase} ===")
        total_tasks = sum(len(home.tasks) for home in self.instance.homes)
        print(f"Homes: {len(self.agents_list)} total, {len(self.task_schedules)}/{total_tasks} tasks scheduled")

        if self.task_schedules:
            print("Current Task Schedules:")
            scheduled_by_home = {}
            for (home_id, task_id), start_time in self.task_schedules.items():
                if home_id not in scheduled_by_home:
                    scheduled_by_home[home_id] = []
                scheduled_by_home[home_id].append(f"{task_id}@{start_time}")

            for home_id, schedules in scheduled_by_home.items():
                print(f"  {home_id}: {', '.join(schedules)}")

        unscheduled_tasks = []
        for home in self.instance.homes:
            for task in home.tasks:
                if (home.id, task.id) not in self.task_schedules:
                    unscheduled_tasks.append(f"{home.id}:{task.id}")

        if unscheduled_tasks:
            print(f"Unscheduled tasks: {', '.join(unscheduled_tasks)}")

        # Calculate and display scores if any tasks are scheduled
        main_grid_draw = 0.0
        if self.task_schedules and self.environment:
            completion_rate = len(self.task_schedules) / total_tasks if total_tasks > 0 else 0
            print(f"Task completion: {len(self.task_schedules)}/{total_tasks} ({completion_rate:.1%})")

            result = self.environment.evaluate(self.task_schedules)
            if result["ok"]:
                main_grid_draw = result["MainGridEnergy"]
                demand_timeseries = self.environment.factor.demand_timeseries(self.task_schedules)
                grid_timeseries, _ = self.environment.factor.main_grid_draw(demand_timeseries)

                print(f"Current Main Grid Draw: {main_grid_draw:.2f} kWh")
                print(f"Peak Grid Draw: {max(grid_timeseries):.2f} kW")
            else:
                print(f"Cannot evaluate current schedules: {result.get('errors', 'Unknown error')}")

        # Always track scores and generate plots/logs for every iteration
        home_metrics = self._calculate_home_energy_consumption()
        self._track_scores(iteration, main_grid_draw, home_metrics)

    # TODO: Verify this calculation is correct and is meaningful
    def _calculate_home_energy_consumption(self) -> Dict[str, float]:
        """
        Calculate total energy consumption for each home based on their scheduled tasks.

        Returns:
            Dictionary mapping home_id -> total_energy_consumption
        """
        home_consumption = {home.id: 0.0 for home in self.instance.homes}

        for (home_id, task_id), start_time in self.task_schedules.items():
            # Find the task to get its consumption and duration
            home = next((h for h in self.instance.homes if h.id == home_id), None)
            if home:
                task = next((t for t in home.tasks if t.id == task_id), None)
                if task:
                    # Total energy = consumption (kW) × duration (hours)
                    total_energy = task.consumption * task.duration
                    home_consumption[home_id] += total_energy

        return home_consumption

    def _track_scores(self, iteration: int, global_score: float, local_scores: Dict[str, float]) -> None:
        """Track scores and generate plots/logs."""
        import json
        from datetime import datetime

        # Update score histories
        self.grid_draw_history.append(global_score)  # global_score is main grid draw
        for home_id, score in local_scores.items():
            if home_id in self.home_scores_history:
                self.home_scores_history[home_id].append(score)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir("SmartGrid", tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": "SmartGrid",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "main_grid_draw": global_score,  # Main grid energy consumption
            "home_metrics": local_scores,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "metadata": {
                "total_homes": len(local_scores),
                "total_tasks_scheduled": len(self.task_schedules),
                "time_horizon": self.instance.T if self.instance else 0
            }
        }

        score_file = log_dir / f"scores_iteration_{iteration}.json"
        with open(score_file, 'w') as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

        # Generate plot
        if self.score_plotter:
            try:
                plot_path = self.score_plotter.plot_scores(
                    self.grid_draw_history,
                    self.home_scores_history,
                    iteration,
                    environment_name="SmartGrid",
                    show=False
                )
            except Exception as e:
                print(f"Warning: Failed to generate score plot: {e}")

    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Extract serializable state for MCP transmission.

        Returns:
            Dictionary with serializable environment state
        """
        # Extract home and task specifications in serializable format
        homes_data = {}
        if hasattr(self, 'instance') and self.instance:
            for home in self.instance.homes:
                tasks_data = []
                for task in home.tasks:
                    tasks_data.append({
                        "id": task.id,
                        "consumption": task.consumption,
                        "duration": task.duration,
                        "allowed_starts": task.allowed_starts
                    })
                homes_data[home.id] = {
                    "tasks": tasks_data
                }

        # Convert tuple keys to string format for JSON serialization
        task_schedules_serializable = {
            f"{home_id}:{task_id}": start_time
            for (home_id, task_id), start_time in self.task_schedules.items()
        }

        return {
            "task_schedules": task_schedules_serializable,
            "agents_list": self.agents_list.copy(),
            "homes_data": homes_data,
            "T": self.instance.T if self.instance else 0
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Args:
            state_updates: Dictionary with state updates to apply
        """
        # Apply task_schedules updates (UPDATE, don't replace!)
        if "task_schedules" in state_updates:
            # Convert string keys "home_id:task_id" back to tuple keys (home_id, task_id)
            for key, value in state_updates["task_schedules"].items():
                home_id, task_id = key.split(":", 1)
                self.task_schedules[(home_id, task_id)] = value

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Post-tool execution callback for SmartGrid-specific processing.

        This is called after state updates are applied to perform environment-specific
        operations like score calculation.

        Args:
            state_updates: Dictionary with state updates that were applied
            response: The response dictionary to potentially modify
        """
        # Recalculate main grid draw after state updates if task schedules were updated
        if "task_schedules" in state_updates:
            if hasattr(self, 'environment') and self.environment and self.task_schedules:
                result = self.environment.evaluate(self.task_schedules)
                if result["ok"]:
                    main_grid_draw = result["MainGridEnergy"]
                    # Add main_grid_draw to result for agent feedback
                    if "result" in response:
                        response["result"]["main_grid_draw"] = main_grid_draw

    def _generate_final_summary(self):
        """Generate final simulation summary."""
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE - FINAL SUMMARY")
        print("=" * 60)

        # Get environment-specific final summary
        final_summary = self.get_final_summary()

        print(f"Total iterations: {self.current_iteration}")

        if final_summary:
            for key, value in final_summary.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")

    def _log_iteration_summary(self, iteration: int):
        """Log the summary of an iteration."""
        print(f"\n--- ITERATION {iteration} SUMMARY ---")
        self.log_state(iteration, "iteration_end")

        # Get environment-specific summary
        summary = self.get_iteration_summary()
        if summary:
            print(f"  Iteration {iteration} summary:")
            for key, value in summary.items():
                print(f"    {key}: {value}")

    def cleanup(self) -> None:
        """Clean up any resources used by the environment."""
        print("SmartGrid environment cleanup")
        if self.task_schedules:
            total_tasks = sum(len(home.tasks) for home in self.instance.homes) if self.instance else 0
            print(f"Final task schedules: {len(self.task_schedules)}/{total_tasks} tasks")

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current iteration for logging."""
        total_tasks = sum(len(home.tasks) for home in self.instance.homes) if self.instance else 0
        return {
            "tasks_scheduled": len(self.task_schedules),
            "total_tasks": total_tasks,
            "total_homes": len(self.agents_list),
            "completion_rate": len(self.task_schedules) / total_tasks if total_tasks > 0 else 0
        }

    def get_final_summary(self) -> Dict[str, Any]:
        """Get a final summary of the entire simulation."""
        total_tasks = sum(len(home.tasks) for home in self.instance.homes) if self.instance else 0
        if not self.instance or not self.environment or len(self.task_schedules) != total_tasks:
            return {
                "status": "incomplete",
                "tasks_scheduled": len(self.task_schedules),
                "total_tasks": total_tasks,
                "total_homes": len(self.agents_list)
            }

        # Calculate final scores
        result = self.environment.evaluate(self.task_schedules)
        if not result["ok"]:
            return {
                "status": "evaluation_failed",
                "tasks_scheduled": len(self.task_schedules),
                "total_tasks": total_tasks,
                "errors": result.get("errors", [])
            }

        main_grid_draw = result["MainGridEnergy"]
        demand_timeseries = self.environment.factor.demand_timeseries(self.task_schedules)
        grid_timeseries, _ = self.environment.factor.main_grid_draw(demand_timeseries)

        return {
            "status": "complete",
            "main_grid_draw": main_grid_draw,
            "peak_grid_draw": max(grid_timeseries),
            "avg_grid_draw": sum(grid_timeseries) / len(grid_timeseries),
            "task_schedules": {
                f"{home_id}:{task_id}": start_time
                for (home_id, task_id), start_time in self.task_schedules.items()
            },
            "total_tasks": total_tasks,
            "total_homes": len(self.agents_list)
        }
