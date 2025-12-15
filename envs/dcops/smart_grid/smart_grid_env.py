"""
SmartGrid Environment Adaptor (CoLLAB v2)

Adaptor to integrate the SmartGrid domain (shared renewable source allocation)
with the black_boards_v5 communication protocol framework.

Agents coordinate to assign their machines to shared renewable sources over a
timeline, minimising overflow to the main grid.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Mapping

import logging

if TYPE_CHECKING:
    from src.agents.base import BaseAgent

from problem_layer.smart_grid import SmartGridConfig, generate_instance
from problem_layer.base import ProblemDefinition

from envs.abstract_environment import AbstractEnvironment
from src.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
)

from .smartgrid_prompts import SmartGridPrompts

logger = logging.getLogger(__name__)


class SmartGridEnvironment(AbstractEnvironment):
    """
    SmartGrid environment adaptor for CoLLAB v2 renewable source allocation.
    """

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]

        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.run_timestamp = get_run_timestamp(self.full_config)
        self.current_iteration = 0
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)

        self.current_seed = int(self.simulation_config["seed"])

        # Partial joint assignment: machine_id -> source_id
        self.assignment: Dict[str, Any] = {}
        self.action_logging_config = ["assign_source"]

        # Clear seed directories FIRST to ensure clean state for this run
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        # ---- Build CoLLAB v2 instance ---------------------------------------------
        network_cfg = config.get("communication_network") or {}
        assert network_cfg is not None and network_cfg != {}, "communication_network config must be specified"
        num_agents = network_cfg.get("num_agents")
        assert num_agents is not None and type(num_agents) == int, "communication_network.num_agents in config must be specified as an integer"

        timeline_length = self.env_config.get("timeline_length", self.env_config.get("T", 24))

        tasks_per_home = self.env_config.get("tasks_per_home", (3, 6))
        try:
            min_tasks, max_tasks = tasks_per_home
        except Exception:
            min_tasks, max_tasks = 3, 6

        collab_cfg = SmartGridConfig(
            num_agents=int(num_agents),
            timeline_length=int(timeline_length),
            min_machines_per_agent=int(self.env_config.get("min_machines_per_agent", min_tasks)),
            max_machines_per_agent=int(self.env_config.get("max_machines_per_agent", max_tasks)),
            min_sources_per_agent=int(self.env_config.get("min_sources_per_agent", 2)),
            max_sources_per_agent=int(self.env_config.get("max_sources_per_agent", 3)),
            rng_seed=int(self.current_seed),
        )

        dcops_root = Path(__file__).resolve().parents[1]
        instance_dir = (
            dcops_root
            / "outputs"
            / "collab_instances"
            / "smart_grid"
            / f"seed_{self.current_seed}"
        )
        self.instance = generate_instance(collab_cfg, instance_dir)
        self.problem: ProblemDefinition = self.instance.problem

        self.agent_names: List[str] = list(self.problem.agents.keys())
        self.max_joint_reward = self.compute_max_joint_reward()
        self.min_possible_score = float(getattr(self.instance, "min_utility", 0.0))

        # Score tracking
        self.joint_reward_history: List[float] = []
        self.agent_rewards_history: Dict[str, List[float]] = {a: [] for a in self.agent_names}

        # Prompts (tools are handled by MCP server)
        self.prompts = SmartGridPrompts(self, self.full_config)

        logger.info("%s initialized with %s agents", self.__class__.__name__, len(self.agent_names))
        logger.info("Agent Names: %s", ", ".join(self.agent_names))
        logger.info("Total machines: %s", len(self.problem.variables))
        logger.info("Timeline length: %s slots", self.instance.timeline_length)

    async def async_init(self):
        await super().async_init()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        if phase == "planning" and iteration > 1 and self.assignment:
            logger.info("%s: Clearing assignments for iteration %s", self.__class__.__name__, iteration)
            self.assignment = {}

        agent_vars = [var.name for var in self.problem.agent_variables(agent_name)]
        agent_choices = {v: self.assignment[v] for v in agent_vars if v in self.assignment}

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "joint_assignment": self.assignment.copy(),
            "agent_variables": agent_vars,
            "agent_choices": agent_choices,
            "timeline_length": self.instance.timeline_length,
            "total_variables": len(self.problem.variables),
            "variables_assigned": len(self.assignment),
        }

        context["max_iterations"] = self.env_config.get("max_iterations", 1)
        for key, value in kwargs.items():
            context[key] = value
        return context

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """Return the (partial) joint reward for a joint assignment."""
        total_reward, _ = self._rewards(actions)
        return total_reward

    def agent_reward(self, actions: Mapping[str, Any], agent: str) -> float:
        """Return the reward attributed to a single agent."""
        _, local_rewards = self._rewards(actions)
        assert agent in local_rewards, f"Agent {agent} not found in local rewards"
        local_reward = local_rewards.get(agent)
        assert local_reward is not None, f"Local reward for agent {agent} is None"
        return local_reward

    def _rewards(self, actions: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute joint reward and per-agent rewards for a given joint assignment."""
        total_reward = 0.0
        local_rewards: Dict[str, float] = {a: 0.0 for a in self.agent_names}

        for factor in self.problem.factors:
            if not all(v in actions for v in factor.scope):
                continue
            try:
                reward = factor.evaluate(actions)
            except Exception:
                continue
            total_reward += reward
            owners = {self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables}
            if owners:
                share = reward / len(owners)
                for owner in owners:
                    local_rewards[owner] += share

        return total_reward, local_rewards

    def done(self, iteration: int) -> bool:
        """Return True when the environment is finished."""
        assert self.env_config is not None, "Config not available"
        max_iterations = self.env_config.get("max_iterations", 1)
        if iteration > max_iterations:
            logger.info("Reached max iterations (%s) - stopping simulation", max_iterations)
            return True

        total_vars = len(self.problem.variables)
        if len(self.assignment) == total_vars:
            joint_reward = self.joint_reward(self.assignment)
            logger.info(
                "All machines assigned with joint reward: %.2f - simulation complete",
                joint_reward,
            )
            return True
        return False

    def compute_max_joint_reward(self) -> float:
        """Return the optimal joint reward for the environment."""
        return float(getattr(self.instance, "max_utility", 0.0))

    def log_iteration(self, iteration: int) -> None:
        logger.info("=== %s State - Iteration %s ===", self.__class__.__name__, iteration)
        total_vars = len(self.problem.variables)
        logger.info("Machines: %s total, %s assigned", total_vars, len(self.assignment))

        if self.assignment:
            logger.info("Current Assignments:")
            for machine_id, source_id in sorted(self.assignment.items()):
                logger.info("  %s -> %s", machine_id, source_id)

        joint_reward, agent_rewards = self._rewards(self.assignment)
        ratio = joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0
        logger.info("Current Joint Reward: %.2f (ratio %.2f%%)", joint_reward, ratio * 100.0)

        self._track_scores(iteration, joint_reward, agent_rewards)

    def _track_scores(self, iteration: int, joint_reward: float, agent_rewards: Dict[str, float]) -> None:
        import json
        from datetime import datetime

        self.joint_reward_history.append(joint_reward)
        for agent, reward in agent_rewards.items():
            self.agent_rewards_history.setdefault(agent, []).append(reward)

        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        score_entry = {
            "environment": self.__class__.__name__,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward,
            "max_joint_reward": self.max_joint_reward,
            "agent_rewards": agent_rewards,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards),
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "total_agents": len(agent_rewards),
            "variables_assigned": len(self.assignment),
            "total_variables": len(self.problem.variables),
        }

        data_file = log_dir / f"data_iteration_{iteration}.json"
        with open(data_file, "w") as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

    def get_final_summary(self) -> Dict[str, Any]:
        total_vars = len(self.problem.variables)
        final_assignments = f"{len(self.assignment)}/{total_vars} machines"
        if not self.instance or len(self.assignment) != total_vars:
            return {
                "status": "incomplete",
                "variables_assigned": len(self.assignment),
                "total_variables": total_vars,
                "total_agents": len(self.agent_names),
                "final_assignments": final_assignments,
            }

        joint_reward, agent_rewards = self._rewards(self.assignment)
        return {
            "status": "complete",
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0.0,
            "agent_rewards": agent_rewards,
            "assignment": self.assignment.copy(),
            "total_variables": total_vars,
            "variables_assigned": len(self.assignment),
            "total_agents": len(self.agent_names),
            "final_assignments": final_assignments,
        }

    #### MCP-specific methods ####

    def get_serializable_state(self) -> Dict[str, Any]:
        machines: Dict[str, Any] = {}
        for machine_id, machine in self.instance.machines.items():
            var_spec = self.problem.variables.get(machine_id)
            machines[machine_id] = {
                "owner": machine.owner,
                "label": machine.label,
                "start": machine.start,
                "end": machine.end,
                "power": float(self.instance.machine_powers.get(machine_id, 0.0)),
                "allowed_sources": list(var_spec.domain) if var_spec else [],
            }

        sources: Dict[str, Any] = {}
        for source_id, source in self.instance.sources.items():
            sources[source_id] = {
                "kind": source.kind,
                "capacity": list(source.capacity),
                "clients": list(source.clients),
            }

        return {
            "machines": machines,
            "sources": sources,
            "assignment": self.assignment.copy(),
            "agent_names": self.agent_names.copy(),
            "timeline_length": self.instance.timeline_length,
            "max_joint_reward": self.max_joint_reward,
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        if "assignment" in state_updates:
            self.assignment.update(state_updates["assignment"])

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        if "assignment" in state_updates:
            joint_reward = self.joint_reward(self.assignment)
            if "result" in response:
                response["result"]["joint_reward"] = joint_reward
