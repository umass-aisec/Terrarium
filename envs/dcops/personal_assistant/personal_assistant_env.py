# pyright: basic
"""
PersonalAssistant Environment Adaptor

Adaptor to integrate the PersonalAssistant domain (outfit coordination)
with the black_boards_v5 communication protocol framework.

The PersonalAssistant environment involves agents coordinating to select outfits
that satisfy personal preferences and inter-agent constraints (color matching).
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Mapping

# CoLLAB v2 problem-layer imports (made available via envs.dcops.__init__)
from problem_layer.personal_assistant import PersonalAssistantConfig, generate_instance
from problem_layer.personal_assistant.problem import Outfit
from problem_layer.base import ProblemDefinition
import logging
# Import abstract environment interface and logger
from envs.abstract_environment import AbstractEnvironment
from src.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
)
from .personal_assistant_tools import PersonalAssistantTools
from .personal_assistant_prompts import PersonalAssistantPrompts

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular import (BaseAgent → ToolsetDiscovery → EnvironmentTools → Environment → BaseAgent)
if TYPE_CHECKING:
    from src.agents.base import BaseAgent


class PersonalAssistantEnvironment(AbstractEnvironment):
    """
    PersonalAssistant environment adaptor for outfit coordination tasks.

    Agents coordinate to select outfits that satisfy personal preferences
    and inter-agent constraints (color matching/avoiding).
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize the PersonalAssistant environment."""
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config = config["simulation"]
        # Get the correct seed from simulation config
        self.current_seed = int(self.simulation_config["seed"])

        # Instance management
        # Partial joint assignment: variable_name -> chosen outfit number (1-based)
        self.assignment: Dict[str, Any] = {}
        self.outfit_selections: Dict[str, Outfit] = {}
        self.tool_logger = tool_logger
        self.agent_names: List[str] = []  # Agent names (renamed from agents_list)
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.current_iteration = 0
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)

        # Clear seed directories FIRST to ensure clean state for this run
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        # ---- Build CoLLAB v2 instance -------------------------------------------------
        network_cfg = config.get("communication_network") or {}
        assert network_cfg is not None and network_cfg != {}, "communication_network config must be specified"
        num_agents = network_cfg.get("num_agents")
        assert num_agents is not None and type(num_agents) == int, "communication_network.num_agents in config must be specified as an integer"

        min_outfits = self.env_config.get("min_outfits_per_agent", 4)
        max_outfits = self.env_config.get("max_outfits_per_agent", 6)
        density = self.env_config.get("density")
        if density is None:
            # Back-compat: approximate density from max_degree if present
            max_degree = self.env_config.get("max_degree", 3)
            try:
                density = float(max_degree) / max(1.0, float(num_agents) - 1.0)
            except Exception:
                density = 0.3
        density = max(0.0, min(1.0, float(density)))

        collab_cfg = PersonalAssistantConfig(
            num_agents=int(num_agents),
            density=float(density),
            min_outfits_per_agent=int(min_outfits),
            max_outfits_per_agent=int(max_outfits),
            rng_seed=int(self.current_seed),
        )

        dcops_root = Path(__file__).resolve().parents[1]
        instance_dir = (
            dcops_root
            / "outputs"
            / "collab_instances"
            / "personal_assistant"
            / f"seed_{self.current_seed}"
        )
        self.instance = generate_instance(collab_cfg, instance_dir)
        self.problem: ProblemDefinition = self.instance.problem

        # Score tracking
        self.joint_reward_history: List[float] = []
        self.agent_names = list(self.problem.agents.keys())
        self.agents: List["BaseAgent"] = []  # Set this later in main.py in case agents get different clients or settings
        self.max_joint_reward = self.compute_max_joint_reward()

        # Initialize prompts (Put this after all other instance variables)
        # Note: tools are now in MCP server, not in environment
        self.prompts = PersonalAssistantPrompts(self, self.full_config)

        # Initialize score tracking
        self.agent_rewards_history: Dict[str, List[float]] = {agent: [] for agent in self.agent_names}

        logger.info("%s initialized with %s agents", self.__class__.__name__, len(self.agent_names))
        logger.info("Agent Names: %s", ", ".join(self.agent_names))

    async def async_init(self):
        """Async initialization - create communication blackboards from the supplied network."""
        await super().async_init()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn. This is used in the planning and execution phases
        in CommunicationProtocol.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            **kwargs: Additional context

        Returns:
            Dictionary with agent context
        """
        # Clear outfit selections at the start of each new iteration's planning phase
        # to allow agents to make new choices
        if phase == "planning" and iteration > 1 and (self.outfit_selections or self.assignment):
            logger.info("%s: Clearing selections for iteration %s", self.__class__.__name__, iteration)
            self.outfit_selections = {}
            self.assignment = {}

        if not self.instance:
            return {"error": "Environment not initialized"}

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "outfit_selections": self.outfit_selections.copy(),
            "total_agents": len(self.agent_names),
            "selections_made": len(self.outfit_selections),
            "selections_remaining": len(self.agent_names) - len(self.outfit_selections)
        }

        # Add configuration info (consistent with trading environment)
        assert self.env_config is not None, "Config not available"
        context["max_iterations"] = self.env_config.get("max_iterations", 1)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            context[key] = value

        return context

    def done(self, iteration: int) -> bool:
        """Return True when the environment is finished."""
        # Check max iterations first (consistent with trading environment)
        assert self.env_config is not None, "Config not available"
        max_iterations = self.env_config.get("max_iterations", 1)
        if iteration > max_iterations:
            logger.info("Reached max iterations (%s) - stopping simulation", max_iterations)
            return True

        # Stop early if all variables assigned and max reward reached
        total_vars = len(self.problem.variables)
        if len(self.assignment) == total_vars and self.instance:
            joint_reward = self.joint_reward(self.assignment)
            if self.max_joint_reward and joint_reward >= self.max_joint_reward:
                logger.info(
                    "All constraints satisfied (score: %s/%s) - simulation complete",
                    joint_reward,
                    self.max_joint_reward,
                )
                return True
            logger.info(
                "All agents selected but constraints not fully satisfied (score: %s/%s) - continuing",
                joint_reward,
                self.max_joint_reward,
            )

        return False

    def compute_max_joint_reward(self) -> float:
        """Return the optimal joint reward for the environment."""
        return float(getattr(self.instance, "max_utility", 0.0))

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
        """
        Compute joint reward and per-agent rewards for a given joint assignment.

        Factors whose full scope has been assigned contribute to the total reward.
        Per-agent rewards are attributed evenly to variable owners in scope.
        """
        total_reward = 0.0
        local_rewards: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}

        for factor in self.problem.factors:
            if not all(var in actions for var in factor.scope):
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

    def log_iteration(self, iteration: int) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
        """
        logger.info("=== %s State - Iteration %s ===", self.__class__.__name__, iteration)
        logger.info(
            "Agents: %s total, %s selected outfits",
            len(self.agent_names),
            len(self.outfit_selections),
        )

        if self.outfit_selections:
            logger.info("Current Selections:")
            for agent, outfit in self.outfit_selections.items():
                logger.info("  %s: %s, %s", agent, outfit.article, outfit.color)

        remaining = [agent for agent in self.agent_names if agent not in self.outfit_selections]
        if remaining:
            logger.info("Remaining agents: %s", ", ".join(remaining))

        joint_reward, agent_rewards = self._rewards(self.assignment)
        ratio = joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0
        logger.info("Current Joint Reward: %.2f (ratio %.2f%%)", joint_reward, ratio * 100.0)

        # Track scores for every iteration
        self._track_scores(iteration, joint_reward, agent_rewards)

    def _track_scores(self, iteration: int, joint_reward: float, agent_rewards: Dict[str, float]) -> None:
        """Track scores and write logs."""
        import json
        from datetime import datetime

        # Update score histories
        self.joint_reward_history.append(joint_reward)
        for agent, reward in agent_rewards.items():
            if agent in self.agent_rewards_history:
                self.agent_rewards_history[agent].append(reward)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
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
        """Get a final summary of the entire simulation."""
        total_vars = len(self.problem.variables)
        final_selections = f"{len(self.outfit_selections)}/{len(self.agent_names)} agents"
        if not self.instance or len(self.assignment) != total_vars:
            return {
                "status": "incomplete",
                "variables_assigned": len(self.assignment),
                "total_variables": total_vars,
                "total_agents": len(self.agent_names),
                "final_selections": final_selections,
            }

        joint_reward, agent_rewards = self._rewards(self.assignment)

        return {
            "status": "complete",
            "joint_reward": joint_reward / self.max_joint_reward if self.max_joint_reward > 0 else 0.0,
            "raw_joint_reward": joint_reward,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0.0,
            "agent_rewards": agent_rewards,
            "outfit_selections": {
                agent: {"article": outfit.article, "color": outfit.color}
                for agent, outfit in self.outfit_selections.items()
            },
            "total_variables": total_vars,
            "variables_assigned": len(self.assignment),
            "total_agents": len(self.agent_names),
            "final_selections": final_selections,
        }

    #### MCP-specific methods ####

    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Extract serializable state for MCP transmission.

        Returns:
            Dictionary with serializable environment state
        """
        # Extract wardrobe options in serializable format
        wardrobe_options: Dict[str, List[Dict[str, str]]] = {}
        if self.instance and getattr(self.instance, "wardrobe", None):
            for agent_name, outfits in self.instance.wardrobe.items():
                wardrobe_options[agent_name] = [
                    {"article": outfit.article, "color": outfit.color}
                    for outfit in outfits
                ]

        # Extract factors in serializable format
        factors: List[Dict[str, Any]] = []
        for factor in self.problem.factors:
            owners = sorted({self.problem.variables[v].owner for v in factor.scope if v in self.problem.variables})
            factors.append({"name": factor.name, "type": factor.factor_type, "owners": owners})

        return {
            "outfit_selections": {
                agent: {"article": outfit.article, "color": outfit.color}
                for agent, outfit in self.outfit_selections.items()
            },
            "agent_names": self.agent_names.copy(),
            "wardrobe_options": wardrobe_options,
            "factors": factors,
            "max_joint_reward": self.max_joint_reward,
            "assignment": self.assignment.copy(),
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Args:
            state_updates: Dictionary with state updates to apply
        """
        if "outfit_selections" in state_updates:
            for agent, outfit_dict in state_updates["outfit_selections"].items():
                outfit = Outfit(
                    article=outfit_dict["article"],
                    color=outfit_dict["color"],
                    image=None,
                )
                self.outfit_selections[agent] = outfit

                # Update assignment using wardrobe index (1-based)
                options = self.instance.wardrobe.get(agent, []) if self.instance else []
                choice_num = None
                for idx, opt in enumerate(options, start=1):
                    if opt.article == outfit.article and opt.color == outfit.color:
                        choice_num = idx
                        break
                if choice_num is not None:
                    try:
                        var_name = self.problem.agent_variables(agent)[0].name
                        self.assignment[var_name] = choice_num
                    except Exception:
                        pass

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Post-tool execution callback for PersonalAssistant-specific processing.

        This is called after state updates are applied to perform environment-specific
        operations like score calculation.

        Args:
            state_updates: Dictionary with state updates that were applied
            response: The response dictionary to potentially modify
        """
        if "outfit_selections" in state_updates:
            joint_reward = self.joint_reward(self.assignment)
            if "result" in response:
                response["result"]["joint_reward"] = joint_reward
                response["result"]["max_joint_reward"] = self.max_joint_reward
