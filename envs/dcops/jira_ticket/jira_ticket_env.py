import logging
import math
import random
from typing import Dict, List, Any, Optional, Mapping, Tuple

from envs.abstract_environment import AbstractEnvironment
from terrarium.utils import (
    clear_seed_directories,
    extract_model_info,
    get_tag_model_subdir,
    get_run_timestamp,
    build_log_dir,
)
from .jira_ticket_prompts import JiraTicketPrompts

logger = logging.getLogger(__name__)


class JiraTicketEnvironment(AbstractEnvironment):
    """JIRA ticket assignment DCOP environment."""

    DEFAULT_MICROTASK_TYPES = ["implement", "review", "test", "docs", "triage"]
    DEFAULT_SKILL_TAGS = ["backend", "frontend", "infrastructure", "machine-learning", "security", "data-science", "api-development", "ui-ux-design", "devops", "mobile-development", "testing", "documentation"]

    PRIORITY_WEIGHTS = {
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
        "critical": 1.0,
    }

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize synthetic JIRA task environment and compute cost tables."""
        self.full_config = config
        self.env_config = config["environment"]
        self.simulation_config = config["simulation"]
        self.current_seed = int(self.simulation_config.get("seed", 42))
        self.rng = random.Random(self.current_seed)

        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.run_timestamp = get_run_timestamp(self.full_config)

        # Clear seed directories to ensure clean state
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        # Build instance data
        network_cfg = config.get("communication_network") or {}
        assert network_cfg is not None and network_cfg != {}, "communication_network config must be specified"
        num_agents = network_cfg.get("num_agents")
        assert num_agents is not None and type(num_agents) == int, "communication_network.num_agents must be int"

        self.microtask_types = self._get_microtask_types()
        self.tasks = self._build_tasks(max_tasks=int(self.env_config.get("max_tasks", 20)))

        self.agent_names, self.agent_private = self._build_agents(num_agents)
        self.assignment: Dict[str, Optional[str]] = {}

        self.priority_bonus = max(0.0, float(self.env_config.get("priority_bonus", 20.0)))

        self.costs = self._compute_costs()
        self.tasks_done_bonus = max(0.0, float(self.env_config.get("tasks_done_bonus", 20.0)))
        self.violation_penalty = float(self.env_config.get("violation_penalty", 20.0))

        # Theoretical upper bound (no solver)
        self.optimal_k = min(len(self.agent_names), len(self.tasks))
        self.optimal_cost = 0.0
        self.max_joint_reward = self.compute_max_joint_reward()

        # Prompts
        self.prompts = JiraTicketPrompts(self, self.full_config)

        # Score tracking
        self.joint_reward_history: List[float] = []
        self.agent_rewards_history: Dict[str, List[float]] = {agent: [] for agent in self.agent_names}

        logger.info("%s initialized with %s agents", self.__class__.__name__, len(self.agent_names))
        logger.info("Total tasks: %s", len(self.tasks))
        logger.info("Microtask types: %s", ", ".join(self.microtask_types))

    def _get_microtask_types(self) -> List[str]:
        """Return ordered list of microtask types."""
        return list(self.DEFAULT_MICROTASK_TYPES)

    def _generate_agent_names(self, num_agents: int) -> List[str]:
        """Generate unique agent first names using the `names` package."""
        try:
            import names  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "JiraTicketEnvironment requires the `names` package for agent naming. "
                "Install it in the active environment (e.g. `pip install names`)."
            ) from exc

        name_pool: set[str] = set()
        for gender in ("male", "female"):
            filename = (names.FILES or {}).get(f"first:{gender}")
            if not filename:
                continue
            with open(filename) as name_file:
                for line in name_file:
                    parts = line.split()
                    if not parts:
                        continue
                    name_pool.add(parts[0].capitalize())

        if num_agents > len(name_pool):
            raise ValueError(
                f"num_agents={num_agents} exceeds available unique first names ({len(name_pool)})."
            )

        name_rng = random.Random(self.current_seed + 1337)
        return name_rng.sample(sorted(name_pool), k=num_agents)

    def _expand_microtasks(self, issues: List[Dict[str, Any]], max_tasks: int) -> Dict[str, Dict[str, Any]]:
        """Expand issues into microtasks with derived effort/metadata."""
        task_map: Dict[str, Dict[str, Any]] = {}
        multipliers = {
            "implement": 1.0,
            "review": 0.5,
            "test": 0.7,
            "docs": 0.5,
            "triage": 0.4,
        }
        for issue in issues:
            for micro in self.microtask_types:
                task_id = f"{issue['issue_id']}::{micro}"
                if task_id in task_map:
                    continue
                effort = issue["effort"] * multipliers.get(micro, 0.6)
                task_map[task_id] = {
                    "id": task_id,
                    "issue_id": issue["issue_id"],
                    "title": f"{issue['summary']} [{micro}]",
                    "tags": list(issue["tags"]),
                    "priority": issue["priority"],
                    "effort": max(1.0, effort),
                    "work_type": micro,
                }
                if len(task_map) >= max_tasks:
                    return task_map
        return task_map

    def _build_tasks(self, max_tasks: int) -> Dict[str, Dict[str, Any]]:
        """Generate synthetic issues then expand into microtasks."""
        return self._expand_microtasks(self._generate_synthetic_issues(max_tasks), max_tasks=max_tasks)

    def _generate_synthetic_issues(self, max_tasks: int) -> List[Dict[str, Any]]:
        """Create a synthetic pool of base issues with tags/effort/priority."""
        tag_pool = list(self.DEFAULT_SKILL_TAGS)
        priority_pool = list(self.PRIORITY_WEIGHTS.keys())

        issues: List[Dict[str, Any]] = []
        issue_count = max(1, int(math.ceil(max_tasks / max(1, len(self.microtask_types)))))
        for idx in range(issue_count):
            issue_id = f"ISSUE-{idx + 1:04d}"
            max_tags = min(2, len(tag_pool))
            tag_count = self.rng.randint(1, max_tags) if max_tags > 0 else 0
            tags = self.rng.sample(tag_pool, k=tag_count)
            priority_label = self.rng.choice(priority_pool)
            effort = float(self.rng.randint(2, 8))
            issues.append({
                "issue_id": issue_id,
                "summary": f"{self.rng.choice(['Build', 'Fix', 'Improve'])} {self.rng.choice(tags)}",
                "tags": tags,
                "priority": priority_label,
                "effort": effort,
            })
        return issues

    def _build_agents(self, num_agents: int) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Sample synthetic agent skill/availability profiles."""
        tag_pool = sorted({tag for task in self.tasks.values() for tag in task.get("tags", [])})

        availability_range = self.env_config.get("availability_range", [4, 10])
        if isinstance(availability_range, (list, tuple)) and len(availability_range) >= 2:
            min_avail, max_avail = availability_range[0], availability_range[1]
        elif isinstance(availability_range, (int, float)):
            min_avail, max_avail = availability_range, availability_range
        else:
            min_avail, max_avail = 4, 10

        agent_names = self._generate_agent_names(num_agents)
        agent_private: Dict[str, Dict[str, Any]] = {}

        for agent in agent_names:
            max_primary = min(2, len(tag_pool))
            primary_count = self.rng.randint(1, max_primary) if max_primary > 0 else 0
            primary_tags = self.rng.sample(tag_pool, k=primary_count) if tag_pool else []
            skills = {tag: self.rng.uniform(0.6, 1.0) for tag in primary_tags}

            agent_private[agent] = {
                "availability": float(self.rng.randint(int(min_avail), int(max_avail))),
                "skills": skills,
            }

        return agent_names, agent_private

    def _compute_costs(self) -> Dict[str, Dict[str, float]]:
        """Compute private cost matrix using skills and availability."""
        weights = self.env_config.get("cost_weights", {})
        load_cost = float(weights.get("load", 1.0))
        eps = float(self.env_config.get("skill_eps", 0.1))

        costs: Dict[str, Dict[str, float]] = {agent: {} for agent in self.agent_names}

        for agent in self.agent_names:
            private = self.agent_private[agent]
            skills = private["skills"]
            availability = float(private["availability"])

            for task_id, task in self.tasks.items():
                tags = task.get("tags", [])
                if tags:
                    match = sum(skills.get(tag, 0.0) for tag in tags) / max(1, len(tags))
                else:
                    match = 0.0

                effort = float(task.get("effort", 1.0))

                skill_adjusted = effort / max(eps, match + eps)
                overload = load_cost * max(0.0, effort - availability)

                cost = skill_adjusted + overload
                if cost < 0:
                    cost = 0.0
                costs[agent][task_id] = float(cost)

        return costs

    def _priority_weight(self, priority: Any) -> float:
        """Map a priority label to a numeric weight in [0, 1]."""
        label = str(priority or "medium").strip().lower()
        return float(self.PRIORITY_WEIGHTS.get(label, self.PRIORITY_WEIGHTS["medium"]))

    def _evaluate_assignment(self, actions: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate tasks_done, total_cost, violations, and lexicographic score."""
        seen_tasks: set[str] = set()
        violations = 0
        total_cost = 0.0
        tasks_done = 0
        priority_sum = 0.0

        for agent, task in actions.items():
            if task in (None, "skip"):
                continue

            task_key = str(task)
            cost = self.costs.get(agent, {}).get(task_key, float("inf"))
            if not math.isfinite(cost):
                violations += 1
                continue
            priority_sum += self._priority_weight(self.tasks.get(task_key, {}).get("priority"))

            if task_key in seen_tasks:
                violations += 1
            else:
                seen_tasks.add(task_key)

            tasks_done += 1
            total_cost += cost

        score = (
            self.tasks_done_bonus * tasks_done
            + self.priority_bonus * priority_sum
            - total_cost
            - self.violation_penalty * violations
        )
        return {
            "score": score,
            "tasks_done": tasks_done,
            "priority_sum": priority_sum,
            "total_cost": total_cost,
            "violations": violations,
        }

    def _normalized_metrics(self, eval_result: Mapping[str, Any]) -> Tuple[float, float, float]:
        """Compute coverage, cost gap, and normalized score against the upper bound."""
        if not self.optimal_k or eval_result.get("violations", 0) > 0:
            return 0.0, 0.0, 0.0

        coverage = float(eval_result["tasks_done"]) / float(self.optimal_k)
        cost_gap = float(eval_result["total_cost"]) - float(self.optimal_cost)
        tau = float(self.env_config.get("norm_temperature", max(1.0, self.optimal_cost or 1.0)))
        norm_score = coverage * math.exp(-max(0.0, cost_gap) / tau)
        return coverage, cost_gap, norm_score

    def _rewards(self, actions: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute joint score and per-agent contributions (additive only)."""
        eval_result = self._evaluate_assignment(actions)
        total_score = float(eval_result["score"])

        # Additive contribution
        local_rewards: Dict[str, float] = {agent: 0.0 for agent in self.agent_names}
        task_groups: Dict[str, List[str]] = {}
        for agent, task in actions.items():
            if task in (None, "skip"):
                continue
            task_groups.setdefault(str(task), []).append(agent)

        for agent, task in actions.items():
            if task in (None, "skip"):
                continue
            task_key = str(task)
            cost = self.costs.get(agent, {}).get(task_key, float("inf"))
            if not math.isfinite(cost):
                local_rewards[agent] -= self.violation_penalty
                continue
            prio = self._priority_weight(self.tasks.get(task_key, {}).get("priority"))
            local_rewards[agent] += self.tasks_done_bonus + self.priority_bonus * prio - cost

        for agents in task_groups.values():
            if len(agents) <= 1:
                continue
            penalty_total = self.violation_penalty * (len(agents) - 1)
            penalty_share = penalty_total / len(agents)
            for agent in agents:
                local_rewards[agent] -= penalty_share

        return total_score, local_rewards

    def get_network_context(self) -> str:
        """Return base context string shown on each blackboard channel."""
        base_context = (
            "JIRA sprint task allocation. Each agent chooses at most one micro-task "
            "(or skips). Maximize tasks completed, prefer higher-priority tasks, then minimize total cost. "
            "Avoid duplicate task selections and infeasible assignments."
        )
        if self.assignment_filling_enabled():
            return f"{base_context}\n\n{self.UNASSIGNED_VARIABLE_FALLBACK_NOTE}"
        return base_context

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        """Assemble per-agent prompt context for the current phase/iteration."""
        private = self.agent_private.get(agent_name, {})
        costs = self.costs.get(agent_name, {})
        task_ids = list(self.tasks)
        cost_table = [
            {"task_id": task_id, "cost": (c := costs.get(task_id, float("inf"))), "feasible": math.isfinite(c)}
            for task_id in task_ids
        ]
        cost_table.sort(key=lambda item: (not item["feasible"], item["cost"]))

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "joint_assignment": self.assignment.copy(),
            "tasks": [self.tasks[task_id] for task_id in task_ids],
            "cost_table": cost_table,
            "private_state": {
                "availability": private.get("availability"),
                "top_skills": sorted(
                    private.get("skills", {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5],
            },
        }

        context.update(kwargs)

        return context

    def done(self, iteration: int) -> bool:
        """Stop when max iterations reached or all agents have decided."""
        max_iterations = self.env_config.get("max_iterations", 1)
        if iteration > max_iterations:
            logger.info("Reached max iterations (%s) - stopping simulation", max_iterations)
            return True

        return len(self.assignment) >= len(self.agent_names)

    def compute_max_joint_reward(self) -> float:
        """Return a theoretical upper bound of the joint reward (no solver)."""
        max_priority = max(self.PRIORITY_WEIGHTS.values(), default=1.0)
        return float((self.tasks_done_bonus + self.priority_bonus * max_priority) * self.optimal_k - self.optimal_cost)

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        """Return joint score for a given partial/complete assignment."""
        return float(self._evaluate_assignment(actions)["score"])

    def agent_reward(self, actions: Mapping[str, Any], agent: str) -> float:
        """Return the reward attributed to a single agent."""
        _, local_rewards = self._rewards(actions)
        return float(local_rewards.get(agent, 0.0))

    def log_iteration(self, iteration: int) -> None:
        """Log current state and scores for this iteration."""
        logger.info("=== %s State - Iteration %s ===", self.__class__.__name__, iteration)
        logger.info("Agents assigned: %s/%s", len(self.assignment), len(self.agent_names))

        if self.assignment:
            logger.info("Current Assignments:")
            for agent, task in sorted(self.assignment.items()):
                if task in (None, "skip"):
                    logger.info("  %s -> skip", agent)
                    continue

                task_id = str(task)
                task_meta = self.tasks.get(task_id)
                if not task_meta:
                    logger.info("  %s -> %s (unknown task)", agent, task_id)
                    continue

                task_tags = [str(tag) for tag in (task_meta.get("tags") or [])]
                skills = (self.agent_private.get(agent, {}) or {}).get("skills") or {}
                top_skills = sorted(skills.items(), key=lambda item: item[1], reverse=True)[:5]
                top_skills_str = ", ".join(f"{tag}:{score:.2f}" for tag, score in top_skills) or "-"
                tag_skill_str = ", ".join(f"{tag}:{float(skills.get(tag, 0.0)):.2f}" for tag in task_tags) or "-"
                tags_str = ", ".join(task_tags) or "-"
                logger.info(
                    "  %s -> %s | task_tags=[%s] | agent_top_skills=[%s] | agent_skill_on_task_tags=[%s]",
                    agent,
                    task_id,
                    tags_str,
                    top_skills_str,
                    tag_skill_str,
                )

        eval_result = self._evaluate_assignment(self.assignment)
        joint_reward, agent_rewards = self._rewards(self.assignment)
        ratio = joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0
        logger.info(
            "Joint Score: %.2f (ratio %.2f%%) | tasks_done=%s prio_sum=%.2f cost=%.2f violations=%s",
            joint_reward,
            ratio * 100.0,
            eval_result["tasks_done"],
            eval_result.get("priority_sum", 0.0),
            eval_result["total_cost"],
            eval_result["violations"],
        )

        self._track_scores(iteration, eval_result, agent_rewards)

    def _track_scores(self, iteration: int, eval_result: Dict[str, Any], agent_rewards: Dict[str, float]) -> None:
        """Persist per-iteration metrics to disk and in-memory histories."""
        import json
        from datetime import datetime

        self.joint_reward_history.append(float(eval_result["score"]))
        for agent, reward in agent_rewards.items():
            if agent in self.agent_rewards_history:
                self.agent_rewards_history[agent].append(float(reward))

        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        coverage, cost_gap, norm_score = self._normalized_metrics(eval_result)

        score_entry = {
            "environment": self.__class__.__name__,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "joint_reward": eval_result["score"],
            "joint_reward_ratio": eval_result["score"] / self.max_joint_reward if self.max_joint_reward else 0.0,
            "tasks_done": eval_result["tasks_done"],
            "priority_sum": eval_result.get("priority_sum", 0.0),
            "total_cost": eval_result["total_cost"],
            "violations": eval_result["violations"],
            "coverage": coverage,
            "cost_gap": cost_gap,
            "normalized_score": norm_score,
            "optimal_k": self.optimal_k,
            "optimal_cost": self.optimal_cost,
            "agent_rewards": agent_rewards,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0.0,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "total_agents": len(agent_rewards),
            "tasks_available": len(self.tasks),
            "decisions_made": len(self.assignment),
        }

        data_file = log_dir / f"data_iteration_{iteration}.json"
        with open(data_file, "w") as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

    def _fill_random_unassigned_agents(self) -> int:
        """Assign uniform-random tasks (or skip) to any agents that never acted."""
        seed = int(getattr(self, "current_seed", 0))
        rng = getattr(self, "rng", None)
        if not isinstance(rng, random.Random):
            rng = random.Random(seed + 99173)

        task_ids = list(self.tasks)
        allowed: List[Optional[str]] = [None] + [str(t) for t in task_ids]
        if not allowed:
            allowed = [None]

        filled = 0
        for agent in self.agent_names:
            if agent not in self.assignment:
                self.assignment[agent] = rng.choice(allowed)
                filled += 1
                continue

            value = self.assignment.get(agent)
            if value == "skip":
                self.assignment[agent] = None
                continue
            if value is not None and str(value) not in self.tasks:
                self.assignment[agent] = rng.choice(allowed)
                filled += 1

        return filled

    def get_final_summary(self) -> Dict[str, Any]:
        """Return run-level summary including coverage and normalized score."""
        assignment_filling = self.assignment_filling_enabled()
        random_fallback_fills = self._fill_random_unassigned_agents() if assignment_filling else 0
        notes = [self.UNASSIGNED_VARIABLE_FALLBACK_NOTE] if assignment_filling else []

        if len(self.assignment) < len(self.agent_names):
            return {
                "status": "incomplete",
                "decisions_made": len(self.assignment),
                "total_agents": len(self.agent_names),
                "tasks_available": len(self.tasks),
                "random_fallback_fills": random_fallback_fills,
                "notes": notes,
            }

        eval_result = self._evaluate_assignment(self.assignment)
        joint_reward, agent_rewards = self._rewards(self.assignment)
        coverage, cost_gap, norm_score = self._normalized_metrics(eval_result)

        return {
            "status": "complete",
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward if self.max_joint_reward else 0.0,
            "tasks_done": eval_result["tasks_done"],
            "priority_sum": eval_result.get("priority_sum", 0.0),
            "total_cost": eval_result["total_cost"],
            "violations": eval_result["violations"],
            "coverage": coverage,
            "cost_gap": cost_gap,
            "normalized_score": norm_score,
            "optimal_k": self.optimal_k,
            "optimal_cost": self.optimal_cost,
            "agent_rewards": agent_rewards,
            "assignment": self.assignment.copy(),
            "tasks_available": len(self.tasks),
            "total_agents": len(self.agent_names),
            "random_fallback_fills": random_fallback_fills,
            "notes": notes,
        }

    #### MCP-specific methods ####

    def get_serializable_state(self) -> Dict[str, Any]:
        """Expose minimal serializable state for MCP tool calls."""
        tasks_serializable = {
            task_id: {
                "id": task.get("id", task_id),
                "title": task.get("title"),
                "tags": task.get("tags", []),
                "priority": task.get("priority"),
                "effort": task.get("effort"),
                "work_type": task.get("work_type"),
            }
            for task_id, task in self.tasks.items()
        }

        return {
            "tasks": tasks_serializable,
            "assignment": self.assignment.copy(),
            "agent_names": self.agent_names.copy(),
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """Apply assignment updates coming back from MCP tool calls."""
        if "assignment" in state_updates:
            self.assignment.update(state_updates["assignment"])

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Attach immediate joint reward feedback after a tool call."""
        if "assignment" in state_updates:
            joint_reward = self.joint_reward(self.assignment)
            if "result" in response:
                response["result"]["joint_reward"] = joint_reward
