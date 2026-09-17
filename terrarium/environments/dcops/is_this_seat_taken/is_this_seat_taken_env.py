from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from terrarium.personas import MAX_LEVEL, NEUTRAL_LEVEL, PRESETS, Persona
from .is_this_seat_taken_prompts import IsThisSeatTakenPrompts

logger = logging.getLogger(__name__)

# Upper bound of base_tolerance. Traits are on a 0-1 scale, so tolerance is divided
# by this before it is compared with a neighbour's traits.
MAX_BASE_TOLERANCE = 2.5


@dataclass
class Seat:
    seat_id: str
    row: int
    col: int
    role: str
    neighbors: List[str] = field(default_factory=list)
    occupied_by: Optional[str] = None


class IsThisSeatTakenEnvironment(AbstractEnvironment):
    """Seat-selection MAS environment inspired by social seating problems."""

    DEFAULT_SCENARIO_COLS = {
        "bus": 2,
        "cinema": 5,
        "wedding": 4,
        "taxi": 3,
        "airplane": 6,
        "default": 4,
    }

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 4))

        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        # Clear seed directories FIRST to ensure clean state for this run.
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")

        self.num_agents = int(num_agents)
        self.scenario_type = str(
            self.env_config.get("scenario_type", self.env_config.get("layout_type", "cinema"))
        ).strip().lower() or "cinema"
        self.group_move_penalty = float(self.env_config.get("group_move_penalty", 0.25))
        self.move_cost = float(self.env_config.get("move_cost", -0.5))
        self.social_action_cost = float(self.env_config.get("social_action_cost", -0.3))
        self.request_pressure = float(self.env_config.get("request_pressure", 0.4))
        self.complaint_pressure = float(self.env_config.get("complaint_pressure", 0.9))

        self.time_step = 0
        self.total_moves = 0

        self.agent_names = [f"agent_{idx}" for idx in range(self.num_agents)]
        self.persona_by_agent = self._resolve_personas()
        self.seats = self._build_layout()
        self.agent_state = self._build_agent_state()
        self._assign_initial_seats()
        self._refresh_rewards(force_reset=True)

        self.prompts = IsThisSeatTakenPrompts(self, self.full_config)

        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()
        self.reward_convergence_threshold = float(self.env_config.get("reward_convergence_threshold", 0.05))
        self.convergence_window = int(self.env_config.get("convergence_window", 3))

        logger.info(
            "%s initialized with %s agents, scenario=%s, seats=%s",
            self.__class__.__name__,
            len(self.agent_names),
            self.scenario_type,
            len(self.seats),
        )

    def _resolve_personas(self) -> Dict[str, Optional[Persona]]:
        """Resolve environment.persona (uniform) or environment.personas (per-agent)
        into a Persona per agent. Shared by prompt-shaping and trait generation so
        a persona's Extraversion also biases that agent's own generated traits.
        """
        persona_name = self.env_config.get("persona")
        persona_map = self.env_config.get("personas")

        def _resolve(name: str) -> Persona:
            if name not in PRESETS:
                raise ValueError(
                    f"Unknown persona '{name}'. Available presets: {sorted(PRESETS)}"
                )
            return PRESETS[name]

        if persona_map:
            missing = [agent for agent in self.agent_names if agent not in persona_map]
            if missing:
                raise ValueError(
                    f"environment.personas is missing entries for agents: {missing}"
                )
            return {agent: _resolve(name) for agent, name in persona_map.items()}
        if persona_name:
            persona = _resolve(persona_name)
            return {agent: persona for agent in self.agent_names}
        return {agent: None for agent in self.agent_names}

    def get_network_context(self) -> str:
        return (
            f"Scenario type: {self.scenario_type}. "
            "Agents are trying to find stable seats that satisfy personal and social constraints. "
            "Seats may be empty or occupied; adjacent occupancies can reveal public traits."
        )
    
    async def async_init(self):
        await super().async_init()
        # The starting layout is recorded for the GUI and analysis rather than posted
        # to agent channels, so agents learn positions only from their own view, their
        # neighbours, and moves announced in the chat.
        self._write_initial_seating()

    def _initial_seating(self) -> Dict[str, str]:
        return {
            seat.occupied_by: seat_id
            for seat_id, seat in self.seats.items()
            if seat.occupied_by is not None
        }

    def _write_initial_seating(self) -> None:
        log_dir = getattr(self.tool_logger, "log_dir", None)
        if log_dir is None:
            return
        path = Path(log_dir) / "initial_seating.json"
        path.write_text(json.dumps(self._initial_seating(), indent=2), encoding="utf-8")


    def _layout_dimensions(self) -> Tuple[int, int]:
        cols = int(
            self.env_config.get(
                "cols",
                self.DEFAULT_SCENARIO_COLS.get(self.scenario_type, self.DEFAULT_SCENARIO_COLS["default"]),
            )
        )
        cols = max(1, cols)
        buffer = int(self.env_config.get("empty_seat_buffer", max(2, math.ceil(self.num_agents * 0.35))))
        seat_count = max(self.num_agents + buffer, self.num_agents)
        rows = int(self.env_config.get("rows", math.ceil(seat_count / cols)))
        rows = max(1, rows)
        return rows, cols

    def _seat_role(self, row: int, col: int, rows: int, cols: int) -> str:
        if self.scenario_type == "bus":
            if col == 0:
                return "window"
            if col == cols - 1:
                return "aisle"
            return "middle"
        if self.scenario_type == "cinema":
            if col in {0, cols - 1}:
                return "window"
            if cols > 2 and col in {1, cols - 2}:
                return "aisle"
            if row == 0:
                return "front"
            if row == rows - 1:
                return "back"
            return "middle"
        if self.scenario_type == "wedding":
            if row in {0, rows - 1} or col in {0, cols - 1}:
                return "edge"
            return "table_center"
        if self.scenario_type == "taxi":
            if row == 0:
                return "front"
            if row == rows - 1:
                return "back"
            return "middle"
        if self.scenario_type == "airplane":
            half = cols // 2
            if col in {0, cols - 1}:
                return "window"
            if cols % 2 == 0:
                if col in {half - 1, half}:
                    return "aisle"
            elif col == half:
                return "aisle"
            return "middle"
        if col in {0, cols - 1}:
            return "window"
        return "middle"

    def _build_layout(self) -> Dict[str, Seat]:
        rows, cols = self._layout_dimensions()
        seats: Dict[str, Seat] = {}

        seat_limit = max(self.num_agents + 2, rows * cols)
        for idx in range(seat_limit):
            row = idx // cols
            col = idx % cols
            if row >= rows:
                break
            seat_id = f"seat_{row + 1}_{col + 1}"
            seats[seat_id] = Seat(
                seat_id=seat_id,
                row=row,
                col=col,
                role=self._seat_role(row, col, rows, cols),
            )

        positions = {(seat.row, seat.col): seat_id for seat_id, seat in seats.items()}
        for seat in seats.values():
            neighbors: List[str] = []
            for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_pos = (seat.row + d_row, seat.col + d_col)
                neighbor_id = positions.get(neighbor_pos)
                if neighbor_id is not None:
                    neighbors.append(neighbor_id)
            seat.neighbors = neighbors

        return seats

    def _generate_agent_public_traits(self, agent_name: str) -> Dict[str, float]:
        # Extraversion nudges loudness/talkativeness toward its pole (real personality
        # research ties both to it) without fully determining them — a random draw is
        # still shifted, not replaced. Scent is left alone: there's no personality basis
        # for tying body odor to a trait, and manufacturing one risks an ugly association.
        persona = self.persona_by_agent.get(agent_name)
        ext_level = persona.traits.get("EXT", NEUTRAL_LEVEL) if persona else NEUTRAL_LEVEL
        ext_bias = (ext_level - NEUTRAL_LEVEL) / (MAX_LEVEL - NEUTRAL_LEVEL)  # -1..1

        def _biased_trait() -> float:
            value = self.rng.uniform(0.0, 1.0) + ext_bias * 0.3
            return round(max(0.0, min(1.0, value)), 3)

        return {
            "loudness": _biased_trait(),
            "scent": round(self.rng.uniform(0.0, 1.0), 3),
            "talkativeness": _biased_trait(),
        }

    def _build_agent_state(self) -> Dict[str, Dict[str, Any]]:
        seat_roles = sorted({seat.role for seat in self.seats.values()})
        agent_state: Dict[str, Dict[str, Any]] = {}

        for agent in self.agent_names:
            preferred_roles = self.rng.sample(seat_roles, k=min(len(seat_roles), 1 + self.rng.randint(0, 1)))
            avoided_pool = [role for role in seat_roles if role not in preferred_roles]
            avoided_roles = self.rng.sample(avoided_pool, k=min(len(avoided_pool), 1)) if avoided_pool else []
            other_agents = [name for name in self.agent_names if name != agent]
            preferred_neighbors = (
                self.rng.sample(other_agents, k=1)
                if other_agents and self.rng.random() < 0.45
                else []
            )
            avoided_neighbor_pool = [name for name in other_agents if name not in preferred_neighbors]
            avoided_neighbors = (
                self.rng.sample(avoided_neighbor_pool, k=1)
                if avoided_neighbor_pool and self.rng.random() < 0.35
                else []
            )

            agent_state[agent] = {
                "preference_profile": {
                    "preferred_roles": preferred_roles,
                    "avoided_roles": avoided_roles,
                    "preferred_neighbors": preferred_neighbors,
                    "avoided_neighbors": avoided_neighbors,
                    "prefer_isolation": self.rng.random() < 0.3,
                },
                "current_seat": None,
                "satisfaction_score": 0.0,
                "last_instant_reward": 0.0,
                "base_tolerance": round(self.rng.uniform(1.0, MAX_BASE_TOLERANCE), 3),
                "settled": False,
                "social_pressure": 0.0,
                "pending_reaction": False,
                "pressure_requests": 0,
                "pressure_complaints": 0,
                "public_traits": self._generate_agent_public_traits(agent),
            }

        return agent_state

    def _assign_initial_seats(self) -> None:
        empty_seats = list(self.seats.keys())
        self.rng.shuffle(empty_seats)
        for agent, seat_id in zip(self.agent_names, empty_seats):
            self.seats[seat_id].occupied_by = agent
            self.agent_state[agent]["current_seat"] = seat_id

    def _occupied_neighbors(self, agent_name: str) -> List[Dict[str, Any]]:
        seat_id = self.agent_state[agent_name]["current_seat"]
        if seat_id is None:
            return []
        seat = self.seats[seat_id]
        observations: List[Dict[str, Any]] = []
        for neighbor_id in seat.neighbors:
            neighbor = self.seats[neighbor_id]
            if neighbor.occupied_by is None:
                continue
            neighbor_agent = neighbor.occupied_by
            observations.append(
                {
                    "seat_id": neighbor_id,
                    "agent_id": neighbor_agent,
                    "role": neighbor.role,
                    "public_traits": dict(self.agent_state[neighbor_agent]["public_traits"]),
                }
            )
        return observations

    def _agent_effective_tolerance(self, agent_name: str) -> float:
        state = self.agent_state[agent_name]
        pressure = float(state["social_pressure"])
        degraded = float(state["base_tolerance"])
        degraded -= pressure * 0.15
        degraded -= float(state.get("pressure_complaints", 0)) * 0.05
        degraded -= float(state.get("pressure_requests", 0)) * 0.02
        return max(0.0, degraded)

    def _compute_instant_reward(self, agent_name: str) -> float:
        state = self.agent_state[agent_name]
        seat_id = state["current_seat"]
        if seat_id is None:
            return 0.0

        seat = self.seats[seat_id]
        profile = state["preference_profile"]
        reward = 0.0
        penalties = 0.0

        if seat.role in profile.get("preferred_roles", []):
            reward += 1.0
        if seat.role in profile.get("avoided_roles", []):
            penalties += 1.0

        neighbors = self._occupied_neighbors(agent_name)
        neighbor_ids = {neighbor["agent_id"] for neighbor in neighbors}

        if profile.get("prefer_isolation"):
            if neighbors:
                penalties += 1.0
            else:
                reward += 1.0

        # Isolation takes precedence: an agent that wants empty seats around it gets no
        # credit for a preferred neighbour, which would otherwise always cancel out.
        if not profile.get("prefer_isolation"):
            for preferred_neighbor in profile.get("preferred_neighbors", []):
                if preferred_neighbor in neighbor_ids:
                    reward += 1.0
        for avoided_neighbor in profile.get("avoided_neighbors", []):
            if avoided_neighbor in neighbor_ids:
                penalties += 1.0

        # Without mapping onto the trait scale, no trait (max 1.0) could exceed a
        # tolerance (min 1.0) unless pressure had already worn it down.
        trait_cutoff = self._agent_effective_tolerance(agent_name) / MAX_BASE_TOLERANCE
        for neighbor in neighbors:
            traits = neighbor["public_traits"]
            if float(traits.get("loudness", 0.0)) > trait_cutoff:
                penalties += 1.0
            if float(traits.get("scent", 0.0)) > trait_cutoff:
                penalties += 1.0
            if float(traits.get("talkativeness", 0.0)) > trait_cutoff:
                penalties += 1.0

        return reward - penalties

    def _refresh_rewards(self, force_reset: bool = False) -> None:
        for agent in self.agent_names:
            instant_reward = self._compute_instant_reward(agent)
            state = self.agent_state[agent]
            previous = 0.0 if force_reset else float(state.get("last_instant_reward", 0.0))
            delta = instant_reward - previous
            state["satisfaction_score"] = float(state.get("satisfaction_score", 0.0)) + delta
            state["last_instant_reward"] = instant_reward

    def _apply_move_cost(self, agent_name: str) -> None:
        self.agent_state[agent_name]["satisfaction_score"] += self.move_cost

    def _apply_social_action_cost(self, agent_name: str) -> None:
        self.agent_state[agent_name]["satisfaction_score"] += self.social_action_cost

    def _current_agent_seat(self, agent_name: str) -> Optional[Seat]:
        seat_id = self.agent_state[agent_name]["current_seat"]
        return self.seats.get(seat_id) if seat_id else None

    def _is_neighbor(self, agent_name: str, target_agent: str) -> bool:
        seat = self._current_agent_seat(agent_name)
        if seat is None:
            return False
        for neighbor_id in seat.neighbors:
            neighbor = self.seats[neighbor_id]
            if neighbor.occupied_by == target_agent:
                return True
        return False

    def _move_agent(self, agent_name: str, seat_id: Optional[str]) -> Dict[str, Any]:
        state = self.agent_state[agent_name]
        current_seat_id = state["current_seat"]

        if seat_id is None:
            if current_seat_id is not None:
                self.seats[current_seat_id].occupied_by = None
            state["current_seat"] = None
            state["settled"] = False
            state["pending_reaction"] = False
            return {
                "status": "success",
                "result": {
                    "agent": agent_name,
                    "action": "stand",
                    "previous_seat": current_seat_id,
                    "current_seat": None,
                },
            }

        target_seat = self.seats.get(seat_id)
        if target_seat is None:
            return {"status": "retry", "reason": f"seat_id {seat_id} not found"}
        if target_seat.occupied_by is not None:
            self._apply_move_cost(agent_name)
            return {
                "status": "failed",
                "reason": f"seat {seat_id} is already occupied",
                "result": {"agent": agent_name, "attempted_seat": seat_id},
            }

        if current_seat_id == seat_id:
            return {"status": "retry", "reason": "Already in the requested seat"}

        if current_seat_id is not None:
            self.seats[current_seat_id].occupied_by = None

        target_seat.occupied_by = agent_name
        state["current_seat"] = seat_id
        state["settled"] = False
        state["pending_reaction"] = False
        self.total_moves += 1
        self._apply_move_cost(agent_name)

        return {
            "status": "success",
            "result": {
                "agent": agent_name,
                "action": "move",
                "from_seat": current_seat_id,
                "to_seat": seat_id,
            },
        }

    def _pressure_level(self, agent_name: str) -> str:
        """Qualitative bucket for social_pressure relative to the agent's own base
        tolerance — this is what gets surfaced to the agent (see build_agent_context),
        never the raw number. Replaces a mechanical forced-move trigger: the agent now
        decides for itself, via its own (persona-shaped) reasoning, how to respond.
        """
        state = self.agent_state[agent_name]
        pressure = float(state["social_pressure"])
        tolerance = max(0.1, float(state["base_tolerance"]))
        ratio = pressure / tolerance
        if ratio < 0.3:
            return "none"
        if ratio < 0.7:
            return "mild"
        if ratio < 1.1:
            return "building"
        return "high"

    def _request_neighbor_move(
        self,
        agent_name: str,
        target_agent: str,
        complaint: bool = False,
    ) -> Dict[str, Any]:
        if target_agent not in self.agent_names:
            return {"status": "failed", "reason": f"Agent {target_agent} not found"}
        if not self._is_neighbor(agent_name, target_agent):
            return {"status": "failed", "reason": f"{target_agent} is not a neighboring agent"}

        target_state = self.agent_state[target_agent]
        target_state["social_pressure"] += self.complaint_pressure if complaint else self.request_pressure
        target_state["pressure_complaints" if complaint else "pressure_requests"] += 1
        if complaint:
            target_state["pending_reaction"] = True

        self._apply_social_action_cost(agent_name)
        return {
            "status": "success",
            "result": {
                "requester": agent_name,
                "target": target_agent,
                "action": "complain" if complaint else "request_move",
                "target_pressure_level": self._pressure_level(target_agent),
            },
        }

    def apply_planning_social_action(
        self,
        agent_name: str,
        action_name: str,
        arguments: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Apply social pressure during planning. The target decides how (and whether)
        to react itself at execution time — nothing here forces a move; see
        _pressure_level for what gets surfaced to the target instead.
        """
        if action_name not in {"request_move", "complain"}:
            return {"error": f"Unknown planning social action: {action_name}"}

        if self.agent_state[agent_name]["settled"]:
            self.agent_state[agent_name]["settled"] = False

        target_agent = arguments.get("agent_id")
        if target_agent is None:
            return {"status": "retry", "reason": "agent_id is required"}

        return self._request_neighbor_move(
            agent_name,
            str(target_agent),
            complaint=action_name == "complain",
        )

    def _settle_agent(self, agent_name: str) -> Dict[str, Any]:
        state = self.agent_state[agent_name]
        if state["current_seat"] is None:
            return {"status": "retry", "reason": "Cannot settle while standing"}
        state["settled"] = True
        state["pending_reaction"] = False
        return {
            "status": "success",
            "result": {
                "agent": agent_name,
                "action": "settle",
                "current_seat": state["current_seat"],
                "settled": True,
            },
        }

    def _apply_agent_action(self, agent_name: str, action_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        if self.agent_state[agent_name]["settled"] and action_name != "settle":
            self.agent_state[agent_name]["settled"] = False

        if action_name == "move":
            seat_id = arguments.get("seat_id")
            if seat_id is None:
                return {"status": "retry", "reason": "seat_id is required for move"}
            result = self._move_agent(agent_name, str(seat_id))
        elif action_name == "settle":
            result = self._settle_agent(agent_name)
        elif action_name == "stand":
            result = self._move_agent(agent_name, None)
        else:
            return {"error": f"Unknown action type: {action_name}"}

        self._refresh_rewards()
        self._advance_time_step()
        return result

    def _advance_time_step(self) -> None:
        self.time_step += 1

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        state = self.agent_state[agent_name]
        true_satisfaction = float(state["satisfaction_score"])
        current_instant_reward = float(state.get("last_instant_reward", 0.0))
        true_tolerance = self._agent_effective_tolerance(agent_name)
        progress = float(iteration) / max(1.0, float(self.max_iterations))

        if progress >= 0.8:
            time_pressure = "high"
        elif progress >= 0.5:
            time_pressure = "medium"
        else:
            time_pressure = "low"

        cumulative_gap = true_satisfaction - true_tolerance
        if cumulative_gap >= 1.0 or current_instant_reward >= 1.0:
            comfort_label = "great"
        elif cumulative_gap >= -0.5 or current_instant_reward >= 0.0:
            comfort_label = "comfortable"
        elif cumulative_gap >= -2.0:
            comfort_label = "uncomfortable"
        else:
            comfort_label = "miserable"

        if true_tolerance < 1.2:
            tol_description = "low (neighbours bother you easily)"
        elif true_tolerance < 1.8:
            tol_description = "moderate"
        else:
            tol_description = "high (neighbours rarely bother you)"

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "scenario_type": self.scenario_type,
            "visible_seats": self._get_visible_seats(agent_name, radius=1),
            "neighbor_observations": self._occupied_neighbors(agent_name),
            "current_seat": state["current_seat"],
            "settled": bool(state["settled"]),
            "pending_reaction": bool(state["pending_reaction"]),
            "preference_profile": dict(state["preference_profile"]),
            "legal_actions": self._legal_actions_for(agent_name, phase=phase),
            "comfort_signal": {
                "label": comfort_label,
                "tolerance_description": tol_description,
                "time_pressure": time_pressure,
            },
            "social_pressure_level": self._pressure_level(agent_name),
        }

        for key, value in kwargs.items():
            context[key] = value

        return context

    def _get_visible_seats(self, agent_name: str, radius: int = 1) -> List[Dict]:
        """Return local seats for seated agents, or the room scan for standing agents."""
        current = self.agent_state[agent_name]["current_seat"]
        if current is None:
            return [
                {
                    "seat_id": seat.seat_id,
                    "row": seat.row,
                    "col": seat.col,
                    "role": seat.role,
                    "occupied": seat.occupied_by is not None,
                    "distance": None,
                }
                for seat in self.seats.values()
            ]
        
        current_seat = self.seats[current]
        visible = []
        
        for seat_id, seat in self.seats.items():
            # Manhattan distance
            dist = abs(seat.row - current_seat.row) + abs(seat.col - current_seat.col)
            if dist <= radius:
                visible.append({
                    "seat_id": seat.seat_id,
                    "row": seat.row,
                    "col": seat.col,
                    "role": seat.role,
                    "occupied": seat.occupied_by is not None,
                    "distance": dist,
                })
        return visible

    def _legal_actions_for(self, agent_name: str, phase: Optional[str] = None) -> List[str]:
        state = self.agent_state[agent_name]
        if phase == "planning":
            return ["request_move", "complain"]
        actions = ["move", "settle", "stand"]
        if state["current_seat"] is None:
            actions.remove("settle")
        return actions

    def done(self, iteration: int) -> bool:
        if iteration > self.max_iterations:
            return True

        # Check if all agents are settled
        all_settled = all(self.agent_state[agent]["settled"] for agent in self.agent_names)

        # Check reward convergence: if reward hasn't improved significantly in recent iterations
        reward_converged = False
        if len(self.joint_reward_history) >= self.convergence_window:
            recent_rewards = self.joint_reward_history[-self.convergence_window:]
            max_recent = max(recent_rewards)
            min_recent = min(recent_rewards)
            # Converged if the range of recent rewards is within threshold
            reward_converged = (max_recent - min_recent) <= self.reward_convergence_threshold

        # Check if current reward is close to theoretical maximum
        current_reward = self.joint_reward_history[-1] if self.joint_reward_history else 0.0
        reward_near_max = False
        if self.max_joint_reward > 0:
            reward_ratio = current_reward / self.max_joint_reward
            # Consider near max if within 10% of theoretical maximum
            reward_near_max = reward_ratio >= 0.9

        # End simulation if agents are settled AND reward has converged OR is near maximum
        if all_settled and (reward_converged or reward_near_max):
            logger.info(
                "Simulation ending: all agents settled=%s, reward_converged=%s, reward_near_max=%s",
                all_settled,
                reward_converged,
                reward_near_max,
            )
            return True

        return False

    def compute_max_joint_reward(self) -> float:
        optimistic = 0.0
        for agent in self.agent_names:
            profile = self.agent_state[agent]["preference_profile"]
            optimistic += 3.0
            optimistic += float(len(profile.get("preferred_roles", [])))
            optimistic += float(len(profile.get("preferred_neighbors", [])))
            optimistic += 1.0 if profile.get("prefer_isolation") else 0.0
            optimistic += 5.0
        optimistic += 2.0 * len(self.agent_names)
        return float(optimistic)

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        reward = sum(float(self.agent_state[agent]["satisfaction_score"]) for agent in self.agent_names)
        reward -= self.group_move_penalty * float(self.total_moves)
        return float(reward)

    def agent_reward(self, agent_name: str, action: Any) -> float:
        return float(self.agent_state[agent_name]["satisfaction_score"])

    def get_final_summary(self) -> Dict[str, Any]:
        return {
            "environment": self.__class__.__name__,
            "scenario_type": self.scenario_type,
            "current_time_step": self.time_step,
            "seat_count": len(self.seats),
            "total_moves": self.total_moves,
            "agents_still_unsettled": sum(1 for agent in self.agent_names if not self.agent_state[agent]["settled"]),
            "joint_reward": self.joint_reward({}),
            "settled": all(self.agent_state[agent]["settled"] for agent in self.agent_names),
            "agent_summaries": {
                agent: {
                    "current_seat": self.agent_state[agent]["current_seat"],
                    "settled": self.agent_state[agent]["settled"],
                    "satisfaction_score": round(float(self.agent_state[agent]["satisfaction_score"]), 3),
                }
                for agent in self.agent_names
            },
            "layout": {
                seat_id: {
                    "row": seat.row,
                    "col": seat.col,
                    "role": seat.role,
                    "occupied_by": seat.occupied_by,
                    "neighbors": list(seat.neighbors),
                }
                for seat_id, seat in self.seats.items()
            },
        }

    def _decay_social_pressure(self, factor: float = 0.7) -> None:
        """Ease off accumulated pressure between iterations.

        social_pressure/pressure_complaints/pressure_requests previously only ever
        increased for the life of the episode, so an agent complained about early on
        stayed permanently more sensitive (lower effective tolerance) even after
        moving away from the source of friction. That ratchet meant more iterations
        or more planning chatter made convergence *less* likely, not more — agents
        kept getting re-triggered into moving instead of settling. Decaying once per
        iteration lets pressure fade if it isn't reinforced by fresh complaints.
        """
        for state in self.agent_state.values():
            state["social_pressure"] = max(0.0, float(state["social_pressure"]) * factor)
            state["pressure_complaints"] = max(0.0, float(state.get("pressure_complaints", 0)) * factor)
            state["pressure_requests"] = max(0.0, float(state.get("pressure_requests", 0)) * factor)

    def log_iteration(self, iteration: int) -> None:
        # Record the joint reward once per iteration. done() reads this history to stop
        # a run early once every agent has settled and the reward has stopped changing.
        self.joint_reward_history.append(self.joint_reward({}))
        self._decay_social_pressure()
        logger.info("=== %s State - Iteration %s ===", self.__class__.__name__, iteration)
        logger.info(
            "Scenario=%s time_step=%s unsettled=%s total_moves=%s",
            self.scenario_type,
            self.time_step,
            sum(1 for agent in self.agent_names if not self.agent_state[agent]["settled"]),
            self.total_moves,
        )

    
