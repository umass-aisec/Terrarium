"""
MeetingScheduling Environment Adaptor of MeetingScheduling Environment in CoLLAB

Adaptor to integrate the MeetingScheduling domain.

The MeetingScheduling environment involves agents coordinating to schedule
meetings while respecting time preferences, travel constraints, and priorities.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from envs.dcops.CoLLAB.MeetingScheduling.data_structure import (
    MeetingId, SlotId, SLOT_LABELS
)

# Use TYPE_CHECKING to avoid circular import (Agent → ToolsetDiscovery → MeetingSchedulingEnvironmentTools → MeetingSchedulingEnvironment → Agent)
if TYPE_CHECKING:
    from src.agent import Agent
from envs.dcops.CoLLAB.MeetingScheduling.generate import build_meeting_env
# Import abstract environment interface and loggers
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
from .meeting_scheduling_prompts import MeetingSchedulingPrompts

class MeetingSchedulingEnvironment(AbstractEnvironment):
    """
    MeetingScheduling environment adaptor for meeting coordination tasks.

    Agents (meeting owners) coordinate to schedule meetings while optimizing
    time preferences, travel constraints, and priority-based attendance.
    """

    def __init__(self, communication_protocol, config, tool_logger):
        """Initialize the MeetingScheduling environment."""
        self.full_config = config
        self.config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        # Get the correct seed from environment config (matches what's used for instance generation)
        self.current_seed = self.config.get("rng_seed", 42)

        # Instance management
        self.meeting_schedules: Dict[MeetingId, SlotId] = {}  # 0-based slots internally
        self.tool_logger = tool_logger
        self.agent_names: List[str] = []  # Meeting owners (agents)
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self # Add bidirectional reference
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.current_iteration = 0
        self.max_iterations = self.simulation_config.get("max_iterations", None)
        self.max_planning_rounds = self.simulation_config.get("max_planning_rounds", None)        
        # Data paths to static CoLLAB data files
        self.data_root = Path(__file__).parent.parent.parent.parent / "envs" / "dcops" / "CoLLAB" / "MeetingScheduling" / "data"

        # Build the CoLLAB meeting environment instance
        self.instance = build_meeting_env(
            n_agents=self.config.get("n_agents", 6),
            n_meetings=self.config.get("n_meetings", 8),
            data_root=self.data_root,
            rng_seed=self.config.get("rng_seed", 42),
            max_attendees_per_meeting=self.config.get("max_attendees_per_meeting", 4),
            p_zoom=self.config.get("p_zoom", 0.4),
            min_prefs_per_agent=self.config.get("min_prefs_per_agent", 4),
            max_prefs_per_agent=self.config.get("max_prefs_per_agent", 7),
            time_match_weight=self.config.get("time_match_weight", 1.0)
        )

        # Score tracking
        self.global_score_history: List[float] = []
        self.local_scores_history: Dict[str, List[float]] = {}
        self.score_plotter: Optional[ScorePlotter] = None
        # Only include agents who actually participate in at least one meeting
        # (some agents may not be selected as attendees in any meeting)
        self.agent_names = list(set(
            agent for meeting in self.instance.graph.meetings.values()
            for agent in meeting.attendees
        ))
        self.agents: List['Agent'] = []

        # Clear seed directories FIRST to ensure clean state for this run
        # This must happen before creating loggers (PromptLogger in MeetingSchedulingPrompts)
        clear_seed_directories("MeetingScheduling", self.current_seed, self.full_config)

        # Initialize prompts (Put this after all other instance variables)
        self.prompts = MeetingSchedulingPrompts(self, self.full_config)

        # Initialize score tracking
        self.local_scores_history = {agent: [] for agent in self.agent_names}

        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        plots_dir = build_plots_dir("MeetingScheduling", tag_model, self.current_seed, self.run_timestamp)
        self.score_plotter = ScorePlotter(save_dir=str(plots_dir))

        print(f"MeetingScheduling environment initialized with {len(self.agent_names)} agents")
        print(f"Agents (meeting owners): {', '.join(self.agent_names)}")
        print(f"Total meetings: {len(self.instance.graph.meetings)}")        
        print("MeetingSchedulingEnvironment initialized")

    async def async_init(self):
        await self.create_blackboards_from_factors()

    def set_agent_clients(self, agents: List['Agent']):
        """Set the agents for the environment."""
        self.agents = agents

    async def create_blackboards_from_factors(self):
        """Create blackboards from MeetingScheduling factors. This happens only during initialization"""

        # Create blackboards for multi-agent coordination factors (MEETING_TIME_MATCH with multiple attendees)
        for factor in self.instance.graph.factors:
            if factor.ftype == "MEETING_TIME_MATCH" and len(factor.agent_scope) > 1:
                # Get the meeting for context
                meeting_id = factor.var_scope[0]
                meeting = self.instance.graph.meetings[meeting_id]

                context = f"Meeting Coordination: {meeting_id} ({meeting.mode}"
                if meeting.location:
                    context += f" at {meeting.location}"
                context += f"). Attendees: {', '.join(factor.agent_scope)}. " \
                          f"Coordinate to find optimal time slot that maximizes preferences."

                blackboard_id = await self.communication_protocol.generate_blackboard_network_from_factor(factor, context)
                print(f"Created Meeting Blackboard {blackboard_id}: {factor.agent_scope} for {meeting_id}")

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

    def get_agent_names(self) -> List[str]:
        """Get list of agent names."""
        return self.agent_names.copy()

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        """
        Build environment-specific context for an agent's turn.

        Args:
            agent_name: Name of the agent
            phase: Current phase ("planning" or "execution")
            iteration: Current iteration number
            **kwargs: Additional context

        Returns:
            Dictionary with agent context
        """
        # Get agent's owned meetings
        agent_meetings = [mid for mid, meeting in self.instance.graph.meetings.items()
                         if meeting.owner == agent_name]

        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "meeting_schedules": self.meeting_schedules.copy(),
            "agent_owned_meetings": agent_meetings,
            "total_meetings": len(self.instance.graph.meetings),
            "meetings_scheduled": len(self.meeting_schedules),
            "meetings_remaining": len(self.instance.graph.meetings) - len(self.meeting_schedules)
        }

        # Add configuration info
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
        # Check max iterations first
        assert self.config is not None, "Config not available"
        max_iterations = self.config.get("max_iterations", 10)
        if iteration > max_iterations:
            print(f"Reached max iterations ({max_iterations}) - stopping simulation")
            return False

        # Stop early if all meetings have been scheduled
        total_meetings = len(self.instance.graph.meetings)
        if len(self.meeting_schedules) == total_meetings:
            global_score = self.instance.graph.global_score(
                self.meeting_schedules,
                self.instance.preferences,
                self.instance.coords
            )
            print(f"All meetings scheduled with global score: {global_score:.2f} - simulation complete")
            return False

        return True


    def log_state(self, iteration: int, phase: str) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
            phase: Current phase
        """
        print(f"=== MeetingScheduling State - Iteration {iteration}, Phase {phase} ===")
        total_meetings = len(self.instance.graph.meetings)
        print(f"Meetings: {total_meetings} total, {len(self.meeting_schedules)} scheduled")

        if self.meeting_schedules:
            print("Current Meeting Schedules:")
            for mid, slot in self.meeting_schedules.items():
                meeting = self.instance.graph.meetings[mid]
                print(f"  {mid}: Slot {slot + 1} ({SLOT_LABELS[slot]}) - {meeting.mode} with {', '.join(meeting.attendees)}")

        unscheduled_meetings = [mid for mid in self.instance.graph.meetings.keys()
                               if mid not in self.meeting_schedules]
        if unscheduled_meetings:
            print(f"Unscheduled meetings: {', '.join(unscheduled_meetings)}")

        # Calculate and display scores if any meetings are scheduled
        global_score = 0.0
        if self.meeting_schedules:
            completion_rate = len(self.meeting_schedules) / total_meetings if total_meetings > 0 else 0
            print(f"Meeting completion: {len(self.meeting_schedules)}/{total_meetings} ({completion_rate:.1%})")

            global_score = self.instance.graph.global_score(
                self.meeting_schedules,
                self.instance.preferences,
                self.instance.coords
            )
            print(f"Current Global Score: {global_score:.2f}")

        # Calculate per-agent scores (based on meetings they own)
        agent_scores = {}
        for agent in self.agent_names:
            # Get meetings owned by this agent that have been scheduled
            agent_meetings = [mid for mid, meeting in self.instance.graph.meetings.items()
                             if meeting.owner == agent and mid in self.meeting_schedules]
            if agent_meetings:
                # Calculate score for this agent's meetings
                agent_schedule = {mid: self.meeting_schedules[mid] for mid in agent_meetings}
                agent_score = self.instance.graph.global_score(
                    agent_schedule,
                    self.instance.preferences,
                    self.instance.coords
                )
                agent_scores[agent] = agent_score
            else:
                agent_scores[agent] = 0.0

        # Track scores and generate plots/logs for every iteration
        self._track_scores(iteration, global_score, agent_scores)

    def _track_scores(self, iteration: int, global_score: float, local_scores: Dict[str, float]) -> None:
        """Track scores and generate plots/logs."""
        import json
        from datetime import datetime

        # Update score histories
        self.global_score_history.append(global_score)
        for agent, score in local_scores.items():
            if agent in self.local_scores_history:
                self.local_scores_history[agent].append(score)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir("MeetingScheduling", tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": "MeetingScheduling",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "global_score": global_score,
            "local_scores": local_scores,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "metadata": {
                "total_agents": len(local_scores),
                "total_meetings_scheduled": len(self.meeting_schedules),
                "total_meetings": len(self.instance.graph.meetings) if self.instance else 0,
                "average_local_score": sum(local_scores.values()) / len(local_scores) if local_scores else 0
            }
        }

        score_file = log_dir / f"scores_iteration_{iteration}.json"
        with open(score_file, 'w') as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

        # Generate plot
        if self.score_plotter:
            try:
                plot_path = self.score_plotter.plot_scores(
                    self.global_score_history,
                    self.local_scores_history,
                    iteration,
                    environment_name="MeetingScheduling",
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
        # Extract meeting information in serializable format
        meetings = {}
        if hasattr(self, 'instance') and self.instance:
            for mid, meeting in self.instance.graph.meetings.items():
                meetings[mid] = {
                    "owner": meeting.owner,
                    "mode": meeting.mode,
                    "location": meeting.location,
                    "attendees": meeting.attendees
                }

        return {
            "meetings": meetings,
            "meeting_schedules": self.meeting_schedules.copy(),
            "agent_names": self.agent_names.copy()
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        """
        Apply state updates from tool execution.

        Args:
            state_updates: Dictionary with state updates to apply
        """
        # Apply meeting_schedules updates (UPDATE, don't replace!)
        if "meeting_schedules" in state_updates:
            self.meeting_schedules.update(state_updates["meeting_schedules"])

    def post_tool_execution_callback(self, state_updates: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Post-tool execution callback for MeetingScheduling-specific processing.
        Useful to get immediate score feedback, but not required.

        This is called after state updates are applied to perform environment-specific
        operations like score calculation.

        Args:
            state_updates: Dictionary with state updates that were applied
            response: The response dictionary to potentially modify
        """
        # Recalculate global_score after state updates if meeting schedules were updated
        if "meeting_schedules" in state_updates:
            if hasattr(self, 'instance') and self.instance:
                global_score = self.instance.graph.global_score(
                    self.meeting_schedules,
                    self.instance.preferences,
                    self.instance.coords
                )
                # Add global_score to result for agent feedback
                if "result" in response:
                    response["result"]["global_score"] = global_score

    def cleanup(self) -> None:
        """Clean up any resources used by the environment."""
        print("MeetingScheduling environment cleanup")
        if self.meeting_schedules:
            total_meetings = len(self.instance.graph.meetings) if self.instance else 0
            print(f"Final meeting schedules: {len(self.meeting_schedules)}/{total_meetings} meetings")

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current iteration for logging."""
        total_meetings = len(self.instance.graph.meetings) if self.instance else 0
        return {
            "meetings_scheduled": len(self.meeting_schedules),
            "total_meetings": total_meetings,
            "total_agents": len(self.agent_names),
            "completion_rate": len(self.meeting_schedules) / total_meetings if total_meetings > 0 else 0
        }

    def get_final_summary(self) -> Dict[str, Any]:
        """Get a final summary of the entire simulation."""
        total_meetings = len(self.instance.graph.meetings) if self.instance else 0

        if not self.instance or len(self.meeting_schedules) != total_meetings:
            return {
                "status": "incomplete",
                "meetings_scheduled": len(self.meeting_schedules),
                "total_meetings": total_meetings,
                "total_agents": len(self.agent_names)
            }

        # Calculate final scores
        global_score = self.instance.graph.global_score(
            self.meeting_schedules,
            self.instance.preferences,
            self.instance.coords
        )

        # Calculate per-agent scores
        agent_scores = {}
        for agent in self.agent_names:
            agent_meetings = [mid for mid, meeting in self.instance.graph.meetings.items()
                             if meeting.owner == agent]
            if agent_meetings:
                agent_schedule = {mid: self.meeting_schedules[mid] for mid in agent_meetings
                                 if mid in self.meeting_schedules}
                if agent_schedule:
                    agent_score = self.instance.graph.global_score(
                        agent_schedule,
                        self.instance.preferences,
                        self.instance.coords
                    )
                    agent_scores[agent] = agent_score
                else:
                    agent_scores[agent] = 0.0
            else:
                agent_scores[agent] = 0.0

        return {
            "status": "complete",
            "global_score": global_score,
            "average_local_score": sum(agent_scores.values()) / len(agent_scores) if agent_scores else 0,
            "local_scores": agent_scores,
            "meeting_schedules": {
                mid: {"slot": slot + 1, "time_label": SLOT_LABELS[slot]}
                for mid, slot in self.meeting_schedules.items()
            },
            "total_meetings": total_meetings,
            "total_agents": len(self.agent_names)
        }
