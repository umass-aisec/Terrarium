"""
Communication protocol for managing multi-agent interactions and phases.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from terrarium.blackboard import Megaboard, format_blackboard_events_for_prompt
from terrarium.communication_protocols.base import BaseCommunicationProtocol
from terrarium.environment_tools import (
    EnvironmentToolsNotFoundError,
    instantiate_environment_tools,
)
from terrarium.logger import BlackboardLogger

if TYPE_CHECKING:
    from terrarium.agents.base import BaseAgent


class SequentialCommunicationProtocol(BaseCommunicationProtocol):
    """
    Manages the overall communication protocol.

    This class is environment-agnostic and handles:
    - Phase management (planning, execution)
    - Agent turn ordering and iteration
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tool_logger: Any,
        run_timestamp: Optional[str] = None,
    ):
        self.config = config
        self.simulation_config = config["simulation"]
        self.tool_logger = tool_logger
        self.run_timestamp = run_timestamp
        self.blackboard_logger = BlackboardLogger(
            self.config, run_timestamp=self.run_timestamp
        )
        self.blackboard_logger.clear_blackboard_logs()

        self.megaboard = Megaboard()
        self.environment = None
        self.environment_tools = None
        self._environment_tools_name: Optional[str] = None

    def bind_environment(self, environment: Any) -> None:
        """
        Bind a concrete environment instance and initialize its tool handlers.

        This resets blackboards so each simulation run starts clean.
        """
        self.environment = environment
        self.megaboard.clear_blackboards()
        self.environment_tools = None
        self._environment_tools_name = None
        self._ensure_environment_tools_initialized()

    def _ensure_environment_tools_initialized(self) -> None:
        if self.environment is None:
            raise ValueError("Environment must be set before using environment tools.")

        env_name = self.environment.__class__.__name__
        if (
            self.environment_tools is not None
            and self._environment_tools_name == env_name
        ):
            return

        try:
            self.environment_tools = instantiate_environment_tools(
                env_name,
                self.megaboard,
                environment=self.environment,
            )
            self._environment_tools_name = env_name
        except EnvironmentToolsNotFoundError as exc:
            raise ValueError(str(exc)) from exc

    def _log_blackboard_states(
        self,
        *,
        iteration: int,
        phase: str,
        agent_name: str,
        planning_round: Optional[int] = None,
    ) -> None:
        if not self.blackboard_logger:
            return
        for blackboard in self.megaboard.blackboards:
            self.blackboard_logger.log_blackboard_state(
                blackboard, iteration, phase, agent_name, planning_round
            )

    async def environment_handle_tool_call(
        self,
        tool_name,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_environment_tools_initialized()
        response = self.environment_tools.handle_tool_call(
            tool_name, agent_name, arguments, phase, iteration
        )
        return response if isinstance(response, dict) else {"result": response}

    async def blackboard_handle_tool_call(
        self,
        tool_name,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        response = self.megaboard.handle_tool_call(
            tool_name, agent_name, arguments, phase, iteration
        )
        return response if isinstance(response, dict) else {"result": response}

    async def get_all_blackboard_ids(self) -> List[str]:
        return self.megaboard.get_blackboard_string_ids()

    async def post_system_message(
        self, blackboard_id: int, kind: str, payload: Optional[Dict[str, Any]] = None
    ) -> str:
        return self.megaboard.post_system_message(blackboard_id, kind, payload)

    async def _prefetch_blackboard_events(
        self,
        agent_name: str,
        *,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, str]:
        blackboard_ids = self.megaboard.get_agent_blackboards(agent_name)
        if not isinstance(blackboard_ids, list):
            return {}

        def _sort_key(bb_id: Any) -> int:
            try:
                return int(bb_id)
            except Exception:
                return 0

        contexts: Dict[str, str] = {}
        for bb_id_str in sorted([str(b) for b in blackboard_ids], key=_sort_key):
            try:
                bb_id_int = int(bb_id_str)
                events = self.megaboard.get(bb_id_int, agent_name, limit=None)
            except Exception:
                continue

            contexts[bb_id_str] = format_blackboard_events_for_prompt(
                events if isinstance(events, list) else []
            )

        return contexts

    async def agent_planning_turn(
        self,
        agent: "BaseAgent",
        agent_name: str,
        agent_context,
        environment,
        iteration: int,
        planning_round: int,
    ):
        """Handle a single agent's planning turn."""
        self.environment = environment
        self._ensure_environment_tools_initialized()
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="planning",
            iteration=iteration,
        )

        prompts = environment.prompts
        await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="planning",
            iteration=iteration,
            round_num=planning_round,
        )

        self._log_blackboard_states(
            iteration=iteration,
            phase="planning",
            agent_name=agent_name,
            planning_round=planning_round,
        )

    async def generate_comm_network(
        self, participants, context: str, template: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new blackboard and seed it with an initial context message.

        Args:
            participants: List of agent names that can access this blackboard.
            context: Initial context message to post as a system "context" event.
            template: Optional blackboard template.
        """
        blackboard_id = self.megaboard.add_blackboard(
            list(participants), template=template
        )
        self.megaboard.post_system_message(
            blackboard_id,
            "context",
            {"message": context},
        )
        return blackboard_id

    async def agent_execution_turn(
        self,
        agent: "BaseAgent",
        agent_name: str,
        agent_context,
        environment,
        iteration: int,
    ):
        """
        Handle a single agent's execution turn with retry logic.
        """
        self.environment = environment
        self._ensure_environment_tools_initialized()
        blackboard_contexts = await self._prefetch_blackboard_events(
            agent_name,
            phase="execution",
            iteration=iteration,
        )
        prompts = environment.prompts
        await agent.generate_response(
            agent_name=agent_name,
            agent_context=agent_context,
            blackboard_context=blackboard_contexts,
            communication_protocol=self,
            prompts=prompts,
            phase="execution",
            iteration=iteration,
        )

        self._log_blackboard_states(
            iteration=iteration,
            phase="execution",
            agent_name=agent_name,
        )
