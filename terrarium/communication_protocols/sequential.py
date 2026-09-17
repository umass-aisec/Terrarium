"""
Communication protocol for managing multi-agent interactions and phases.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from terrarium.core.blackboard import Megaboard, format_blackboard_events_for_prompt
from terrarium.communication_protocols.base import BaseCommunicationProtocol
from terrarium.tools.environment import (
    EnvironmentToolsNotFoundError,
    instantiate_environment_tools,
)
from terrarium.core.logger import BlackboardLogger
from terrarium.compaction import compact_events, CompactionLogger
from terrarium.utils import get_client_instance, get_generation_params, get_model_name

logger = logging.getLogger(__name__)

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

        _compaction_config = (self.config.get("llm") or {}).get("compaction")
        _compaction_config = _compaction_config if isinstance(_compaction_config, dict) else {}
        # Compaction is opt-in: with no llm.compaction block the agent sees the raw
        # transcript, as it did before compaction existed. Without this, every
        # environment that never configured compaction silently got lossy,
        # LLM-summarized prompts once a transcript passed the default threshold.
        self.compaction_enabled = bool(_compaction_config)
        self.compaction_logger = None
        if self.compaction_enabled:
            self.compaction_logger = CompactionLogger(
                self.config, run_timestamp=self.run_timestamp
            )
            self.compaction_logger.reset_log()
        self._compaction_pin_context = bool(_compaction_config.get("pin_context_events", False))
        self._compaction_token_threshold = int(_compaction_config.get("token_threshold", 3000))
        self._compaction_keep_recent = int(_compaction_config.get("keep_recent", 3))
        self._compaction_mechanism = str(_compaction_config.get("mechanism", "baseline"))
        self._compaction_extract_limit = int(_compaction_config.get("extract_limit", 5))
        _evict_kinds = _compaction_config.get("evict_kinds")
        self._compaction_evict_kinds = set(_evict_kinds) if isinstance(_evict_kinds, list) else None
        # Per-blackboard mutable state for mechanism="anchored" — carries the
        # running summary across turns instead of re-summarizing everything.
        self._compaction_caches: Dict[int, Dict[str, Any]] = {}

        self.megaboard = Megaboard()
        self.environment = None
        self.environment_tools = None
        self._environment_tools_name: Optional[str] = None
        self._compaction_client = None
        self._compaction_model_name: Optional[str] = None
        self._compaction_params: Dict[str, Any] = {}

    def _get_compaction_client_and_model(self):
        llm_config = self.config.get("llm") or {}
        compaction_config = llm_config.get("compaction")
        if not isinstance(compaction_config, dict) or not compaction_config:
            return None, None

        if self._compaction_client is not None and self._compaction_model_name is not None:
            return self._compaction_client, self._compaction_model_name

        compaction_provider = str(
            compaction_config.get("provider") or llm_config.get("provider") or ""
        ).strip().lower()
        if not compaction_provider:
            return None, None

        try:
            self._compaction_client = get_client_instance(compaction_config)
            self._compaction_model_name = get_model_name(compaction_provider, compaction_config)
            self._compaction_params = get_generation_params(
                {**compaction_config, "provider": compaction_provider}
            )
        except Exception as exc:
            logger.warning(
                "Compaction client initialization failed; falling back to main agent client: %s",
                exc,
            )
            self._compaction_client = None
            self._compaction_model_name = None
            self._compaction_params = {}
            return None, None

        return self._compaction_client, self._compaction_model_name

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
        llm_client=None,
        model_name=None,
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

            if not self.compaction_enabled:
                body = format_blackboard_events_for_prompt(
                    events if isinstance(events, list) else []
                )
                contexts[bb_id_str] = self._label_channel(bb_id_str, bb_id_int, agent_name, body)
                continue

            compaction_client = llm_client
            compaction_model_name = model_name
            compaction_params = None
            override_client, override_model = self._get_compaction_client_and_model()
            if override_client is not None and override_model is not None:
                compaction_client = override_client
                compaction_model_name = override_model
                compaction_params = self._compaction_params

            body = compact_events(
                events if isinstance(events, list) else [],
                llm_client=compaction_client,
                model_name=compaction_model_name,
                token_threshold=self._compaction_token_threshold,
                keep_recent=self._compaction_keep_recent,
                mechanism=self._compaction_mechanism,
                pin_context_events=self._compaction_pin_context,
                cache=self._compaction_caches.setdefault(bb_id_int, {}),
                evict_kinds=self._compaction_evict_kinds,
                extract_limit=self._compaction_extract_limit,
                compaction_logger=self.compaction_logger,
                agent_name=agent_name,
                blackboard_id=bb_id_int,
                phase=phase,
                iteration=iteration,
                params=compaction_params,
            )
            contexts[bb_id_str] = self._label_channel(bb_id_str, bb_id_int, agent_name, body)

        return contexts

    def _label_channel(self, bb_id_str: str, bb_id_int: int, agent_name: str, body: str) -> str:
        """Label a channel so agents can address a specific blackboard_id.

        Only environments that let agents open private channels need the label;
        other environments keep their channel history exactly as before.
        """
        if not getattr(self.environment_tools, "supports_private_channels", False):
            return body
        blackboard = self.megaboard.blackboards[bb_id_int]
        kind = (
            "PRIVATE channel"
            if blackboard.template.get("private_channel")
            else "channel"
        )
        others = ", ".join(sorted(a for a in blackboard.agents if a != agent_name))
        return f"[{kind} {bb_id_str} — with {others}]\n{body}"

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
            llm_client=agent.client,
            model_name=agent.model_name,
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
            llm_client=agent.client,
            model_name=agent.model_name,
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
