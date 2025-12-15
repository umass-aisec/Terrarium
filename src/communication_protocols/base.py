from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from src.agents.base import BaseAgent


class BaseCommunicationProtocol(ABC):
    """Abstract interface for communication protocols.

    A protocol coordinates agent turns, manages environment tool calls, and
    provides access to shared state (e.g., blackboards).
    """

    environment: Any = None

    @abstractmethod
    async def environment_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def blackboard_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_all_blackboard_ids(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    async def post_system_message(
        self,
        blackboard_id: int,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_comm_network(
        self,
        participants: Sequence[str],
        context: str,
        template: Optional[Dict[str, Any]] = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    async def agent_planning_turn(
        self,
        agent: "BaseAgent",
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        planning_round: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def agent_execution_turn(
        self,
        agent: "BaseAgent",
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
    ) -> None:
        raise NotImplementedError
