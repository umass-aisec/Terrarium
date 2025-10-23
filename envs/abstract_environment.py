"""
Abstract base class for environments.

This module defines the interface that all environments must implement to work
with the CommunicationProtocol. It separates environment-specific logic from
the generic communication and phase management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from src.logger import BlackboardLogger, PromptLogger

class AbstractEnvironment(ABC):
    """
    Abstract base class for environments that can be used with CommunicationProtocol.

    This interface defines the minimal set of methods that any environment must
    implement to integrate with the communication protocol system. The protocol
    handles phases, iterations, blackboard management, and LLM interactions,
    while the environment handles domain-specific logic, actions, and context.

    Attributes:
        blackboard_logger: Optional logger for tracking blackboard state changes.
            Environments should initialize this in their initialize() method if they
            want blackboard logging. Set to None to disable logging.
        prompt_logger: Optional logger for tracking agent prompts (system and user).
            Environments should initialize this in their initialize() method if they
            want prompt logging. Set to None to disable logging.
    """

    # Standard attributes - environments should initialize these in initialize()
    blackboard_logger: Optional[BlackboardLogger] = None
    prompt_logger: Optional[PromptLogger] = None

    @abstractmethod
    def get_agent_names(self) -> List[str]:
        """
        Get the list of all agent names in this environment.

        Returns:
            List of agent name strings

        This is used by the protocol to iterate through agents during phases.
        """
        pass

    @abstractmethod
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
            **kwargs: Additional context (planning_round, etc.)

        Returns:
            Dictionary with agent context including:
            - Agent state (budget, inventory, utilities)
            - Environment state (store, available items)
            - Phase-specific information
            - Any other relevant context

        The protocol will pass this context to get_user_prompt along with
        blackboard contexts and available tools.
        """
        pass

    @abstractmethod
    def should_continue_simulation(self, iteration: int) -> bool:
        """
        Check if the simulation should continue running.

        Args:
            iteration: Current iteration number

        Returns:
            True if simulation should continue, False to stop

        This allows the environment to implement custom stopping conditions
        beyond the maximum iteration limit.
        """
        pass

    @abstractmethod
    def log_state(self, iteration: int, phase: str) -> None:
        """
        Log the current state of the environment.

        Args:
            iteration: Current iteration number
            phase: Current phase

        This should log environment-specific state information separately
        from the conversation logging handled by the protocol.
        """
        pass

    @abstractmethod
    def cleanup(self, iteration: int) -> None:
        """
        Clean up any resources used by the environment.

        Args:
            iteration: Current iteration number (for final logging)

        This is called when the simulation ends and should handle:
        - Closing files or connections
        - Saving final state
        - Releasing any held resources
        """
        pass


    # Optional methods with default implementations

    def handle_pending_response(self, response_type: str, target_agent: str,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle responses for multi-agent interactions (trades, invitations).

        Args:
            response_type: Type of response needed ("trade", "invitation")
            target_agent: Agent who should respond
            context: Context for the response

        Returns:
            Dictionary with response handling result

        Default implementation returns an error. Environments can override
        if they need custom handling for these cases.
        """
        return {
            "success": False,
            "reason": f"Environment does not support {response_type} responses"
        }

    def get_iteration_summary(self, iteration: int) -> Dict[str, Any]:
        """
        Get a summary of the current iteration for logging.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary with iteration summary

        Default implementation returns empty summary. Environments can override
        to provide meaningful iteration statistics.
        """
        return {}

    def get_final_summary(self) -> Dict[str, Any]:
        """
        Get a final summary of the entire simulation.

        Returns:
            Dictionary with final simulation results

        Default implementation returns empty summary. Environments can override
        to provide meaningful final statistics and results.
        """
        return {}