"""
Agent factory to build agents based on provider configuration.

Centralises agent construction for runner scripts.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Type

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


def build_agents(
    agent_names: Sequence[str],
    *,
    agent_cls: Type[BaseAgent] = BaseAgent,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    provider: str,
    provider_label: str,
    llm_config: Dict[str, Any],
    model_name: str,
    max_conversation_steps: int,
    tool_logger: Any,
    trajectory_logger: Any,
    environment: Any,
    generation_params: Dict[str, Any],
    vllm_runtime: Any = None,
    shuffle: bool = True,
) -> List[BaseAgent]:
    """Build a list of Agents (or Agent subclasses) given agent names and provider config."""
    if not issubclass(agent_cls, BaseAgent):
        raise TypeError(f"agent_cls must be a subclass of BaseAgent, got: {agent_cls}")

    if environment is None:
        raise ValueError("environment is required")
    environment_name = environment.__class__.__name__

    init_kwargs: Dict[str, Any] = dict(agent_kwargs or {})
    init_kwargs.setdefault("generation_params", generation_params)

    logger.info(
        "Agent factory using agent_cls=%s.%s",
        agent_cls.__module__,
        getattr(agent_cls, "__qualname__", agent_cls.__name__),
    )
    logger.info("Agent factory using environment=%s", environment_name)

    agents: List[BaseAgent] = []
    for name in agent_names:
        if provider == "vllm":
            if not vllm_runtime:
                raise ValueError("vllm_runtime is required when provider == 'vllm'")
            client, agent_model_name = vllm_runtime.create_client(name)
        else:
            from src.utils import get_client_instance

            client = get_client_instance(llm_config)
            agent_model_name = model_name

        agent = agent_cls(
            client,
            name,
            agent_model_name,
            max_conversation_steps,
            tool_logger,
            trajectory_logger,
            environment_name,
            **init_kwargs,
        )
        logger.info(
            "Initialized %s: %s with %s - %s",
            agent.__class__.__name__,
            name,
            provider_label,
            agent_model_name,
        )
        agents.append(agent)

    if shuffle:
        random.shuffle(agents)

    return agents

