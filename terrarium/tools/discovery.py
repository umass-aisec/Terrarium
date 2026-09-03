from typing import Any, Dict, List, Optional, Set

from terrarium.tools.environment import (
    EnvironmentToolsNotFoundError,
    instantiate_environment_tools,
)


class ToolsetDiscovery:
    def __init__(self):
        self._tools_by_environment: Dict[str, Any] = {}

    def _get_tools_instance(self, environment_name: str) -> Optional[Any]:
        if not environment_name:
            return None
        cached = self._tools_by_environment.get(environment_name)
        if cached is not None:
            return cached
        try:
            tools = instantiate_environment_tools(
                environment_name, blackboard_manager=None
            )
        except EnvironmentToolsNotFoundError:
            return None
        except Exception:
            # Treat toolset discovery as best-effort; tool execution will still
            # error if the environment tries to call a missing tool.
            return None
        self._tools_by_environment[environment_name] = tools
        return tools

    def get_tools_for_environment(
        self, environment_name: str, phase: str
    ) -> List[Dict[str, Any]]:
        """
        Get the toolset for a specific environment.

        Args:
            environment_name: Canonical environment identifier (prefer passing environment.__class__.__name__)
            phase: Current phase ("planning" or "execution")
        """
        tools = self._get_tools_instance(environment_name)
        if not tools:
            return []
        return tools.get_tools(phase)

    def get_env_tool_names(self, environment_name: str) -> Set[str]:
        """
        Get the set of tool names that this environment supports.

        Args:
            environment_name: Canonical environment identifier (prefer passing environment.__class__.__name__)
        """
        tools = self._get_tools_instance(environment_name)
        if not tools:
            return set()
        return tools.get_tool_names()

    def supports_private_channels(self, environment_name: str = "") -> bool:
        """
        Whether an environment lets agents open their own private channels.

        Environments opt in by setting `supports_private_channels = True` on
        their Tools class; the channel machinery itself is environment-agnostic.
        """
        tools = self._get_tools_instance(environment_name)
        return bool(getattr(tools, "supports_private_channels", False))

    def get_blackboard_tool_names(
        self, environment_name: str = "", retrieval_enabled: bool = False
    ) -> Set[str]:
        """
        Get the set of tool names that this blackboard manager supports.

        Returns:
            Set of supported tool names
        """
        names = {"post_message"}
        if self.supports_private_channels(environment_name):
            names.add("create_channel")
        if retrieval_enabled:
            names.add("recall")
        return names

    def get_tools_for_blackboard(
        self, phase: str, environment_name: str = "", retrieval_enabled: bool = False
    ) -> List[Dict[str, Any]]:
        """Get blackboard specific tools for the given phase. This is different from Environment tools."""
        # Add phase-specific tools
        if phase == "planning":
            planning_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "post_message",
                        "description": "Post a communication message to agents on the blackboard",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message to communicate to other agents",
                                },
                                "blackboard_id": {
                                    "type": "integer",
                                    "description": "ID of the blackboard you are posting to",
                                },
                            },
                            "required": ["message"],
                        },
                    },
                },
            ]
            if self.supports_private_channels(environment_name):
                planning_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "create_channel",
                            "description": (
                                "Open a private side channel with one or more other agents. "
                                "Only the listed agents (and you) can read or post to it. "
                                "Returns a blackboard_id to use with post_message."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "agent_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Agents to invite. You are added automatically.",
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": (
                                            "Optional first message. Posted immediately — do NOT "
                                            "call post_message again to repeat it."
                                        ),
                                    },
                                },
                                "required": ["agent_ids"],
                            },
                        },
                    }
                )
            if retrieval_enabled:
                planning_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "description": (
                                "Search the FULL, uncompacted history of a channel for "
                                "something the summarized context may have dropped — a "
                                "specific number, identifier, or commitment. Use this when you "
                                "suspect you're missing a detail, not as a substitute for "
                                "reading the chat normally."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Keywords to search for, e.g. a specific identifier or 'Bob agreed'.",
                                    },
                                    "blackboard_id": {
                                        "type": "integer",
                                        "description": "Which channel to search. Defaults to your main channel.",
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                    }
                )
            return planning_tools

        return []
