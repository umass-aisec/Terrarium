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

    def get_blackboard_tool_names(self) -> Set[str]:
        """
        Get the set of tool names that this blackboard manager supports.

        Returns:
            Set of supported tool names
        """
        return {"post_message"}

    def get_tools_for_blackboard(self, phase: str) -> List[Dict[str, Any]]:
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
            return planning_tools

        return []
