from typing import Dict, List, Any, Optional, Set

class PersonalAssistantEnvironmentTools:
    def __init__(self):
        pass

    def get_supported_tools(self) -> Set[str]:
        """Return set of tool names this environment supports."""
        return {"choose_outfit"}

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get environment-specific tools available for a phase.

        Args:
            phase: Current phase ("planning" or "execution")

        Returns:
            List of tool definitions
        """
        if phase == "execution":
            # During execution phase, provide the choose_outfit tool
            return [{
                "type": "function",
                "function": {
                    "name": "choose_outfit",
                    "description": "Choose your final outfit from your wardrobe options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "outfit_number": {
                                "type": "integer",
                                "description": "The number of the outfit to choose (1-based index from your wardrobe)"
                            }
                        },
                        "required": ["outfit_number"]
                    }
                }
            }]
        # During planning phase, no environment-specific tools
        # Agents use blackboard tools for communication
        return []
