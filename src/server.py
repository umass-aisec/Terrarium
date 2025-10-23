from fastmcp import FastMCP
import json
from typing import Any, Dict, List, Optional
import sys
import os

# Add project root to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import with src. prefix
from src.blackboard import Megaboard, Blackboard
from src.logger import BlackboardLogger
# Constructing a MCP server to interact with the Megaboard

mcp = FastMCP("blackboard_mcp")

megaboard = Megaboard()

# Environment tools will be set dynamically based on environment type
environment_tools = None

# Initialize later
blackboard_logger: Optional[BlackboardLogger] = None

def set_environment_tools(environment_name: str):
    """
    Dynamically load and set environment-specific tools.

    Args:
        environment_name: Name of the environment (e.g., "MeetingScheduling", "PersonalAssistant", "SmartGrid")
    """
    global environment_tools

    # Normalize environment name
    normalized = environment_name.lower().replace("_", "")

    if normalized == "meetingscheduling":
        from envs.dcops.meeting_scheduling import MeetingSchedulingTools
        environment_tools = MeetingSchedulingTools(megaboard)
    elif normalized == "personalassistant":
        from envs.dcops.personal_assistant.personal_assistant_tools import PersonalAssistantTools
        environment_tools = PersonalAssistantTools(megaboard)
    elif normalized == "smartgrid":
        from envs.dcops.smart_grid.smartgrid_tools import SmartGridTools
        environment_tools = SmartGridTools(megaboard)
    else:
        raise ValueError(f"Unknown environment: {environment_name}. Supported: MeetingScheduling, PersonalAssistant, SmartGrid")

@mcp.tool()
def handle_environment_tool_call(tool_name: str, agent_name: str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None, env_state: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Handle environment tool calls using serializable state."""
    if environment_tools is None:
        return {"error": "Environment tools not initialized. Call set_environment_tools first."}
    return environment_tools.handle_tool_call(tool_name, agent_name, arguments, phase, iteration, env_state)

@mcp.tool()
def initialize_environment_tools(environment_name: str) -> Dict[str, str]:
    """Initialize environment-specific tools based on environment name."""
    try:
        # Clear all blackboards when starting a new environment
        megaboard.clear_blackboards()
        set_environment_tools(environment_name)
        return {"status": "success", "message": f"Environment tools initialized for {environment_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def return_blackboards() -> List[Blackboard]:
    """Return the blackboards"""
    return megaboard.return_blackboards()

@mcp.tool()
def add_blackboard(agents: List[str], template: Optional[Dict[str, Any]] = None) -> int:
    """Add a new blackboard"""
    return megaboard.add_blackboard(agents, template)

@mcp.tool()
def clear_blackboards() -> Dict[str, str]:
    """Clear all blackboards. Used when starting a new simulation."""
    try:
        megaboard.clear_blackboards()
        return {"status": "success", "message": "All blackboards cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def post(blackboard_id: int, agent: str, kind: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """Post an event to the blackboard"""
    return megaboard.post(blackboard_id, agent, kind, payload)

@mcp.tool()
def post_system_message(blackboard_id: int, kind: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """Post a system message to the blackboard"""
    return megaboard.post_system_message(blackboard_id, kind, payload)

@mcp.tool()
def get(blackboard_id: int, agent: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get the recent events from the blackboard"""
    return megaboard.get(blackboard_id, agent, limit)

@mcp.tool()
def get_blackboard_by_string_id(blackboard_id: str) -> Optional[Blackboard]:
    """Get a blackboard by string ID"""
    return megaboard.get_blackboard_by_string_id(blackboard_id)

@mcp.tool()
def get_agent_blackboards(agent_name: str) -> List[str]:
    """Get the blackboards for an agent"""
    return megaboard.get_agent_blackboards(agent_name)

@mcp.tool()
def get_agent_blackboard_contexts(agent_name: str) -> Dict[str, str]:
    """Get the contexts for an agent's blackboards"""
    return megaboard.get_agent_blackboard_contexts(agent_name)

@mcp.tool()
def get_blackboard_string_ids() -> List[str]:
    return megaboard.get_blackboard_string_ids()

@mcp.tool()
def handle_blackboard_tool_call(tool_name, agent_name:str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
    return megaboard.handle_tool_call(tool_name, agent_name, arguments, phase, iteration)

@mcp.tool()
def log_action_to_blackboards(agent_name: str, action: Dict[str, Any], result: Dict[str, Any],
                                   phase: Optional[str] = None, iteration: Optional[int] = None) -> None:
    return megaboard.log_action_to_blackboards(agent_name, action, result, phase, iteration)

@mcp.tool()
def initialize_blackboard_logger(config: Dict[str, Any]) -> Dict[str, str]:
    global blackboard_logger
    blackboard_logger = BlackboardLogger(config)
    blackboard_logger.clear_blackboard_logs()
    return {"status": "success", "message": "Blackboard logger initialized"}

@mcp.tool()
def log_blackboard_states(iteration: int, phase: str, agent_name: str, planning_round: Optional[int] = None) -> Dict[str, str]:
    global blackboard_logger
    if blackboard_logger is None:
        return {"status": "error", "message": "Blackboard logger not initialized"}
    try:
        # Log all blackboards
        for i, blackboard in enumerate(megaboard.blackboards):
            blackboard_logger.log_blackboard_state(
                blackboard, iteration, phase, agent_name, planning_round
            )
        return {"status": "success", "message": f"Logged {len(megaboard.blackboards)} blackboards"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to log blackboards: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)