"""
Centralized user prompt template for agent responses.

This module provides a single location for generating user prompts,
eliminating duplication across client implementations.
"""

from typing import Dict, Any, List, Optional


def _format_blackboard_memberships(agent_context: Dict[str, Any]) -> str:
    """Format blackboard memberships for display in user prompt."""
    memberships = agent_context.get('blackboard_memberships', [])
    
    if not memberships:
        return "- You are not currently in any blackboards"
    
    formatted = []
    for membership in memberships:
        bb_id = membership.get('blackboard_id', 'unknown')
        participants = membership.get('participants', [])
        context = membership.get('initial_context', '')
        recent_events = membership.get('recent_events', 0)

        # Format with clear blackboard ID for tools
        info = f"- Blackboard {bb_id} (use blackboard_id: \"{bb_id}\"): {', '.join(participants)}"
        
        if context:
            info += f" (Purpose: {context})"
        info += f" - {recent_events} recent messages"
        formatted.append(info)
    
    return '\n'.join(formatted)


def _format_pending_invites(agent_context: Dict[str, Any]) -> str:
    """Format pending blackboard invitations for display in user prompt."""
    invites = agent_context.get('pending_blackboard_invites', [])
    
    if not invites:
        return "- No pending blackboard invitations"
    
    formatted = []
    for invite in invites:
        bb_id = invite.get('blackboard_id', 'unknown')
        inviter = invite.get('inviter', 'unknown')
        participants = invite.get('participants', [])
        context = invite.get('initial_context', '')
        
        info = f"- Invitation to blackboard '{bb_id}' from {inviter}"
        info += f" (Participants: {', '.join(participants)})"
        if context:
            info += f" (Purpose: {context})"
        formatted.append(info)
    
    return '\n'.join(formatted)


def _format_agent_ordering(agent_context: Dict[str, Any], agent_name: str) -> str:
    """Format agent ordering for display in user prompt."""
    agent_order = agent_context.get('agent_order', [])
    
    if not agent_order:
        return "- Agent turn order not available"
    
    formatted = []
    for i, name in enumerate(agent_order, 1):
        if name == agent_name:
            formatted.append(f"{i}. {name} (you)")
        else:
            formatted.append(f"{i}. {name}")
    
    return '\n'.join(formatted)


def _format_blackboard_contexts(blackboard_contexts: Dict[str, str]) -> str:
    """Format multiple blackboard contexts for display in user prompt."""
    if not blackboard_contexts:
        return "No blackboard activity available."

    formatted_sections = []
    for bb_id, context in blackboard_contexts.items():
        section = f"<blackboard '{bb_id}' context>\n{context}\n</blackboard '{bb_id}' context>"
        formatted_sections.append(section)

    return '\n\n'.join(formatted_sections)


def generate_user_prompt(agent_name: str, agent_context: Dict[str, Any],
                        blackboard_contexts: Dict[str, str], 
                        available_tools: Optional[List[str]] = None) -> str:
    """
    Generate a comprehensive user prompt for an agent.
    
    Args:
        agent_name: Name of the agent
        agent_context: Agent's private context (budget, inventory, utilities, phase)
        blackboard_contexts: Dictionary mapping blackboard_id -> context_summary for agent's blackboards
        available_tools: List of available tool names for this phase
        
    Returns:
        Formatted user prompt string
    """
    # Get current phase and check for special states
    current_phase = agent_context.get('phase', 'unknown')
    # Check if any blackboard context contains trade proposal or blackboard invitation
    all_contexts = ' '.join(blackboard_contexts.values()) if blackboard_contexts else ''
    is_trade_response = "TRADE PROPOSAL TO RESPOND TO" in all_contexts
    is_blackboard_invitation_response = "BLACKBOARD INVITATION TO RESPOND TO" in all_contexts
    
    # Build tool usage instructions based on current state and available tools
    if is_trade_response:
        # Trade response mode
        tool_instructions = """You need to respond to an incoming trade proposal. Use one of these tools:
- accept_trade_proposal: Accept the trade if you find it favorable
- decline_trade_proposal: Decline the trade if you don't find it favorable
Always provide a clear reason for your decision."""
        
    elif is_blackboard_invitation_response:
        # Blackboard invitation response mode
        tool_instructions = """You need to respond to a blackboard invitation. Use this tool:
- respond_to_channel_invite: Accept (True) or decline (False) the invitation
Consider the purpose and participants before deciding."""
    
    elif current_phase == "planning":
        # Planning phase
        tool_instructions = """You are in the PLANNING phase. You MUST actively communicate with other agents using the post_message tool.

CRITICAL: You MUST call tools to take actions. Do NOT just describe what you intend to do.

REQUIRED ACTIONS:
1. FIRST: Use 'get_blackboard_events' to check what other agents have said
2. THEN: Use 'post_message' to post a message to the blackboard announcing your intentions, sharing information, proposing trades

Available tools may include:
- post_message: Post messages to blackboards (CALL THIS TOOL - don't just describe it)
- get_blackboard_events: Read recent messages (CALL THIS TOOL)
- get_item_prices: View current item prices

WRONG: "I will communicate my strategy to other agents"
RIGHT: Actually call the post_message tool with your message

Remember: The planning phase is for communication and strategy. You must actively participate in blackboard discussions by CALLING TOOLS."""
    
    else:
        # Execution phase
        tool_instructions = """You are in the EXECUTION phase. Use available tools to take concrete actions:
- propose_trade: Propose trades with other agents
- buy_item: Buy items from the store
- do_nothing: Skip your turn if you don't want to act

You can also use information tools to help make decisions:
- get_item_prices: View current item prices
- get_blackboard_events: Read recent messages"""
    
    # Format available tools if provided
    if available_tools:
        # Handle case where available_tools might contain dicts instead of strings
        tool_names = []
        for tool in available_tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            else:
                raise ValueError(f"Unexpected tool format: {tool}")

        tools_list = ", ".join(tool_names)
        tool_availability = f"\nAvailable tools for this turn: {tools_list}"
    else:
        tool_availability = ""

    # Build game info section conditionally on the phase
    game_info_lines = [
        f"- Phase: {current_phase}",
        f"- Current iteration: {agent_context.get('iteration', 0)}"
    ]
    # Add planning round only during planning phase
    if current_phase == "planning":
        game_info_lines.insert(1, f"- Current planning round: {agent_context.get('planning_round', 'N/A')}")
    game_info = '\n'.join(game_info_lines)
    user_prompt = f"""You are {agent_name}.

<what you have>
- Budget (USD): ${agent_context.get('budget', 0)}
- Current Inventory: {agent_context.get('inventory', [])}
- Current Total Inventory Utility: {agent_context.get('current_utility', 0)}
- Item Utilities: {agent_context.get('utilities', {})}
</what you have>

<agent ordering>
{_format_agent_ordering(agent_context, agent_name)}
</agent ordering>

<blackboard memberships>
{_format_blackboard_memberships(agent_context)}
</blackboard memberships>

<pending blackboard invitations>
{_format_pending_invites(agent_context)}
</pending blackboard invitations>

<environment rules and properties>
- All agents have different utilities for all items, and the utilities for other agents are not known.
- All agents have different store item prices that depends on the agent's utility for the item. For example, if Kyle has a high utility for an apple then the apple will be more expensive for Kyle than for Zoe who has a low utility for an apple.
- The store shares the same inventory for all other traders, so if Kyle buys a bannana from the store, then the store will have one less bannana for Zoe to purchase during her turn.
- Only one item can be purchased from the store at a time if you select the **buy** action.
- There are {agent_context.get('max_planning_rounds', 'unknown')} planning rounds and one action execution round per iteration and {agent_context.get('max_iterations', 'unknown')} total iterations in the game.
</environment rules and properties>

{_format_blackboard_contexts(blackboard_contexts)}

<game info>
{game_info}
</game info>

<tool instructions>
{tool_instructions}{tool_availability}
</tool instructions>

**Read** the blackboard contexts carefully to understand the current situations. Consider your utilities and budget carefully. Maximize your total utility score through strategic trading and communication.
**Think** and **consider** the actions taken by other agents.

Use the available tools to take your action. You can call multiple tools if needed, but focus on taking one primary action per turn.

It is now your turn:"""

    return user_prompt