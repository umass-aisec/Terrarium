"""
Trading game environment implementation.

This module implements a trading simulation environment where agents can
buy items, propose trades, and communicate through blackboards. It implements
the AbstractEnvironment interface to work with the CommunicationProtocol.
"""

import os
from typing import Dict, List, Any, Optional, Set

from envs.abstract_environment import AbstractEnvironment
from .store import Store
from .agent import Agent, create_agent_pool, create_agents_with_names
from .trade import TradeManager
from envs.negotiation.plotter import UtilityPlotter
from envs.dcops.plotter import ScorePlotter
from src.logger import BlackboardLogger, PromptLogger
from src.utils import clear_seed_directories, extract_model_info, get_tag_model_subdir
from .prompts.user_prompt import generate_user_prompt


class TradingGameEnvironment(AbstractEnvironment):
    """
    Trading game environment where agents buy items, trade, and optimize utility.

    This environment implements the AbstractEnvironment interface and provides:
    - Store with items and pricing
    - Agent management with utilities and budgets
    - Trade proposals and execution
    - Action validation and retry hints
    - Game-specific context building
    """

    def __init__(self):
        """Initialize the trading game environment."""
        # Action types to log to blackboards (empty list = no logging)
        self.action_logging_config = ["buy", "trade", "approve_trade", "reject_trade"]

        # Core components - initialized in initialize()
        self.config = None
        self.store = None
        self.agents = []
        self.agents_dict = {}
        self.trade_manager = None
        self.blackboard_manager = None

        # Logging and tracking
        self.blackboard_logger = None
        self.utility_history = {}
        self.budget_history = {}
        self.plotter = None

        # Factor graph support
        self._factor_graph_blackboards = []

    def initialize(self, config: Dict[str, Any], blackboard_manager=None) -> None:
        """
        Initialize the trading game with configuration and blackboard access.

        Args:
            config: Full configuration dictionary containing all simulation settings
            blackboard_manager: Blackboard manager for posting events
        """
        self.full_config = config
        self.config = config["environment"]  # Extract environment-specific config
        self.blackboard_manager = blackboard_manager

        # Extract and store seed for reproducibility
        self.seed = self.full_config.get("_current_seed", self.config.get("rng_seed", 42))

        # Initialize store with seed for reproducible inventory
        self.store = Store(seed=self.seed)

        # Initialize trade manager
        self.trade_manager = TradeManager()

        # Get list of items from store
        items_list = list(self.store.item_prices.keys())

        # Create agents and generate random blackboards for trading coordination
        min_budget = self.config.get("min_budget", 25)
        max_budget = self.config.get("max_budget", 25)
        num_agents = self.config.get("num_agents", 4)

        # Create agents first
        self.agents = create_agent_pool(
            items_list=items_list,
            num_agents=num_agents,
            min_budget=min_budget,
            max_budget=max_budget,
            initial_items=config.get("environment", {}).get("initial_items", 3),
            seed=self.seed
        )

        # Generate random blackboards for agent coordination with seeded randomization
        if self.blackboard_manager:
            import random
            rng = random.Random(self.seed)

            # Create random blackboards: roughly num_agents // 2 blackboards
            num_blackboards = max(2, num_agents // 2)

            for i in range(num_blackboards):
                # Randomly select 2-3 agents for this blackboard
                num_participants = rng.randint(2, min(3, num_agents))
                participants = rng.sample([a.name for a in self.agents], num_participants)

                # Create blackboard with these agents
                blackboard_id = self.blackboard_manager.add_blackboard(participants, None)

                print(f"Created random blackboard {blackboard_id}: {participants}")

            # Store factor graph blackboards for reference
            self._factor_graph_blackboards = self.blackboard_manager.get_blackboard_string_ids()
            print(f"Generated {len(self._factor_graph_blackboards)} random blackboards for {num_agents} agents")
        else:
            self._factor_graph_blackboards = []
            print(f"Created {num_agents} agents without blackboards (no blackboard manager)")

        # Create agents dictionary for quick lookup
        self.agents_dict = {agent.name: agent for agent in self.agents}

        # Add agents to their factor graph blackboards
        if self.blackboard_manager and self._factor_graph_blackboards:
            print(f"Registering agents to {len(self._factor_graph_blackboards)} factor graph blackboards...")
            for bb_string_id in self._factor_graph_blackboards:
                blackboard = self.blackboard_manager.get_blackboard_by_string_id(bb_string_id)
                if blackboard:
                    for agent_name in blackboard.participants:
                        # Find the agent and add membership
                        agent = next((a for a in self.agents if a.name == agent_name), None)
                        if agent:
                            agent.join_blackboard(bb_string_id)
                            print(f"  Added {agent_name} to blackboard {bb_string_id}")


        # Initialize loggers
        self.blackboard_logger = BlackboardLogger("Trading", self.seed, self.full_config)
        self.prompt_logger = PromptLogger("Trading", self.seed, self.full_config)

        # Clear logs at simulation start
        self.blackboard_logger.clear_blackboard_logs()
        self.prompt_logger.reset_log()

        # Clear seed directories to ensure clean state for this run
        clear_seed_directories("Trading", self.seed, self.full_config)

        # Initialize utility and budget tracking and plotter
        self.utility_history = {agent.name: [] for agent in self.agents}
        self.budget_history = {agent.name: [] for agent in self.agents}

        # Setup seed-based directory structure for plots
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        plots_dir = f"plots/Trading/{tag_model}/seed_{self.seed}"
        self.plotter = UtilityPlotter(save_dir=plots_dir)

        # Initialize score tracking for JSON logging (similar to other environments)
        self.global_score_history: List[float] = []
        self.local_scores_history: Dict[str, List[float]] = {agent.name: [] for agent in self.agents}
        self.score_plotter = ScorePlotter(save_dir=plots_dir)

        # Record initial utilities and budgets (before any trading)
        for agent in self.agents:
            self.utility_history[agent.name].append(agent.state.current_utility)
            self.budget_history[agent.name].append(agent.state.budget)

        print(f"Initialized trading game with {len(self.agents)} agents and {len(items_list)} items")

        # Print agent list
        for agent in self.agents:
            print(f"  {agent.name}")

    def get_system_prompt(self) -> str:
        """Get the system prompt for trading game agents."""
        return self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt for agents from file with validation."""
        # Get prompt path from config
        prompt_path = self.config.get("system_prompt_path", "src/trading/prompts/system_prompt.txt") if self.config else "src/trading/prompts/system_prompt.txt"

        # Convert to absolute path relative to the project root directory
        if not os.path.isabs(prompt_path):
            # Go up three levels from trading_environment.py to get to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            prompt_path = os.path.join(project_root, prompt_path)

        # Validate file extension
        if not prompt_path.endswith('.txt'):
            raise ValueError(f"System prompt file must be a .txt file, got: {prompt_path}")

        # Validate file exists
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

        # Load and validate content
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Validate content is not empty
        if not content:
            raise ValueError(f"System prompt file is empty: {prompt_path}")

        return content

    def _get_user_prompt_impl(self, agent_name: str, agent_context: Dict[str, Any],
                             blackboard_context: Dict[str, Any], available_tools: List[Dict]) -> str:
        """Generate user prompt for trading game using existing prompt generation."""
        return generate_user_prompt(agent_name, agent_context, blackboard_context, [tool['function']['name'] for tool in available_tools] if available_tools else None)

    def execute_action(self, agent_name: str, action: Dict[str, Any], log_to_blackboards: bool = True) -> Dict[str, Any]:
        """
        Execute a trading game action with validation and retry hints.

        Args:
            agent_name: Name of the agent performing the action
            action: Dictionary containing action type and parameters
            log_to_blackboards: Whether to log the action to blackboards (default: True)

        Returns:
            Dictionary with execution result and status
        """
        if agent_name not in self.agents_dict:
            return {"status": "failed", "reason": f"Agent {agent_name} not found"}

        agent = self.agents_dict[agent_name]
        action_type = action.get("action")

        if not action_type:
            return {"status": "failed", "reason": "No action type specified"}

        # Handle different action types
        if action_type == "buy":
            result = self._execute_buy_action(agent, action)
        elif action_type == "trade":
            result = self._execute_trade_action(agent, action)
        elif action_type == "approve_trade":
            result = self._execute_approve_trade_action(agent, action)
        elif action_type == "reject_trade":
            result = self._execute_reject_trade_action(agent, action)
        elif action_type == "skip":
            result = self._execute_skip_action(agent, action)
        else:
            result = {"status": "failed", "reason": f"Unknown action type: {action_type}"}

        # Validate result format
        assert isinstance(result, dict), "Action result must be a dictionary"
        assert "status" in result, "Action result must contain 'status' field"
        assert result["status"] in ["success", "retry", "failed", "pending_response"], f"Invalid status: {result['status']}"

        # Log action to blackboards if enabled and action type is in config
        if log_to_blackboards and self.blackboard_manager and self.action_logging_config:
            action_type = action.get("action")
            # Only log if action type is in the list
            if action_type in self.action_logging_config:
                self._log_action_to_blackboards(agent_name, action, result)

        return result

    def _log_action_to_blackboards(self, agent_name: str, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Log an action to all blackboards that the agent belongs to.

        Args:
            agent_name: Name of the agent performing the action
            action: Dictionary containing action type and parameters
            result: Result from action execution
        """
        if not self.blackboard_manager:
            return

        # Find all blackboards the agent belongs to
        agent_blackboards = []
        for i, blackboard in enumerate(self.blackboard_manager.blackboards):
            if agent_name in blackboard.agents:
                agent_blackboards.append(i)

        if not agent_blackboards:
            print(f"DEBUG: Agent {agent_name} is not in any blackboards, skipping action logging")
            return

        # Create payload for the event
        payload = {
            "action_type": action.get("action"),
            "action_params": action.copy(),
            "result_status": result.get("status"),
            "details": result.copy()
        }

        # Post to all agent's blackboards
        for blackboard_id in agent_blackboards:
            try:
                self.blackboard_manager.post(
                    blackboard_id=blackboard_id,
                    agent=agent_name,
                    kind="action_executed",
                    payload=payload
                )
            except Exception as e:
                print(f"ERROR: Failed to log action to blackboard {blackboard_id}: {e}")

    def _execute_buy_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a buy action with budget validation."""
        item = action.get("item")
        if not item:
            return {"status": "failed", "reason": "No item specified for buy action"}

        # Check if item exists and is available
        if not self.store or not self.store.is_item_available(item):
            return {"status": "failed", "reason": f"Item {item} is not available"}

        # Get agent-specific cost
        agent_cost = self.store.get_item_cost_for_agent(item, agent) if self.store else None
        if agent_cost is None:
            return {"status": "failed", "reason": f"Could not determine cost for item {item}"}

        # Check budget
        if agent.state.budget < agent_cost:
            # Return retry hint for insufficient budget
            affordable_items = self.store.get_items_under_budget_for_agent(agent.state.budget, agent) if self.store else []
            return {
                "status": "retry",
                "reason": "insufficient_budget",
                "current_budget": agent.state.budget,
                "item_cost": agent_cost,
                "suggestions": [f"Try buying a cheaper item like: {', '.join([item['item'] for item in affordable_items[:3]])}"]
            }

        # Execute the purchase directly
        agent.state.budget = int(agent.state.budget - agent_cost)
        assert isinstance(agent.state.inventory, dict), f"Agent inventory must be dict, got {type(agent.state.inventory)}"
        agent.state.inventory[item] = agent.state.inventory.get(item, 0) + 1
        # Update utility using existing method
        agent.state.current_utility = agent.get_utility_for_items(list(agent.state.inventory.keys()))

        return {"status": "success", "result": {"item": item, "cost": agent_cost}}

    def _execute_trade_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade action by creating a trade proposal."""
        target = action.get("target")
        give_items = action.get("give_items", [])
        request_items = action.get("request_items", [])
        money_delta = action.get("money_delta", 0)

        if not target:
            return {"status": "failed", "reason": "No target agent specified for trade"}

        if target not in self.agents_dict:
            return {"status": "failed", "reason": f"Target agent {target} not found"}

        # Validate agent has items to give
        for item in give_items:
            assert isinstance(agent.state.inventory, dict), f"Agent inventory must be dict, got {type(agent.state.inventory)}"
            if agent.state.inventory.get(item, 0) <= 0:
                return {"status": "failed", "reason": f"Agent does not have {item} to trade"}

        # Validate money delta doesn't exceed budget
        if money_delta > 0 and agent.state.budget < money_delta:
            return {
                "status": "retry",
                "reason": "insufficient_budget_for_trade",
                "current_budget": agent.state.budget,
                "required_money": money_delta,
                "suggestions": ["Reduce the money amount in the trade", "Trade with a different agent", "Trade different items"]
            }

        # Create trade proposal
        try:
            assert self.trade_manager is not None, "Trade manager not available"
            trade_id = self.trade_manager.create_trade_proposal(
                proposer=agent.name,
                target=target,
                give_items=give_items,
                request_items=request_items,
                money_delta=money_delta
            )

            return {
                "status": "pending_response",
                "response_type": "trade",
                "target": target,
                "context": {
                    "trade_id": trade_id,
                    "give_items": give_items,
                    "request_items": request_items,
                    "money_delta": money_delta
                }
            }
        except Exception as e:
            return {"status": "failed", "reason": f"Error creating trade: {str(e)}"}

    def _execute_approve_trade_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an approve trade action."""
        if not self.trade_manager:
            return {"status": "failed", "reason": "Trade manager not available"}

        # Auto-find pending trade for this agent
        pending_trades = self.get_pending_trades_for_agent(agent.name)

        if not pending_trades:
            return {"status": "failed", "reason": "No pending trades to approve"}

        # Use first (and should be only) pending trade
        assert len(pending_trades) == 1, f"Agent {agent.name} has {len(pending_trades)} pending trades, expected exactly 1"
        trade_id = pending_trades[0]["trade_id"]

        # Get the trade record
        trade_record = self.trade_manager.active_trades[trade_id]

        # Get optional reason
        reason = action.get("reason", "Approved by target agent")

        # Approve the trade
        success = self.trade_manager.approve_trade(trade_id, reason)

        if success:
            # Execute the trade between the agents
            proposer_agent = self.agents_dict.get(trade_record.proposer)
            target_agent = self.agents_dict.get(trade_record.target)

            if proposer_agent and target_agent:
                execution_result = self.trade_manager.execute_trade(trade_id, proposer_agent, target_agent)
                if execution_result["success"]:
                    return {"status": "success", "result": {"trade_id": trade_id, "outcome": "approved_and_executed"}}
                else:
                    return {"status": "failed", "reason": f"Trade approved but execution failed: {execution_result.get('reason', 'Unknown error')}"}
            else:
                return {"status": "failed", "reason": "Could not find agents for trade execution"}
        else:
            return {"status": "failed", "reason": f"Failed to approve trade {trade_id}"}

    def _execute_reject_trade_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reject trade action."""
        if not self.trade_manager:
            return {"status": "failed", "reason": "Trade manager not available"}

        # Auto-find pending trade for this agent
        pending_trades = self.get_pending_trades_for_agent(agent.name)

        if not pending_trades:
            return {"status": "failed", "reason": "No pending trades to reject"}

        # Use first (and should be only) pending trade
        assert len(pending_trades) == 1, f"Agent {agent.name} has {len(pending_trades)} pending trades, expected exactly 1"
        trade_id = pending_trades[0]["trade_id"]

        # Reject the trade
        reason = action.get("reason", "Rejected by target agent")
        success = self.trade_manager.reject_trade(trade_id, reason)

        if success:
            return {"status": "success", "result": {"trade_id": trade_id, "outcome": "rejected", "reason": reason}}
        else:
            return {"status": "failed", "reason": f"Failed to reject trade {trade_id}"}

    def _execute_skip_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a skip action."""
        return {"status": "success", "result": {"action": "skip", "agent": agent.name}}


    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get agent information (used by MCP tools)."""
        if agent_name not in self.agents_dict:
            return {"success": False, "error": f"Agent {agent_name} not found"}

        agent = self.agents_dict[agent_name]
        return {
            "success": True,
            "name": agent.name,
            "budget": agent.state.budget,
            "inventory": agent.state.inventory.copy(),
            "current_utility": agent.state.current_utility,
            "blackboard_memberships": getattr(agent, 'blackboard_memberships', [])
        }

    def get_store_info(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get store information with optional agent-specific pricing."""
        assert self.store is not None, "Store not available"

        # Base store info
        result = {
            "success": True,
            "inventory": self.store.inventory.copy(),
            "total_items": len(self.store.item_prices)
        }

        # Add prices - agent-specific if agent_name provided, otherwise base prices
        if agent_name and agent_name in self.agents_dict:
            agent = self.agents_dict[agent_name]
            agent_prices = {}
            for item in self.store.item_prices.keys():
                if self.store.is_item_available(item):
                    agent_cost = self.store.get_item_cost_for_agent(item, agent)
                    if agent_cost is not None:
                        agent_prices[item] = agent_cost
            result["item_prices"] = agent_prices
        else:
            result["item_prices"] = self.store.item_prices.copy()

        return result

    def execute_tool(self, tool_name: str, agent_name: str, _params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading game specific tool calls."""
        if agent_name not in self.agents_dict:
            return {"success": False, "error": f"Agent {agent_name} not found"}

        agent = self.agents_dict[agent_name]

        if tool_name == "get_store_info":
            return self.get_store_info(agent_name)

        elif tool_name == "get_agent_info":
            return {
                "success": True,
                "name": agent.name,
                "budget": agent.state.budget,
                "inventory": agent.state.inventory.copy(),
                "current_utility": agent.state.current_utility,
                "blackboard_memberships": getattr(agent, 'blackboard_memberships', [])
            }

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def get_agents(self) -> List[str]:
        """Get list of agent names."""
        return [agent.name for agent in self.agents]

    def get_pending_trades_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get pending trade proposals that target the specified agent."""
        if not self.trade_manager:
            return []

        pending_trades = []
        for trade_id, trade_record in self.trade_manager.active_trades.items():
            if trade_record.target == agent_name and trade_record.status.value == "pending":
                pending_trades.append({
                    "trade_id": trade_id,
                    "proposer": trade_record.proposer,
                    "give_items": trade_record.give_items,
                    "request_items": trade_record.request_items,
                    "money_delta": trade_record.money_delta,
                    "timestamp": trade_record.timestamp
                })

        return pending_trades

    def build_agent_context(self, agent_name: str, phase: str, iteration: int,
                          _blackboard_contexts: Optional[Dict[str, str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Build comprehensive context for a trading game agent."""
        if agent_name not in self.agents_dict:
            raise ValueError(f"Agent {agent_name} not found")

        agent = self.agents_dict[agent_name]

        # Build base agent context
        agent_context = agent.get_full_context(phase, iteration, self.blackboard_manager)

        # Add additional context from kwargs (like planning_round)
        for key, value in kwargs.items():
            agent_context[key] = value

        # Add store information based on phase
        if phase == "planning":
            assert self.store is not None, "Store not available during planning phase"
            # During planning, provide all store items for strategic planning
            all_store_items = []
            for item in self.store.item_prices:
                if self.store.is_item_available(item):
                    agent_cost = self.store.get_item_cost_for_agent(item, agent)
                    agent_utility = agent.state.utilities.get(item, 0)
                    all_store_items.append({
                        "item": item,
                        "price": agent_cost,
                        "stock": self.store.inventory.get(item, 0),
                        "utility": agent_utility
                    })
            all_store_items.sort(key=lambda x: x["price"])
            agent_context["all_store_items"] = all_store_items

        elif phase == "execution":
            assert self.store is not None, "Store not available during execution phase"
            # During execution, provide only affordable items
            affordable_items = self.store.get_items_under_budget_for_agent(agent.state.budget, agent)
            agent_context["affordable_store_items"] = affordable_items[:10]

        # Add game structure information
        assert self.config is not None, "Config not available"
        agent_context["max_iterations"] = self.config.get("max_iterations", 3)
        agent_context["max_planning_rounds"] = self.config.get("max_planning_rounds", 5)

        return agent_context

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        """Get trading game specific tools for the given phase. This is different from Blackboard tools."""
        # Define base tools available in all phases
        base_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_store_info",
                    "description": "Get store inventory and agent-specific item pricing information",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_agent_info",
                    "description": "Get current agent state and information",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        # Add phase-specific tools
        if phase == "execution":
            execution_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "buy_item",
                        "description": "Buy an item from the store",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "item": {"type": "string", "description": "Name of item to buy"}
                            },
                            "required": ["item"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "propose_trade",
                        "description": "Propose a trade with another agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_agent": {"type": "string", "description": "Target agent name"},
                                "give_items": {"type": "array", "items": {"type": "string"}, "description": "Items to give"},
                                "request_items": {"type": "array", "items": {"type": "string"}, "description": "Items to request"},
                                "money_delta": {"type": "number", "description": "Money to give (positive) or receive (negative)"}
                            },
                            "required": ["target_agent", "give_items", "request_items"]
                        }
                    }
                }
            ]
            return base_tools + execution_tools

        elif phase == "trade_response":
            # Special phase for responding to trade proposals - no other actions allowed
            trade_response_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "approve_trade",
                        "description": "Approve a trade proposal made to you",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Optional reason for approving the trade"}
                            },
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "reject_trade",
                        "description": "Reject a trade proposal made to you",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Optional reason for rejecting the trade"}
                            },
                            "required": []
                        }
                    }
                }
            ]
            return base_tools + trade_response_tools

        return base_tools

    def get_supported_tools(self) -> Set[str]:
        """Return set of tool names this environment supports."""
        return {"get_agent_info", "get_store_info", "buy_item", "propose_trade", "approve_trade", "reject_trade"}

    def handle_tool_call(self, tool_name: str, agent_name: str, arguments: Dict[str, Any],
                        phase: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle tool calls by routing to execute_action or execute_tool as appropriate.

        Args:
            tool_name: Name of the tool to execute
            agent_name: Name of the agent calling the tool
            arguments: Parameters for the tool call
            phase: Current simulation phase (planning/execution) - not used for trading but accepted for compatibility
            iteration: Current iteration number - not used for trading but accepted for compatibility

        Returns:
            Dictionary with tool execution result
        """
        # Check if tool is supported
        if tool_name not in self.get_supported_tools():
            return {"error": f"Tool '{tool_name}' not supported by TradingGameEnvironment"}

        # Tools that are actually actions and modify state
        if tool_name == "buy_item":
            item = arguments.get("item")
            if not item:
                return {"error": "item is required for buy_item"}

            action = {"action": "buy", "item": item}
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True)

            # Convert result format for compatibility
            if env_result["status"] == "success":
                return {"success": True, "item": item}
            else:
                return {"success": False, "reason": env_result.get("reason", "")}

        elif tool_name == "propose_trade":
            target_agent = arguments.get("target_agent")
            if not target_agent:
                return {"error": "target_agent is required for propose_trade"}

            action = {
                "action": "trade",
                "target": target_agent,
                "give_items": arguments.get("give_items", []),
                "request_items": arguments.get("request_items", []),
                "money_delta": arguments.get("money_delta", 0)
            }
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True)

            # Convert result format
            if env_result["status"] == "pending_response":
                return {"success": True, "trade_id": env_result["context"]["trade_id"]}
            else:
                return {"success": False, "reason": env_result.get("reason", "")}

        elif tool_name == "approve_trade":
            action = {
                "action": "approve_trade",
                "reason": arguments.get("reason", "Approved by target agent")
            }
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True)

            # Convert result format
            if env_result["status"] == "success":
                return {"success": True, "message": "Trade approved and executed"}
            else:
                return {"error": env_result.get("reason", "")}

        elif tool_name == "reject_trade":
            action = {
                "action": "reject_trade",
                "reason": arguments.get("reason", "Rejected by target agent")
            }
            env_result = self.execute_action(agent_name, action, log_to_blackboards=True)

            # Convert result format
            if env_result["status"] == "success":
                return {"success": True, "message": "Trade rejected"}
            else:
                return {"error": env_result.get("reason", "")}

        # Information tools that don't modify state
        else:
            return self.execute_tool(tool_name, agent_name, arguments)

    def should_continue_simulation(self, iteration: int) -> bool:
        """Check if trading game should continue."""
        # Continue if we haven't reached max iterations
        assert self.config is not None, "Config not available"
        max_iterations = self.config.get("max_iterations", 3)
        return iteration <= max_iterations

    def log_state(self, iteration: int, phase: str) -> None:
        """Log trading game state."""
        # Log game state information
        if phase == "initialization":
            self._log_initial_state()
        elif phase == "iteration_end":
            self._log_iteration_state(iteration)

        # Update utility and budget tracking
        if phase == "iteration_end":
            for agent in self.agents:
                self.utility_history[agent.name].append(agent.state.current_utility)
                self.budget_history[agent.name].append(agent.state.budget)

            # Generate utility plot
            if self.plotter:
                plot_path = self.plotter.plot_utilities(
                    self.utility_history,
                    iteration,
                    budget_history=self.budget_history,
                    show=False
                )
                print(f"  📊 Utility plot saved: {plot_path}")

            # Calculate and track scores like other environments
            global_score = sum(agent.state.current_utility for agent in self.agents)
            local_scores = {agent.name: float(agent.state.current_utility) for agent in self.agents}

            # Track scores and generate JSON logs and plots
            self._track_scores(iteration, global_score, local_scores)

    def _track_scores(self, iteration: int, global_score: float, local_scores: Dict[str, float]) -> None:
        """Track scores and generate plots/logs similar to other environments."""
        import json
        from datetime import datetime
        from pathlib import Path

        # Update score histories
        self.global_score_history.append(global_score)
        for agent, score in local_scores.items():
            if agent in self.local_scores_history:
                self.local_scores_history[agent].append(score)

        # Create logs directory with seed subdirectory
        # Get tag_model subdirectory
        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = Path(__file__).parent.parent.parent / "logs" / "Trading" / tag_model / f"seed_{self.seed}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log scores to JSON
        score_entry = {
            "environment": "Trading",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "global_score": global_score,
            "local_scores": local_scores,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "metadata": {
                "total_agents": len(local_scores),
                "average_local_score": sum(local_scores.values()) / len(local_scores) if local_scores else 0,
                "total_utility": global_score,
                "total_budget": sum(agent.state.budget for agent in self.agents)
            }
        }

        score_file = log_dir / f"scores_iteration_{iteration}.json"
        with open(score_file, 'w') as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

        # Generate score plot using ScorePlotter
        if self.score_plotter:
            try:
                plot_path = self.score_plotter.plot_scores(
                    self.global_score_history,
                    self.local_scores_history,
                    iteration,
                    environment_name="Trading",
                    show=False
                )
            except Exception as e:
                print(f"Warning: Failed to generate score plot: {e}")

    def _log_initial_state(self):
        """Log initial trading game state."""
        # Post initial inventories to factor graph blackboards
        if self.blackboard_manager and self._factor_graph_blackboards:
            for bb_string_id in self._factor_graph_blackboards:
                blackboard = self.blackboard_manager.get_blackboard_by_string_id(bb_string_id)
                if blackboard:
                    for agent_name in blackboard.participants:
                        agent = next((a for a in self.agents if a.name == agent_name), None)
                        if agent:
                            # Create template-compliant payload
                            payload = {
                                "message": f"inventory - {agent.state.inventory}",
                                "phase": "initial"
                            }

                            self.blackboard_manager.post(
                                blackboard_id=int(bb_string_id),
                                agent=agent.name,
                                kind="inventory_update",
                                payload=payload
                            )

            # Log initial state of all factor graph blackboards
            for bb_string_id in self._factor_graph_blackboards:
                blackboard = self.blackboard_manager.get_blackboard_by_string_id(bb_string_id)
                if blackboard and self.blackboard_logger:
                    self.blackboard_logger.log_blackboard_state(
                        blackboard, 0, "initialization", "SYSTEM"
                    )

    def _log_iteration_state(self, _iteration: int):
        """Log iteration summary for trading game."""
        # Log trade statistics
        if self.trade_manager:
            trade_stats = self.trade_manager.get_trade_statistics()
            print(f"  Trade summary:")
            print(f"    Total trades: {trade_stats['total_trades']}")
            print(f"    Executed trades: {trade_stats['status_breakdown']['executed']}")
            print(f"    Success rate: {trade_stats['success_rate']:.1%}")

    def handle_post_turn_responses(self, agent_name: str, iteration: int, protocol) -> None:
        """
        Handle pending trade responses after an agent's turn.

        This is a trading-specific method that checks for pending trades
        and queries target agents for their responses.

        Args:
            agent_name: Agent who just completed their turn
            iteration: Current iteration
            protocol: CommunicationProtocol instance for accessing blackboard and LLM
        """
        # Get all agents to check for pending trades targeting them
        all_agents = self.get_agents()

        for target_agent in all_agents:
            if target_agent == agent_name:
                continue  # Skip the agent who just acted

            # Check for pending trades targeting this agent
            pending_trades = self.get_pending_trades_for_agent(target_agent)

            for trade in pending_trades:
                # Only handle trades proposed by the agent who just acted
                if trade['proposer'] == agent_name:
                    print(f"      --> Immediate trade response query for {target_agent} (trade {trade['trade_id']})")
                    self._query_agent_for_trade_response(target_agent, trade, iteration, protocol)

    def _query_agent_for_trade_response(self, target_agent: str, trade: Dict[str, Any],
                                       iteration: int, protocol) -> None:
        """
        Query the target agent for their response to a trade proposal.

        Args:
            target_agent: Agent who should respond to the trade
            trade: Trade proposal details
            iteration: Current iteration
            protocol: CommunicationProtocol instance for accessing blackboard and LLM
        """
        # Get blackboard contexts for the target agent
        blackboard_contexts = protocol.blackboard_manager.get_agent_blackboard_contexts(target_agent)

        # Build special context for trade response
        trade_context = {
            "agent_name": target_agent,
            "phase": "trade_response",
            "iteration": iteration,
            "pending_trade": trade,
            "trade_details": {
                "trade_id": trade["trade_id"],
                "proposer": trade["proposer"],
                "give_items": trade["give_items"],  # Items proposer will give to you
                "request_items": trade["request_items"],  # Items proposer wants from you
                "money_delta": trade["money_delta"],  # Money proposer will give (+) or receive (-)
                "timestamp": trade["timestamp"]
            }
        }

        # Get trade response tools (only approve/reject)
        environment_tools = self.get_tools("trade_response")
        blackboard_tools = protocol.blackboard_manager.get_tools("trade_response")
        tools = environment_tools + blackboard_tools

        # Get response from target agent using protocol's LLM method
        print(f"        Querying {target_agent} for trade response...")
        response_data = protocol._get_agent_response(
            agent_name=target_agent,
            agent_context=trade_context,
            blackboard_contexts=blackboard_contexts,
            available_tools=tools,
            phase="trade_response",
            iteration=iteration
        )

        if response_data:
            print(f"        {target_agent} responded to trade {trade['trade_id']}")
        else:
            print(f"        Failed to get response from {target_agent} for trade {trade['trade_id']}")

    def cleanup(self, iteration: int) -> None:
        """Clean up trading game resources."""
        print("Cleaning up trading game environment...")

        # Generate final plots if plotter available
        if self.plotter and self.utility_history:
            # Get final iteration from utility history length
            final_iteration = max(len(utils) for utils in self.utility_history.values() if utils)
            final_plot = self.plotter.plot_utilities(
                self.utility_history,
                iteration=final_iteration,
                budget_history=self.budget_history,
                show=False
            )
            print(f"Final utility plot saved: {final_plot}")

        # Log final blackboard state
        if self.blackboard_logger and self.blackboard_manager and self._factor_graph_blackboards:
            for bb_string_id in self._factor_graph_blackboards:
                blackboard = self.blackboard_manager.get_blackboard_by_string_id(bb_string_id)
                if blackboard:
                    self.blackboard_logger.log_blackboard_state(
                        blackboard, iteration, "final", "SIMULATION_END"
                    )
            # Get any log file from the blackboard_log_files dict
            log_files = getattr(self.blackboard_logger, 'blackboard_log_files', {})
            if log_files:
                first_log_file = next(iter(log_files.values()))
                print(f"Final blackboard state saved: {first_log_file}")
            else:
                print("Final blackboard state saved")

    def handle_pending_response(self, response_type: str, target_agent: str,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle responses for trade proposals.

        Args:
            response_type: Type of response needed (should be "trade")
            target_agent: Agent who should respond
            context: Context containing trade details

        Returns:
            Dictionary with response handling result
        """
        if response_type != "trade":
            return {
                "success": False,
                "reason": f"Unsupported response type: {response_type}"
            }

        if target_agent not in self.agents_dict:
            return {
                "success": False,
                "reason": f"Target agent {target_agent} not found"
            }

        trade_id = context.get("trade_id")
        if not trade_id:
            return {
                "success": False,
                "reason": "No trade_id provided in context"
            }

        if not self.trade_manager or trade_id not in self.trade_manager.active_trades:
            return {
                "success": False,
                "reason": f"Trade {trade_id} not found"
            }

        # Return success - the actual response will be handled by the protocol
        # when it calls the target agent with trade response tools
        return {
            "success": True,
            "trade_id": trade_id,
            "target_agent": target_agent,
            "context": context
        }

    def get_iteration_summary(self, _iteration: int) -> Dict[str, Any]:
        """Get trading game iteration summary."""
        summary = {}

        if self.trade_manager:
            trade_stats = self.trade_manager.get_trade_statistics()
            summary.update({
                "total_trades": trade_stats['total_trades'],
                "executed_trades": trade_stats['status_breakdown']['executed'],
                "success_rate": f"{trade_stats['success_rate']:.1%}"
            })

        return summary

    def get_final_summary(self) -> Dict[str, Any]:
        """Get final trading game summary."""
        # Calculate final utilities and budgets
        final_utilities = {}
        for agent in self.agents:
            final_utilities[agent.name] = {
                "final_utility": agent.state.current_utility,
                "final_budget": agent.state.budget,
                "final_inventory": agent.state.inventory
            }

        # Trade statistics
        trade_stats = {}
        if self.trade_manager:
            trade_stats = self.trade_manager.get_trade_statistics()

        # Store statistics
        store_summary = {}
        if self.store:
            store_summary = self.store.get_inventory_summary()

        summary = {
            "agents": final_utilities,
            "trade_statistics": trade_stats,
            "store_summary": store_summary,
            "metrics": {
                "total_utility": sum(agent.state.current_utility for agent in self.agents),
                "total_remaining_budget": sum(agent.state.budget for agent in self.agents)
            }
        }

        return summary