import json
import random
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Container for agent's private state information."""
    name: str
    budget: int # TODO Put this in environment.py
    inventory: Dict[str, int] # TODO Put this in environment.py
    utilities: Dict[str, int]  # item -> utility score (1-100) # TODO Put this in environment.py
    item_costs: Dict[str, float]  # agent-specific item costs # TODO Put this in environment.py
    current_utility: int = 0 # TODO Put this in environment.py
    blackboard_memberships: List[str] = field(default_factory=list)  # List of blackboard IDs this agent belongs to

class Agent:
    """
    Agent class representing a trading agent with explicitly defined private and public information.

    Each agent has a budget, inventory, utility function, and can participate
    in trading negotiations and actions through the blackboard system.
    """
    
    # Pool of realistic human names for agents
    AGENT_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George"
    ]
    
    def __init__(self, name: Optional[str] = None, budget: Optional[int] = None, 
                 initial_inventory: Optional[List[str]] = None,
                 utilities: Optional[Dict[str, int]] = None,
                 items_list: Optional[List[str]] = None):
        """
        Initialize an agent with private information.
        
        Args:
            name: Agent's name (random if None)
            budget: Starting budget (random if None)
            initial_inventory: Starting inventory (random if None)
            utilities: Utility scores for items (random if None)
            items_list: List of all possible items (for utility generation)
        """
        self.name = name or self._get_random_name()
        self.state = AgentState(
            name=self.name,
            budget=budget or self._generate_random_budget(),
            inventory=self._convert_inventory_to_dict(initial_inventory or []),
            utilities=utilities or {},
            item_costs={},
            current_utility=0
        )
        
        if items_list and not utilities:
            self.state.utilities = self._generate_random_utilities(items_list)
        
        # Calculate agent-specific costs based on utilities
        if items_list and self.state.utilities:
            self._calculate_item_costs(items_list)
        
        self._calculate_current_utility()
        
        # Agent memory and context
        self.conversation_history: List[Dict[str, Any]] = []
        self.pending_trades: List[Dict[str, Any]] = []
        self.last_action: Optional[Dict[str, Any]] = None
        
        # Save utilities to file for logging
        self._save_utilities()
    
    @classmethod
    def _get_random_name(cls) -> str:
        """Get a random name from the pool."""
        return random.choice(cls.AGENT_NAMES)
    
    def _generate_random_budget(self, min_budget: int = 100, max_budget: int = 1000) -> int:
        """
        Generate a random budget for the agent.
        
        Args:
            min_budget: Minimum budget
            max_budget: Maximum budget
            
        Returns:
            Random budget amount
        """
        return random.randint(min_budget, max_budget)
    
    def _generate_random_utilities(self, items_list: List[str]) -> Dict[str, int]:
        """
        Generate random utility scores for all items.
        
        Args:
            items_list: List of all available items
            
        Returns:
            Dictionary mapping items to utility scores (1-100)
        """
        utilities = {}
        for item in items_list:
            utilities[item] = random.randint(1, 100)
        return utilities

    def _convert_inventory_to_dict(self, inventory_list: List[str]) -> Dict[str, int]:
        """Convert a list of items to a dict with counts."""
        inventory_dict = {}
        for item in inventory_list:
            inventory_dict[item] = inventory_dict.get(item, 0) + 1
        return inventory_dict

    def _calculate_current_utility(self):
        """Calculate current total utility from inventory."""
        total_utility = 0
        for item, quantity in self.state.inventory.items():
            utility = self.state.utilities.get(item, 0) * quantity
            total_utility += utility
        self.state.current_utility = total_utility
    
    def _calculate_item_costs(self, items_list: List[str], base_price_dict: Optional[Dict[str, int]] = None):
        """
        Calculate agent-specific costs for items based on utilities.
        Higher utility items cost more for this agent.
        
        Args:
            items_list: List of all available items
            base_price_dict: Base prices for items (default uses standard prices)
        """
        from .store import Store  # Import here to avoid circular imports
        
        # Use provided base prices or default store prices
        if base_price_dict is None:
            # Load base prices from items config
            try:
                import json
                with open("src/trading/items.json", 'r') as f:
                    base_price_dict = json.load(f)
            except:
                # Fallback to simple pricing
                base_price_dict = {item: 10 for item in items_list}
        
        self.state.item_costs = {}
        for item in items_list:
            if item in self.state.utilities:
                utility = self.state.utilities[item]
                base_price = base_price_dict.get(item, 10)
                
                # Cost formula: higher utility = higher cost
                # Scale factor based on utility (utility/50 gives 0.5x to 2x multiplier)
                cost_multiplier = utility / 50.0
                agent_cost = base_price * cost_multiplier
                
                self.state.item_costs[item] = round(agent_cost, 2)
    
    def get_item_cost(self, item: str) -> float:
        """
        Get the agent-specific cost for an item.
        
        Args:
            item: Item name
            
        Returns:
            Cost of the item for this agent
        """
        return self.state.item_costs.get(item, 0.0)
    
    def _save_utilities(self):
        """Save agent utilities to JSON file for logging."""
        utilities_dir = "src/trading/utility"
        os.makedirs(utilities_dir, exist_ok=True)
        
        filename = os.path.join(utilities_dir, f"{self.name}_utilities.json")
        utility_data = {
            "agent_name": self.name,
            "utilities": self.state.utilities,
            "generation_timestamp": random.random()  # Simple timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(utility_data, f, indent=2)
    
    def get_public_info(self) -> Dict[str, Any]:
        """
        Get public information about the agent.
        
        Returns:
            Dictionary containing publicly visible information
        """
        return {
            "name": self.name,
            "inventory": self.state.inventory.copy()
        }
    
    def get_private_info(self) -> Dict[str, Any]:
        """
        Get private information about the agent.
        
        Returns:
            Dictionary containing private information
        """
        return {
            "name": self.name,
            "budget": self.state.budget,
            "inventory": self.state.inventory.copy(),
            "utilities": self.state.utilities.copy(),
            "current_utility": self.state.current_utility
        }
    
    def get_full_context(self, phase: str, iteration: int,
                        blackboard_manager=None) -> Dict[str, Any]:
        """
        Get complete agent context for LLM prompting.

        Args:
            phase: Current game phase
            iteration: Current iteration number
            blackboard_manager: BlackboardManager instance to get blackboard info

        Returns:
            Complete context dictionary
        """
        context = {
            "name": self.name,
            "budget": self.state.budget,
            "inventory": self.state.inventory.copy(),
            "utilities": self.state.utilities.copy(),
            "current_utility": self.state.current_utility,
            "phase": phase,
            "iteration": iteration,
            "pending_trades": self.pending_trades.copy(),
            "last_action": self.last_action,
            "blackboard_memberships": [],
            "blackboard_id_mapping": {}
        }
        
        # Add detailed blackboard membership information
        if blackboard_manager:
            for bb_id in self.state.blackboard_memberships:
                blackboard = blackboard_manager.get_blackboard_by_string_id(bb_id)
                if blackboard:
                    # Get recent events count for context
                    recent_events = len(blackboard.logs[-10:])  # Get last 10 events
                    
                    # Get initial context from first initialize event if available
                    initial_context = ""
                    for event in blackboard.logs:
                        if event.get('kind') == 'initialize' and event.get('payload', {}).get('message'):
                            initial_context = event['payload']['message']
                            break
                    
                    # bb_id is already a string integer ID
                    membership_info = {
                        "blackboard_id": bb_id,
                        "participants": list(blackboard.agents),
                        "initial_context": initial_context,
                        "recent_events": recent_events,
                        "total_events": len(blackboard.logs)
                    }
                    context["blackboard_memberships"].append(membership_info)

        return context
    
    def add_item_to_inventory(self, item: str):
        """
        Add an item to the agent's inventory.

        Args:
            item: Item to add
        """
        self.state.inventory[item] = self.state.inventory.get(item, 0) + 1
        self._calculate_current_utility()
    
    def remove_item_from_inventory(self, item: str) -> bool:
        """
        Remove an item from the agent's inventory.

        Args:
            item: Item to remove

        Returns:
            True if item was removed, False if not found
        """
        if item in self.state.inventory and self.state.inventory[item] > 0:
            self.state.inventory[item] -= 1
            if self.state.inventory[item] == 0:
                del self.state.inventory[item]
            self._calculate_current_utility()
            return True
        return False
    
    def has_item(self, item: str) -> bool:
        """
        Check if agent has a specific item.
        
        Args:
            item: Item to check
            
        Returns:
            True if agent has the item
        """
        return item in self.state.inventory
    
    def has_items(self, items: List[str]) -> bool:
        """
        Check if agent has all specified items.
        
        Args:
            items: List of items to check
            
        Returns:
            True if agent has all items
        """
        for item in items:
            if not self.has_item(item):
                return False
        return True
    
    def can_afford(self, amount: int) -> bool:
        """
        Check if agent can afford a specific amount.
        
        Args:
            amount: Amount to check
            
        Returns:
            True if agent can afford it
        """
        return self.state.budget >= amount
    
    def adjust_budget(self, amount: int) -> bool:
        """
        Adjust agent's budget by a specific amount.
        
        Args:
            amount: Amount to add (positive) or subtract (negative)
            
        Returns:
            True if adjustment was successful, False if would result in negative budget
        """
        new_budget = self.state.budget + amount
        if new_budget < 0:
            return False
        
        self.state.budget = new_budget
        return True
    
    def get_utility_for_item(self, item: str) -> int:
        """
        Get utility score for a specific item.
        
        Args:
            item: Item name
            
        Returns:
            Utility score (0 if item not in utilities)
        """
        return self.state.utilities.get(item, 0)
    
    def get_utility_for_items(self, items: List[str]) -> int:
        """
        Get total utility for a list of items.
        
        Args:
            items: List of items
            
        Returns:
            Total utility score
        """
        total = 0
        for item in items:
            total += self.get_utility_for_item(item)
        return total
    
    def evaluate_trade_utility(self, give_items: List[str], receive_items: List[str], 
                              money_delta: int) -> Dict[str, Any]:
        """
        Evaluate the utility impact of a potential trade.
        
        Args:
            give_items: Items to give away
            receive_items: Items to receive
            money_delta: Money change (positive = receive, negative = give)
            
        Returns:
            Dictionary with trade evaluation
        """
        current_utility = self.state.current_utility
        
        # Calculate utility without given items
        utility_lost = self.get_utility_for_items(give_items)
        
        # Calculate utility gained from received items
        utility_gained = self.get_utility_for_items(receive_items)
        
        # Net utility change
        net_utility_change = utility_gained - utility_lost
        
        # Check budget impact
        new_budget = self.state.budget + money_delta
        budget_feasible = new_budget >= 0
        
        # Check inventory feasibility
        inventory_feasible = self.has_items(give_items)
        
        return {
            "current_utility": current_utility,
            "utility_lost": utility_lost,
            "utility_gained": utility_gained,
            "net_utility_change": net_utility_change,
            "new_total_utility": current_utility + net_utility_change,
            "budget_change": money_delta,
            "new_budget": new_budget,
            "budget_feasible": budget_feasible,
            "inventory_feasible": inventory_feasible,
            "trade_feasible": budget_feasible and inventory_feasible,
            "trade_beneficial": net_utility_change > 0
        }
    
    def add_conversation_memory(self, event_type: str, content: str, context: Optional[Dict[str, Any]] = None):
        """
        Add an event to the agent's conversation memory.
        
        Args:
            event_type: Type of event (response, action, observation)
            content: Content of the event
            context: Additional context
        """
        memory_entry = {
            "type": event_type,
            "content": content,
            "context": context or {},
            "timestamp": random.random()  # Simple timestamp
        }
        self.conversation_history.append(memory_entry)
        
        # Keep only recent memories (max 50)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def add_pending_trade(self, trade_id: str, trade_details: Dict[str, Any]):
        """
        Add a pending trade to the agent's memory.
        
        Args:
            trade_id: Unique trade identifier
            trade_details: Trade details
        """
        self.pending_trades.append({
            "trade_id": trade_id,
            "details": trade_details,
            "timestamp": random.random()
        })
    
    def remove_pending_trade(self, trade_id: str):
        """
        Remove a pending trade from the agent's memory.
        
        Args:
            trade_id: Trade identifier to remove
        """
        self.pending_trades = [
            trade for trade in self.pending_trades 
            if trade["trade_id"] != trade_id
        ]
    
    def join_blackboard(self, blackboard_id: str):
        """
        Add agent to a blackboard's membership.
        
        Args:
            blackboard_id: ID of the blackboard to join
        """
        if blackboard_id not in self.state.blackboard_memberships:
            self.state.blackboard_memberships.append(blackboard_id)
    
    def leave_blackboard(self, blackboard_id: str):
        """
        Remove agent from a blackboard's membership.
        
        Args:
            blackboard_id: ID of the blackboard to leave
        """
        if blackboard_id in self.state.blackboard_memberships:
            self.state.blackboard_memberships.remove(blackboard_id)
    
    def is_member_of_blackboard(self, blackboard_id: str) -> bool:
        """
        Check if agent is a member of a specific blackboard.
        
        Args:
            blackboard_id: ID of the blackboard to check
            
        Returns:
            True if agent is a member, False otherwise
        """
        return blackboard_id in self.state.blackboard_memberships
    
    def get_blackboard_memberships(self) -> List[str]:
        """
        Get list of blackboard IDs this agent belongs to.
        
        Returns:
            List of blackboard IDs
        """
        return self.state.blackboard_memberships.copy()
    
    def get_blackboard_count(self) -> int:
        """
        Get number of blackboards this agent belongs to.
        
        Returns:
            Number of blackboard memberships
        """
        return len(self.state.blackboard_memberships)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary for serialization.
        
        Returns:
            Dictionary representation of agent
        """
        return {
            "name": self.name,
            "state": {
                "budget": self.state.budget,
                "inventory": self.state.inventory,
                "utilities": self.state.utilities,
                "current_utility": self.state.current_utility
            },
            "conversation_history": self.conversation_history,
            "pending_trades": self.pending_trades,
            "last_action": self.last_action
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"Agent {self.name}: ${self.state.budget} budget, "
                f"{len(self.state.inventory)} items, "
                f"{self.state.current_utility} utility")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Agent(name='{self.name}', budget={self.state.budget}, utility={self.state.current_utility})"


def generate_shared_base_utilities(items_list: List[str], seed: Optional[int] = None) -> List[int]:
    """
    Generate a single set of base utility values to be shared across all agents.

    Args:
        items_list: List of all available items
        seed: Random seed for reproducible utility generation

    Returns:
        List of utility values (1-100) in same order as items_list
    """
    # Use seeded random generator for reproducible utilities
    rng = random.Random(seed) if seed is not None else random
    return [rng.randint(1, 100) for _ in items_list]


def create_agent_utilities_from_shared(items_list: List[str], base_utilities: List[int], agent_index: int) -> Dict[str, int]:
    """
    Create agent-specific utility mapping by shuffling the shared base utilities.
    
    Args:
        items_list: List of all available items
        base_utilities: Shared base utility values
        agent_index: Index to create deterministic but different shuffling
        
    Returns:
        Dictionary mapping items to utility scores for this agent
    """
    if len(items_list) != len(base_utilities):
        raise ValueError("items_list and base_utilities must have same length")
    
    # Create a deterministic shuffle based on agent index
    random.seed(42 + agent_index)  # Use fixed seed + agent index for reproducibility
    shuffled_utilities = base_utilities.copy()
    random.shuffle(shuffled_utilities)
    random.seed()  # Reset random seed
    
    # Map items to shuffled utilities
    utilities = {}
    for item, utility in zip(items_list, shuffled_utilities):
        utilities[item] = utility
    
    return utilities


def select_initial_items_for_equal_utility(items_list: List[str], utilities: Dict[str, int],
                                          target_utility: int, max_items: int = 3, seed: Optional[int] = None) -> List[str]:
    """
    Select initial items to achieve a target total utility value.
    
    Args:
        items_list: List of all available items
        utilities: Agent's utility mapping
        target_utility: Target total utility to achieve
        max_items: Maximum number of items to select
        
    Returns:
        List of selected items
    """
    # Sort items by utility (descending) for greedy selection
    sorted_items = sorted(items_list, key=lambda x: utilities.get(x, 0), reverse=True)
    
    selected_items = []
    current_utility = 0
    
    # Greedy approach: select highest utility items first
    for item in sorted_items:
        if len(selected_items) >= max_items:
            break
        
        item_utility = utilities.get(item, 0)
        if current_utility + item_utility <= target_utility:
            selected_items.append(item)
            current_utility += item_utility
            
            if current_utility == target_utility:
                break
    
    # Log if we didn't reach the target utility
    if current_utility != target_utility:
        print(f"Warning: Could not reach target utility {target_utility}, achieved {current_utility}")
    
    return selected_items


def create_random_agent(items_list: List[str],
                       min_budget: int = 100, max_budget: int = 1000,
                       initial_items: int = 3,
                       name: Optional[str] = None,
                       base_utilities: Optional[List[int]] = None,
                       agent_index: int = 0,
                       target_utility: Optional[int] = None,
                       seed: Optional[int] = None) -> Agent:
    """
    Create a random agent with specified parameters.

    Args:
        items_list: List of all available items
        min_budget: Minimum budget
        max_budget: Maximum budget
        initial_items: Number of initial items to give agent
        name: Specific name for agent (random if None)
        base_utilities: Shared base utility values for fair distribution
        agent_index: Agent index for deterministic utility shuffling
        target_utility: Target starting utility for equal initial positions
        seed: Random seed for reproducible agent generation

    Returns:
        Initialized Agent instance
    """
    # Use seeded random generator for reproducible agent creation
    rng = random.Random(seed) if seed is not None else random

    budget = rng.randint(min_budget, max_budget)

    # Generate utilities using shared base or random
    if base_utilities is not None:
        utilities = create_agent_utilities_from_shared(items_list, base_utilities, agent_index)
    else:
        utilities = {}
        for item in items_list:
            utilities[item] = rng.randint(1, 100)

    # Select initial inventory to achieve equal starting utility
    if target_utility is not None:
        inventory = select_initial_items_for_equal_utility(items_list, utilities, target_utility, initial_items, seed=seed)
    else:
        inventory = rng.sample(items_list, min(initial_items, len(items_list)))

    return Agent(
        name=name,
        budget=budget,
        initial_inventory=inventory,
        utilities=utilities,
        items_list=items_list
    )


def create_agent_pool(items_list: List[str], num_agents: int = 4,
                     min_budget: int = 100, max_budget: int = 1000,
                     initial_items: int = 3, seed: Optional[int] = None) -> List[Agent]:
    """
    Create a pool of agents for the simulation with shared utility structure.
    All agents have same utility values but mapped to different items (name swapping).

    Args:
        items_list: List of all available items
        num_agents: Number of agents to create
        min_budget: Minimum budget per agent
        max_budget: Maximum budget per agent
        initial_items: Number of initial items per agent
        seed: Random seed for reproducible agent generation

    Returns:
        List of initialized Agent instances
    """
    # Use seeded random generator for reproducible agent creation
    rng = random.Random(seed) if seed is not None else random

    agents = []
    used_names = set()

    # Generate shared base utilities for fair distribution
    base_utilities = generate_shared_base_utilities(items_list, seed=seed)
    
    # Calculate target utility for equal starting positions
    # Use the top N utility values (where N = initial_items) from base_utilities
    sorted_base_utilities = sorted(base_utilities, reverse=True)
    target_utility = sum(sorted_base_utilities[:initial_items])
    
    # Select equal budget for all agents using seeded random
    equal_budget = rng.randint(min_budget, max_budget)

    for i in range(num_agents):
        # Ensure unique names
        available_names = [name for name in Agent.AGENT_NAMES if name not in used_names]
        if not available_names:
            name = f"Agent_{i+1}"
        else:
            name = rng.choice(available_names)

        used_names.add(name)

        agent = create_random_agent(
            items_list=items_list,
            min_budget=equal_budget,
            max_budget=equal_budget,
            initial_items=initial_items,
            name=name,
            base_utilities=base_utilities,
            agent_index=i,
            target_utility=target_utility,
            seed=seed
        )
        
        agents.append(agent)
    
    return agents


def create_agents_with_names(items_list: List[str], agent_names: List[str],
                           min_budget: int = 100, max_budget: int = 1000,
                           initial_items: int = 3, seed: Optional[int] = None) -> List[Agent]:
    """
    Create agents with specific names for factor graph initialization.

    Args:
        items_list: List of all available items
        agent_names: List of specific agent names to create
        min_budget: Minimum budget per agent
        max_budget: Maximum budget per agent
        initial_items: Number of initial items per agent
        seed: Random seed for reproducible agent generation

    Returns:
        List of initialized Agent instances with specified names
    """
    # Use seeded random generator for reproducible agent creation
    rng = random.Random(seed) if seed is not None else random

    agents = []

    # Generate shared base utilities for fair distribution
    base_utilities = generate_shared_base_utilities(items_list, seed=seed)

    # Calculate target utility for equal starting positions
    sorted_base_utilities = sorted(base_utilities, reverse=True)
    target_utility = sum(sorted_base_utilities[:initial_items])

    # Select equal budget for all agents using seeded random
    equal_budget = rng.randint(min_budget, max_budget)

    for i, name in enumerate(agent_names):
        agent = create_random_agent(
            items_list=items_list,
            min_budget=equal_budget,
            max_budget=equal_budget,
            initial_items=initial_items,
            name=name,
            base_utilities=base_utilities,
            agent_index=i,
            target_utility=target_utility,
            seed=seed
        )
        
        agents.append(agent)
    
    return agents