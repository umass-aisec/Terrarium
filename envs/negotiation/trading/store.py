import json
import random
from typing import Dict, List, Optional, Any


class Store:
    """
    Store class for managing item inventory and purchases.
    
    Maintains a fixed inventory of items with limited quantities that do not refresh
    during a problem instance. Handles purchase validation and inventory tracking.
    """
    
    def __init__(self, items_file: str = "src/trading/items.json", min_stock: int = 1, max_stock: int = 3, seed: Optional[int] = None):
        """
        Initialize the store with items from JSON file.

        Args:
            items_file: Path to JSON file containing item prices
            min_stock: Minimum initial stock per item
            max_stock: Maximum initial stock per item
            seed: Random seed for reproducible inventory generation
        """
        self.items_file = items_file
        self.item_prices: Dict[str, int] = {}
        self.inventory: Dict[str, int] = {}
        self.seed = seed

        self._load_items()
        self._initialize_inventory(min_stock, max_stock)
    
    def _load_items(self):
        """Load item prices from JSON file."""
        try:
            with open(self.items_file, 'r') as f:
                self.item_prices = json.load(f)
            print(f"Loaded {len(self.item_prices)} items from {self.items_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Items file not found: {self.items_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in items file: {self.items_file}")
    
    def _initialize_inventory(self, min_stock: int, max_stock: int):
        """
        Initialize store inventory with random quantities.

        Args:
            min_stock: Minimum stock per item
            max_stock: Maximum stock per item
        """
        # Use seeded random generator for reproducible inventory
        rng = random.Random(self.seed) if self.seed is not None else random

        for item in self.item_prices:
            self.inventory[item] = rng.randint(min_stock, max_stock)
        
        print(f"Initialized store inventory with {sum(self.inventory.values())} total items")
    
    def get_available_items(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available items with their prices and stock.
        
        Returns:
            Dictionary mapping item names to their info (price, stock)
        """
        available = {}
        for item, price in self.item_prices.items():
            stock = self.inventory.get(item, 0)
            if stock > 0:
                available[item] = {
                    "price": price,
                    "stock": stock
                }
        return available
    
    def get_item_price(self, item: str) -> Optional[int]:
        """
        Get the base price of a specific item.
        
        Args:
            item: Item name
            
        Returns:
            Item price or None if item doesn't exist
        """
        return self.item_prices.get(item)
    
    def get_item_cost_for_agent(self, item: str, agent) -> Optional[float]:
        """
        Get the agent-specific cost for an item.
        
        Args:
            item: Item name
            agent: Agent object with agent-specific pricing
            
        Returns:
            Agent-specific item cost or None if item doesn't exist
        """
        if item not in self.item_prices:
            return None
        return agent.get_item_cost(item)
    
    def get_item_stock(self, item: str) -> int:
        """
        Get the current stock of a specific item.
        
        Args:
            item: Item name
            
        Returns:
            Current stock quantity
        """
        return self.inventory.get(item, 0)
    
    def is_item_available(self, item: str) -> bool:
        """
        Check if an item is available for purchase.
        
        Args:
            item: Item name
            
        Returns:
            True if item is in stock, False otherwise
        """
        return self.get_item_stock(item) > 0
    
    def can_afford_item(self, item: str, budget: int, agent=None) -> bool:
        """
        Check if a buyer can afford an item.
        
        Args:
            item: Item name
            budget: Available budget
            agent: Agent object for agent-specific pricing (optional)
            
        Returns:
            True if item can be afforded, False otherwise
        """
        if agent is not None:
            cost = self.get_item_cost_for_agent(item, agent)
        else:
            cost = self.get_item_price(item)
        return cost is not None and budget >= cost
    
    def validate_purchase(self, item: str, budget: int, agent=None) -> Dict[str, Any]:
        """
        Validate a purchase attempt.
        
        Args:
            item: Item name to purchase
            budget: Available budget
            agent: Agent object for agent-specific pricing (optional)
            
        Returns:
            Dictionary with validation result and details
        """
        # Get appropriate cost for this agent
        if agent is not None:
            cost = self.get_item_cost_for_agent(item, agent)
        else:
            cost = self.get_item_price(item)
        
        result = {
            "valid": False,
            "reason": "",
            "item": item,
            "price": cost,
            "stock": self.get_item_stock(item)
        }
        
        # Check if item exists
        if item not in self.item_prices:
            result["reason"] = f"Item '{item}' does not exist in store"
            return result
        
        # Check if item is in stock
        if not self.is_item_available(item):
            result["reason"] = f"Item '{item}' is out of stock"
            return result
        
        # Check if buyer can afford it
        if not self.can_afford_item(item, budget, agent):
            result["reason"] = f"Insufficient budget: need ${cost}, have ${budget}"
            return result
        
        result["valid"] = True
        result["reason"] = "Purchase is valid"
        return result
    
    def purchase_item(self, item: str, budget: int, agent=None) -> Dict[str, Any]:
        """
        Attempt to purchase an item from the store.
        
        Args:
            item: Item name to purchase
            budget: Available budget
            agent: Agent object for agent-specific pricing (optional)
            
        Returns:
            Dictionary with purchase result and updated inventory
        """
        # Validate the purchase
        validation = self.validate_purchase(item, budget, agent)
        
        if not validation["valid"]:
            return {
                "success": False,
                "reason": validation["reason"],
                "item": item,
                "cost": 0,
                "remaining_budget": budget
            }
        
        # Execute the purchase - use agent-specific cost
        if agent is not None:
            cost = self.get_item_cost_for_agent(item, agent)
        else:
            cost = self.get_item_price(item)
        self.inventory[item] -= 1
        
        return {
            "success": True,
            "reason": f"Successfully purchased {item}",
            "item": item,
            "cost": cost,
            "remaining_budget": budget - cost
        }
    
    def get_inventory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current store inventory.
        
        Returns:
            Dictionary containing inventory statistics
        """
        total_items = sum(self.inventory.values())
        total_value = sum(
            self.item_prices[item] * stock 
            for item, stock in self.inventory.items()
        )
        
        in_stock_items = {
            item: {"price": self.item_prices[item], "stock": stock}
            for item, stock in self.inventory.items()
            if stock > 0
        }
        
        out_of_stock_items = [
            item for item, stock in self.inventory.items()
            if stock == 0
        ]
        
        return {
            "total_items": total_items,
            "total_value": total_value,
            "unique_items": len(self.item_prices),
            "in_stock_count": len(in_stock_items),
            "out_of_stock_count": len(out_of_stock_items),
            "in_stock_items": in_stock_items,
            "out_of_stock_items": out_of_stock_items
        }
    
    def get_items_under_budget(self, budget: int) -> List[Dict[str, Any]]:
        """
        Get all items that can be purchased within a given budget.
        
        Args:
            budget: Available budget
            
        Returns:
            List of affordable items with their details
        """
        affordable_items = []
        
        for item, price in self.item_prices.items():
            if price <= budget and self.is_item_available(item):
                affordable_items.append({
                    "item": item,
                    "price": price,
                    "stock": self.inventory[item]
                })
        
        # Sort by price (ascending)
        affordable_items.sort(key=lambda x: x["price"])
        
        return affordable_items
    
    def get_items_under_budget_for_agent(self, budget: int, agent) -> List[Dict[str, Any]]:
        """
        Get all items that can be purchased within a given budget using agent-specific pricing.
        
        Args:
            budget: Available budget
            agent: Agent object for agent-specific pricing and utilities
            
        Returns:
            List of affordable items with agent-specific details
        """
        affordable_items = []
        
        for item, base_price in self.item_prices.items():
            if self.is_item_available(item):
                agent_cost = self.get_item_cost_for_agent(item, agent)
                if agent_cost is not None and budget >= agent_cost:
                    agent_utility = agent.state.utilities.get(item, 0)
                    affordable_items.append({
                        "item": item,
                        "price": agent_cost,  # Agent-specific transformed price
                        "stock": self.inventory[item],
                        "utility": agent_utility  # Agent's utility for this item
                    })
        
        # Sort by price (ascending)
        affordable_items.sort(key=lambda x: x["price"])
        
        return affordable_items
    
    def reset_inventory(self, min_stock: int = 1, max_stock: int = 3):
        """
        Reset store inventory to initial random state.

        Args:
            min_stock: Minimum stock per item
            max_stock: Maximum stock per item
        """
        self._initialize_inventory(min_stock, max_stock)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert store state to dictionary for serialization.
        
        Returns:
            Dictionary representation of store state
        """
        return {
            "item_prices": self.item_prices,
            "inventory": self.inventory,
            "summary": self.get_inventory_summary()
        }
    
    def __str__(self) -> str:
        """String representation of the store."""
        summary = self.get_inventory_summary()
        return (f"Store: {summary['in_stock_count']} items in stock, "
                f"${summary['total_value']} total value")
    
    def __repr__(self) -> str:
        """Detailed string representation of the store."""
        return f"Store(items={len(self.item_prices)}, total_stock={sum(self.inventory.values())})"