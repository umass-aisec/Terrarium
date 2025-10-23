import json
import time
import uuid
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict


class TradeStatus(Enum):
    """Enum for trade status values."""
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    EXECUTED = "executed"
    INVALID = "invalid"


@dataclass
class TradeRecord:
    """Record of a trade transaction for logging."""
    trade_id: str
    timestamp: float
    proposer: str
    target: str
    give_items: List[str]
    request_items: List[str]
    money_delta: int  # positive = proposer receives money, negative = proposer pays
    status: TradeStatus
    reason: str = ""
    execution_timestamp: Optional[float] = None


class TradeManager:
    """
    Manages trade proposals, validation, and execution between agents.
    
    Handles trade logging, validation logic, and maintains trade history
    in JSONL format for analysis.
    """
    
    def __init__(self, trade_log_file: str = "src/trading/trades.jsonl"):
        """
        Initialize the trade manager.
        
        Args:
            trade_log_file: Path to JSONL file for trade logging
        """
        self.trade_log_file = trade_log_file
        self.active_trades: Dict[str, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []
        
        # Initialize trade log file
        self._initialize_trade_log()
    
    def _initialize_trade_log(self):
        """Initialize the trade log file with header information."""
        try:
            # Create file if it doesn't exist
            with open(self.trade_log_file, 'a') as f:
                pass  # Just ensure file exists
        except Exception as e:
            print(f"Warning: Could not initialize trade log file: {e}")
    
    def _log_trade(self, trade_record: TradeRecord):
        """
        Log a trade record to the JSONL file.
        
        Args:
            trade_record: Trade record to log
        """
        try:
            with open(self.trade_log_file, 'a') as f:
                trade_data = asdict(trade_record)
                # Convert enum to string for JSON serialization
                trade_data['status'] = trade_record.status.value
                f.write(json.dumps(trade_data) + '\n')
        except Exception as e:
            print(f"Warning: Could not log trade: {e}")
    
    def create_trade_proposal(self, proposer: str, target: str,
                            give_items: List[str], request_items: List[str],
                            money_delta: int) -> str:
        """
        Create a new trade proposal.
        
        Args:
            proposer: Agent making the proposal
            target: Agent receiving the proposal
            give_items: Items proposer will give
            request_items: Items proposer wants to receive
            money_delta: Money change (positive = proposer receives, negative = proposer pays)
            
        Returns:
            Unique trade ID
        """
        trade_id = str(uuid.uuid4())
        
        trade_record = TradeRecord(
            trade_id=trade_id,
            timestamp=time.time(),
            proposer=proposer,
            target=target,
            give_items=give_items.copy(),
            request_items=request_items.copy(),
            money_delta=money_delta,
            status=TradeStatus.PENDING
        )
        
        self.active_trades[trade_id] = trade_record
        self._log_trade(trade_record)
        
        return trade_id
    
    def validate_trade_proposal(self, trade_id: str, proposer_agent, target_agent) -> Dict[str, Any]:
        """
        Validate a trade proposal against agent inventories and budgets.
        
        Args:
            trade_id: Trade ID to validate
            proposer_agent: Proposer agent object
            target_agent: Target agent object
            
        Returns:
            Dictionary with validation results
        """
        if trade_id not in self.active_trades:
            return {
                "valid": False,
                "reason": "Trade ID not found",
                "trade_id": trade_id
            }
        
        trade = self.active_trades[trade_id]
        
        # Check if proposer has the items they want to give
        proposer_has_items = proposer_agent.has_items(trade.give_items)
        if not proposer_has_items:
            missing_items = [item for item in trade.give_items if not proposer_agent.has_item(item)]
            return {
                "valid": False,
                "reason": f"Proposer {trade.proposer} missing items: {missing_items}",
                "trade_id": trade_id
            }
        
        # Check if target has the items proposer wants to receive
        target_has_items = target_agent.has_items(trade.request_items)
        if not target_has_items:
            missing_items = [item for item in trade.request_items if not target_agent.has_item(item)]
            return {
                "valid": False,
                "reason": f"Target {trade.target} missing items: {missing_items}",
                "trade_id": trade_id
            }
        
        # Check budget constraints
        # If money_delta is positive, proposer receives money (target pays)
        # If money_delta is negative, proposer pays money (target receives)
        if trade.money_delta > 0:
            # Target needs to pay proposer
            if not target_agent.can_afford(trade.money_delta):
                return {
                    "valid": False,
                    "reason": f"Target {trade.target} cannot afford to pay ${trade.money_delta}",
                    "trade_id": trade_id
                }
        elif trade.money_delta < 0:
            # Proposer needs to pay target
            if not proposer_agent.can_afford(abs(trade.money_delta)):
                return {
                    "valid": False,
                    "reason": f"Proposer {trade.proposer} cannot afford to pay ${abs(trade.money_delta)}",
                    "trade_id": trade_id
                }
        
        return {
            "valid": True,
            "reason": "Trade is valid",
            "trade_id": trade_id,
            "trade_details": trade
        }
    
    def approve_trade(self, trade_id: str, reason: str = "") -> bool:
        """
        Approve a trade proposal.
        
        Args:
            trade_id: Trade ID to approve
            reason: Reason for approval
            
        Returns:
            True if approval was successful
        """
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        trade.status = TradeStatus.APPROVED
        trade.reason = reason or "Trade approved"
        
        self._log_trade(trade)
        return True
    
    def reject_trade(self, trade_id: str, reason: str = "") -> bool:
        """
        Reject a trade proposal.
        
        Args:
            trade_id: Trade ID to reject
            reason: Reason for rejection
            
        Returns:
            True if rejection was successful
        """
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        trade.status = TradeStatus.REJECTED
        trade.reason = reason or "Trade rejected"
        
        self._log_trade(trade)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        self.trade_history.append(trade)
        
        return True
    
    def execute_trade(self, trade_id: str, proposer_agent, target_agent) -> Dict[str, Any]:
        """
        Execute an approved trade between agents.
        
        Args:
            trade_id: Trade ID to execute
            proposer_agent: Proposer agent object
            target_agent: Target agent object
            
        Returns:
            Dictionary with execution results
        """
        if trade_id not in self.active_trades:
            return {
                "success": False,
                "reason": "Trade ID not found",
                "trade_id": trade_id
            }
        
        trade = self.active_trades[trade_id]
        
        if trade.status != TradeStatus.APPROVED:
            return {
                "success": False,
                "reason": f"Trade not approved (status: {trade.status.value})",
                "trade_id": trade_id
            }
        
        # Final validation before execution
        validation = self.validate_trade_proposal(trade_id, proposer_agent, target_agent)
        if not validation["valid"]:
            trade.status = TradeStatus.INVALID
            trade.reason = validation["reason"]
            self._log_trade(trade)
            return {
                "success": False,
                "reason": f"Trade validation failed: {validation['reason']}",
                "trade_id": trade_id
            }
        
        try:
            # Execute the trade
            
            # Transfer items from proposer to target
            for item in trade.give_items:
                proposer_agent.remove_item_from_inventory(item)
                target_agent.add_item_to_inventory(item)
            
            # Transfer items from target to proposer
            for item in trade.request_items:
                target_agent.remove_item_from_inventory(item)
                proposer_agent.add_item_to_inventory(item)
            
            # Handle money transfer
            if trade.money_delta != 0:
                # Proposer's budget adjustment
                proposer_agent.adjust_budget(trade.money_delta)
                # Target's budget adjustment (opposite of proposer)
                target_agent.adjust_budget(-trade.money_delta)
            
            # Update trade status
            trade.status = TradeStatus.EXECUTED
            trade.execution_timestamp = time.time()
            trade.reason = "Trade executed successfully"
            
            self._log_trade(trade)
            
            # Move to history and remove from active
            del self.active_trades[trade_id]
            self.trade_history.append(trade)
            
            return {
                "success": True,
                "reason": "Trade executed successfully",
                "trade_id": trade_id,
                "trade_details": trade
            }
            
        except Exception as e:
            trade.status = TradeStatus.INVALID
            trade.reason = f"Execution error: {str(e)}"
            self._log_trade(trade)
            
            return {
                "success": False,
                "reason": f"Trade execution failed: {str(e)}",
                "trade_id": trade_id
            }
    
    def get_active_trades_for_agent(self, agent_name: str) -> List[TradeRecord]:
        """
        Get all active trades involving a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of active trade records involving the agent
        """
        agent_trades = []
        for trade in self.active_trades.values():
            if trade.proposer == agent_name or trade.target == agent_name:
                agent_trades.append(trade)
        return agent_trades
    
    def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """
        Get a trade record by ID.
        
        Args:
            trade_id: Trade ID to lookup
            
        Returns:
            Trade record or None if not found
        """
        # Check active trades first
        if trade_id in self.active_trades:
            return self.active_trades[trade_id]
        
        # Check history
        for trade in self.trade_history:
            if trade.trade_id == trade_id:
                return trade
        
        return None
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all trades.
        
        Returns:
            Dictionary containing trade statistics
        """
        all_trades = list(self.active_trades.values()) + self.trade_history
        
        status_counts = {}
        for status in TradeStatus:
            status_counts[status.value] = 0
        
        for trade in all_trades:
            status_counts[trade.status.value] += 1
        
        # Calculate success rate
        total_decided = status_counts["approved"] + status_counts["rejected"] + status_counts["executed"]
        success_rate = (status_counts["approved"] + status_counts["executed"]) / max(total_decided, 1)
        
        return {
            "total_trades": len(all_trades),
            "active_trades": len(self.active_trades),
            "completed_trades": len(self.trade_history),
            "status_breakdown": status_counts,
            "success_rate": success_rate,
            "execution_rate": status_counts["executed"] / max(len(all_trades), 1)
        }
    
    def clear_completed_trades(self):
        """Clear completed trades from memory (but keep in log file)."""
        self.trade_history.clear()
        
        # Remove non-pending trades from active trades
        active_pending = {
            trade_id: trade for trade_id, trade in self.active_trades.items()
            if trade.status == TradeStatus.PENDING
        }
        self.active_trades = active_pending
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade manager state to dictionary.
        
        Returns:
            Dictionary representation of trade manager state
        """
        return {
            "active_trades": {
                trade_id: {
                    "trade_id": trade.trade_id,
                    "proposer": trade.proposer,
                    "target": trade.target,
                    "give_items": trade.give_items,
                    "request_items": trade.request_items,
                    "money_delta": trade.money_delta,
                    "status": trade.status.value,
                    "timestamp": trade.timestamp
                }
                for trade_id, trade in self.active_trades.items()
            },
            "statistics": self.get_trade_statistics()
        }
    
    def __str__(self) -> str:
        """String representation of trade manager."""
        stats = self.get_trade_statistics()
        return (f"TradeManager: {stats['active_trades']} active, "
                f"{stats['completed_trades']} completed, "
                f"{stats['success_rate']:.1%} success rate")


def parse_trade_format(trade_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse trade format from agent response.
    
    Expected format: {"give_items": [...], "request_items": [...], "money_delta": 0, "target_agent": "name"}
    
    Args:
        trade_string: String containing trade proposal
        
    Returns:
        Parsed trade dictionary or None if parsing failed
    """
    try:
        # Try to extract JSON from the string
        import re
        
        # Look for JSON-like structure
        json_match = re.search(r'\{[^}]*\}', trade_string)
        if not json_match:
            return None
        
        json_str = json_match.group()
        trade_data = json.loads(json_str)
        
        # Validate required fields
        required_fields = ["give_items", "request_items", "money_delta", "target_agent"]
        for field in required_fields:
            if field not in trade_data:
                return None
        
        # Ensure lists are actually lists
        if not isinstance(trade_data["give_items"], list):
            return None
        if not isinstance(trade_data["request_items"], list):
            return None
        
        # Ensure money_delta is a number
        if not isinstance(trade_data["money_delta"], (int, float)):
            return None
        
        return trade_data
        
    except (json.JSONDecodeError, AttributeError):
        return None