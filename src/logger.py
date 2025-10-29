import time
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict
from .blackboard import Event, Blackboard
from .utils import get_tag_model_subdir, get_run_timestamp, build_log_dir


def _resolve_run_timestamp(config: Optional[Dict[str, Any]], run_timestamp: Optional[str]) -> Optional[str]:
    if run_timestamp:
        return run_timestamp
    if config:
        return get_run_timestamp(config)
    return None


class AttackLogger:
    """Structured logger for attack executions and outcomes."""

    def __init__(
        self,
        environment_name: str,
        seed: int,
        config: Dict[str, Any],
        run_timestamp: Optional[str] = None,
    ):
        self.environment_name = environment_name
        self.seed = seed
        self.config = config
        self.tag_model = get_tag_model_subdir(config or {})
        self.run_timestamp = _resolve_run_timestamp(config, run_timestamp)
        self.log_dir = build_log_dir(
            self.environment_name,
            self.tag_model,
            self.seed,
            self.run_timestamp,
        )
        self.events_path = self.log_dir / "attack_events.jsonl"
        self.text_path = self.log_dir / "attack_events.log"
        self.summary_path = self.log_dir / "attack_summary.json"
        self.stats: Dict[str, Dict[str, int]] = {}
        self.reset_log()

    def reset_log(self) -> None:
        """Reset attack logs for a fresh simulation run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")
        self.text_path.write_text("", encoding="utf-8")
        self.stats = {}
        self._flush_summary()
        self._snapshot_config()

    def _snapshot_config(self) -> None:
        """Persist a copy of the source config alongside the run logs."""
        config_path = self.config.get("_config_path")
        if not config_path:
            return
        source = Path(config_path)
        if not source.exists():
            return
        destination = self.log_dir / source.name
        try:
            if source.resolve() == destination.resolve():
                return
        except OSError:
            # Fall back to string comparison if resolution fails
            if source == destination:
                return
        try:
            shutil.copy2(source, destination)
        except OSError:
            # Silently continue if we cannot copy; logging should not fail the run
            pass

    def log_agent_attack(
        self,
        attack_type: str,
        attacker: str,
        *,
        target: Optional[Any] = None,
        target_type: Optional[str] = None,
        success: bool = True,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        round_num: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "category": "agent",
            "attack_type": attack_type,
            "attacker": attacker,
            "target": target,
            "target_type": target_type,
            "success": success,
            "phase": phase,
            "iteration": iteration,
            "round": round_num,
            "metadata": metadata or {},
        }
        self._log_event(entry)

    def log_protocol_attack(
        self,
        attack_type: str,
        *,
        attacker: str,
        target: Optional[Any],
        success: bool,
        trigger: Optional[str],
        iteration: Optional[int],
        phase: Optional[str],
        round_num: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "category": "protocol",
            "attack_type": attack_type,
            "attacker": attacker,
            "target": target,
            "target_type": "blackboard" if target is not None else None,
            "success": success,
            "phase": phase,
            "iteration": iteration,
            "round": round_num,
            "trigger": trigger,
            "metadata": metadata or {},
        }
        self._log_event(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_event(self, entry: Dict[str, Any]) -> None:
        timestamp = datetime.now().isoformat()
        entry["timestamp"] = timestamp

        # JSON line log
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Human-readable log line
        text_line = self._format_text_entry(entry)
        with self.text_path.open("a", encoding="utf-8") as f:
            f.write(text_line + "\n")

        self._update_stats(entry.get("attack_type"), bool(entry.get("success")))

    def _format_text_entry(self, entry: Dict[str, Any], metadata_limit: int = 4) -> str:
        meta_preview = self._format_metadata(entry.get("metadata") or {}, metadata_limit)
        parts = [
            f"[{entry['timestamp']}]",
            f"cat={entry.get('category')}",
            f"attack={entry.get('attack_type')}",
            f"attacker={entry.get('attacker','-')}",
            f"target={entry.get('target','-')}",
            f"phase={entry.get('phase','unknown')}",
            f"iter={entry.get('iteration')}",
            f"round={entry.get('round')}",
            f"success={entry.get('success')}",
        ]
        if trigger := entry.get("trigger"):
            parts.append(f"trigger={trigger}")
        if meta_preview:
            parts.append(f"metrics={meta_preview}")
        return " ".join(parts)

    @staticmethod
    def _format_metadata(metadata: Dict[str, Any], limit: int) -> str:
        if not metadata:
            return ""
        items = []
        for idx, (key, value) in enumerate(metadata.items()):
            if idx >= limit:
                items.append("…")
                break
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            items.append(f"{key}={value}")
        return ", ".join(items)

    def _update_stats(self, attack_type: Optional[str], success: bool) -> None:
        if not attack_type:
            return
        stats = self.stats.setdefault(attack_type, {"success": 0, "failure": 0})
        stats["success" if success else "failure"] += 1
        self._flush_summary()

    def _flush_summary(self) -> None:
        payload = {
            "environment": self.environment_name,
            "seed": self.seed,
            "run_timestamp": self.run_timestamp,
            "updated_at": datetime.now().isoformat(),
            "attack_counts": self.stats,
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)



class BlackboardLogger:
    """
    Logger specifically for tracking the complete blackboard state.

    Saves the entire blackboard after each agent interaction, providing
    a complete view of all agent communications and their evolution.
    Now supports multiple blackboards with separate log files.
    """

    def __init__(self, config: Dict[str, Any], run_timestamp: Optional[str] = None):
        """
        Initialize the blackboard logger.

        Args:
            environment_name: Name of the environment (e.g., "Trading", "SmartGrid")
            seed: Simulation seed for unique log directories
            config: Full configuration dictionary containing all simulation settings
        """
        self.session_start = time.time()
        self.config = config
        self.environment_name = self.config["environment"]["name"]
        self.seed = self.config["environment"]["rng_seed"]
        self.tag_model = get_tag_model_subdir(self.config)
        self.run_timestamp = _resolve_run_timestamp(self.config, run_timestamp)
        self.log_root = build_log_dir(
            self.environment_name,
            self.tag_model,
            self.seed,
            self.run_timestamp,
        )
        self.log_root.mkdir(parents=True, exist_ok=True)

        # Track log files for each blackboard
        self.blackboard_log_files: Dict[str, str] = {}

    def clear_blackboard_logs(self):
        """
        Clear all blackboard log files and reset tracking.
        Called at the start of each simulation.
        """
        if self.log_root.exists():
            shutil.rmtree(self.log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.blackboard_log_files.clear()
        print(f"Cleared blackboard logs directory: {self.log_root}")

    def _initialize_log(self, log_file: str, blackboard_id: str = ""):
        """Initialize log file with header."""
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            if blackboard_id:
                f.write(f"BLACKBOARD STATE LOG - {blackboard_id.upper()}\n")
            else:
                f.write("BLACKBOARD STATE LOG\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

            # Only show trade-specific format notes for Trading environment
            if self.environment_name == "Trading":
                f.write("FORMAT NOTES:\n")
                f.write("- Trade actions show: Trade: agent1 → agent2\n")
                f.write("- Money Exchange: +amount (received) or -amount (paid) from proposer's perspective\n")
                f.write("- Items: -[items given] +[items received] from proposer's perspective\n")

            f.write("=" * 80 + "\n\n")
    
    def _get_log_file_for_blackboard(self, blackboard_id: str) -> str:
        """
        Get the appropriate log file path for a blackboard.

        Args:
            blackboard_id: ID of the blackboard

        Returns:
            Path to the log file for this blackboard
        """
        if blackboard_id not in self.blackboard_log_files:
            log_file = self.log_root / f"blackboard_{blackboard_id}.txt"
            self.blackboard_log_files[blackboard_id] = str(log_file)

            if not log_file.exists():
                self._initialize_log(str(log_file), blackboard_id)

        return self.blackboard_log_files[blackboard_id]

    def _get_event_property(self, event, property_name: str, default=None):
        """
        Safely get property from event, handling both dict and Event object types.

        Args:
            event: Event object (dict or dataclass)
            property_name: Name of the property to get
            default: Default value if property not found

        Returns:
            Property value or default
        """
        if isinstance(event, dict):
            return event.get(property_name, default)
        else:
            return getattr(event, property_name, default)

    def log_blackboard_state(self, blackboard: Blackboard, iteration: int = 0,
                           phase: str = "unknown", agent_name: str = "", round_num: Optional[int] = None):
        """
        Log the complete current state of the blackboard.

        Args:
            blackboard: The blackboard instance to log
            iteration: Current iteration number
            phase: Current phase (planning/execution)
            agent_name: Name of agent that just acted (optional)
            round_num: Round number within the phase (optional)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get the appropriate log file for this blackboard
        log_file = self._get_log_file_for_blackboard(blackboard.blackboard_id)
        
        # Get all events from the blackboard
        all_events = list(blackboard.logs)
        
        # Calculate statistics
        event_stats = self._calculate_event_stats(all_events)
        
        # Build the log content with permanent header
        log_content = "=" * 80 + "\n"
        if blackboard.blackboard_id:
            log_content += f"BLACKBOARD STATE LOG - {blackboard.blackboard_id.upper()}\n"
        else:
            log_content += "BLACKBOARD STATE LOG\n"
        log_content += f"Session started: {datetime.fromtimestamp(self.session_start).strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += "=" * 80 + "\n"

        # Only show trade-specific format notes for Trading environment
        if self.environment_name == "Trading":
            log_content += "FORMAT NOTES:\n"
            log_content += "- Trade actions show: Trade: agent1 → agent2\n"
            log_content += "- Money Exchange: +amount (received) or -amount (paid) from proposer's perspective\n"
            log_content += "- Items: -[items given] +[items received] from proposer's perspective\n"

        # Add blackboard specific information
        log_content += f"\nBLACKBOARD INFO:\n"
        log_content += f"- ID: {blackboard.blackboard_id}\n"
        log_content += f"- Participants: {', '.join(blackboard.participants)}\n"
        if blackboard.initial_context:
            log_content += f"- Context: {blackboard.initial_context}\n"
        
        log_content += "=" * 80 + "\n\n"
        
        log_content += "=" * 80 + "\n"
        log_content += f"BLACKBOARD STATE UPDATE\n"
        log_content += f"Updated: {timestamp}\n"
        log_content += f"Iteration: {iteration} | Phase: {phase}"
        if agent_name:
            log_content += f" | Last Agent: {agent_name}"
        log_content += "\n"
        log_content += f"Total Events: {len(all_events)}\n"
        log_content += "=" * 80 + "\n\n"
        
        # Add event summary
        log_content += "EVENT SUMMARY:\n"
        log_content += f"- Total Events: {len(all_events)}\n"
        
        if event_stats['by_type']:
            type_summary = ", ".join([f"{k}({v})" for k, v in event_stats['by_type'].items()])
            log_content += f"- By Type: {type_summary}\n"
        
        if event_stats['by_agent']:
            agent_summary = ", ".join([f"{k}({v})" for k, v in event_stats['by_agent'].items()])
            log_content += f"- By Agent: {agent_summary}\n"
        
        log_content += "\n" + "-" * 80 + "\n"
        log_content += "FULL EVENT LOG:\n\n"
        
        # Add all events in chronological order
        for i, event in enumerate(all_events, 1):
            try:
                formatted_event = self._format_event(event, i, iteration, phase)
                log_content += formatted_event
                log_content += "\n"
            except Exception as e:
                log_content += f"[ERROR FORMATTING EVENT {i}: {e}]\n"

        log_content += "\n" + "=" * 80 + "\n\n"

        # Write to file (overwrite completely)
        with open(log_file, 'w') as f:
            f.write(log_content)
    
    def log_blackboard_creation(self, blackboard_id: str, participants: List[str], 
                              initial_context: str = "", inviter: str = ""):
        """
        Log the creation of a new blackboard.
        
        Args:
            blackboard_id: ID of the new blackboard
            participants: List of agents participating in the blackboard
            initial_context: Purpose/context of the blackboard
            inviter: Agent who created the blackboard (if applicable)
        """
        log_file = self._get_log_file_for_blackboard(blackboard_id)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        creation_log = f"\n[{timestamp}] BLACKBOARD CREATED\n"
        creation_log += f"ID: {blackboard_id}\n"
        creation_log += f"Participants: {', '.join(participants)}\n"
        if inviter:
            creation_log += f"Created by: {inviter}\n"
        if initial_context:
            creation_log += f"Context: {initial_context}\n"
        creation_log += "=" * 50 + "\n\n"
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(creation_log)
    
    def log_blackboard_join(self, blackboard_id: str, agent_name: str):
        """
        Log an agent joining a blackboard.
        
        Args:
            blackboard_id: ID of the blackboard
            agent_name: Name of the agent joining
        """
        log_file = self._get_log_file_for_blackboard(blackboard_id)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        join_log = f"[{timestamp}] AGENT JOINED: {agent_name}\n\n"
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(join_log)
    
    def log_blackboard_exit(self, blackboard_id: str, agent_name: str):
        """
        Log an agent exiting a blackboard.
        
        Args:
            blackboard_id: ID of the blackboard
            agent_name: Name of the agent exiting
        """
        log_file = self._get_log_file_for_blackboard(blackboard_id)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        exit_log = f"[{timestamp}] AGENT EXITED: {agent_name}\n\n"
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(exit_log)
    
    def _calculate_event_stats(self, events: List) -> Dict[str, Dict[str, int]]:
        """Calculate statistics about events."""
        stats = {
            'by_type': {},
            'by_agent': {}
        }

        for event in events:
            # Count by type
            event_type = self._get_event_property(event, 'kind', 'unknown')
            stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1

            # Count by agent
            agent = self._get_event_property(event, 'agent', 'unknown')
            stats['by_agent'][agent] = stats['by_agent'].get(agent, 0) + 1

        return stats
    
    def _format_event(self, event, event_num: int, iteration: int = 0,
                     phase: str = "unknown") -> str:
        """Format a single event for logging."""
        event_ts = self._get_event_property(event, 'ts', time.time())
        timestamp = time.strftime("%H:%M:%S", time.localtime(event_ts))

        # Build event header with iteration and phase information
        # Use event's stored phase if available, otherwise fall back to current phase parameter
        event_payload = self._get_event_property(event, 'payload', {})
        event_phase = event_payload.get('phase', phase) if isinstance(event_payload, dict) else phase
        phase_display = f"{event_phase.title()}" if event_phase != "unknown" else "Unknown"

        # Use event's stored iteration if available, otherwise fall back to current iteration parameter
        event_iteration = event_payload.get('iteration', iteration) if isinstance(event_payload, dict) else iteration

        # Create header with iteration and round info
        header_parts = [f"Event #{event_num}"]
        if event_iteration > 0:
            header_parts.append(f"Iteration: {event_iteration}")

        # Use event's stored round if available, otherwise fall back to current round parameter
        event_round = event_payload.get('round', None) if isinstance(event_payload, dict) else None
        if event_round is not None:
            header_parts.append(f"Round: {event_round}")

        event_agent = self._get_event_property(event, 'agent', 'unknown')
        event_kind = self._get_event_property(event, 'kind', 'unknown')
        formatted = f"[{', '.join(header_parts)}] [{timestamp}] [{phase_display}] {event_agent} ({event_kind})"

        # Add references if present
        event_refs = self._get_event_property(event, 'refs', [])
        if event_refs:
            ref_list = ", ".join(event_refs[:3])
            if len(event_refs) > 3:
                ref_list += "..."
            formatted += f" [refs: {ref_list}]"
        
        # Add payload content based on event type
        payload = self._get_event_property(event, 'payload', {})

        assert isinstance(payload, dict), f"Payload is not a dict: {payload}"

        if event_kind == "message":
            if "message" in payload:
                formatted += f"  Message: \"{payload['message']}\"\n"

        elif event_kind in ["proposal", "counter-offer"]:
            if "trade_details" in payload:
                trade = payload["trade_details"]
                give_items = trade.get("give_items", [])
                receive_items = trade.get("receive_items", [])
                money_delta = trade.get("money_delta", 0)

                formatted += f"  Trade: Give {give_items}"
                if money_delta < 0:
                    formatted += f" + ${abs(money_delta)}"
                formatted += f" → Receive {receive_items}"
                if money_delta > 0:
                    formatted += f" + ${money_delta}"
                formatted += "\n"

            if "message" in payload:
                formatted += f"  Message: \"{payload['message']}\"\n"

        elif event_kind == "negotiate":
            if "message" in payload:
                formatted += f"  Message: \"{payload['message']}\"\n"

        elif event_kind == "action":
            if "action_type" in payload:
                action_type = payload["action_type"]
                formatted += f"  Action: {action_type}\n"
                
                if action_type == "trade":
                    # Display verbose trade details
                    target_agent = payload.get("target_agent", "unknown")
                    give_items = payload.get("give_items", [])
                    request_items = payload.get("request_items", [])
                    money_delta = payload.get("money_delta", 0)
                    
                    formatted += f"  Trade: {event_agent} → {target_agent}\n"
                    
                    if money_delta > 0:
                        formatted += f"  Money Exchange: +{money_delta}\n"
                    elif money_delta < 0:
                        formatted += f"  Money Exchange: {money_delta}\n"
                    else:
                        formatted += f"  Money Exchange: +0\n"
                    
                    formatted += f"  Items: -{give_items} +{request_items}\n"
                    
                    # Add agent message if available
                    if "agent_message" in payload:
                        formatted += f"  Agent Message: \"{payload['agent_message']}\"\n"
                
                elif action_type == "buy" and "item" in payload:
                    item = payload["item"]
                    formatted += f"  Item: {item}\n"
                    
                    # Add agent message if available
                    if "agent_message" in payload:
                        formatted += f"  Agent Message: \"{payload['agent_message']}\"\n"
                
                elif action_type in ["approve", "disapprove"]:
                    # Handle trade approval/disapproval
                    if "trade_id" in payload:
                        formatted += f"  Trade ID: {payload['trade_id']}\n"
                    
                    # Add agent reasoning (stored as "reason" in payload)
                    if "reason" in payload:
                        formatted += f"  Agent Reason: \"{payload['reason']}\"\n"
        
        elif event_kind == "inventory_update":
            # Specific handling for inventory updates
            if "message" in payload:
                formatted += f"  Message: \"{payload['message']}\"\n"
            if "items" in payload:
                formatted += f"  Current Inventory: {payload['items']}\n"
        
        else:
            # Generic payload handling for other event types
            if "message" in payload:
                formatted += f"  Message: \"{payload['message']}\"\n"
            
            # Add other important payload fields (excluding metadata already shown in header)
            metadata_fields = {"phase", "iteration", "round", "message"}
            for key, value in payload.items():
                if key not in metadata_fields and value:
                    formatted += f"  {key.title()}: {value}\n"

        return formatted


class ToolCallLogger:
    """
    Logger for capturing all tool calls with metadata.

    Logs all tool invocations including agent name, phase, iteration/round,
    function name, parameters, result, and timing information.
    """

    def __init__(
        self,
        environment_name: str = "Unknown",
        seed: int = 0,
        config: Dict[str, Any] = None,
        run_timestamp: Optional[str] = None,
    ):
        """
        Initialize the tool call logger.

        Args:
            environment_name: Name of the environment (e.g., "Trading", "SmartGrid")
            seed: Simulation seed for unique log files
            config: Full configuration dictionary containing all simulation settings
        """
        self.environment_name = environment_name
        self.seed = seed
        tag_model = get_tag_model_subdir(config or {})
        self.run_timestamp = _resolve_run_timestamp(config, run_timestamp)
        self.log_dir = build_log_dir(environment_name, tag_model, seed, self.run_timestamp)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "tool_calls.json"

        if not self.log_file_path.exists():
            with self.log_file_path.open('w') as f:
                json.dump([], f)

    def log_tool_call(self,
                     agent_name: str,
                     phase: str,
                     tool_name: str,
                     parameters: Dict[str, Any],
                     result: Dict[str, Any],
                     iteration: Optional[int] = None,
                     round_num: Optional[int] = None,
                     duration_ms: Optional[float] = None) -> None:
        """
        Log a tool call with all relevant metadata.

        Args:
            agent_name: Name of the agent making the tool call
            phase: Current phase (planning, execution, etc.)
            tool_name: Name of the tool/function being called
            parameters: Parameters passed to the tool
            result: Result returned from the tool
            iteration: Current iteration number (for execution phase)
            round_num: Current round number (for planning phase)
            duration_ms: Time taken to execute the tool in milliseconds
        """
        try:
            # Determine success from result
            if isinstance(result, dict):
                success = result.get("success", True) if "error" not in result else not bool(result.get("error"))
            else:
                success = True

            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "round": round_num,
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result,
                "success": success,
                "duration_ms": duration_ms
            }

            # Read existing entries, append new one, and write back
            try:
                with self.log_file_path.open('r') as f:
                    entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []

            entries.append(log_entry)

            with self.log_file_path.open('w') as f:
                json.dump(entries, f, indent=2)

        except Exception as e:
            print(f"ERROR: Failed to log tool call: {e}")

    def reset_log(self):
        """Reset the tool call log by clearing all entries."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_file_path.open('w') as f:
            json.dump([], f)

    def log_adversarial_agent_action(self, agent_name: str, original_message: str,
                                     replaced_message: str, phase: str,
                                     iteration: Optional[int] = None, round_num: Optional[int] = None):
        """Log when an adversarial_agent agent's message is replaced."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": time.time(),
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "round_num": round_num,
                "action_type": "adversarial_agent_message_replacement",
                "original_message": original_message,
                "replaced_message": replaced_message,
                "environment": self.environment_name,
                "seed": self.seed
            }
            try:
                with self.log_file_path.open('r') as f:
                    entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []
            entries.append(log_entry)
            with self.log_file_path.open('w') as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            print(f"ERROR: Failed to log adversarial_agent action: {e}")


class PromptLogger:
    """Logger for capturing agent prompts (system and user) in both JSON and Markdown formats."""

    def __init__(
        self,
        environment_name: str = "Unknown",
        seed: int = 0,
        config: Dict[str, Any] = None,
        run_timestamp: Optional[str] = None,
    ):
        self.environment_name = environment_name
        self.seed = seed
        tag_model = get_tag_model_subdir(config or {})
        self.run_timestamp = _resolve_run_timestamp(config, run_timestamp)
        self.log_dir = build_log_dir(environment_name, tag_model, seed, self.run_timestamp)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # JSON log for programmatic access
        self.log_file_path = self.log_dir / "agent_prompts.json"
        # Markdown log for human readability
        self.md_log_path = self.log_dir / "agent_prompts.md"

        if not self.log_file_path.exists():
            with self.log_file_path.open('w') as f:
                json.dump([], f)
        if not self.md_log_path.exists():
            with self.md_log_path.open('w') as f:
                f.write(f"# Agent Prompts Log - {environment_name} (Seed: {seed})\n\n")

    def log_prompts(self, agent_name: str, system_prompt: str, user_prompt: str,
                   phase: str, iteration: Optional[int] = None, round_num: Optional[int] = None) -> None:
        """Log both system and user prompts with metadata in JSON and Markdown formats."""
        try:
            timestamp = datetime.now().isoformat()

            # Create JSON log entry
            log_entry = {
                "timestamp": timestamp,
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "round": round_num,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }

            # Write to JSON file (for programmatic access)
            try:
                with self.log_file_path.open('r') as f:
                    entries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []

            entries.append(log_entry)

            with self.log_file_path.open('w') as f:
                json.dump(entries, f, indent=2)

            # Write to Markdown file (for human readability)
            # Build metadata string
            metadata_parts = [f"**Phase:** {phase}"]
            if iteration is not None:
                metadata_parts.append(f"**Iteration:** {iteration}")
            if round_num is not None:
                metadata_parts.append(f"**Round:** {round_num}")
            metadata = " | ".join(metadata_parts)

            # Create markdown entry
            md_entry = f"""## {agent_name} - {metadata}
**Timestamp:** {timestamp}

### System Prompt
```
{system_prompt}
```

### User Prompt
```
{user_prompt}
```

---

"""

            # Append to markdown file
            with self.md_log_path.open('a') as f:
                f.write(md_entry)

        except Exception as e:
            print(f"ERROR: Failed to log prompts: {e}")

    def reset_log(self):
        """Reset both JSON and Markdown prompt logs by clearing all entries."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with self.log_file_path.open('w') as f:
            json.dump([], f)

        with self.md_log_path.open('w') as f:
            f.write(f"# Agent Prompts Log - {self.environment_name} (Seed: {self.seed})\n\n")


class AgentTrajectoryLogger:
    """
    Logger for capturing agent reasoning and tool call trajectories.

    Logs the internal decision-making process of agents including:
    - Reasoning text (from response.output_text)
    - Tool calls with parameters
    - Organized by agent → iteration → phase → trajectory steps
    """

    def __init__(
        self,
        environment_name: str = "Unknown",
        seed: int = 0,
        config: Dict[str, Any] = None,
        run_timestamp: Optional[str] = None,
    ):
        """
        Initialize the agent trajectory logger.

        Args:
            environment_name: Name of the environment (e.g., "Trading", "SmartGrid")
            seed: Simulation seed for unique log files
            config: Full configuration dictionary containing all simulation settings
        """
        self.environment_name = environment_name
        self.seed = seed
        tag_model = get_tag_model_subdir(config or {})
        self.run_timestamp = _resolve_run_timestamp(config, run_timestamp)
        self.log_dir = build_log_dir(environment_name, tag_model, seed, self.run_timestamp)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "agent_trajectories.json"

        if not self.log_file_path.exists():
            with self.log_file_path.open('w') as f:
                json.dump({}, f)

    def log_trajectory(self,
                      agent_name: str,
                      iteration: int,
                      phase: str,
                      trajectory_dict: Dict[str, Any],
                      round_num: Optional[int] = None) -> None:
        """
        Log a trajectory for an agent during a specific iteration and phase.

        Args:
            agent_name: Name of the agent
            iteration: Current iteration number
            phase: Current phase (planning, execution)
            trajectory_dict: Dictionary of trajectory steps with keys like "step_1", "step_2"
                Each step contains:
                - tools: List of tool calls formatted as "{tool_name} -- {parameters}"
                - reasoning: Reasoning text (from response.output_text)
            round_num: Round number (for planning phase only, None for execution)
        """
        try:
            # Read existing data
            try:
                with self.log_file_path.open('r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}

            # Initialize nested structure if needed
            if agent_name not in data:
                data[agent_name] = {}

            iteration_key = f"iteration_{iteration}"
            if iteration_key not in data[agent_name]:
                data[agent_name][iteration_key] = {}

            # Handle round-based structure (planning) vs direct structure (execution)
            if round_num is not None:
                # Planning phase with rounds
                if phase not in data[agent_name][iteration_key]:
                    data[agent_name][iteration_key][phase] = {}

                round_key = f"round_{round_num}"
                if round_key not in data[agent_name][iteration_key][phase]:
                    data[agent_name][iteration_key][phase][round_key] = {"trajectory": {}}

                # Merge trajectory dict
                data[agent_name][iteration_key][phase][round_key]["trajectory"].update(trajectory_dict)
            else:
                # Execution phase (no rounds)
                if phase not in data[agent_name][iteration_key]:
                    data[agent_name][iteration_key][phase] = {"trajectory": {}}

                # Merge trajectory dict
                data[agent_name][iteration_key][phase]["trajectory"].update(trajectory_dict)

            # Write back to file
            with self.log_file_path.open('w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"ERROR: Failed to log agent trajectory: {e}")

    def reset_log(self):
        """Reset the agent trajectory log by clearing all entries."""
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        with open(self.log_file_path, 'w') as f:
            json.dump({}, f)
        print("Agent trajectory log has been reset for new simulation")
