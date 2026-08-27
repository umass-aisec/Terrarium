from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import json
import time
import uuid


@dataclass
class Event:
    """
    Event object for blackboard communication between agents.

    Attributes:
        id: Unique identifier for the event
        ts: Timestamp when event was created
        agent: Name of agent that created the event
        kind: Type of event (proposal, counter-offer, introduction, explanation, etc.)
        payload: Free-form dictionary containing event-specific data
        refs: List of event IDs this event references
        blackboard_id: ID of the blackboard this event belongs to
    """

    id: str
    ts: float
    agent: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    refs: List[str] = field(default_factory=list)
    blackboard_id: Optional[int] = None


@dataclass
class Blackboard:
    """
    Append-only event log.

    Provides methods for posting, scanning, and creating text summaries for LLM prompting.
    Each blackboard has a unique ID and tracks its participants.
    """

    blackboard_id: str = "-1"
    participants: List[str] = field(default_factory=list)
    initial_context: str = ""

    # For Megaboard compatibility
    agents: Set[str] = field(default_factory=set)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    template: Dict[str, Any] = field(default_factory=dict)
    verify_template: int = 0
    suspended: bool = False


class Megaboard:
    """
    Factor graph-based blackboard system for multi-agent communication.

    Maintains multiple blackboards with template verification and factor graph initialization.
    Designed for structured agent communication and optional external tool integration.
    """

    def __init__(self):
        """Initialize the Megaboard system."""
        self.blackboards: List[Blackboard] = []

    def return_blackboards(self):
        return self.blackboards

    def clear_blackboards(self) -> None:
        """Clear all blackboards. Used when starting a new simulation."""
        self.blackboards = []

    def _validate_blackboard_id(self, blackboard_id: int) -> None:
        if blackboard_id < 0 or blackboard_id >= len(self.blackboards):
            raise ValueError(f"Blackboard {blackboard_id} does not exist")

    def _prepare_payload(
        self,
        payload: Optional[Dict[str, Any]],
        phase: Optional[str],
        iteration: Optional[int],
    ) -> Dict[str, Any]:
        """Prepare event payload with metadata."""
        if payload is None:
            payload = {}

        # Add phase and iteration metadata if provided
        if phase is not None:
            payload["phase"] = phase
        if iteration is not None:
            payload["iteration"] = iteration

        return payload

    def add_blackboard(
        self, agents: List[str], template: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new blackboard with the specified agents.

        Args:
            agents: List of agent names who will participate
            template: Optional template for event validation

        Returns:
            The blackboard ID (index in the list)
        """
        raw_template = template
        template = template or {}
        allow_duplicate = bool(template.get("allow_duplicate")) or bool(
            template.get("secret_channel")
        )

        # Check if blackboard with same agents already exists. Secret/private
        # overlays must be allowed to duplicate participant sets so a hidden
        # channel can coexist with a public channel over the same agents.
        agents_set = set(agents)
        if not allow_duplicate:
            for i, blackboard in enumerate(self.blackboards):
                if blackboard.agents == agents_set:
                    return i

        # Create new blackboard
        verify_template = 1 if raw_template is not None else 0
        blackboard_id = len(self.blackboards)
        blackboard = Blackboard(
            blackboard_id=str(blackboard_id),  # Convert to string for compatibility
            participants=agents.copy(),
            agents=agents_set,
            template=template,
            verify_template=verify_template,
            logs=[],
        )

        self.blackboards.append(blackboard)
        return len(self.blackboards) - 1

    def is_private_channel(self, blackboard_id: Any) -> bool:
        """Whether a blackboard is an agent-created private channel."""
        blackboard = self.get_blackboard_by_string_id(str(blackboard_id))
        return bool(blackboard and blackboard.template.get("private_channel"))

    def _reachable_agents(self, agent: str) -> Set[str]:
        """Agents that already share at least one channel with `agent`."""
        peers: Set[str] = set()
        for blackboard in self.blackboards:
            if agent in blackboard.agents:
                peers |= blackboard.agents
        peers.discard(agent)
        return peers

    def create_channel(
        self,
        agent_name: str,
        participants: List[str],
        message: str = "",
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Open a private channel between the caller and the given agents.

        The channel is a normal blackboard, so only its participants can read or
        post to it. Reuses an existing private channel with the same membership.
        """
        invited = {str(a).strip() for a in participants if str(a).strip()}
        invited.discard(agent_name)
        if not invited:
            return {"error": "create_channel requires at least one other agent in agent_ids"}

        unreachable = sorted(invited - self._reachable_agents(agent_name))
        if unreachable:
            return {
                "error": (
                    f"Cannot open a channel with {', '.join(unreachable)}: "
                    "you share no existing channel with them."
                )
            }

        members = sorted(invited | {agent_name})
        blackboard_id = next(
            (
                i
                for i, bb in enumerate(self.blackboards)
                if bb.template.get("private_channel") and bb.agents == set(members)
            ),
            None,
        )
        created = blackboard_id is None
        if created:
            blackboard_id = self.add_blackboard(
                members,
                # allow_duplicate so a private channel can coexist with a public
                # channel over the same set of agents.
                template={
                    "private_channel": True,
                    "created_by": agent_name,
                    "allow_duplicate": True,
                },
            )
            self.post_system_message(
                blackboard_id,
                "context",
                {
                    "message": (
                        f"Private channel opened by {agent_name}. "
                        f"Participants: {', '.join(members)}. "
                        "Only these agents can see what is posted here."
                    )
                },
                phase=phase,
                iteration=iteration,
            )

        if message:
            self.post(
                blackboard_id,
                agent_name,
                "communication",
                {"content": message},
                phase=phase,
                iteration=iteration,
            )

        return {
            "status": "success",
            "blackboard_id": blackboard_id,
            "participants": members,
            "created": created,
            "note": "Post here with post_message(message, blackboard_id).",
        }

    def post(
        self,
        blackboard_id: int,
        agent: str,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Post a new event to the specified blackboard.

        Args:
            blackboard_id: ID of the target blackboard
            agent: Name of the agent posting the event
            kind: Type of event being posted
            payload: Event data and content
            phase: Current simulation phase (planning/execution) - will be stored in payload
            iteration: Current iteration number - will be stored in payload

        Returns:
            The unique ID of the created event
        """
        self._validate_blackboard_id(blackboard_id)
        blackboard = self.blackboards[blackboard_id]

        if blackboard.suspended:
            raise ValueError(f"Blackboard {blackboard_id} is suspended")

        if agent not in blackboard.agents:
            raise ValueError(f"Agent {agent} is not in blackboard {blackboard_id}")

        payload = self._prepare_payload(payload, phase, iteration)

        event_id = str(uuid.uuid4())
        event = {
            "blackboard_id": blackboard_id,
            "id": event_id,
            "ts": time.time(),
            "agent": agent,
            "kind": kind,
            "payload": payload,
        }

        blackboard.logs.append(event)

        return event_id

    def post_system_message(
        self,
        blackboard_id: int,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        Post a system message to the blackboard (bypasses agent membership check).

        Args:
            blackboard_id: ID of the blackboard
            kind: Type of event being posted
            payload: Event data and content
            phase: Current simulation phase (planning/execution) - will be stored in payload
            iteration: Current iteration number - will be stored in payload

        Returns:
            The unique ID of the created event
        """
        self._validate_blackboard_id(blackboard_id)
        blackboard = self.blackboards[blackboard_id]

        if blackboard.suspended:
            raise ValueError(f"Blackboard {blackboard_id} is suspended")

        payload = self._prepare_payload(payload, phase, iteration)

        event_id = str(uuid.uuid4())
        event = {
            "blackboard_id": blackboard_id,
            "id": event_id,
            "ts": time.time(),
            "agent": "SYSTEM",
            "kind": kind,
            "payload": payload,
        }

        # System messages bypass template verification
        blackboard.logs.append(event)

        return event_id

    def get(
        self, blackboard_id: int, agent: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent events from the blackboard for the specified agent.

        Args:
            blackboard_id: ID of the blackboard
            agent: Name of the agent requesting events
            limit: Maximum number of recent events to return

        Returns:
            List of recent events
        """
        self._validate_blackboard_id(blackboard_id)
        blackboard = self.blackboards[blackboard_id]

        if agent not in blackboard.agents:
            raise ValueError(f"Agent {agent} is not in blackboard {blackboard_id}")

        events = blackboard.logs
        return events[-limit:] if limit else events

    # Adapter methods for BlackboardManager compatibility
    def get_blackboard_by_string_id(self, blackboard_id: str) -> Optional[Blackboard]:
        """Get a Blackboard object given a string ID. Returns None if not found or invalid ID."""
        try:
            int_id = int(blackboard_id)
            if int_id < 0 or int_id >= len(self.blackboards):
                return None
            return self.blackboards[int_id]
        except (ValueError, TypeError):
            return None

    def get_agent_blackboards(self, agent_name: str) -> List[str]:
        """Get list of blackboard string IDs that an agent participates in."""
        result = []
        for i, blackboard in enumerate(self.blackboards):
            if agent_name in blackboard.agents:
                result.append(str(i))
        return result

    def recall(
        self, blackboard_id: int, agent: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the FULL, uncompacted event log for a blackboard for events
        matching `query` — the retrieval-backed compaction technique. The
        underlying `blackboard.logs` list is never pruned by compaction (only
        the prompt string built from it is lossy), so this recovers detail a
        compacted summary may have dropped, scoped to a keyword query rather
        than dumping the entire history back.

        Matching is a simple case-insensitive substring match against each
        event's formatted text, keeping the most recent matches.
        """
        events = self.get(blackboard_id, agent, limit=None)
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []

        matches: List[Dict[str, Any]] = []
        for event in events:
            text = format_blackboard_events_for_prompt([event]).lower()
            if any(term in text for term in terms):
                matches.append(event)

        return matches[-limit:] if limit else matches

    def get_agent_blackboard_contexts(self, agent_name: str) -> Dict[str, str]:
        """
        Get context summaries for all blackboards that an agent participates in. This is used as context in agent prompts.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary mapping string blackboard_id -> context_summary
        """
        agent_blackboards = self.get_agent_blackboards(agent_name)
        contexts = {}

        for bb_string_id in agent_blackboards:
            try:
                int_id = int(bb_string_id)
                if int_id >= 0 and int_id < len(self.blackboards):
                    blackboard = self.blackboards[int_id]
                    # Get all events and create a simple context summary
                    events = blackboard.logs
                    context_parts = []
                    for event in events:
                        if event.get("kind") == "context" and event.get(
                            "payload", {}
                        ).get("message"):
                            context_parts.append(
                                f"Initial: {event['payload']['message']}"
                            )
                        elif event.get("kind") == "communication" and event.get(
                            "payload", {}
                        ).get("content"):
                            context_parts.append(
                                f"{event.get('agent', 'Unknown')}: {event['payload']['content']}"
                            )

                    contexts[bb_string_id] = (
                        "\n".join(context_parts)
                        if context_parts
                        else "No recent activity"
                    )
            except (ValueError, TypeError):
                continue

        return contexts

    def get_blackboard_string_ids(self) -> List[str]:
        """
        Get all blackboard IDs as strings.

        Returns:
            List of string blackboard IDs for all existing non-suspended blackboards
        """
        return [
            str(i)
            for i in range(len(self.blackboards))
            if not self.blackboards[i].suspended
        ]

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Handle blackboard tool calls by routing to appropriate methods.

        Args:
            tool_name: Name of the tool to execute
            agent_name: Name of the agent calling the tool
            arguments: Parameters for the tool call
            phase: Current simulation phase (planning/execution)
            iteration: Current iteration number

        Returns:
            Dictionary with tool execution result
        """
        try:
            if tool_name == "get_blackboard_events":
                blackboard_id = arguments.get("blackboard_id")

                if blackboard_id is None:
                    return {
                        "error": "blackboard_id is required for get_blackboard_events"
                    }

                # Convert to int to handle float values from LLM responses (e.g., Gemini UltraThink)
                blackboard_id = int(blackboard_id)

                result = self.get(blackboard_id, agent_name, limit=None)
                return {"events": result}

            elif tool_name == "recall":
                query = str(arguments.get("query", "")).strip()
                if not query:
                    return {"error": "query is required for recall"}

                blackboard_id = arguments.get("blackboard_id")
                if blackboard_id is None:
                    agent_blackboards = self.get_agent_blackboards(agent_name)
                    if not agent_blackboards:
                        return {"error": "No blackboards available to recall from"}
                    blackboard_id = int(agent_blackboards[0])
                else:
                    blackboard_id = int(blackboard_id)

                limit = arguments.get("limit")
                limit = int(limit) if limit else 5
                matches = self.recall(blackboard_id, agent_name, query, limit=limit)
                if not matches:
                    return {"result": "No matching events found in the full archive."}
                return {
                    "events": matches,
                    "result": format_blackboard_events_for_prompt(matches),
                }

            elif tool_name == "post_message":
                # Handle post_message as a special case of post_event
                blackboard_id = arguments.get("blackboard_id")
                message = arguments.get("message", "")

                # Use the agent's first blackboard if none specified
                if blackboard_id is None:
                    agent_blackboards = self.get_agent_blackboards(agent_name)
                    if not agent_blackboards:
                        return {"error": "No blackboards available for communication"}
                    blackboard_id = int(agent_blackboards[0])
                else:
                    # Convert to int to handle float values from LLM responses (e.g., Gemini UltraThink)
                    blackboard_id = int(blackboard_id)

                payload = {"content": message}
                # Include phase and iteration metadata in the posted event
                result = self.post(
                    blackboard_id,
                    agent_name,
                    "communication",
                    payload,
                    phase=phase,
                    iteration=iteration,
                )
                return {"event_id": result}

            elif tool_name == "create_channel":
                agent_ids = arguments.get("agent_ids") or arguments.get("agent_id") or []
                if isinstance(agent_ids, str):
                    agent_ids = [a for a in agent_ids.split(",")]
                return self.create_channel(
                    agent_name,
                    list(agent_ids),
                    message=str(arguments.get("message") or "").strip(),
                    phase=phase,
                    iteration=iteration,
                )

            else:
                return {"error": f"Unknown blackboard tool: {tool_name}"}

        except Exception as e:
            return {"error": f"Error executing blackboard tool {tool_name}: {str(e)}"}

    def log_action_to_blackboards(
        self,
        agent_name: str,
        action: Dict[str, Any],
        result: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Log an action to all blackboards that the agent belongs to.

        Args:
            agent_name: Name of the agent performing the action
            action: Dictionary containing action type and parameters
            result: Result from action execution
            phase: Current simulation phase (planning/execution)
            iteration: Current iteration number
        """
        # Find all blackboards the agent belongs to
        agent_blackboards = []
        for i, blackboard in enumerate(self.blackboards):
            if agent_name in blackboard.agents:
                agent_blackboards.append(i)

        if not agent_blackboards:
            print(
                f"DEBUG: Agent {agent_name} is not in any blackboards, skipping action logging"
            )
            return

        # Create payload for the event
        payload = {
            "action_type": action.get("action"),
            "action_params": action.copy(),
            "result_status": result.get("status"),
            "details": result.copy(),
        }

        # Post to all agent's blackboards with phase and iteration metadata
        for blackboard_id in agent_blackboards:
            try:
                self.post(
                    blackboard_id=blackboard_id,
                    agent=agent_name,
                    kind="action_executed",
                    payload=payload,
                    phase=phase,
                    iteration=iteration,
                )
            except Exception as e:
                print(f"ERROR: Failed to log action to blackboard {blackboard_id}: {e}")


def format_blackboard_events_for_prompt(events: List[Dict[str, Any]]) -> str:
    """Render blackboard event logs into a compact, readable transcript for prompts."""
    if not events:
        return "No recent activity"

    def _compact(obj: Any) -> str:
        try:
            return json.dumps(
                obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except Exception:
            return repr(obj)

    lines: List[str] = []
    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            lines.append(f"[{idx}] {_compact(event)}")
            continue

        kind = str(event.get("kind") or "event")
        agent = str(event.get("agent") or "Unknown")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        if kind == "context" and payload.get("message"):
            lines.append(f"[{idx}] [context] {payload.get('message')}")
            continue

        if kind == "communication" and payload.get("content"):
            meta_parts: List[str] = []
            if payload.get("phase") is not None:
                meta_parts.append(f"phase={payload.get('phase')}")
            if payload.get("iteration") is not None:
                meta_parts.append(f"iter={payload.get('iteration')}")
            meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"[{idx}] {agent}: {payload.get('content')}{meta}")
            continue

        event_id = event.get("id")
        id_str = f" id={event_id}" if event_id else ""
        lines.append(f"[{idx}] [{kind}] {agent}{id_str} payload={_compact(payload)}")

    return "\n".join(lines)
