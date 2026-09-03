"""Public JiraTicket environment namespace for Terrarium."""

from __future__ import annotations

from terrarium.environments._lazy import lazy_namespace

__all__ = ["JiraTicketEnvironment", "JiraTicketTools", "JiraTicketPrompts"]

_LAZY_ATTRS = {
    "JiraTicketEnvironment": ".jira_ticket_env:JiraTicketEnvironment",
    "JiraTicketTools": ".jira_ticket_tools:JiraTicketTools",
    "JiraTicketPrompts": ".jira_ticket_prompts:JiraTicketPrompts",
}


__getattr__, __dir__ = lazy_namespace(__name__, _LAZY_ATTRS, globals(), __all__)
