---
title: Tools
---

# Tools

Terrarium tools are OpenAI-style function schemas exposed to agents during a
simulation. Planning phases use blackboard communication tools, while execution
phases add environment-specific action tools.

## Built-In Tool Surfaces

| Scope | Class or Module | Tool names |
| --- | --- | --- |
| Blackboard planning | `ToolsetDiscovery` | `post_message` |
| Meeting scheduling | `MeetingSchedulingTools` | `attend_meeting` |
| Personal assistant | `PersonalAssistantTools` | `choose_outfit` |
| Smart grid | `SmartGridTools` | `assign_source` |
| Hospital | `HospitalTools` | `find_available_slots`, `get_job_queue`, `transfer_resources`, `broadcast_message`, `schedule_patient` |
| Jira ticket | `JiraTicketTools` | `assign_task` |
| External MCP servers | `ExternalMCPToolRegistry` | Discovered from configured MCP servers |

## Discovery and Registry

```{eval-rst}
.. autoclass:: terrarium.tools.discovery.ToolsetDiscovery
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: terrarium.tools.environment
   :members:
   :undoc-members:
```

## vLLM Tool Prompt Helpers

```{eval-rst}
.. automodule:: terrarium.tools.prompts
   :members:
```

## Environment Tool Classes

```{eval-rst}
.. autoclass:: terrarium.environments.dcops.meeting_scheduling.meeting_scheduling_tools.MeetingSchedulingTools
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.personal_assistant.personal_assistant_tools.PersonalAssistantTools
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.smart_grid.smartgrid_tools.SmartGridTools
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.hospital.hospital_tools.HospitalTools
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.jira_ticket.jira_ticket_tools.JiraTicketTools
   :members:
   :undoc-members:
   :show-inheritance:
```

## External MCP Tools

```{eval-rst}
.. autoclass:: terrarium.llm.clients.external_mcp.ExternalMCPServerConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: terrarium.llm.clients.external_mcp.ExternalMCPToolRegistry
   :members:
   :undoc-members:
   :show-inheritance:
```
