---
title: Environments API
---

# Environments API

## Base Environment

```{eval-rst}
.. automodule:: terrarium.environments.abstract_environment
   :members:
   :show-inheritance:
```

## Implemented Environments

```{eval-rst}
.. autoclass:: terrarium.environments.dcops.meeting_scheduling.meeting_scheduling_env.MeetingSchedulingEnvironment
   :members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.personal_assistant.personal_assistant_env.PersonalAssistantEnvironment
   :members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.smart_grid.smart_grid_env.SmartGridEnvironment
   :members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.hospital.hospital_env.HospitalEnvironment
   :members:
   :show-inheritance:

.. autoclass:: terrarium.environments.dcops.jira_ticket.jira_ticket_env.JiraTicketEnvironment
   :members:
   :show-inheritance:
```

## Prompt Helpers

```{eval-rst}
.. autoclass:: terrarium.environments.dcops.meeting_scheduling.meeting_scheduling_prompts.MeetingSchedulingPrompts
   :members:

.. autoclass:: terrarium.environments.dcops.personal_assistant.personal_assistant_prompts.PersonalAssistantPrompts
   :members:

.. autoclass:: terrarium.environments.dcops.smart_grid.smartgrid_prompts.SmartGridPrompts
   :members:

.. autoclass:: terrarium.environments.dcops.hospital.hospital_prompts.HospitalPrompts
   :members:

.. autoclass:: terrarium.environments.dcops.jira_ticket.jira_ticket_prompts.JiraTicketPrompts
   :members:
```
