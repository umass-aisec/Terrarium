---
title: Tools
---

# Tools

Tools are how an agent changes something outside its prompt. In Terrarium there are three tool surfaces: blackboard tools for communication, environment tools for domain actions, and optional external MCP tools for provider-backed integrations.

The agent does not decide which surface owns a tool. `ToolsetDiscovery` collects tools for the current environment and phase, and `BaseAgent` routes each tool call through the right handler.

```python
from terrarium.tools.discovery import ToolsetDiscovery

discovery = ToolsetDiscovery()

planning_tools = discovery.get_tools_for_blackboard("planning")
execution_tools = discovery.get_tools_for_environment(
    "MeetingSchedulingEnvironment",
    "execution",
)
```

## Blackboard Tools

The main blackboard tool is `post_message`. It is available during planning and writes an event to a blackboard the agent can access.

```python
await communication_protocol.blackboard_handle_tool_call(
    "post_message",
    "agent_0",
    {
        "blackboard_id": 0,
        "message": "I can cover this task if agent_1 takes the backup task.",
    },
    phase="planning",
    iteration=1,
)
```

Because blackboard messages are logged as events, they can be inspected after a run and used to study coordination behavior.

## Environment Tools

Environment tools are domain actions. In meeting scheduling, an execution tool can commit attendance. In Jira ticket coordination, an execution tool can assign work. These tools live beside the environment and follow a naming convention:

`FooEnvironment` resolves to `FooTools`.

```python
class WarehouseTools:
    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tools(self, phase):
        if phase != "execution":
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": "assign_package",
                    "description": "Assign a package to a warehouse agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "package_id": {"type": "string"},
                            "agent_message": {"type": "string"},
                        },
                        "required": ["package_id"],
                    },
                },
            }
        ]

    def get_tool_names(self):
        return {"assign_package"}

    def handle_tool_call(self, tool_name, agent_name, arguments, phase=None, iteration=None):
        if tool_name != "assign_package":
            return {"error": f"Unknown tool: {tool_name}"}
        self.environment.assignments[agent_name] = arguments["package_id"]
        return {"status": "success", "assigned_to": agent_name}
```

Place the class in an environment `*_tools.py` module under `terrarium/environments/...`. Discovery imports those modules and maps the tool class to the environment class name.

## External MCP Tools

```{warning}
This is experimental and may not work for your particular configuration.
```

External MCP tools are configured under `llm.external_mcp_servers`. They are discovered by the client and added to the same tool list as local tools.

```yaml
llm:
  provider: openai
  external_mcp_servers:
    - name: filesystem
      url: http://127.0.0.1:9000/mcp
      enabled: true
      tool_prefix: fs_
```

Use environment tools for simulation state. Use MCP tools when the agent should call an external service that is not part of the environment model.

See the [Tools API](../api/tools.md) for generated reference details.
