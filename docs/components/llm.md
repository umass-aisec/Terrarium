---
title: Inference Clients
---

# Inference Clients

Inference clients are the adapter layer between Terrarium agents and model providers. The rest of the framework talks to the same client interface: initialize a conversation, generate a response, detect tool calls, execute those tools through a callback, and append the results to the next turn.

Most users configure clients in YAML and let the agent factory create them.

```yaml
llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
    params:
      max_tokens: 1024
      temperature: 0.7
```

The base runner then extracts the model and generation parameters:

```python
from terrarium.utils import get_generation_params, get_model_name

llm_config = config["llm"]
provider = llm_config.get("provider", "openai").lower()
model_name = get_model_name(provider, llm_config)
generation_params = get_generation_params(llm_config)
```

## Provider Clients

Terrarium includes clients for OpenAI-compatible and provider-specific APIs, including OpenAI, Foundry, Grok, Anthropic, Gemini, Together, Fireworks, and vLLM. API keys and endpoints are usually resolved from environment variables so configs can be shared without secrets.

```bash
cp -n .env.example .env
```

Set only the keys for the providers you plan to use.

## Local vLLM

For local inference, Terrarium can build a vLLM runtime and give each agent a client pointed at the right local server.

```python
from terrarium.utils import build_vllm_runtime

vllm_runtime = build_vllm_runtime(config["llm"])
client, model_name = vllm_runtime.create_client("agent_0")
```

```yaml
llm:
  provider: vllm
  vllm:
    auto_start_server: true
    persistent_server: false
    models:
      - checkpoint: Qwen/Qwen2.5-7B-Instruct
        served_model_name: Qwen2.5-7B-Instruct
        host: 127.0.0.1
        port: 8001
        tensor_parallel_size: 1
```

If `auto_start_server` is true, the runtime starts the server before agents use it and shuts it down at the end unless `persistent_server` is enabled.

## Tool Calling

Every client implements provider-specific tool parsing. The agent sees one interface, but each provider may return tool calls in a different shape. That is why tool execution lives in `process_tool_calls()` on the client rather than in the protocol.

```python
response, text = client.generate_response(input=context, params=params)

tools_executed, context, tool_names = await client.process_tool_calls(
    response,
    context,
    execute_tool_callback,
)
```

This keeps provider quirks from leaking into environments and communication protocols.

See the [LLM Clients API](../api/llm.md) for generated reference details.
