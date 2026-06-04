---
title: Configuration
---

# Configuration

Terrarium runs are configured with YAML files under `examples/configs`.

## LLM Provider

Set `llm.provider` to select the client implementation, then configure the matching provider block:

```yaml
llm:
  provider: gemini
  gemini:
    model: gemini-2.5-flash
```

## External MCP Servers

External MCP servers can be attached per LLM client:

```yaml
llm:
  provider: openai
  external_mcp_servers:
    - name: filesystem
      url: http://127.0.0.1:9000/mcp
      enabled: true
      tool_prefix: fs_
```

## vLLM

For local model serving, set `llm.provider` to `vllm` and configure a single server under `llm.vllm`:

```yaml
llm:
  provider: vllm
  vllm:
    auto_start_server: true
    persistent_server: false
    startup_timeout: 180
    models:
      - checkpoint: /data/models/Qwen2-7B-Instruct
        served_model_name: Qwen2-7B-Instruct
        host: 127.0.0.1
        port: 8001
        tensor_parallel_size: 1
        trust_remote_code: true
```
