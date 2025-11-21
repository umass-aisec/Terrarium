# vLLM Tool-Calling Compatibility Guide

Terrarium auto-configures vLLM tool-calling flags (parser, chat template, reasoning parser) based on the checkpoint name.
Be sure to track the success rate which is the limiting factor. Some model are not able to execute tool calls despite vLLM docs.

## Supported Models

The following substrings in the checkpoint (or served name) trigger automatic tool-calling support. 

- `hermes`
- `mistral`
- `granite-4.0`, `granite-3.1`, `granite-3.0`, `granite-20b`
- `internlm`, `internlm2.5`
- `jamba` / `ai21`
- `xlam-llama`, `xlam-2-8b`, `xlam-2-70b`
- `xlam-qwen`, `xlam-1b`, `xlam-3b`, `xlam-32b`
- `qwen2.5`, `qwen-2.5`, `qwen2_5`
- `qwen3`
- `qwq`, `qwq-32b`
- `minimax`, `minimaxai`
- `deepseek-v3`, `deepseek-v3-0324`, `deepseek-r1-0528`
- `deepseek-v3.1`
- `kimi-k2`, `moonshotai/kimi`, `moonshot kimi`
- `hunyuan-a13b`
- `glm-4.5`, `glm45`

For these models, Terrarium will:

1. stop any running vLLM server for other checkpoints,
2. launch the new model with `--enable-auto-tool-choice`, the proper `--tool-call-parser`, and any required `--chat-template`, and
3. attach reasoning parsers when they exist (e.g., Qwen3, QwQ, DeepSeek R1 variants).

## Unsupported / Blocked Models

The following checkpoints do **not** support auto tool choice in vLLM yet, so Terrarium refuses to launch them unless you explicitly override the parser (and even then tool calling may fail):

- `deepseek r1`, `deepseek-r1`
- `ernie-4.5`, `ernie45`

If you need one of these, consult vLLM’s documentation for future parser support and manually set `tool_call_parser` once it becomes available.

## Unknown Models

Any checkpoint whose name does not match the substrings above is treated as "unknown". Terrarium will:

- warn that it could not auto-configure tool calling, and
- skip `--enable-auto-tool-choice` unless you explicitly supply `tool_call_parser`/`chat_template` within the `llm.vllm.models` entry.

To add support for a new model:

1. Confirm vLLM exposes a parser for it (see https://docs.vllm.ai/en/latest/features/tool_calling/).
2. Add the new rule in `llm_server/vllm/runtime.py`’s `TOOL_CALLING_RULES` list (with substrings, parser, template, etc.).
3. Re-run Terrarium; the auto-config flow will pick up the new preset.

## Tips
- Large checkpoints (e.g., Qwen3-VL or DeepSeek R1) may require higher `request_timeout` values in their spec (`request_timeout: 180`) to avoid client-side read timeouts.
