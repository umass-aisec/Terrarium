"""Utilities for injecting tool-calling instructions into prompts. Used only for vLLM provider."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import warnings

from src.utils import extract_model_info

DEFAULT_PLANNING_TOOL_LINES = [
    "- get_blackboard_events(blackboard_id: int): Inspect recent coordination notes.",
    "- post_message(message: str, blackboard_id?: int): Broadcast updates to collaborators.",
]


def _needs_tool_instructions(full_config: Dict[str, any]) -> bool:
    llm_config = (full_config or {}).get("llm") or {}
    return (llm_config.get("provider") or "").lower() == "vllm"


def _build_format_hint(full_config: Dict[str, any]) -> str:
    model_label = (extract_model_info(full_config) or "").lower()
    if "qwen3" in model_label or "qwen 3" in model_label:
        # Reference: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
        return (
            "Follow the Qwen3 Hermes-style template: list tool schemas inside <tools></tools>, emit each call as <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>, "
            "and echo tool outputs under <tool_response>. Do not mix prose outside those tags until tool calls finish."
        )
    if "qwen" in model_label:
        return (
            "Wrap every tool invocation inside <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> blocks and avoid plain text outside those tags. "
            "Example: <tool_call>{\"name\":\"post_message\",\"arguments\":{\"message\":\"Checking slot 4?\",\"blackboard_id\":2}}</tool_call>."
        )
    if "mistral" in model_label:
        # Reference: https://docs.mistral.ai/capabilities/function_calling/
        return (
            "Return only JSON tool_calls using the standard `{\"type\":\"function\",\"function\":{\"name\":...,\"arguments\":{...}}}` schema so vLLM can parse them. "
            "Never add narration outside that JSON blob and keep every argument valid per the provided JSON schema."
        )
    if "gpt-oss" in model_label or "gptoss" in model_label:
        # Reference: https://github.com/openai/harmony
        return (
            "Use the Harmony prompt format: emit tool calls inside `<|start|>assistant<|message|><|channel|>commentary ... <|end|>` and include only JSON such as "
            "{\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":...,\"arguments\":{...}}}]}. No free-form prose outside Harmony tags."
        )
    if "glm" in model_label:
        # Reference: https://hwcoder.top/Tool-Call-Format
        return (
            "Produce native GLM tool envelopes: `<tool_call>function_name<arg_key>param</arg_key><arg_value>value</arg_value>...</tool_call>` for each invocation and "
            "report tool outputs via `<observation>{...}</observation>` with zero extra narration."
        )
    if "deepseek" in model_label:
        # Reference: https://api-docs.deepseek.com/
        return (
            "Respond with OpenAI-compatible JSON tool calls (tool_calls array with type=function/function.name/function.arguments) because DeepSeek's API mirrors that wire format. "
            "Keep everything machine-parsable JSON and avoid text outside the payload."
        )
    if "kimi" in model_label or "moonshot" in model_label or "k2" in model_label:
        # Reference: https://github.com/MoonshotAI/Kimi-K2
        return (
            "Invoke tools exactly the way Kimi K2 expects: supply the `tools=[{\"type\":\"function\",...}]` metadata and reply only with tool_calls JSON "
            "(each entry has `function.name`, `function.arguments`, and `tool_call_id`) before waiting for tool outputs."
        )

    warnings.warn(
        "Tool prompts are using the generic function-call format; consider adding a model-specific branch for better reliability.",
        stacklevel=3,
    )
    # Reference: https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
    return (
        "Respond with a Python list containing only function calls, e.g. [post_message(message=\"Checking slot 4?\", blackboard_id=2)]."
    )


def build_vllm_tool_instructions(
    full_config: Dict[str, any],
    *,
    execution_tool_lines: Iterable[str],
    system_note: Optional[str] = None,
    planning_header: str = "Planning phase tools (coordination only):",
    execution_header: str = "Execution phase tools:",
    planning_tool_lines: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """Return formatted tool-instruction blocks (planning/execution/system)."""

    if not _needs_tool_instructions(full_config):
        return {}

    planning_lines = list(planning_tool_lines or DEFAULT_PLANNING_TOOL_LINES)
    format_hint = _build_format_hint(full_config) + " Always ensure argument names exactly match the tool schema."

    def _render_block(header: str, lines: List[str]) -> str:
        body = "\n".join(lines)
        return f"{header}\n{body}\nFORMAT:\n{format_hint}"

    planning_block = _render_block(planning_header, planning_lines)

    exec_lines = planning_lines + list(execution_tool_lines)
    execution_block = _render_block(execution_header, exec_lines)

    default_system = (
        "Planning: only blackboard tools are available.\n"
        "Execution: the environment-specific tools listed above become available in addition to blackboard tools."
    )

    return {
        "planning": planning_block,
        "execution": execution_block,
        "system": system_note or default_system,
    }


def get_phase_tool_instructions(instruction_map: Optional[Dict[str, str]], phase: Optional[str]) -> str:
    if not instruction_map:
        return ""
    return instruction_map.get((phase or "").lower(), "")
