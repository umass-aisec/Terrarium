"""LLM client implementations."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AbstractClient",
    "AnthropicClient",
    "ExternalMCPToolRegistry",
    "FireworksClient",
    "GeminiClient",
    "OpenAIClient",
    "TogetherClient",
    "VLLMClient",
]

_LAZY_ATTRS = {
    "AbstractClient": ".abstract_client:AbstractClient",
    "AnthropicClient": ".anthropic_client:AnthropicClient",
    "ExternalMCPToolRegistry": ".external_mcp:ExternalMCPToolRegistry",
    "FireworksClient": ".fireworks_client:FireworksClient",
    "GeminiClient": ".gemini_client:GeminiClient",
    "OpenAIClient": ".openai_client:OpenAIClient",
    "TogetherClient": ".together_client:TogetherClient",
    "VLLMClient": ".vllm_client:VLLMClient",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
