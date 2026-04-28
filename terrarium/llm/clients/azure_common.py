from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_openai_v1_base_url(endpoint: Any, *, setting_name: str) -> str:
    """
    Normalize a Microsoft-hosted OpenAI-compatible endpoint into an OpenAI v1 base URL.

    Accepts either:
    - project endpoint (e.g. https://...services.ai.azure.com/api/projects/<project>)
    - full OpenAI v1 endpoint (e.g. https://.../openai/v1/)
    """
    raw = str(endpoint or "").strip()
    if not raw:
        raise ValueError(f"{setting_name} is required.")

    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"{setting_name} must be an absolute URL, got: {raw!r}"
        )

    path = (parsed.path or "").rstrip("/")
    normalized_path = path or ""
    if not normalized_path.endswith("/openai/v1"):
        normalized_path = f"{normalized_path}/openai/v1"

    return urlunsplit(
        (parsed.scheme, parsed.netloc, normalized_path + "/", "", "")
    )


def build_entra_token_provider(scope: str):
    """Create a token provider callable for OpenAI SDK `api_key` parameter."""
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:  # pragma: no cover - import behavior is tested with mocks
        raise ImportError(
            "Microsoft Entra authentication requires 'azure-identity'. "
            "Install with: pip install azure-identity"
        ) from exc

    credential = DefaultAzureCredential()
    return get_bearer_token_provider(credential, scope)
