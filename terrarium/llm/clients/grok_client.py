from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv

from terrarium.llm.clients.openai_client import OpenAIClient


_DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"


class GrokClient(OpenAIClient):
    """First-party xAI Grok client using xAI's OpenAI-compatible Responses API."""

    def __init__(
        self,
        *,
        api_key: Optional[Any] = None,
        api_key_env_var: str = "GROK_API_KEY",
        base_url: Optional[str] = None,
        timeout: float = 3600.0,
        base_model: Optional[str] = None,
    ):
        load_dotenv(override=True)
        resolved_api_key = (
            api_key
            or os.getenv(api_key_env_var)
            or os.getenv("GROK_API_KEY")
            or os.getenv("XAI_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "xAI Grok API key not found. Set GROK_API_KEY in .env file"
            )

        super().__init__(
            api_key=resolved_api_key,
            api_key_env_var=api_key_env_var,
            base_url=base_url or _DEFAULT_GROK_BASE_URL,
            timeout=timeout,
            provider_name="xAI Grok",
            capability_model=base_model,
        )
