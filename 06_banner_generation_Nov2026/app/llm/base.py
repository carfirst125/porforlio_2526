"""Abstract interface every text-LLM provider under `app/llm/providers/`
must implement (nhantn-mlai-project-structure standard). This is the
SWAPPABLE text-generation layer used for prompt refinement
(`app/core/prompt_builder.py`) -- distinct from `app/providers/`, this app's
own domain-specific image-generation model layer (see
2-project-technote.md for why the two are kept separate).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    """Raised by any `app/llm/providers/*.py` client on a failed call
    (missing credentials, HTTP error, malformed response, ...). Callers
    always catch this and fall back gracefully -- prompt refinement is an
    optional enhancement, never something that should crash the pipeline."""


def record_usage(
    sink: dict | None,
    *,
    model: str,
    provider: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Shared, shape-agnostic setter for the `usage_sink` dict every
    `generate()`/`generate_vision()` call optionally accepts (2026-09-06 user
    request: per-call token/cost accounting -- see `app/core/usage_tracker.py`).

    Each provider's response shape is different (OpenAI Responses API vs.
    Chat Completions vs. Claude's Messages API vs. Ollama's top-level counters
    all name/nest their usage fields differently), so there is deliberately
    no generic "parse this payload" helper here -- every provider extracts
    its OWN usage fields from its OWN response payload, then calls this one
    shared setter to record them in a uniform shape. No-ops when `sink` is
    `None` (the caller passed no sink, e.g. every existing call site that
    predates this feature), so this is always safe to call unconditionally."""
    if sink is None:
        return
    sink["model"] = model
    sink["provider"] = provider
    sink["input_tokens"] = int(input_tokens or 0)
    sink["output_tokens"] = int(output_tokens or 0)


class BaseLLMProvider(ABC):
    """Interface chung mọi provider phải implement."""

    def __init__(self, settings) -> None:  # noqa: ANN001 -- app.settings.Settings, avoids import cycle
        self.settings = settings

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """Synchronous text generation. `system` is an optional system/
        instruction message; `kwargs` may include `temperature`/`max_tokens`
        overrides (falls back to `settings.llm_temperature`/`llm_max_tokens`
        when omitted)."""

    @abstractmethod
    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """Async counterpart of `generate()`. A provider without native
        async support may implement this by running `generate()` in a
        thread (see `OllamaProvider` for an example)."""
