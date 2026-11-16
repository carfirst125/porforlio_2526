"""DEPRECATED -- moved to `app/llm/providers/openai_provider.py` as part of
the `nhantn-mlai-project-structure` refactor (2026-08-20), so gpt-5-mini
prompt refinement goes through the standard `BaseLLMProvider` + `LLMFactory`
swappable-LLM pattern instead of a one-off function. Nothing in `app/`
imports this file anymore (see `app/core/prompt_builder.py`, which now calls
`app.llm.factory.LLMFactory`). Safe to delete; kept only as a pointer in
case something external still references the old path.

See: `4-project-evaluation-results.md` (2026-08-20 entry) for the full
rationale, and `app/llm/providers/openai_provider.py` for the current
implementation (same Azure AI Foundry endpoint, same behavior).
"""
from __future__ import annotations

from app.llm.providers.openai_provider import OpenAIProvider  # noqa: F401 -- re-exported for old imports

__all__ = ["OpenAIProvider"]
