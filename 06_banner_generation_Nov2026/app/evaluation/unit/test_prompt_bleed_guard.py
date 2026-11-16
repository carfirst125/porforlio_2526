"""Unit tests -- app/core/prompt_bleed_guard.py (2026-08-20, user request:
a final LLM pass over the WHOLE assembled prompt, specifically hunting for
and removing any wording that lets the PRODUCT asset bleed past the safe
area, after a real gemini-3.1-flash run showed exactly that -- see
`4-project-evaluation-results.md`'s 2026-08-20 (13) entry)."""
from __future__ import annotations

import pytest

from app.core import prompt_bleed_guard
from app.llm.base import LLMProviderError

_PROMPT_TEXT = "CANVAS & SAFE AREA: ...\n\nFORMAT DESIGN GUIDELINE: phone imagery may extend past the margin.\n\nCRITICAL REMINDER: ..."


class _FakeProvider:
    def __init__(self, response_text: str | None = None, raise_error: bool = False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        self.calls.append((prompt, system))
        if self._raise_error:
            raise LLMProviderError("simulated LLM failure")
        return self._response_text


@pytest.fixture
def fake_llm_factory(monkeypatch):
    provider_holder = {"provider": _FakeProvider(response_text="CORRECTED PROMPT -- product must stay in the safe area, no exceptions.")}

    def _fake_create(settings):
        return provider_holder["provider"]

    monkeypatch.setattr(prompt_bleed_guard.LLMFactory, "create", staticmethod(_fake_create))
    return provider_holder


def test_returns_none_for_empty_prompt_text():
    assert prompt_bleed_guard.enforce_product_containment("") is None
    assert prompt_bleed_guard.enforce_product_containment("   ") is None


def test_calls_the_llm_with_the_full_prompt_and_a_system_instruction(fake_llm_factory):
    result = prompt_bleed_guard.enforce_product_containment(_PROMPT_TEXT)
    assert result == "CORRECTED PROMPT -- product must stay in the safe area, no exceptions."

    calls = fake_llm_factory["provider"].calls
    assert len(calls) == 1
    prompt_sent, system_sent = calls[0]
    assert prompt_sent == _PROMPT_TEXT  # the whole prompt is handed over verbatim, not paraphrased first
    assert system_sent and "product" in system_sent.lower()


def test_falls_back_to_none_when_the_llm_call_fails(monkeypatch):
    failing_provider = _FakeProvider(raise_error=True)
    monkeypatch.setattr(prompt_bleed_guard.LLMFactory, "create", staticmethod(lambda settings: failing_provider))
    assert prompt_bleed_guard.enforce_product_containment(_PROMPT_TEXT) is None


def test_falls_back_to_none_when_the_llm_returns_an_empty_response(monkeypatch):
    empty_provider = _FakeProvider(response_text="   ")
    monkeypatch.setattr(prompt_bleed_guard.LLMFactory, "create", staticmethod(lambda settings: empty_provider))
    assert prompt_bleed_guard.enforce_product_containment(_PROMPT_TEXT) is None
