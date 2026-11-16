"""Unit tests -- app/core/guideline_rewriter.py (2026-08-20, user request:
rewrite a FORMAT DESIGN GUIDELINE's margin/frame numbers via GPT-5-mini so
they describe the SAFE AREA correctly instead of contradicting it -- see
that module's own docstring, and `4-project-evaluation-results.md`'s
2026-08-20 (11) entry, for the full "banner_popup_noti allows phone/app-UI
imagery to bleed past its margin" investigation this fixes)."""
from __future__ import annotations

import pytest

from app.core import guideline_rewriter
from app.llm.base import LLMProviderError

_ORIGINAL_TEXT = (
    "### Guideline: banner_popup_noti\n"
    "4. BIEN AN TOAN\n"
    "- Giu khoang cach giua vung minh hoa den 4 canh cua banner it nhat 48px\n"
    "- Doi voi hinh anh dien thoai co giao dien app, co the phong to vuot ra ngoai bien an toan"
)


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
    """Replaces `LLMFactory.create()` (imported into guideline_rewriter.py
    as `LLMFactory`) with a controllable fake -- no real network/credential
    dependency, mirrors the pattern already used for image-model mocking
    (`fake_generate_image` in conftest.py)."""
    provider_holder = {"provider": _FakeProvider(response_text="REWRITTEN (fake LLM output)")}

    def _fake_create(settings):
        return provider_holder["provider"]

    monkeypatch.setattr(guideline_rewriter.LLMFactory, "create", staticmethod(_fake_create))
    return provider_holder


def test_rewritten_guideline_filename_matches_the_users_convention():
    name = guideline_rewriter.rewritten_guideline_filename("banner_popup_noti", (1536, 1024), (1035, 420))
    assert name == "banner_popup_noti_1536x1024_1035x420.txt"


def test_rewrite_returns_none_when_guideline_text_is_empty(isolated_safe_area_guide_dir):
    assert guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", "", (1536, 1024), (250, 302, 1035, 420)) is None
    assert guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", "   ", (1536, 1024), (250, 302, 1035, 420)) is None


def test_rewrite_calls_the_llm_and_caches_the_result(isolated_safe_area_guide_dir, fake_llm_factory):
    result = guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    assert result == "REWRITTEN (fake LLM output)"
    assert len(fake_llm_factory["provider"].calls) == 1

    dest = isolated_safe_area_guide_dir / "banner_popup_noti_1536x1024_1035x420.txt"
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "REWRITTEN (fake LLM output)"


def test_rewrite_reuses_the_cached_file_without_calling_the_llm_again(isolated_safe_area_guide_dir, fake_llm_factory):
    first = guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    second = guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    assert first == second
    # The whole point (user's explicit ask, "tiết kiệm thời gian gọi LLM"):
    # only 1 real LLM call across both invocations.
    assert len(fake_llm_factory["provider"].calls) == 1


def test_rewrite_prompt_gives_the_llm_precomputed_per_side_offsets(isolated_safe_area_guide_dir, fake_llm_factory):
    """The LLM must be handed already-correct arithmetic (left/right/top/
    bottom pixel offsets), not asked to reconstruct the geometry itself --
    canvas 1536x1024, safe area 1035x420 at (250, 302) -> left=250,
    right=1536-(250+1035)=251, top=302, bottom=1024-(302+420)=302."""
    guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    prompt_sent, system_sent = fake_llm_factory["provider"].calls[0]
    assert "left=250px" in prompt_sent
    assert "right=251px" in prompt_sent
    assert "top=302px" in prompt_sent
    assert "bottom=302px" in prompt_sent
    assert system_sent  # a system instruction was provided, not just a bare prompt


def test_rewrite_falls_back_to_none_when_the_llm_call_fails(isolated_safe_area_guide_dir, monkeypatch):
    failing_provider = _FakeProvider(raise_error=True)
    monkeypatch.setattr(guideline_rewriter.LLMFactory, "create", staticmethod(lambda settings: failing_provider))

    result = guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    assert result is None
    # A failed attempt must NOT leave a cache file behind for the next call
    # to (incorrectly) treat as a successful rewrite.
    dest = isolated_safe_area_guide_dir / "banner_popup_noti_1536x1024_1035x420.txt"
    assert not dest.exists()


def test_rewrite_is_shared_across_different_format_types_and_geometries(isolated_safe_area_guide_dir, fake_llm_factory):
    """2 different format_types (or geometries) must NOT collide on the
    same cache file -- each gets its own."""
    guideline_rewriter.rewrite_guideline_for_safe_area("banner_popup_noti", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))
    guideline_rewriter.rewrite_guideline_for_safe_area("banner_other_format", _ORIGINAL_TEXT, (1536, 1024), (250, 302, 1035, 420))

    assert (isolated_safe_area_guide_dir / "banner_popup_noti_1536x1024_1035x420.txt").exists()
    assert (isolated_safe_area_guide_dir / "banner_other_format_1536x1024_1035x420.txt").exists()
    assert len(fake_llm_factory["provider"].calls) == 2
