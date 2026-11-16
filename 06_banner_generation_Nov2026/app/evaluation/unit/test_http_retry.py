from __future__ import annotations

import pytest

from app.providers._http import _post_with_retry, fail_fast_on_quota_exhaustion
from app.providers.base import ProviderError


class _QuotaResp:
    status_code = 429
    text = '{"error": {"message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}}'


class _QuotaSession:
    def __init__(self, calls: dict):
        self._calls = calls

    def post(self, *_args, **_kwargs):
        self._calls["n"] += 1
        return _QuotaResp()


def test_fail_fast_on_quota_does_not_backoff(monkeypatch):
    from app.providers import _http
    from app.settings import get_settings

    calls = {"n": 0}
    slept: list[float] = []

    monkeypatch.setattr(_http, "_get_session", lambda: _QuotaSession(calls))
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    monkeypatch.setattr(get_settings(), "http_max_retries", 3)
    monkeypatch.setattr(get_settings(), "http_timeout_seconds", 1)

    with pytest.raises(ProviderError, match="429"):
        with fail_fast_on_quota_exhaustion():
            _post_with_retry("https://example.test/generate", headers={}, json={})

    assert calls["n"] == 1
    assert slept == []


def test_quota_without_fail_fast_still_retries(monkeypatch):
    from app.providers import _http
    from app.settings import get_settings

    calls = {"n": 0}
    slept: list[float] = []

    monkeypatch.setattr(_http, "_get_session", lambda: _QuotaSession(calls))
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    monkeypatch.setattr(get_settings(), "http_max_retries", 3)
    monkeypatch.setattr(get_settings(), "http_timeout_seconds", 1)

    with pytest.raises(ProviderError, match="failed after 3 attempts"):
        _post_with_retry("https://example.test/generate", headers={}, json={})

    assert calls["n"] == 3
    assert slept == [8, 16]
