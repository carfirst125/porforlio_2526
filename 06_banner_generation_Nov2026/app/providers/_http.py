"""Small shared HTTP helper: POST with retry/backoff, used by every provider
client in `app/providers/*.py`. Centralized so `HTTP_TIMEOUT_SECONDS` /
`HTTP_MAX_RETRIES` (`.env`) are honored consistently everywhere.

2026-08-25: `requests` verifies TLS against `certifi`'s bundled CA list by
default, which does NOT include a corporate network's own root CA -- on a
machine behind a TLS-inspecting proxy this fails with
`SSLCertVerificationError: self-signed certificate in certificate chain`
(seen consistently calling the Azure LLM endpoint from `app/llm/providers/*`,
and intermittently calling Gemini). `truststore` makes `ssl` use the OS's own
trust store instead (where that corporate root CA is normally already
trusted), fixing verification properly -- this does NOT disable verification
(never use `verify=False` here)."""
from __future__ import annotations

import time
import threading
from contextlib import contextmanager
from contextvars import ContextVar

import requests
import truststore

truststore.inject_into_ssl()

from app.logging_conf import get_logger
from app.providers.base import ProviderError
from app.settings import get_settings

logger = get_logger("providers.http")

# `requests.Session` reuses TCP/TLS connections, which matters because one
# generated image can trigger several consecutive LLM/vision calls. Sessions
# are thread-local: version generation is concurrent, while requests.Session
# itself is not documented as thread-safe.
_session_local = threading.local()

# Repair/follow-up image calls already have a keep-best canvas. Retrying a
# 429 for 8s+16s just stalls the run; first-pass generate still uses the
# normal backoff below.
_fail_fast_on_429: ContextVar[bool] = ContextVar("fail_fast_on_429", default=False)


@contextmanager
def fail_fast_on_quota_exhaustion():
    token = _fail_fast_on_429.set(True)
    try:
        yield
    finally:
        _fail_fast_on_429.reset(token)


def _get_session() -> requests.Session:
    session = getattr(_session_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session_local.session = session
    return session


def post_json(url: str, *, headers: dict, json_body: dict, timeout: int | None = None) -> dict:
    return _post_with_retry(url, headers=headers, json=json_body, timeout=timeout).json()


def post_multipart(url: str, *, headers: dict, data: dict, files: list[tuple[str, tuple[str, bytes, str]]], timeout: int | None = None) -> dict:
    return _post_with_retry(url, headers=headers, data=data, files=files, timeout=timeout).json()


def _post_with_retry(url: str, *, timeout: int | None = None, **kwargs) -> requests.Response:
    settings = get_settings()
    timeout = timeout or settings.http_timeout_seconds
    max_retries = max(1, settings.http_max_retries)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _get_session().post(url, timeout=timeout, **kwargs)
            if resp.status_code >= 400:
                body_preview = resp.text[:500]
                raise ProviderError(f"HTTP {resp.status_code} from {url}: {body_preview}")
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("POST %s failed (attempt %d/%d): %s", url, attempt, max_retries, exc)
            if attempt < max_retries:
                delay = min(2 ** attempt, 10)
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if _fail_fast_on_429.get():
                        logger.warning(
                            "Quota/rate-limit on a follow-up generate — not retrying so keep-best can export the earlier image."
                        )
                        raise
                    delay = min(8 * (2 ** (attempt - 1)), 30)
                    logger.warning("Quota/rate-limit — waiting %ss before retry.", delay)
                time.sleep(delay)

    raise ProviderError(f"POST {url} failed after {max_retries} attempts: {last_exc}") from last_exc
