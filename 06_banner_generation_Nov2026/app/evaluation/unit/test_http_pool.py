from __future__ import annotations

from app.providers import _http


class _Response:
    status_code = 200

    def json(self):
        return {"ok": True}


class _Session:
    def __init__(self):
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return _Response()


def test_post_json_reuses_thread_local_session(monkeypatch):
    session = _Session()
    monkeypatch.setattr(_http._session_local, "session", session, raising=False)

    assert _http.post_json("https://example.invalid/a", headers={}, json_body={}) == {"ok": True}
    assert _http.post_json("https://example.invalid/b", headers={}, json_body={}) == {"ok": True}
    assert session.calls == 2
