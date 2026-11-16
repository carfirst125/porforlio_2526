"""Anthropic Claude provider -- calls the public Messages API
(`https://api.anthropic.com/v1/messages`) directly over HTTP (no `anthropic`
SDK dependency, consistent with the rest of `app/` using plain `requests`).
Requires `ANTHROPIC_API_KEY` in `.env`; not configured by default in this
project (its real deployed LLM is Azure `gpt-5-mini`, see
`openai_provider.py`) -- selectable any time by setting `LLM_PROVIDER=claude`.
"""
from __future__ import annotations

import asyncio

from app.llm.base import BaseLLMProvider, LLMProviderError, record_usage

_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class ClaudeProvider(BaseLLMProvider):
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        from app.providers._http import post_json

        s = self.settings
        if not s.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY not configured in .env")

        headers = {
            "x-api-key": s.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        body: dict = {
            "model": kwargs.get("model", s.llm_model_name),
            "max_tokens": kwargs.get("max_tokens", s.llm_max_tokens),
            "temperature": kwargs.get("temperature", s.llm_temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        payload = post_json(s.llm_base_url or _MESSAGES_URL, headers=headers, json_body=body)
        usage = payload.get("usage") or {}
        record_usage(
            kwargs.get("usage_sink"),
            model=body["model"],
            provider="claude",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        try:
            return "\n".join(block["text"] for block in payload["content"] if block.get("type") == "text").strip()
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Claude response shape: {exc}; payload keys={list(payload.keys())}") from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.llm.providers.claude_provider --prompt "Xin chào"
#
# Yêu cầu trước khi chạy:
#   - Biến ANTHROPIC_API_KEY trong .env đã có giá trị thật (chưa được cấu
#     hình mặc định trong project này).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    provider = ClaudeProvider(get_settings())
    print(provider.generate(args.prompt, system=args.system))
