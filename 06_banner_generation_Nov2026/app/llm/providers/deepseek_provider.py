"""DeepSeek provider -- calls DeepSeek's OpenAI-chat-compatible API
(`https://api.deepseek.com/chat/completions` by default, overridable via
`LLM_BASE_URL`). Requires `DEEPSEEK_API_KEY` in `.env`; not configured by
default in this project -- selectable any time by setting
`LLM_PROVIDER=deepseek`.
"""
from __future__ import annotations

import asyncio

from app.llm.base import BaseLLMProvider, LLMProviderError, record_usage

_DEFAULT_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekProvider(BaseLLMProvider):
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        from app.providers._http import post_json

        s = self.settings
        if not s.deepseek_api_key:
            raise LLMProviderError("DEEPSEEK_API_KEY not configured in .env")

        headers = {"Authorization": f"Bearer {s.deepseek_api_key}", "Content-Type": "application/json"}
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": kwargs.get("model", s.llm_model_name if s.llm_model_name != "gpt-5-mini" else "deepseek-chat"),
            "messages": messages,
            "temperature": kwargs.get("temperature", s.llm_temperature),
            "max_tokens": kwargs.get("max_tokens", s.llm_max_tokens),
        }
        payload = post_json(s.llm_base_url or _DEFAULT_URL, headers=headers, json_body=body)
        usage = payload.get("usage") or {}
        record_usage(
            kwargs.get("usage_sink"),
            model=body["model"],
            provider="deepseek",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected DeepSeek response shape: {exc}; payload keys={list(payload.keys())}") from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.llm.providers.deepseek_provider --prompt "Xin chào"
#
# Yêu cầu trước khi chạy:
#   - Biến DEEPSEEK_API_KEY trong .env đã có giá trị thật (chưa được cấu
#     hình mặc định trong project này).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    provider = DeepSeekProvider(get_settings())
    print(provider.generate(args.prompt, system=args.system))
