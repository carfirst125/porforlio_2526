"""Hosted-cloud Llama provider -- targets any OpenAI-chat-compatible hosted
Llama endpoint (Groq, Together AI, AWS Bedrock's OpenAI-compatible gateway,
etc.) via `LLM_BASE_URL` + `LLAMA_API_KEY`. Not configured by default in
this project -- selectable any time by setting `LLM_PROVIDER=llama` plus
those two `.env` values.
"""
from __future__ import annotations

import asyncio

from app.llm.base import BaseLLMProvider, LLMProviderError, record_usage


class LlamaProvider(BaseLLMProvider):
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        from app.providers._http import post_json

        s = self.settings
        if not s.llm_base_url or not s.llama_api_key:
            raise LLMProviderError("LLM_BASE_URL / LLAMA_API_KEY not configured in .env for llm_provider=llama")

        headers = {"Authorization": f"Bearer {s.llama_api_key}", "Content-Type": "application/json"}
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": kwargs.get("model", s.llm_model_name),
            "messages": messages,
            "temperature": kwargs.get("temperature", s.llm_temperature),
            "max_tokens": kwargs.get("max_tokens", s.llm_max_tokens),
        }
        url = s.llm_base_url.rstrip("/") + "/chat/completions"
        payload = post_json(url, headers=headers, json_body=body)
        usage = payload.get("usage") or {}
        record_usage(
            kwargs.get("usage_sink"),
            model=body["model"],
            provider="llama",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Unexpected Llama response shape: {exc}; payload keys={list(payload.keys())}") from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.llm.providers.llama_provider --prompt "Xin chào"
#
# Yêu cầu trước khi chạy:
#   - LLM_BASE_URL trỏ tới 1 endpoint chat/completions tương thích OpenAI
#     (vd Groq: https://api.groq.com/openai/v1) + LLAMA_API_KEY tương ứng
#     trong .env (chưa được cấu hình mặc định trong project này).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    provider = LlamaProvider(get_settings())
    print(provider.generate(args.prompt, system=args.system))
