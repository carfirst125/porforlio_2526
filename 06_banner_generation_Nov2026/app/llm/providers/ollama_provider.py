"""Self-hosted Ollama provider -- calls a local (or LAN) Ollama instance's
`/api/chat` endpoint (`OLLAMA_BASE_URL` in `.env`, default
`http://localhost:11434`). No API key needed. This is the recommended
provider for testing the `app/llm/` abstraction without spending any API
credit -- `LLM_PROVIDER=ollama`, `LLM_MODEL_NAME=llama3` (or whatever model
you've `ollama pull`ed).
"""
from __future__ import annotations

import asyncio

from app.llm.base import BaseLLMProvider, LLMProviderError, record_usage


class OllamaProvider(BaseLLMProvider):
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        from app.providers._http import post_json

        s = self.settings
        base_url = s.llm_base_url or s.ollama_base_url
        if not base_url:
            raise LLMProviderError("OLLAMA_BASE_URL not configured in .env")

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": kwargs.get("model", s.llm_model_name),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", s.llm_temperature),
                "num_predict": kwargs.get("max_tokens", s.llm_max_tokens),
            },
        }
        url = base_url.rstrip("/") + "/api/chat"
        payload = post_json(url, headers={"Content-Type": "application/json"}, json_body=body)
        # Ollama reports usage counters at the TOP LEVEL of the response
        # (not nested under a "usage" key like every OpenAI-shaped API).
        record_usage(
            kwargs.get("usage_sink"),
            model=body["model"],
            provider="ollama",
            input_tokens=payload.get("prompt_eval_count", 0),
            output_tokens=payload.get("eval_count", 0),
        )
        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(
                f"Unexpected Ollama response shape: {exc}; payload keys={list(payload.keys())} -- "
                f"is the model pulled? try: ollama pull {body['model']}"
            ) from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.llm.providers.ollama_provider --prompt "xin chào" --model llama3
#
# Yêu cầu trước khi chạy:
#   - Đã cài và chạy Ollama local (`ollama serve`), model đã pull sẵn
#     (`ollama pull llama3`)
#   - Biến OLLAMA_BASE_URL trong .env trỏ đúng tới instance đang chạy
#     (mặc định http://localhost:11434 nếu chạy local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--model", default=None, help="Override LLM_MODEL_NAME for this call")
    args = parser.parse_args()

    provider = OllamaProvider(get_settings())
    kwargs = {"model": args.model} if args.model else {}
    print(provider.generate(args.prompt, system=args.system, **kwargs))
