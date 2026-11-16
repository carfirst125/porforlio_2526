"""OpenAI-compatible provider -- talks to the real public OpenAI API when
`llm_base_url`/`openai_api_key` are set in `.env`, otherwise falls back to
this project's actual deployed LLM: Azure AI Foundry's `gpt-5-mini`
(`AZ_GPT5MINI_ENDPOINT`/`AZ_GPT5MINI_KEY`), since Azure OpenAI / AI
Foundry speaks the same OpenAI-compatible Responses API shape. This is why
`llm_provider=openai` (the default) works out of the box with the
credentials already in `.env` without any extra configuration.

2026-08-20 (user-reported bug): these 2 env var names used to read
`AZ_GPI_5_MINI_ENDPOINT`/`AZ_GPI_5_MINI_KEY` here (and in `.env.example`,
and in a real user `.env`) -- a typo/inconsistent naming that did NOT
match what pydantic-settings actually derives from
`Settings.az_gpt5mini_endpoint`/`az_gpt5mini_key` (this project's own
convention: env var name = uppercased field name). A `.env` filled in
under the old names was silently ignored by `Settings` -- real credentials
sat there unused while every GPT-5-mini call kept failing with "No
OpenAI-compatible endpoint configured", which is exactly what happened.
Fixed here and in `.env.example`; see `4-project-evaluation-results.md`
for the full investigation.
"""
from __future__ import annotations

import asyncio
import base64

from app.llm.base import BaseLLMProvider, LLMProviderError, record_usage

_PUBLIC_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIProvider(BaseLLMProvider):
    def _resolve_endpoint_and_key(self) -> tuple[str, str]:
        s = self.settings
        if s.llm_base_url and s.openai_api_key:
            return s.llm_base_url, s.openai_api_key
        if s.llm_base_url and not s.openai_api_key:
            raise LLMProviderError("llm_base_url is set but openai_api_key is empty in .env")
        if s.az_gpt5mini_endpoint and s.az_gpt5mini_key:
            return s.az_gpt5mini_endpoint, s.az_gpt5mini_key
        if s.openai_api_key:
            return _PUBLIC_OPENAI_RESPONSES_URL, s.openai_api_key
        raise LLMProviderError(
            "No OpenAI-compatible endpoint configured -- set either "
            "(AZ_GPT5MINI_ENDPOINT + AZ_GPT5MINI_KEY) for the Azure deployment, "
            "or (LLM_BASE_URL + OPENAI_API_KEY) for a custom endpoint, "
            "or just OPENAI_API_KEY to use the public OpenAI API."
        )

    def _extract_text(self, payload: dict) -> str:
        text = payload.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        chunks: list[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                t = content.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()
        raise LLMProviderError(f"Could not find text output in response: keys={list(payload.keys())}")

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        from app.providers._http import post_json  # local import: avoid app.llm <-> app.providers cycle at module load

        endpoint, api_key = self._resolve_endpoint_and_key()
        model = kwargs.get("model", self.settings.llm_model_name)
        # 2026-08-25: the Responses API rejects an `input` item with no
        # explicit `type` (400 invalid_value on `input[1]`, once SSL
        # verification itself was fixed and requests actually reached the
        # server) -- every item needs `"type": "message"` set explicitly.
        messages = ([{"type": "message", "role": "system", "content": system}] if system else []) + [
            {"type": "message", "role": "user", "content": prompt}
        ]
        body = {
            "model": model,
            "input": messages,
        }
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        payload = post_json(endpoint, headers=headers, json_body=body)
        usage = payload.get("usage") or {}
        record_usage(
            kwargs.get("usage_sink"),
            model=model,
            provider="openai",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return self._extract_text(payload)

    async def agenerate(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, system=system, **kwargs)

    def generate_vision(
        self,
        prompt: str,
        images: list[tuple[bytes, str]],
        *,
        system: str | None = None,
        usage_sink: dict | None = None,
    ) -> str:
        """2026-08-25: image-input counterpart of `generate()`, added for
        `app/core/output_validator.py`'s post-generation presence checks
        (verifying a required bucket asset actually shows up in the FINAL
        image) -- NOT part of `BaseLLMProvider`'s abstract interface, since
        only this project's real deployed model (Azure `gpt-5-mini` via the
        Responses API) is known to support image input the way it's called
        here; callers must `hasattr(provider, "generate_vision")` and skip
        the check gracefully on any other configured provider.

        `images` is an ORDERED list of `(image_bytes, mime_type)` -- e.g.
        [generated_image, reference_asset] so the prompt can refer to "the
        FIRST image" / "the SECOND image". `usage_sink`, when passed, is
        filled in with this call's token usage (2026-09-06 user request --
        `app/core/output_validator.py` can call this several times per
        (model, version) generation, each one its own billable call)."""
        from app.providers._http import post_json  # local import: avoid app.llm <-> app.providers cycle at module load

        endpoint, api_key = self._resolve_endpoint_and_key()
        image_parts = [
            {"type": "input_image", "image_url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"}
            for data, mime_type in images
        ]
        messages = ([{"type": "message", "role": "system", "content": system}] if system else []) + [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, *image_parts],
            }
        ]
        # 2026-08-25: verified against a real run that these simple yes/no
        # classification calls take long enough at the deployment's default
        # reasoning effort ("medium") to occasionally exceed HTTP_TIMEOUT_SECONDS
        # (a real `Read timed out` was observed) -- "low" is plenty for a
        # single yes/no judgment and noticeably faster.
        body = {"model": self.settings.llm_model_name, "input": messages, "reasoning": {"effort": "low"}}
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        payload = post_json(endpoint, headers=headers, json_body=body)
        usage = payload.get("usage") or {}
        record_usage(
            usage_sink,
            model=self.settings.llm_model_name,
            provider="openai",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return self._extract_text(payload)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   python -m app.llm.providers.openai_provider --prompt "Xin chào, giới thiệu ngắn về bạn"
#
# Yêu cầu trước khi chạy:
#   - `.env` có AZ_GPT5MINI_ENDPOINT + AZ_GPT5MINI_KEY (deployment thật của
#     project), HOẶC LLM_BASE_URL + OPENAI_API_KEY, HOẶC chỉ OPENAI_API_KEY
#     để gọi thẳng public OpenAI API.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    provider = OpenAIProvider(get_settings())
    print(provider.generate(args.prompt, system=args.system))
