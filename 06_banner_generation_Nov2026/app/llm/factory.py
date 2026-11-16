"""Creates the configured `BaseLLMProvider` based on `settings.llm_provider`
(nhantn-mlai-project-structure standard). Adding a new provider: write a new
class implementing `BaseLLMProvider` under `app/llm/providers/`, register it
in `_REGISTRY` below -- nothing else in `app/` needs to change since callers
only ever go through `LLMFactory.create(settings)`.
"""
from __future__ import annotations

from app.llm.base import BaseLLMProvider
from app.llm.providers.claude_provider import ClaudeProvider
from app.llm.providers.deepseek_provider import DeepSeekProvider
from app.llm.providers.llama_provider import LlamaProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.settings import Settings


class LLMFactory:
    """Tạo provider phù hợp dựa trên settings.llm_provider."""

    _REGISTRY: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "llama": LlamaProvider,  # hosted cloud (Groq/Together/Bedrock...)
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider,  # self-host
        # Backward-compat alias: the real `.env` on disk still has the
        # pre-refactor value `LLM_PROVIDER=azure_gpt5mini` (this tool can't
        # write `.env` remotely -- see 4-project-evaluation-results.md) --
        # `OpenAIProvider` already resolves the Azure AI Foundry gpt-5-mini
        # deployment on its own (see its docstring), so this alias keeps that
        # existing value working without editing `.env` by hand.
        "azure_gpt5mini": OpenAIProvider,
    }

    @classmethod
    def create(cls, settings: Settings) -> BaseLLMProvider:
        provider_cls = cls._REGISTRY.get(settings.llm_provider)
        if provider_cls is None:
            raise ValueError(f"Unsupported llm_provider={settings.llm_provider!r}. Available: {list(cls._REGISTRY)}")
        return provider_cls(settings)

    @classmethod
    def available_providers(cls) -> list[str]:
        return list(cls._REGISTRY)


# ---------------------------------------------------------------------------
# Chạy độc lập file này (in provider hiện đang được chọn theo .env, và thử
# gọi generate() 1 lần để xác nhận cấu hình đúng):
#   python -m app.llm.factory --prompt "Xin chào"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from app.settings import get_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    settings = get_settings()
    print(f"llm_provider = {settings.llm_provider!r} (available: {LLMFactory.available_providers()})")
    provider = LLMFactory.create(settings)
    print(f"Using {type(provider).__name__} ...")
    print(provider.generate(args.prompt))
