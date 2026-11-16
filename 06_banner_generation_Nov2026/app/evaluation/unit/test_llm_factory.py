"""Unit tests -- app/llm/factory.py + app/llm/base.py (nhantn-mlai-project-structure
BaseLLMProvider + LLMFactory pattern: swap LLM by config, no code change)."""
from __future__ import annotations

import pytest

from app.llm.base import BaseLLMProvider
from app.llm.factory import LLMFactory
from app.llm.providers.claude_provider import ClaudeProvider
from app.llm.providers.deepseek_provider import DeepSeekProvider
from app.llm.providers.llama_provider import LlamaProvider
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.openai_provider import OpenAIProvider
from app.settings import get_settings


@pytest.mark.parametrize(
    "provider_name,expected_cls",
    [
        ("openai", OpenAIProvider),
        ("claude", ClaudeProvider),
        ("llama", LlamaProvider),
        ("deepseek", DeepSeekProvider),
        ("ollama", OllamaProvider),
        # Backward-compat alias for the value already baked into the real
        # `.env` on disk (`LLM_PROVIDER=azure_gpt5mini`), see factory.py.
        ("azure_gpt5mini", OpenAIProvider),
    ],
)
def test_factory_creates_expected_provider(provider_name, expected_cls):
    settings = get_settings().model_copy(update={"llm_provider": provider_name})
    provider = LLMFactory.create(settings)
    assert isinstance(provider, expected_cls)
    assert isinstance(provider, BaseLLMProvider)


def test_factory_unknown_provider_raises_value_error():
    settings = get_settings().model_copy(update={"llm_provider": "not-a-real-provider"})
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        LLMFactory.create(settings)


def test_available_providers_lists_every_registered_name():
    names = LLMFactory.available_providers()
    for expected in ("openai", "claude", "llama", "deepseek", "ollama"):
        assert expected in names


def test_openai_provider_raises_clear_error_without_credentials():
    from app.llm.base import LLMProviderError

    settings = get_settings().model_copy(
        update={"llm_provider": "openai", "az_gpt5mini_endpoint": "", "az_gpt5mini_key": "", "llm_base_url": None, "openai_api_key": None}
    )
    provider = LLMFactory.create(settings)
    with pytest.raises(LLMProviderError):
        provider.generate("hello")
