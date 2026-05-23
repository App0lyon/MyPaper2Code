from __future__ import annotations

import os

import pytest

from mypaper2code.providers.base import (
    NVIDIA_DEFAULT_MODEL,
    NvidiaProvider,
    OllamaProvider,
    ProviderError,
    load_env_file,
    provider_for,
)


def test_ollama_provider_posts_chat_payload() -> None:
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"message": {"content": "ok"}}

    provider = OllamaProvider(
        model="llama3",
        base_url="http://localhost:11434",
        transport=transport,
    )

    assert provider.complete("hello", system="system") == "ok"
    url, payload, headers, _ = calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "llama3"
    assert payload["messages"][0]["role"] == "system"
    assert headers["Content-Type"] == "application/json"


def test_nvidia_provider_posts_openai_compatible_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = NvidiaProvider(
        model="meta/test",
        base_url="https://integrate.api.nvidia.com/v1",
        transport=transport,
    )

    assert provider.complete("hello") == "ok"
    url, payload, headers, _ = calls[0]
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert payload["model"] == "meta/test"
    assert headers["Authorization"] == "Bearer secret"


def test_nvidia_provider_requires_api_key(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    provider = NvidiaProvider(model="meta/test")

    with pytest.raises(ProviderError):
        provider.complete("hello")


def test_provider_for_uses_requested_provider() -> None:
    assert provider_for("ollama", "llama3").name == "ollama"
    assert provider_for("nvidia", "meta/test").name == "nvidia"


def test_provider_for_rejects_stub_provider() -> None:
    with pytest.raises(ProviderError):
        provider_for("stub", "stub")


def test_provider_for_uses_nvidia_default_model_for_legacy_stub_model() -> None:
    provider = provider_for("nvidia", "stub")

    assert provider.model == NVIDIA_DEFAULT_MODEL


def test_nvidia_provider_reads_api_key_from_dotenv(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('NVIDIA_API_KEY="from-dotenv"\n', encoding="utf-8")

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = NvidiaProvider(model="meta/test", transport=transport)

    assert provider.complete("hello") == "ok"
    assert calls[0][2]["Authorization"] == "Bearer from-dotenv"


def test_load_env_file_does_not_override_existing_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "from-env")
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=from-file\n", encoding="utf-8")

    load_env_file(start=tmp_path)

    assert os.environ["NVIDIA_API_KEY"] == "from-env"
