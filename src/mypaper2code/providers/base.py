from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str) -> str:
        """Return a completion for the prompt."""


@dataclass
class StubProvider:
    model: str = "stub"
    name: str = "stub"

    def complete(self, prompt: str) -> str:
        del prompt
        return "LLM generation is disabled in the default stub provider."


@dataclass
class OllamaProvider:
    model: str = "llama3"
    name: str = "ollama"

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Ollama execution is reserved for a later integration pass.")


@dataclass
class NvidiaProvider:
    model: str = "nvidia"
    name: str = "nvidia"

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "NVIDIA Build execution is reserved for a later integration pass."
        )


def provider_for(name: str, model: str) -> LLMProvider:
    normalized = name.lower()
    if normalized == "ollama":
        return OllamaProvider(model=model)
    if normalized == "nvidia":
        return NvidiaProvider(model=model)
    return StubProvider(model=model)
