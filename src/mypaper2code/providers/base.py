from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

JsonDict = dict[str, Any]
Transport = Callable[[str, JsonDict, dict[str, str], float], JsonDict]
NVIDIA_DEFAULT_MODEL = "mistralai/mistral-medium-3.5-128b"
OLLAMA_DEFAULT_MODEL = "llama3"
SUPPORTED_PROVIDERS = {"nvidia", "ollama"}


class ProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Return a completion for the prompt."""


def _post_json(url: str, payload: JsonDict, headers: dict[str, str], timeout: float) -> JsonDict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc.reason}") from exc
    return json.loads(raw)


@dataclass
class OllamaProvider:
    model: str = OLLAMA_DEFAULT_MODEL
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    transport: Transport = _post_json
    name: str = "ollama"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = self.transport(
            f"{self.base_url.rstrip('/')}/api/chat",
            payload,
            {"Content-Type": "application/json"},
            self.timeout,
        )
        try:
            return str(data["message"]["content"])
        except KeyError as exc:
            raise ProviderError(f"Unexpected Ollama response shape: {data}") from exc


@dataclass
class NvidiaProvider:
    model: str = NVIDIA_DEFAULT_MODEL
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    env_file: str = ".env"
    timeout: float = 120.0
    transport: Transport = _post_json
    name: str = "nvidia"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        load_env_file(self.env_file)
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"NVIDIA provider requires `{self.api_key_env}` in the environment "
                f"or in `{self.env_file}`."
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self.transport(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            self.timeout,
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected NVIDIA response shape: {data}") from exc


def provider_for(
    name: str,
    model: str,
    ollama_base_url: str | None = None,
    nvidia_base_url: str | None = None,
    nvidia_api_key_env: str | None = None,
) -> LLMProvider:
    normalized = name.lower()
    if normalized == "ollama":
        return OllamaProvider(
            model=model if model and model != "stub" else OLLAMA_DEFAULT_MODEL,
            base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if normalized == "nvidia":
        return NvidiaProvider(
            model=model if model and model != "stub" else NVIDIA_DEFAULT_MODEL,
            base_url=nvidia_base_url
            or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key_env=nvidia_api_key_env or "NVIDIA_API_KEY",
        )
    raise ProviderError(
        f"Unsupported provider `{name}`. Supported providers are: "
        f"{', '.join(sorted(SUPPORTED_PROVIDERS))}."
    )


def load_env_file(env_file: str = ".env", start: Path | None = None) -> None:
    path = find_env_file(env_file, start=start)
    if not path:
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_env_value(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def find_env_file(env_file: str = ".env", start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    candidates = [current] if current.is_dir() else [current.parent]
    candidates.extend(candidates[0].parents)
    for directory in candidates:
        path = directory / env_file
        if path.exists() and path.is_file():
            return path
    return None


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
