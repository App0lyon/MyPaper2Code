from __future__ import annotations

import re
import unicodedata


def slugify(value: str, fallback: str = "paper") -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return slug or fallback


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def excerpt(text: str, max_chars: int = 420) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())
