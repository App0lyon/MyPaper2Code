from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import read_json
from mypaper2code.core.models import ImplementationTrace
from mypaper2code.core.text import excerpt, tokenize

CODE_EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".txt"}


class CodeQuestionAnswerer:
    def answer(self, workspace: Path, question: str, limit: int = 5) -> list[str]:
        generated = workspace / "generated_code"
        if not generated.exists():
            raise FileNotFoundError(f"Generated code not found: {generated}")

        query_terms = set(tokenize(question))
        answers = self._trace_matches(workspace, query_terms)
        answers.extend(
            self._file_matches(generated, query_terms, limit=max(limit - len(answers), 0))
        )
        return answers[:limit]

    @staticmethod
    def _trace_matches(workspace: Path, query_terms: set[str]) -> list[str]:
        trace_path = workspace / "analysis" / "implementation_trace.json"
        if not trace_path.exists():
            return []
        trace = ImplementationTrace.model_validate(read_json(trace_path))
        matches: list[str] = []
        for entry in trace.entries:
            haystack = " ".join(
                [
                    entry.step_id,
                    entry.paper_claim,
                    " ".join(location.file for location in entry.implemented_in),
                    " ".join(location.symbol or "" for location in entry.implemented_in),
                ]
            )
            if query_terms & set(tokenize(haystack)):
                locations = ", ".join(
                    f"{location.file}"
                    + (f":{location.line_start}" if location.line_start else "")
                    + (f" `{location.symbol}`" if location.symbol else "")
                    for location in entry.implemented_in
                )
                source = ""
                if entry.source:
                    source = (
                        f" Source papier: page {entry.source.page}, "
                        f"section {entry.source.section}."
                    )
                matches.append(f"{entry.step_id}: {entry.paper_claim} -> {locations}.{source}")
        return matches

    @staticmethod
    def _file_matches(generated: Path, query_terms: set[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        scored: list[tuple[int, str]] = []
        for path in generated.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
                continue
            relative = path.relative_to(generated).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                score = len(query_terms & set(tokenize(relative + " " + line)))
                if score:
                    scored.append((score, f"{relative}:{line_number}: {excerpt(line, 220)}"))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in scored[:limit]]
