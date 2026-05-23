from __future__ import annotations

from pathlib import Path

from mypaper2code.core.io import read_json, write_json
from mypaper2code.core.models import ResearchAmbiguity, ResearchUnderstanding, UserDecision


def decisions_path(workspace: Path) -> Path:
    return workspace / "decisions" / "decisions.json"


def approvals_path(workspace: Path) -> Path:
    return workspace / "decisions" / "approvals.json"


def load_decisions(workspace: Path) -> dict[str, UserDecision]:
    path = decisions_path(workspace)
    if not path.exists():
        return {}
    return {
        item["decision_id"]: UserDecision.model_validate(item)
        for item in read_json(path)
    }


def save_decision(workspace: Path, decision_id: str, value: str) -> UserDecision:
    decisions = load_decisions(workspace)
    decision = UserDecision(decision_id=decision_id, value=value)
    decisions[decision_id] = decision
    write_json(
        decisions_path(workspace),
        [item.model_dump(mode="json") for item in decisions.values()],
    )
    return decision


def load_approvals(workspace: Path) -> dict[str, bool]:
    path = approvals_path(workspace)
    if not path.exists():
        return {}
    return {str(key): bool(value) for key, value in read_json(path).items()}


def save_approval(workspace: Path, approval_id: str) -> None:
    approvals = load_approvals(workspace)
    approvals[approval_id] = True
    write_json(approvals_path(workspace), approvals)


def unresolved_blocking_ambiguities(
    understanding: ResearchUnderstanding,
    workspace: Path,
) -> list[ResearchAmbiguity]:
    decisions = load_decisions(workspace)
    approvals = load_approvals(workspace)
    unresolved: list[ResearchAmbiguity] = []
    for ambiguity in understanding.ambiguities:
        if ambiguity.severity != "blocking":
            continue
        if ambiguity.ambiguity_id in decisions or approvals.get(ambiguity.ambiguity_id):
            continue
        unresolved.append(ambiguity)
    return unresolved
