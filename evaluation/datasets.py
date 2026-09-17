import json
from pathlib import Path

from evaluation.models import RetrievalGoldCase


def load_retrieval_gold(path: Path) -> list[RetrievalGoldCase]:
    cases: list[RetrievalGoldCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(RetrievalGoldCase.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid retrieval case at {path}:{line_number}") from exc
    return cases

