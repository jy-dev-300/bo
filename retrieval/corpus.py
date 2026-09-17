import json
from pathlib import Path

from ingestion.models import Chunk


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Pre-chunked corpus not found: {path}")

    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid chunk at {path}:{line_number}") from exc
    return chunks

