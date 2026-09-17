from dataclasses import dataclass
from pathlib import Path

from ingestion.interfaces import DocumentParser, OCREngine
from ingestion.models import NormalizedDocument


@dataclass(frozen=True)
class IngestionDependencies:
    parsers: tuple[DocumentParser, ...]
    ocr: OCREngine | None = None


def normalize_file(path: Path, media_type: str, deps: IngestionDependencies) -> NormalizedDocument:
    """Select an adapter and normalize a file without deciding chunk boundaries."""
    parser = next((candidate for candidate in deps.parsers if candidate.supports(path, media_type)), None)
    if parser is None:
        raise ValueError(f"No parser configured for {media_type}: {path}")
    raise NotImplementedError(
        "Parser adapters and checksum persistence are infrastructure follow-up work; "
        "chunking remains separately student-owned."
    )

