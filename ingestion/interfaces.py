from pathlib import Path
from typing import Protocol

from ingestion.models import NormalizedBlock


class DocumentParser(Protocol):
    def supports(self, path: Path, media_type: str) -> bool:
        """Return whether this parser can read the given file and media type."""
        ...

    def parse(self, path: Path) -> list[NormalizedBlock]:
        """Convert a supported file into ordered normalized text blocks."""
        ...


class OCREngine(Protocol):
    def extract(self, image_bytes: bytes, *, language_hint: str | None = None) -> str:
        """Extract readable text from image bytes, optionally using a language hint."""
        ...
