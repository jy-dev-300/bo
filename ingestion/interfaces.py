from pathlib import Path
from typing import Protocol

from ingestion.models import NormalizedBlock


class DocumentParser(Protocol):
    def supports(self, path: Path, media_type: str) -> bool: ...

    def parse(self, path: Path) -> list[NormalizedBlock]: ...


class OCREngine(Protocol):
    def extract(self, image_bytes: bytes, *, language_hint: str | None = None) -> str: ...

