"""Normalize files explicitly selected in the testing dashboard."""

from __future__ import annotations

import hashlib
import io
import mimetypes
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree

from ingestion.models import NormalizedBlock, NormalizedDocument, SourceLocation

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".html", ".htm",
}
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".sql", ".sh", ".ps1", ".css", ".scss", ".xml",
}


def _safe_relative_path(relative_path: str) -> str:
    """Keep browser-supplied paths relative and readable in search results."""
    normalized = relative_path.replace("\\", "/").lstrip("/")
    parts = [part for part in PurePosixPath(normalized).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError("file path is empty")
    return "/".join(parts)


def _text_document(path: str, media_type: str, content: bytes) -> NormalizedDocument:
    """Decode a text or source-code file into one provenance-aware block."""
    text = content.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ValueError("file contains no searchable text")
    extension = PurePosixPath(path).suffix.lower()
    block_kind = "code" if extension in CODE_EXTENSIONS else "paragraph"
    checksum = hashlib.sha256(content).hexdigest()
    return NormalizedDocument(
        id=f"local-{hashlib.sha256((path + checksum).encode()).hexdigest()[:16]}",
        source_path=path,
        media_type=media_type,
        checksum=checksum,
        blocks=[
            NormalizedBlock(
                kind=block_kind,
                text=text,
                location=SourceLocation(
                    source_path=path,
                    line_start=1,
                    line_end=text.count("\n") + 1,
                    char_start=0,
                    char_end=len(text),
                ),
                attributes={"source_path": path},
            )
        ],
    )


def _pdf_document(path: str, content: bytes) -> NormalizedDocument:
    """Extract searchable text page by page from a selected PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared by the project
        raise ValueError("PDF support is not installed; install project dependencies") from exc

    reader = PdfReader(io.BytesIO(content))
    blocks: list[NormalizedBlock] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(
                NormalizedBlock(
                    kind="paragraph",
                    text=text,
                    location=SourceLocation(source_path=path, page=page_number),
                )
            )
    if not blocks:
        raise ValueError("PDF contains no extractable text; scanned PDFs need OCR")
    checksum = hashlib.sha256(content).hexdigest()
    return NormalizedDocument(
        id=f"local-{hashlib.sha256((path + checksum).encode()).hexdigest()[:16]}",
        source_path=path,
        media_type="application/pdf",
        checksum=checksum,
        blocks=blocks,
    )


def _docx_document(path: str, content: bytes) -> NormalizedDocument:
    """Read paragraph text from a DOCX file without storing the upload on disk."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("DOCX file is invalid or unreadable") from exc

    root = ElementTree.fromstring(xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks: list[NormalizedBlock] = []
    for paragraph_index, paragraph in enumerate(root.iter(f"{namespace}p"), start=1):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()
        if text:
            blocks.append(
                NormalizedBlock(
                    kind="paragraph",
                    text=text,
                    location=SourceLocation(
                        source_path=path,
                        section=f"Paragraph {paragraph_index}",
                    ),
                )
            )
    if not blocks:
        raise ValueError("DOCX contains no searchable paragraph text")
    checksum = hashlib.sha256(content).hexdigest()
    return NormalizedDocument(
        id=f"local-{hashlib.sha256((path + checksum).encode()).hexdigest()[:16]}",
        source_path=path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        checksum=checksum,
        blocks=blocks,
    )


def normalize_local_file(
    relative_path: str,
    media_type: str | None,
    content: bytes,
) -> NormalizedDocument:
    """Convert one selected local file into BO's common document model."""
    path = _safe_relative_path(relative_path)
    extension = PurePosixPath(path).suffix.lower()
    detected_media_type = media_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
    if extension == ".pdf":
        return _pdf_document(path, content)
    if extension == ".docx":
        return _docx_document(path, content)
    is_text = (
        extension in TEXT_EXTENSIONS
        or extension in CODE_EXTENSIONS
        or detected_media_type.startswith("text/")
    )
    if is_text:
        return _text_document(path, detected_media_type, content)
    raise ValueError(f"unsupported file type: {extension or detected_media_type}")
