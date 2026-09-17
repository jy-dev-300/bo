from typing import Any, Literal

from pydantic import BaseModel, Field


# Describes where a piece of text came from within its original source.
class SourceLocation(BaseModel):
    # Stores the path of the original file; this value is always required.
    source_path: str
    # Stores the source page number when the document has pages.
    page: int | None = None
    # Stores the heading or section name containing the text when available.
    section: str | None = None
    # Stores the first source line containing the text when line numbers are known.
    line_start: int | None = None
    # Stores the last source line containing the text when line numbers are known.
    line_end: int | None = None
    # Stores the text's starting character offset within the source when known.
    char_start: int | None = None
    # Stores the text's ending character offset within the source when known.
    char_end: int | None = None


# Represents one normalized piece of a document before pieces are grouped into chunks.
class NormalizedBlock(BaseModel):
    # Identifies whether this block is a heading, paragraph, code, or image-derived text.
    kind: Literal["heading", "paragraph", "code", "image_text"]
    # Contains the actual text extracted from this block.
    text: str
    # Records where this block appeared in the original source.
    location: SourceLocation
    # Holds optional format-specific details without sharing one mutable dictionary between instances.
    attributes: dict[str, Any] = Field(default_factory=dict)


# Represents a complete source document after normalization and before chunking.
class NormalizedDocument(BaseModel):
    # Provides the document's stable unique identifier.
    id: str
    # Stores the path of the original source file.
    source_path: str
    # Describes the source format with a MIME type such as "text/plain".
    media_type: str
    # Stores a content checksum used to detect whether the document has changed.
    checksum: str
    # Contains the document's normalized blocks in their original order.
    blocks: list[NormalizedBlock]
    # Holds optional document-level details without sharing one mutable dictionary between instances.
    metadata: dict[str, Any] = Field(default_factory=dict)


# Extends source-location information with metadata needed by a searchable chunk.
class ChunkMetadata(SourceLocation):
    # Describes the original document format when it is known.
    media_type: str | None = None
    # Stores the text's language code when it is known.
    language: str | None = None
    # Stores the source creation date or timestamp when it is known.
    created_at: str | None = None
    # Holds optional chunk-specific details without sharing one mutable dictionary between instances.
    attributes: dict[str, Any] = Field(default_factory=dict)


# Represents one searchable text unit produced from a normalized document.
class Chunk(BaseModel):
    # Provides the chunk's stable unique identifier.
    id: str
    # Links the chunk back to the normalized document that produced it.
    document_id: str
    # Records the chunk's zero-based position within its document and forbids negative values.
    ordinal: int = Field(ge=0)
    # Contains the searchable chunk text and forbids an empty string.
    text: str = Field(min_length=1)
    # Preserves the chunk's source location and other provenance information.
    metadata: ChunkMetadata
