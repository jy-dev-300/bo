"""Structure-, syntax-, and semantics-aware document chunking.

The original Student Task A implementation manually split prose with a regular
expression, packed character windows, and added overlap from both neighbours.
That version was useful practice, but it treated token budgets as characters
and could double a middle chunk's overlap. The production path below delegates
prose boundary detection and token accounting to Chonkie while keeping this
project's provenance model and deterministic identifiers in our code.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from chonkie import SemanticChunker, SentenceChunker, TokenChunker
from pydantic import BaseModel, Field, model_validator
from tree_sitter_language_pack import (
    ProcessConfig,
    detect_language,
    detect_language_from_content,
    process,
)

from ingestion.models import (
    Chunk,
    ChunkMetadata,
    NormalizedBlock,
    NormalizedDocument,
    SourceLocation,
)


@dataclass(frozen=True)
class LocatedText:
    text: str
    locations: tuple[SourceLocation, ...]
    kind: Literal["prose", "code"] = "prose"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSpan:
    """Map a slice of composed prose back to one normalized source block."""

    start: int
    end: int
    block: NormalizedBlock


@dataclass(frozen=True)
class ProseGroup:
    text: str
    spans: tuple[SourceSpan, ...]


class ChunkingPolicy(BaseModel):
    """Select a chunking engine and its hard size budget.

    ``sentence`` is the fast, deterministic default. ``semantic`` uses local
    embeddings to split on topic changes and may download its model the first
    time it is used. Character mode always uses Chonkie's character tokenizer;
    token mode uses the explicitly named tokenizer (``word`` is offline-safe,
    while a Hugging Face tokenizer name can match a downstream model exactly).
    """

    target_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    size_unit: Literal["characters", "tokens"] = "characters"
    tokenizer: str = "word"
    strategy: Literal["sentence", "semantic"] = "sentence"
    preserve_code_blocks: bool = True
    semantic_embedding_model: str = "minishlab/potion-base-32M"
    semantic_threshold: float = Field(default=0.8, gt=0, lt=1)
    semantic_similarity_window: int = Field(default=3, gt=0)
    semantic_skip_window: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_budget(self) -> "ChunkingPolicy":
        if self.overlap >= self.target_size:
            raise ValueError("overlap must be smaller than target_size")
        if self.strategy == "semantic" and self.size_unit != "tokens":
            raise ValueError(
                "semantic chunking uses the embedding model tokenizer; "
                "set size_unit='tokens'"
            )
        return self

    @property
    def resolved_tokenizer(self) -> str:
        return "character" if self.size_unit == "characters" else self.tokenizer


def location_for_text_span(
    location: SourceLocation,
    source_text: str,
    relative_start: int,
    relative_end: int,
) -> SourceLocation:
    """Narrow character and line provenance to an exact block substring."""
    updates: dict[str, int] = {}
    if location.char_start is not None:
        updates["char_start"] = location.char_start + relative_start
        updates["char_end"] = location.char_start + relative_end

    if location.line_start is not None:
        line_start = location.line_start + source_text[:relative_start].count("\n")
        line_end = line_start + source_text[relative_start:relative_end].count("\n")
        selected_text = source_text[relative_start:relative_end]
        if relative_end > relative_start and selected_text.endswith("\n"):
            line_end -= 1
        updates["line_start"] = line_start
        updates["line_end"] = line_end

    return location.model_copy(update=updates)


def shared_location_value(
    locations: tuple[SourceLocation, ...],
    field_name: Literal["page", "section"],
) -> int | str | None:
    values = {
        getattr(location, field_name)
        for location in locations
        if getattr(location, field_name) is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def chunk_metadata(
    document: NormalizedDocument,
    locations: tuple[SourceLocation, ...],
    attributes: dict[str, Any] | None = None,
) -> ChunkMetadata:
    line_starts = [location.line_start for location in locations if location.line_start is not None]
    line_ends = [location.line_end for location in locations if location.line_end is not None]
    char_starts = [location.char_start for location in locations if location.char_start is not None]
    char_ends = [location.char_end for location in locations if location.char_end is not None]

    chunk_attributes = dict(attributes or {})
    chunk_attributes["source_locations"] = [
        location.model_dump(exclude_none=True) for location in locations
    ]

    return ChunkMetadata(
        source_path=document.source_path,
        page=shared_location_value(locations, "page"),
        section=shared_location_value(locations, "section"),
        line_start=min(line_starts) if len(line_starts) == len(locations) else None,
        line_end=max(line_ends) if len(line_ends) == len(locations) else None,
        char_start=min(char_starts) if len(char_starts) == len(locations) else None,
        char_end=max(char_ends) if len(char_ends) == len(locations) else None,
        media_type=document.media_type,
        language=chunk_attributes.get("code_language"),
        attributes=chunk_attributes,
    )


def locations_are_compatible(left: SourceLocation, right: SourceLocation) -> bool:
    if left.source_path != right.source_path:
        return False
    for field_name in ("page", "section"):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


def compose_prose_groups(document: NormalizedDocument) -> list[ProseGroup | NormalizedBlock]:
    """Group compatible prose while keeping headings attached to following text."""
    output: list[ProseGroup | NormalizedBlock] = []
    blocks: list[NormalizedBlock] = []

    def flush() -> None:
        if not blocks:
            return
        pieces: list[str] = []
        spans: list[SourceSpan] = []
        cursor = 0
        for block in blocks:
            if pieces:
                separator = "\n\n"
                pieces.append(separator)
                cursor += len(separator)
            pieces.append(block.text)
            spans.append(SourceSpan(cursor, cursor + len(block.text), block))
            cursor += len(block.text)
        output.append(ProseGroup("".join(pieces), tuple(spans)))
        blocks.clear()

    for block in document.blocks:
        if not block.text.strip():
            continue
        if block.kind == "code":
            flush()
            output.append(block)
            continue
        if block.kind == "heading" and blocks:
            flush()
        if blocks and not locations_are_compatible(blocks[-1].location, block.location):
            flush()
        blocks.append(block)

    flush()
    return output


def locations_for_group_span(
    group: ProseGroup,
    start: int,
    end: int,
) -> tuple[SourceLocation, ...]:
    locations: list[SourceLocation] = []
    for span in group.spans:
        selected_start = max(start, span.start)
        selected_end = min(end, span.end)
        if selected_start >= selected_end:
            continue
        locations.append(
            location_for_text_span(
                span.block.location,
                span.block.text,
                selected_start - span.start,
                selected_end - span.start,
            )
        )
    return tuple(locations)


def exact_token_chunk_spans(
    source_text: str, external_chunks: list[Any]
) -> list[tuple[Any, int, int]]:
    """Realign token chunks because some tokenizers report decoded-text offsets."""
    spans: list[tuple[Any, int, int]] = []
    cursor = 0
    for external in external_chunks:
        match = re.search(re.escape(external.text), source_text[cursor:], re.IGNORECASE)
        if match is None:
            raise ValueError("token chunk text could not be mapped back to its source")
        start = cursor + match.start()
        end = cursor + match.end()
        spans.append((external, start, end))
        cursor = end
    return spans


def sentence_units(group: ProseGroup, policy: ChunkingPolicy) -> list[LocatedText]:
    chunker = SentenceChunker(
        tokenizer=policy.resolved_tokenizer,
        chunk_size=policy.target_size,
        chunk_overlap=policy.overlap,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=1,
    )
    hard_limit_chunker = TokenChunker(
        tokenizer=policy.resolved_tokenizer,
        chunk_size=policy.target_size,
        chunk_overlap=0,
    )
    units: list[LocatedText] = []
    for external in chunker(group.text):
        # SentenceChunker intentionally preserves an oversized single sentence.
        # Our public contract is a hard budget, so only that exceptional case
        # falls through to the exact tokenizer window chunker.
        oversized = external.token_count > policy.target_size
        external_parts = hard_limit_chunker(external.text) if oversized else [external]
        located_parts = (
            exact_token_chunk_spans(external.text, external_parts)
            if oversized
            else [(external, external.start_index, external.end_index)]
        )
        for part, part_start, part_end in located_parts:
            start = external.start_index + part_start if oversized else part_start
            end = external.start_index + part_end if oversized else part_end
            text = group.text[start:end] if oversized else part.text
            locations = locations_for_group_span(group, start, end)
            if not text.strip() or not locations:
                continue
            units.append(
                LocatedText(
                    text=text,
                    locations=locations,
                    attributes={
                        "content_kind": "prose",
                        "chunker": (
                            "chonkie-sentence"
                            if len(external_parts) == 1
                            else "chonkie-sentence+token-hard-limit"
                        ),
                        "size_unit": policy.size_unit,
                        "tokenizer": policy.resolved_tokenizer,
                        "external_token_count": part.token_count,
                    },
                )
            )
    return units


def semantic_units(group: ProseGroup, policy: ChunkingPolicy) -> list[LocatedText]:
    # Reserve space before adding prefix overlap so final chunks keep the hard budget.
    base_size = policy.target_size - policy.overlap
    chunker = SemanticChunker(
        embedding_model=policy.semantic_embedding_model,
        threshold=policy.semantic_threshold,
        chunk_size=base_size,
        similarity_window=policy.semantic_similarity_window,
        min_sentences_per_chunk=1,
        min_characters_per_sentence=1,
        skip_window=policy.semantic_skip_window,
    )
    external_chunks = chunker(group.text)
    tokenizer = chunker.tokenizer
    hard_limit_chunker = TokenChunker(
        tokenizer=tokenizer,
        chunk_size=base_size,
        chunk_overlap=0,
    )
    bounded_spans: list[tuple[str, int, int]] = []
    for external in external_chunks:
        if external.token_count <= base_size:
            bounded_spans.append((external.text, external.start_index, external.end_index))
            continue
        hard_limit_parts = hard_limit_chunker(external.text)
        for _part, part_start, part_end in exact_token_chunk_spans(
            external.text, hard_limit_parts
        ):
            bounded_spans.append(
                (
                    external.text[part_start:part_end],
                    external.start_index + part_start,
                    external.start_index + part_end,
                )
            )
    units: list[LocatedText] = []

    for index, (_external_text, external_start, external_end) in enumerate(bounded_spans):
        start = external_start
        if index > 0 and policy.overlap:
            previous_text, previous_start, previous_end = bounded_spans[index - 1]
            overlap_tokens = tokenizer.encode(previous_text)[-policy.overlap :]
            overlap_text = tokenizer.decode(overlap_tokens)
            overlap_start = group.text.rfind(overlap_text, previous_start, previous_end)
            if overlap_start >= 0:
                start = overlap_start

        text = group.text[start:external_end]
        locations = locations_for_group_span(group, start, external_end)
        if text.strip() and locations:
            units.append(
                LocatedText(
                    text=text,
                    locations=locations,
                    attributes={
                        "content_kind": "prose",
                        "chunker": "chonkie-semantic",
                        "size_unit": "tokens",
                        "tokenizer": policy.semantic_embedding_model,
                        "semantic_threshold": policy.semantic_threshold,
                        "semantic_similarity_window": policy.semantic_similarity_window,
                    },
                )
            )
    return units


def detect_code_block_language(
    document: NormalizedDocument,
    block: NormalizedBlock,
) -> str | None:
    explicit_language = block.attributes.get("language")
    if isinstance(explicit_language, str) and explicit_language.strip():
        language_hint = explicit_language.strip().lower()
        return detect_language(f"snippet.{language_hint}") or language_hint
    content_language = detect_language_from_content(block.text)
    if content_language:
        return content_language
    source_path = block.attributes.get("source_path", document.source_path)
    path_language = detect_language(source_path) if isinstance(source_path, str) else None
    return None if path_language in {"markdown", "markdown_inline"} else path_language


def code_unit(
    block: NormalizedBlock,
    text: str,
    start: int,
    end: int,
    language: str | None,
    chunker_name: str,
) -> LocatedText:
    return LocatedText(
        text=text,
        locations=(location_for_text_span(block.location, block.text, start, end),),
        kind="code",
        attributes={
            "content_kind": "code",
            "code_language": language,
            "chunker": chunker_name,
        },
    )


def fallback_code_units(
    block: NormalizedBlock,
    policy: ChunkingPolicy,
    language: str | None,
    reason: str,
) -> list[LocatedText]:
    chunker = TokenChunker(
        tokenizer=policy.resolved_tokenizer,
        chunk_size=policy.target_size,
        chunk_overlap=0,
    )
    external_chunks = chunker(block.text)
    return [
        code_unit(
            block,
            block.text[start:end],
            start,
            end,
            language,
            f"chonkie-token-fallback:{reason}",
        )
        for external, start, end in exact_token_chunk_spans(block.text, external_chunks)
        if external.text
    ]


def code_block_into_units(
    document: NormalizedDocument,
    block: NormalizedBlock,
    policy: ChunkingPolicy,
) -> list[LocatedText]:
    """Prefer Tree-sitter AST boundaries, then enforce the declared size unit."""
    language = detect_code_block_language(document, block)
    if language is None:
        return fallback_code_units(block, policy, None, "language-not-detected")

    token_chunker = TokenChunker(
        tokenizer=policy.resolved_tokenizer,
        chunk_size=policy.target_size,
        chunk_overlap=0,
    )
    syntax_budget = (
        policy.target_size
        if policy.size_unit == "characters"
        else policy.target_size * 4
    )
    try:
        result = process(
            block.text,
            ProcessConfig(language=language, chunk_max_size=syntax_budget),
        )
    except Exception as error:
        return fallback_code_units(block, policy, language, type(error).__name__)
    if not result.chunks:
        return fallback_code_units(block, policy, language, "no-syntax-chunks")

    source_bytes = block.text.encode("utf-8")
    units: list[LocatedText] = []
    for syntax_chunk in result.chunks:
        start = len(source_bytes[: syntax_chunk.start_byte].decode("utf-8"))
        end = len(source_bytes[: syntax_chunk.end_byte].decode("utf-8"))
        syntax_text = block.text[start:end]
        if token_chunker.tokenizer.count_tokens(syntax_text) <= policy.target_size:
            units.append(code_unit(block, syntax_text, start, end, language, "tree-sitter"))
            continue
        external_chunks = token_chunker(syntax_text)
        for _external, external_start, external_end in exact_token_chunk_spans(
            syntax_text, external_chunks
        ):
            units.append(
                code_unit(
                    block,
                    block.text[start + external_start : start + external_end],
                    start + external_start,
                    start + external_end,
                    language,
                    "tree-sitter+chonkie-token",
                )
            )
    return units or fallback_code_units(block, policy, language, "no-text-chunks")


def deterministic_chunk_id(
    document: NormalizedDocument,
    policy: ChunkingPolicy,
    ordinal: int,
    draft: LocatedText,
) -> str:
    identity = {
        "checksum": document.checksum,
        "ordinal": ordinal,
        "policy": policy.model_dump(mode="json"),
        "text": draft.text,
        "locations": [location.model_dump(mode="json") for location in draft.locations],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{document.id}:{ordinal:04d}:{digest}"


def chunk_document(document: NormalizedDocument, policy: ChunkingPolicy) -> list[Chunk]:
    """Transform normalized blocks into stable, provenance-rich retrieval chunks."""
    drafts: list[LocatedText] = []
    for item in compose_prose_groups(document):
        if isinstance(item, NormalizedBlock):
            if policy.preserve_code_blocks:
                drafts.extend(code_block_into_units(document, item, policy))
            else:
                group = ProseGroup(item.text, (SourceSpan(0, len(item.text), item),))
                drafts.extend(sentence_units(group, policy))
        elif policy.strategy == "semantic":
            drafts.extend(semantic_units(item, policy))
        else:
            drafts.extend(sentence_units(item, policy))

    return [
        Chunk(
            id=deterministic_chunk_id(document, policy, ordinal, draft),
            document_id=document.id,
            ordinal=ordinal,
            text=draft.text,
            metadata=chunk_metadata(document, draft.locations, draft.attributes),
        )
        for ordinal, draft in enumerate(drafts)
        if draft.text.strip()
    ]
