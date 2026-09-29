"""Turn a raw search request into an inspectable, evidence-preserving plan."""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field

from ingestion.models import Chunk

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
IDENTIFIER_SEPARATOR = re.compile(r"[_./\\-]+")
WHITESPACE = re.compile(r"\s+")
QUOTED_TEXT = re.compile(r"([\"'])(.+?)\1")
IDENTIFIER = re.compile(
    r"(?<!\w)(?:[A-Za-z]:[\\/][^\s]+|[^\s]+[\\/][^\s]+|"
    r"[A-Za-z][\w-]*\.[A-Za-z0-9]{1,12}|[A-Za-z_]\w*(?:_\w+)+)(?!\w)"
)
FILE_TYPE = re.compile(
    r"(?:filetype|type|ext(?:ension)?)\s*:\s*\.?([a-z0-9]+)|"
    r"\b(pdf|docx?|text|txt|markdown|md|csv|json|python|py)\b(?:\s+files?)?",
    re.IGNORECASE,
)
MONTHS = {
    month.casefold(): number
    for number, month in enumerate(
        (
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
    if month
}
MONTH_REFERENCE = re.compile(
    rf"\b(?:(last)\s+)?({'|'.join(MONTHS)})\b(?:\s+(20\d{{2}}))?",
    re.IGNORECASE,
)
EXTENSION_ALIASES = {
    "document": ".docx",
    "doc": ".doc",
    "docx": ".docx",
    "markdown": ".md",
    "md": ".md",
    "pdf": ".pdf",
    "python": ".py",
    "py": ".py",
    "text": ".txt",
    "txt": ".txt",
    "csv": ".csv",
    "json": ".json",
}


class MetadataFilter(BaseModel):
    """Describe one visible metadata constraint derived from the search request."""

    field: Literal["file_extension", "source_path", "media_type", "created_at"]
    operator: Literal["equals", "ends_with", "gte", "lt"]
    value: str
    source: Literal["explicit", "inferred"]
    evidence: str


class QueryTransformation(BaseModel):
    """Record why one alternate search form was added."""

    kind: Literal["normalize", "identifier_expansion", "supplied_variant"]
    input: str
    output: str
    reason: str


class QueryPlan(BaseModel):
    """Keep the original search request beside all derived search behavior."""

    raw_query: str
    normalized_query: str
    search_queries: list[str]
    exact_terms: list[str] = Field(default_factory=list)
    filters: list[MetadataFilter] = Field(default_factory=list)
    transformations: list[QueryTransformation] = Field(default_factory=list)

    def active_filters(self, *, include_inferred: bool = False) -> list[MetadataFilter]:
        """Return safe hard filters, excluding inferred constraints by default."""
        return [
            item
            for item in self.filters
            if include_inferred or item.source == "explicit"
        ]


def expand_query(query: str, supplied_variants: Sequence[str] = ()) -> list[str]:
    """Keep the original query and add deterministic identifier/path variants."""
    original = WHITESPACE.sub(" ", query).strip()
    if not original:
        return []

    expanded_identifiers = CAMEL_BOUNDARY.sub(" ", original)
    expanded_identifiers = IDENTIFIER_SEPARATOR.sub(" ", expanded_identifiers)
    expanded_identifiers = WHITESPACE.sub(" ", expanded_identifiers).strip()

    variants = [original, expanded_identifiers, *supplied_variants]
    return list(
        dict.fromkeys(
            WHITESPACE.sub(" ", variant).strip()
            for variant in variants
            if variant and variant.strip()
        )
    )


def extract_exact_terms(query: str) -> list[str]:
    """Find quoted phrases, filenames, paths, and code-style identifiers to preserve."""
    terms = [match.group(2).strip() for match in QUOTED_TEXT.finditer(query)]
    terms.extend(match.group(0).rstrip(".,;:!?") for match in IDENTIFIER.finditer(query))
    return list(dict.fromkeys(term for term in terms if term))


def _month_filters(query: str, *, today: date) -> list[MetadataFilter]:
    """Convert visible month references into an inspectable half-open date range."""
    filters: list[MetadataFilter] = []
    for match in MONTH_REFERENCE.finditer(query):
        month = MONTHS[match.group(2).casefold()]
        year_text = match.group(3)
        source: Literal["explicit", "inferred"] = "explicit" if year_text else "inferred"
        if year_text:
            year = int(year_text)
        elif match.group(1):
            year = today.year - 1 if month >= today.month else today.year
        else:
            year = today.year
        start = date(year, month, 1)
        end = date(year, month, monthrange(year, month)[1])
        evidence = match.group(0)
        filters.extend(
            [
                MetadataFilter(
                    field="created_at",
                    operator="gte",
                    value=start.isoformat(),
                    source=source,
                    evidence=evidence,
                ),
                MetadataFilter(
                    field="created_at",
                    operator="lt",
                    value=date.fromordinal(end.toordinal() + 1).isoformat(),
                    source=source,
                    evidence=evidence,
                ),
            ]
        )
    return filters


def derive_metadata_filters(query: str, *, today: date | None = None) -> list[MetadataFilter]:
    """Extract explicit file constraints and visible date clues from a search request."""
    filters: list[MetadataFilter] = []
    for match in FILE_TYPE.finditer(query):
        name = (match.group(1) or match.group(2)).casefold()
        extension = EXTENSION_ALIASES.get(name, f".{name.lstrip('.')}")
        filters.append(
            MetadataFilter(
                field="file_extension",
                operator="equals",
                value=extension,
                source="explicit",
                evidence=match.group(0),
            )
        )

    for term in extract_exact_terms(query):
        suffix = PurePosixPath(term.replace("\\", "/")).suffix
        if suffix:
            filters.append(
                MetadataFilter(
                    field="source_path",
                    operator="ends_with",
                    value=term.replace("\\", "/"),
                    source="explicit",
                    evidence=term,
                )
            )

    filters.extend(_month_filters(query, today=today or date.today()))
    unique: list[MetadataFilter] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in filters:
        identity = (item.field, item.operator, item.value, item.source)
        if identity not in seen:
            seen.add(identity)
            unique.append(item)
    return unique


def process_query(
    query: str,
    *,
    supplied_variants: Sequence[str] = (),
    today: date | None = None,
) -> QueryPlan:
    """Build a traceable search plan without replacing the user's original request."""
    normalized = WHITESPACE.sub(" ", query).strip()
    exact_terms = extract_exact_terms(query)
    transformations: list[QueryTransformation] = []
    if normalized != query:
        transformations.append(
            QueryTransformation(
                kind="normalize",
                input=query,
                output=normalized,
                reason="Collapsed whitespace without changing words.",
            )
        )

    deterministic = expand_query(normalized)
    if len(deterministic) > 1:
        transformations.append(
            QueryTransformation(
                kind="identifier_expansion",
                input=normalized,
                output=deterministic[1],
                reason="Split filename, path, underscore, hyphen, and camel-case boundaries.",
            )
        )

    protected_variants: list[str] = []
    for supplied in supplied_variants:
        cleaned = WHITESPACE.sub(" ", supplied).strip()
        missing_terms = [term for term in exact_terms if term.casefold() not in cleaned.casefold()]
        protected = " ".join([cleaned, *missing_terms]).strip()
        if protected:
            protected_variants.append(protected)
            transformations.append(
                QueryTransformation(
                    kind="supplied_variant",
                    input=supplied,
                    output=protected,
                    reason="Kept externally supplied wording while restoring exact evidence.",
                )
            )

    return QueryPlan(
        raw_query=query,
        normalized_query=normalized,
        search_queries=list(dict.fromkeys([*deterministic, *protected_variants])),
        exact_terms=exact_terms,
        filters=derive_metadata_filters(normalized, today=today),
        transformations=transformations,
    )


def _metadata_value(chunk: Chunk, field: str) -> Any:
    """Read one filterable value from standard chunk metadata."""
    if field == "file_extension":
        return PurePosixPath(chunk.metadata.source_path.replace("\\", "/")).suffix.casefold()
    return getattr(chunk.metadata, field, chunk.metadata.attributes.get(field))


def chunk_matches_filters(
    chunk: Chunk,
    filters: Sequence[MetadataFilter] | Mapping[str, object] | None,
) -> bool:
    """Apply the same metadata constraints to any retrieval implementation."""
    if not filters:
        return True
    if isinstance(filters, Mapping):
        return all(_metadata_value(chunk, field) == expected for field, expected in filters.items())
    for item in filters:
        actual = _metadata_value(chunk, item.field)
        if actual is None:
            return False
        actual_text = str(actual).casefold()
        expected = item.value.casefold()
        if item.operator == "equals" and actual_text != expected:
            return False
        if item.operator == "ends_with" and not actual_text.replace("\\", "/").endswith(expected):
            return False
        if item.operator == "gte" and actual_text < expected:
            return False
        if item.operator == "lt" and actual_text >= expected:
            return False
    return True


def filter_chunks(
    chunks: Sequence[Chunk],
    filters: Sequence[MetadataFilter] | Mapping[str, object] | None,
) -> list[Chunk]:
    """Return only chunks that satisfy every supplied metadata constraint."""
    return [chunk for chunk in chunks if chunk_matches_filters(chunk, filters)]
