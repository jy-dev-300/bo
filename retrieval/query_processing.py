import re
from collections.abc import Sequence

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
IDENTIFIER_SEPARATOR = re.compile(r"[_./\\-]+")
WHITESPACE = re.compile(r"\s+")


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
