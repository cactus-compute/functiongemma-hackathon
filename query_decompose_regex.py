"""Regex-based query decomposition. Splits compound queries into single-action sub-queries."""

import re

# Phase 1: split on conjunction phrases (order matters: Oxford comma before bare "and")
_CONJUNCTION_PATTERN = re.compile(
    r"\s*(?:,\s*and\s+|\s+and\s+|\s+then\s+|\s+also\s+|\s+after\s+)\s*",
    re.IGNORECASE,
)
# Phase 2: split on list separators
_LIST_SEP_PATTERN = re.compile(r"\s*[,;]\s*")
# Strip leading connector words from segments
_LEADING_CONNECTOR = re.compile(r"^\s*(?:and|then|also|after)\s+", re.IGNORECASE)


def _strip_connector(s: str) -> str:
    return _LEADING_CONNECTOR.sub("", s).strip()


def decompose_query(user_text: str) -> list[str]:
    """Split a compound query into single-action sub-queries.

    Input: raw user query string.
    Output: list of sub-queries. Single-hop returns [user_text]. Empty input returns [].
    """
    if not user_text or not user_text.strip():
        return []

    text = user_text.strip()
    # Phase 1: split on conjunctions
    segments = _CONJUNCTION_PATTERN.split(text)
    # Phase 2: split each segment on comma/semicolon
    flat = []
    for seg in segments:
        flat.extend(_LIST_SEP_PATTERN.split(seg))
    # Post-process: strip, remove leading connectors, filter empty
    result = [_strip_connector(s) for s in flat if s and s.strip()]

    if not result:
        return []
    return result
