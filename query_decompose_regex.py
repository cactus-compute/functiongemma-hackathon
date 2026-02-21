"""Regex-based query decomposition. Splits compound queries into single-action sub-queries."""

import re

# Split only when the next fragment looks like a new action.
_ACTION_HINT = r"(?:set|play|remind|send|text|message|check|get|find|look\s+up|search|create|wake)\b"
# Phase 1: split on conjunction phrases for action transitions.
_CONJUNCTION_PATTERN = re.compile(
    rf"\s*(?:,\s*and\s+(?={_ACTION_HINT})|\s+and\s+(?={_ACTION_HINT})|\s+then\s+(?={_ACTION_HINT})|\s+also\s+(?={_ACTION_HINT})|\s+after\s+(?={_ACTION_HINT}))\s*",
    re.IGNORECASE,
)
# Phase 2: split list separators only when followed by an action.
_LIST_SEP_PATTERN = re.compile(rf"\s*[,;]\s*(?={_ACTION_HINT})", re.IGNORECASE)
# Strip leading connector words from segments
_LEADING_CONNECTOR = re.compile(r"^\s*(?:and|then|also|after)\s+", re.IGNORECASE)
_TRAILING_PUNCT = re.compile(r"^[\s,;:.!?]+|[\s,;:.!?]+$")


def _strip_connector(s: str) -> str:
    return _TRAILING_PUNCT.sub("", _LEADING_CONNECTOR.sub("", s).strip())


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
