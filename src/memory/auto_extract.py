"""auto_extract — LLM-powered memory extraction from session briefs.

Calls the Anthropic API to extract structured memories from a brief text,
returning a list of validated entry dicts ready for bulk insert.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Valid memory types (mirrors DECAY_RATES keys + "todo")
_VALID_TYPES = frozenset({"decision", "learning", "error", "pattern", "reference", "todo"})

_REQUIRED_FIELDS = frozenset({"content", "memory_type", "tags", "importance"})

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_SYSTEM_PROMPT = """\
You extract structured memories from session briefs.

Output ONLY a valid JSON array — no prose, no markdown, no explanation.

Schema for each element:
{
  "content": "<concise fact, decision, or pattern — 1-3 sentences>",
  "memory_type": "decision|learning|error|pattern|reference|todo",
  "tags": ["tag1", "tag2"],
  "importance": <float 0.0–1.0>,
  "rationale": "<one sentence: why this is worth remembering>"
}

Rules:
- Include ONLY entries with clear recall value (skip trivial observations).
- content must be self-contained — no pronouns without referents.
- tags must be lowercase slug strings (e.g. "project:infra", "tool:terraform").
- importance 0.8+ reserved for CRITICAL/BREAKING/MUST-NEVER-FORGET facts.
- Respond with the JSON array only. No other text."""


def _strip_fences(text: str) -> str:
    """Remove ```json``` fences if present, returning the inner content."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _validate_entry(entry: object) -> dict | None:
    """Validate a single extracted entry. Returns cleaned dict or None."""
    if not isinstance(entry, dict):
        return None
    # Required fields present?
    if not _REQUIRED_FIELDS.issubset(entry.keys()):
        return None
    content = entry.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    memory_type = entry.get("memory_type", "")
    if memory_type not in _VALID_TYPES:
        return None
    tags = entry.get("tags", [])
    if not isinstance(tags, list):
        return None
    importance = entry.get("importance", 0.0)
    if not isinstance(importance, (int, float)) or not (0.0 <= float(importance) <= 1.0):
        return None

    return {
        "content": content.strip(),
        "memory_type": memory_type,
        "tags": [str(t) for t in tags if isinstance(t, str)],
        "importance": float(importance),
        "rationale": str(entry.get("rationale", "")),
    }


def extract_memories(
    brief_text: str,
    *,
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
    max_entries: int = 20,
    timeout_s: float = 30.0,
) -> list[dict]:
    """Call the Anthropic API to extract structured memories from a brief.

    Returns a list of dicts: {content, memory_type, tags, importance, rationale}.
    Empty list on any failure (network, rate limit, parse, missing key).
    """
    # Resolve API key: explicit arg → env var → graceful []
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.debug("extract_memories: no API key available, returning []")
        return []

    if not brief_text or not brief_text.strip():
        return []

    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        logger.warning("extract_memories: anthropic package not installed")
        return []

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            timeout=timeout_s,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extract memories from the following session brief.\n\n"
                        f"<brief>\n{brief_text}\n</brief>"
                    ),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_memories: API call failed: %s", exc)
        return []

    # Extract text from first content block
    try:
        raw_text = response.content[0].text
    except (AttributeError, IndexError, TypeError) as exc:
        logger.warning("extract_memories: unexpected response shape: %s", exc)
        return []

    # Parse JSON, tolerating ```json fences
    json_text = _strip_fences(raw_text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("extract_memories: JSON parse failed: %s", exc)
        return []

    if not isinstance(parsed, list):
        logger.warning("extract_memories: expected JSON array, got %s", type(parsed).__name__)
        return []

    results: list[dict] = []
    for raw_entry in parsed[:max_entries]:
        validated = _validate_entry(raw_entry)
        if validated is None:
            logger.debug("extract_memories: dropped invalid entry: %r", raw_entry)
            continue
        # Always add source:auto tag
        if "source:auto" not in validated["tags"]:
            validated["tags"].append("source:auto")
        results.append(validated)

    return results
