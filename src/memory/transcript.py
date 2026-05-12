"""transcript — minimal trimmer for Claude Code session JSONL files.

Reads a session JSONL, drops noise, and returns compact text suitable for
feeding to extract_memories() without requiring the handoff plugin.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Noise entry types to skip entirely
_SKIP_TYPES = frozenset(
    {
        "file-history-snapshot",
        "attachment",
        "permission-mode",
        "last-prompt",
        "summary",
    }
)

# Regex: strip ANSI escape sequences from tool output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _iter_jsonl(path: str | Path):
    """Yield parsed JSON objects from a JSONL file, skipping blank/bad lines."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _extract_user_text(entry: dict) -> str | None:
    """Return user-visible text from a 'user' type entry, or None."""
    msg = entry.get("message", {})
    if not isinstance(msg, dict):
        return None
    content = msg.get("content", "")

    # Simple string content
    if isinstance(content, str):
        text = content.strip()
        return text[:2000] if text else None

    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text[:2000])
        # Skip tool_result blocks entirely (their bodies are noise)
    return "\n".join(parts).strip() or None


def _extract_assistant_text(entry: dict, max_chars: int = 4000) -> str | None:
    """Return assistant text from an 'assistant' type entry.

    Includes "[Tool: name(...)]" markers for tool_use blocks.
    Skips thinking blocks.
    """
    msg = entry.get("message", {})
    if not isinstance(msg, dict):
        return None
    content = msg.get("content", [])
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            # Drop thinking blocks
            continue
        elif btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text[:max_chars])
        elif btype == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            # Build a short preview of the arguments
            try:
                preview = json.dumps(inp, ensure_ascii=False)
            except (TypeError, ValueError):
                preview = str(inp)
            preview = _ANSI_RE.sub("", preview)
            # Truncate argument preview to 200 chars
            if len(preview) > 200:
                preview = preview[:197] + "..."
            parts.append(f"[Tool: {name}({preview})]")

    result = "\n".join(parts).strip()
    return result or None


def trim_transcript(jsonl_path: str | Path, max_chars: int = 100_000) -> str:
    """Read a Claude Code session JSONL, drop noise, return compact text.

    Drops:
      - top-level file-history-snapshot and other non-conversation entries
      - tool_result bodies (keep tool name + short input preview only)
      - thinking blocks
      - empty/whitespace text turns
      - exact duplicate user messages

    Keeps:
      - real user prompts (verbatim, truncated to 2000 chars each)
      - assistant text content (verbatim, truncated to 4000 chars each)
      - tool_use names + 200-char arg preview (no result)

    Output format: simple U: <text> / A: <text> lines, blank-line separated.
    Caps total output at max_chars; appends continuation marker if hit.
    """
    lines: list[str] = []
    seen_user: set[str] = set()
    total_chars = 0

    for entry in _iter_jsonl(jsonl_path):
        if not isinstance(entry, dict):
            continue

        etype = entry.get("type")

        # Skip noise types
        if etype in _SKIP_TYPES:
            continue

        if etype == "user":
            text = _extract_user_text(entry)
            if not text:
                continue
            # Deduplicate exact user messages
            if text in seen_user:
                continue
            seen_user.add(text)
            line = f"U: {text}"
        elif etype == "assistant":
            text = _extract_assistant_text(entry)
            if not text:
                continue
            line = f"A: {text}"
        else:
            # Unknown type — skip
            continue

        # Cap total output
        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        if len(line) > remaining:
            # Count lines we're about to drop
            remaining_lines = sum(
                1
                for e in _iter_jsonl(jsonl_path)
                if e.get("type") in ("user", "assistant")
            )
            lines.append(line[:remaining])
            lines.append(f"\n…[transcript continues {remaining_lines} lines]")
            total_chars = max_chars
            break

        lines.append(line)
        total_chars += len(line) + 2  # +2 for blank-line separator

    return "\n\n".join(lines)
