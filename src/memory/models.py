"""Data models for memory service."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    """A single memory entry."""

    content: str
    content_hash: str = ""
    tags: list[str] = field(default_factory=list)
    memory_type: str = "note"
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    created_at_iso: str = ""
    updated_at_iso: str = ""
    confidence: float = 1.0
    importance: float = 0.5

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.created_at_iso:
            self.created_at_iso = datetime.fromtimestamp(
                self.created_at, tz=timezone.utc
            ).isoformat()
        if not self.updated_at_iso:
            self.updated_at_iso = datetime.fromtimestamp(
                self.updated_at, tz=timezone.utc
            ).isoformat()

    def to_row(self) -> tuple:
        """Return values for SQL INSERT."""
        return (
            self.content_hash,
            self.content,
            json.dumps(self.tags) if self.tags else "[]",
            self.memory_type,
            json.dumps(self.metadata) if self.metadata else "{}",
            self.created_at,
            self.updated_at,
            self.created_at_iso,
            self.updated_at_iso,
            self.confidence,
            self.importance,
        )

    @classmethod
    def from_row(cls, row: dict) -> Memory:
        """Create Memory from a database row dict."""
        tags_raw = row.get("tags", "[]")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            tags = tags_raw or []

        meta_raw = row.get("metadata", "{}")
        if isinstance(meta_raw, str):
            try:
                metadata = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        else:
            metadata = meta_raw or {}

        return cls(
            content=row["content"],
            content_hash=row["content_hash"],
            tags=tags,
            memory_type=row.get("memory_type", "note"),
            metadata=metadata,
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
            created_at_iso=row.get("created_at_iso", ""),
            updated_at_iso=row.get("updated_at_iso", ""),
            confidence=row.get("confidence", 1.0) or 1.0,
            importance=row.get("importance", 0.5) or 0.5,
        )

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "content_hash": self.content_hash,
            "content": self.content,
            "tags": self.tags,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "created_at": self.created_at_iso,
            "updated_at": self.updated_at_iso,
            "confidence": round(self.confidence, 4),
            "importance": round(self.importance, 4),
        }
