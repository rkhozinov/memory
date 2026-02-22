"""Tests for Memory dataclass (no DB or embeddings needed)."""

import hashlib
import json
import time

from memory.models import Memory


def test_memory_auto_hash():
    """__post_init__ computes SHA256 hash from content."""
    m = Memory(content="hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert m.content_hash == expected


def test_memory_auto_timestamps():
    """__post_init__ sets created_at and updated_at to current time."""
    before = time.time()
    m = Memory(content="test")
    after = time.time()
    assert before <= m.created_at <= after
    assert before <= m.updated_at <= after
    assert m.created_at_iso  # non-empty ISO string
    assert m.updated_at_iso


def test_memory_preserves_explicit_hash():
    """Explicit hash is not overwritten by __post_init__."""
    m = Memory(content="test", content_hash="custom_hash")
    assert m.content_hash == "custom_hash"


def test_to_row_roundtrip():
    """to_row() -> from_row() preserves all fields."""
    original = Memory(
        content="roundtrip test",
        tags=["a", "b"],
        memory_type="fact",
        metadata={"key": "value"},
    )
    row_tuple = original.to_row()
    # Simulate what a DB row dict would look like
    keys = [
        "content_hash", "content", "tags", "memory_type", "metadata",
        "created_at", "updated_at", "created_at_iso", "updated_at_iso",
    ]
    row_dict = dict(zip(keys, row_tuple))
    restored = Memory.from_row(row_dict)

    assert restored.content == original.content
    assert restored.content_hash == original.content_hash
    assert restored.tags == original.tags
    assert restored.memory_type == original.memory_type
    assert restored.metadata == original.metadata
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at


def test_from_row_bad_tags_json():
    """Malformed tags JSON falls back to CSV splitting."""
    row = {
        "content": "test",
        "content_hash": "abc",
        "tags": "terraform, aws",
        "memory_type": "note",
        "metadata": "{}",
    }
    m = Memory.from_row(row)
    assert m.tags == ["terraform", "aws"]


def test_from_row_bad_metadata_json():
    """Malformed metadata JSON falls back to empty dict."""
    row = {
        "content": "test",
        "content_hash": "abc",
        "tags": "[]",
        "memory_type": "note",
        "metadata": "{invalid json",
    }
    m = Memory.from_row(row)
    assert m.metadata == {}


def test_to_dict_structure():
    """to_dict() returns all expected keys."""
    m = Memory(content="test", tags=["x"], memory_type="fact", metadata={"k": 1})
    d = m.to_dict()
    expected_keys = {"content_hash", "content", "tags", "memory_type", "metadata", "created_at", "updated_at", "confidence", "importance"}
    assert set(d.keys()) == expected_keys
    assert d["tags"] == ["x"]
    assert d["memory_type"] == "fact"
    assert d["metadata"] == {"k": 1}
