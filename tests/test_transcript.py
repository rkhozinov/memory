"""Tests for transcript.py trimmer and auto_extract_pending core method."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from memory.transcript import trim_transcript


# ---------------------------------------------------------------------------
# JSONL fixture builder
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write a list of dicts as JSONL to path."""
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _make_fixture(tmp_path: Path) -> Path:
    """Build a small representative session JSONL fixture."""
    entries = [
        # Noise: file-history-snapshot
        {"type": "file-history-snapshot", "files": ["foo.tf"]},
        # Noise: attachment
        {"type": "attachment", "data": "ignored"},
        # Real user prompt
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "What is the Terraform state locking mechanism?",
            },
        },
        # Assistant with thinking (should be dropped) + text
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me reason about this..."},
                    {"type": "text", "text": "Terraform uses DynamoDB for state locking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Bash",
                        "input": {"command": "terraform plan"},
                    },
                ],
            },
        },
        # User turn with tool_result (body should be dropped)
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "Plan: 2 to add, 0 to change, 0 to destroy.",
                    }
                ],
            },
        },
        # Second real user prompt
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "Show me the provider configuration.",
            },
        },
        # Duplicate of first user message — should be deduped
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "What is the Terraform state locking mechanism?",
            },
        },
    ]
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, entries)
    return path


# ---------------------------------------------------------------------------
# Test 1 — basic trim: noise dropped, content kept
# ---------------------------------------------------------------------------

def test_trim_drops_noise_keeps_content(tmp_path):
    fixture = _make_fixture(tmp_path)
    result = trim_transcript(fixture)

    # Should contain real user prompts
    assert "What is the Terraform state locking mechanism?" in result
    assert "Show me the provider configuration." in result

    # Should contain assistant text
    assert "Terraform uses DynamoDB for state locking." in result

    # Should include tool_use marker
    assert "[Tool: Bash(" in result

    # Should NOT contain thinking content
    assert "Let me reason about this" not in result

    # Should NOT contain tool_result body
    assert "Plan: 2 to add" not in result

    # Should NOT contain file-history-snapshot noise
    assert "file-history-snapshot" not in result


# ---------------------------------------------------------------------------
# Test 2 — trim_transcript caps at max_chars and appends continuation marker
# ---------------------------------------------------------------------------

def test_trim_caps_at_max_chars(tmp_path):
    # Make a fixture with enough content to exceed small cap
    entries = []
    for i in range(50):
        entries.append({
            "type": "user",
            "message": {"role": "user", "content": f"User message number {i}: " + "x" * 100},
        })
        entries.append({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Assistant reply {i}: " + "y" * 200}],
            },
        })

    path = tmp_path / "long_session.jsonl"
    _write_jsonl(path, entries)

    result = trim_transcript(path, max_chars=500)
    assert len(result) <= 600  # some slack for the marker text
    assert "…[transcript continues" in result


# ---------------------------------------------------------------------------
# Test 3 — exact duplicate user messages are deduped
# ---------------------------------------------------------------------------

def test_trim_deduplicates_user_messages(tmp_path):
    entries = [
        {"type": "user", "message": {"role": "user", "content": "Hello world"}},
        {"type": "user", "message": {"role": "user", "content": "Hello world"}},
        {"type": "user", "message": {"role": "user", "content": "Hello world"}},
        {"type": "user", "message": {"role": "user", "content": "Different message"}},
    ]
    path = tmp_path / "dedup_session.jsonl"
    _write_jsonl(path, entries)

    result = trim_transcript(path)
    # "Hello world" should appear exactly once
    assert result.count("Hello world") == 1
    assert "Different message" in result


# ---------------------------------------------------------------------------
# Test 4 — empty/blank user turns are skipped
# ---------------------------------------------------------------------------

def test_trim_skips_empty_user_turns(tmp_path):
    entries = [
        {"type": "user", "message": {"role": "user", "content": "   "}},
        {"type": "user", "message": {"role": "user", "content": ""}},
        {"type": "user", "message": {"role": "user", "content": "Real message"}},
    ]
    path = tmp_path / "blank_session.jsonl"
    _write_jsonl(path, entries)

    result = trim_transcript(path)
    assert result.startswith("U: Real message")


# ---------------------------------------------------------------------------
# Test 5 — auto_extract_pending skips sessions younger than min_age_minutes
# ---------------------------------------------------------------------------

def test_auto_extract_pending_skips_too_recent(store, tmp_path, monkeypatch):
    # Create a fake projects directory with a very new JSONL
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)
    session_file = projects_dir / "session-abc.jsonl"
    _write_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "hello"}}
    ])
    # Touch it to ensure it's "now"
    session_file.touch()

    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Patch pathlib.Path.home to return tmp_path
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_extract_pending(
            cwd="/Users/test/project",
            min_age_minutes=60,  # 1 hour min age — file is brand new
            marker_dir=marker_dir,
        )

    assert result["skipped_too_recent"] >= 1
    assert result["processed"] == 0


# ---------------------------------------------------------------------------
# Test 6 — auto_extract_pending skips sessions with existing marker
# ---------------------------------------------------------------------------

def test_auto_extract_pending_skips_existing_marker(store, tmp_path, monkeypatch):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)
    session_file = projects_dir / "session-xyz.jsonl"
    _write_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "hello"}}
    ])

    # Make it old enough
    old_time = time.time() - 7200  # 2 hours ago
    os.utime(session_file, (old_time, old_time))

    # Write the marker
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir(parents=True)
    (marker_dir / "session-xyz.marker").write_text("ok\n")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_extract_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    assert result["skipped_marked"] >= 1
    assert result["processed"] == 0


# ---------------------------------------------------------------------------
# Test 7 — auto_extract_pending writes marker after successful extract
# ---------------------------------------------------------------------------

def test_auto_extract_pending_writes_marker_on_success(store, tmp_path, monkeypatch):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    entries = [
        {"type": "user", "message": {"role": "user", "content": "Tell me about Terraform state locking."}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Terraform uses DynamoDB for state locking."}],
            },
        },
    ]
    session_file = projects_dir / "session-marker-test.jsonl"
    _write_jsonl(session_file, entries)

    old_time = time.time() - 600  # 10 minutes ago
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    fake_entries = [
        {
            "content": "Terraform uses DynamoDB for state locking.",
            "memory_type": "pattern",
            "tags": ["tool:terraform"],
            "importance": 0.7,
            "rationale": "Important pattern.",
        }
    ]

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("memory.auto_extract.extract_memories", return_value=fake_entries),
        patch("memory.core.MemoryStore.store_auto_extracted", return_value={
            "attempted": 1, "stored": 1, "dedup_skipped": 0,
            "rejected_invalid": 0, "rejected_injection": 0, "stored_hashes": ["abc"],
        }),
    ):
        result = store.auto_extract_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    # Marker should have been written
    marker_file = marker_dir / "session-marker-test.marker"
    assert marker_file.exists(), "Marker file should exist after successful extraction"
    assert result["processed"] == 1
    assert result["stored_total"] == 1


# ---------------------------------------------------------------------------
# Test 8 — auto_extract_pending does NOT write marker when API key missing
# ---------------------------------------------------------------------------

def test_auto_extract_pending_no_marker_without_api_key(store, tmp_path, monkeypatch):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    session_file = projects_dir / "session-nokey.jsonl"
    _write_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "hello"}}
    ])
    old_time = time.time() - 600
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_extract_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    # Should return early with zero processed
    assert result["processed"] == 0
    assert result["scanned"] == 0
    # No marker should be written
    if marker_dir.exists():
        markers = list(marker_dir.glob("*.marker"))
        assert len(markers) == 0


# ---------------------------------------------------------------------------
# Test 9 — auto_extract_pending respects max_sessions cap
# ---------------------------------------------------------------------------

def test_auto_extract_pending_respects_max_sessions(store, tmp_path, monkeypatch):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    # Create 5 old sessions
    for i in range(5):
        session_file = projects_dir / f"session-{i:03d}.jsonl"
        _write_jsonl(session_file, [
            {"type": "user", "message": {"role": "user", "content": f"message {i}"}}
        ])
        old_time = time.time() - 3600 - i  # Different ages
        os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("memory.auto_extract.extract_memories", return_value=[]),
    ):
        result = store.auto_extract_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            max_sessions=3,  # Cap at 3
            marker_dir=marker_dir,
        )

    assert result["processed"] == 3  # Only 3 processed despite 5 eligible


# ---------------------------------------------------------------------------
# Test 10 — CLI auto-extract-pending --dry-run returns JSON without LLM calls
# ---------------------------------------------------------------------------

def test_cli_auto_extract_pending_dry_run(tmp_path, monkeypatch, capsys):
    """CLI dry-run should return JSON with no LLM calls made."""
    from memory.cli import main

    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    entries = [
        {"type": "user", "message": {"role": "user", "content": "What is Kubernetes?"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Kubernetes is a container orchestrator."}],
            },
        },
    ]
    session_file = projects_dir / "session-dryrun.jsonl"
    _write_jsonl(session_file, entries)
    old_time = time.time() - 600
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    mock_called = []

    def mock_extract(text, **kwargs):
        mock_called.append(text)
        return []

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("memory.auto_extract.extract_memories", side_effect=mock_extract),
    ):
        main([
            "admin", "auto-extract-pending",
            "--cwd", "/Users/test/project",
            "--min-age-minutes", "5",
            "--dry-run",
        ])

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    # dry_run → sessions should have status=dry_run
    assert "sessions" in output
    assert output["processed"] >= 1
    session = output["sessions"][0]
    assert session["status"] == "dry_run"

    # No marker files should be written in dry-run mode
    if marker_dir.exists():
        markers = list(marker_dir.glob("*.marker"))
        assert len(markers) == 0
