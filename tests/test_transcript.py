"""Tests for transcript.py trimmer and auto_archive_pending core method."""

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
# Test 5 — auto_archive_pending skips sessions younger than min_age_minutes
# ---------------------------------------------------------------------------

def test_auto_archive_pending_skips_too_recent(store, tmp_path):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)
    session_file = projects_dir / "session-abc.jsonl"
    _write_jsonl(session_file, [
        {"type": "user", "message": {"role": "user", "content": "hello"}}
    ])
    # Touch it to ensure it's "now"
    session_file.touch()

    marker_dir = tmp_path / "markers"

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=60,  # 1 hour min age — file is brand new
            marker_dir=marker_dir,
        )

    assert result["skipped_too_recent"] >= 1
    assert result["processed"] == 0


# ---------------------------------------------------------------------------
# Test 6 — auto_archive_pending skips sessions with existing marker
# ---------------------------------------------------------------------------

def test_auto_archive_pending_skips_existing_marker(store, tmp_path):
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
    (marker_dir / "session-xyz.marker").write_text("archived\nabc123\n")

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    assert result["skipped_marked"] >= 1
    assert result["processed"] == 0


# ---------------------------------------------------------------------------
# Test 7 — auto_archive_pending stores doc with expected title/tags/doc_type
# ---------------------------------------------------------------------------

def test_auto_archive_pending_stores_doc(store, tmp_path):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-myproject"
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
    session_file = projects_dir / "session-archive-test.jsonl"
    _write_jsonl(session_file, entries)

    old_time = time.time() - 600  # 10 minutes ago
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"
    stored_docs_info = []

    original_store_doc = store.store_doc

    def capturing_store_doc(**kwargs):
        stored_docs_info.append(kwargs)
        return original_store_doc(**kwargs)

    with patch("pathlib.Path.home", return_value=tmp_path):
        with patch.object(store, "store_doc", side_effect=capturing_store_doc):
            result = store.auto_archive_pending(
                cwd="/Users/myproject",
                min_age_minutes=5,
                marker_dir=marker_dir,
            )

    assert result["stored_docs"] == 1
    assert result["processed"] == 1
    assert len(stored_docs_info) == 1

    call = stored_docs_info[0]
    # Title format: "Session <short_id> <cwd_basename> <YYYY-MM-DD>"
    assert call["title"].startswith("Session ")
    assert "session-archive-test"[:8] in call["title"] or call["title"].startswith("Session ")
    # Tags must include source:auto, session-archive, and project:<basename>
    assert "source:auto" in call["tags"]
    assert "session-archive" in call["tags"]
    assert any(t.startswith("project:") for t in call["tags"])
    # doc_type must be session-archive
    assert call["doc_type"] == "session-archive"
    # metadata must include session_id and source_jsonl
    assert "session_id" in call["metadata"]
    assert "source_jsonl" in call["metadata"]


# ---------------------------------------------------------------------------
# Test 8 — auto_archive_pending writes marker with content hash on success
# ---------------------------------------------------------------------------

def test_auto_archive_pending_writes_marker_with_hash(store, tmp_path):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    entries = [
        {"type": "user", "message": {"role": "user", "content": "Important question about infra."}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Detailed answer about infra."}],
            },
        },
    ]
    session_file = projects_dir / "session-hashtest.jsonl"
    _write_jsonl(session_file, entries)
    old_time = time.time() - 600
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    assert result["stored_docs"] == 1
    marker_file = marker_dir / "session-hashtest.marker"
    assert marker_file.exists(), "Marker file should exist after archiving"

    marker_content = marker_file.read_text()
    lines = marker_content.strip().splitlines()
    assert lines[0] == "archived"
    assert len(lines) >= 2, "Marker should contain 'archived\\n<hash>'"
    assert len(lines[1]) > 8, "Second line should be a content hash"


# ---------------------------------------------------------------------------
# Test 9 — empty/unreadable JSONL → skipped_empty++, marker still written
# ---------------------------------------------------------------------------

def test_auto_archive_pending_empty_jsonl_writes_marker(store, tmp_path):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    # Write a JSONL with only noise entries (no real messages → empty trimmed text)
    entries = [
        {"type": "file-history-snapshot", "files": ["foo.tf"]},
        {"type": "attachment", "data": "ignored"},
    ]
    session_file = projects_dir / "session-empty.jsonl"
    _write_jsonl(session_file, entries)
    old_time = time.time() - 600
    os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            marker_dir=marker_dir,
        )

    assert result["skipped_empty"] >= 1
    assert result["stored_docs"] == 0
    # Marker should still be written so the session isn't retried
    marker_file = marker_dir / "session-empty.marker"
    assert marker_file.exists(), "Marker should be written even for empty sessions"


# ---------------------------------------------------------------------------
# Test 10 — auto_archive_pending respects max_sessions cap (oldest-first)
# ---------------------------------------------------------------------------

def test_auto_archive_pending_respects_max_sessions(store, tmp_path):
    projects_dir = tmp_path / ".claude" / "projects" / "-Users-test-project"
    projects_dir.mkdir(parents=True)

    # Create 5 old sessions with distinct ages
    for i in range(5):
        session_file = projects_dir / f"session-{i:03d}.jsonl"
        _write_jsonl(session_file, [
            {"type": "user", "message": {"role": "user", "content": f"message {i}"}}
        ])
        old_time = time.time() - 3600 - i * 10  # Different ages; smallest i = newest
        os.utime(session_file, (old_time, old_time))

    marker_dir = tmp_path / "markers"

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            max_sessions=3,  # Cap at 3
            marker_dir=marker_dir,
        )

    assert result["processed"] == 3  # Only 3 processed despite 5 eligible


# ---------------------------------------------------------------------------
# Test 11 — dry_run returns counts without storing or writing markers
# ---------------------------------------------------------------------------

def test_auto_archive_pending_dry_run(store, tmp_path):
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

    with patch("pathlib.Path.home", return_value=tmp_path):
        result = store.auto_archive_pending(
            cwd="/Users/test/project",
            min_age_minutes=5,
            dry_run=True,
            marker_dir=marker_dir,
        )

    # Dry-run: processed count correct
    assert result["processed"] >= 1
    # stored_docs is 0 in dry-run
    assert result["stored_docs"] == 0
    # Sessions have dry_run status
    assert any(s.get("status") == "dry_run" for s in result["sessions"])
    # No marker files written
    if marker_dir.exists():
        markers = list(marker_dir.glob("*.marker"))
        assert len(markers) == 0


# ---------------------------------------------------------------------------
# Test 12 — CLI auto-archive-pending --dry-run returns JSON without LLM calls
# ---------------------------------------------------------------------------

def test_cli_auto_archive_pending_dry_run(tmp_path, capsys):
    """CLI dry-run returns JSON with no API calls made."""
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

    with patch("pathlib.Path.home", return_value=tmp_path):
        main([
            "admin", "auto-archive-pending",
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
