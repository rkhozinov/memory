"""Tests for the session→memory extractor.

The model is never invoked: `run_extractor` is monkeypatched everywhere. What is
tested is the part that decides whether to spend money and what is allowed to
reach the store.
"""

import json

import pytest

from memory import extract

TRANSCRIPT = """U: the deploy keeps failing
A: [Tool: Edit(src/app.py)]
U: still broken
A: the issue was a stale lock file in /var/run; removing it fixed the deploy
U: great, and what about staging
A: [Tool: Write(deploy/staging.yaml)]
""" + ("filler line\n" * 400)


def fact(**kw):
    base = {
        "content": "The deploy failed because a stale lock file in /var/run blocked the release step.",
        "type": "learning",
        "tags": ["project:demo"],
        "evidence": "the issue was a stale lock file in /var/run",
    }
    base.update(kw)
    return base


# --- Gate: the cost lever ---


def test_gate_accepts_a_substantive_session():
    ok, why = extract.signal_gate(TRANSCRIPT)
    assert ok, why


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("U: hi\nA: hello\n", "too_short"),
        ("U: only one turn\n" + "A: [Tool: Edit(x)] the issue was x\n" + "x" * 5000, "too_few_turns"),
        ("U: a\nU: b\nU: c\nA: the issue was x\n" + "x" * 5000, "no_mutations"),
        ("U: a\nU: b\nU: c\nA: [Tool: Edit(x)] ok\n" + "x" * 5000, "no_signal_tokens"),
    ],
)
def test_gate_rejects_low_signal_sessions(text, reason):
    ok, why = extract.signal_gate(text)
    assert not ok
    assert why == reason


def test_clip_keeps_head_and_tail():
    text = "H" * extract.HEAD_CHARS + "M" * 50_000 + "T" * extract.TAIL_CHARS
    out = extract.clip(text)
    assert out.startswith("H" * 100)
    assert out.endswith("T" * 100)
    assert "elided" in out
    assert len(out) < len(text)


# --- Filters: what is allowed to reach the store ---


def test_ungrounded_facts_are_dropped():
    """The guard against confident summaries of things that did not happen."""
    kept, dropped = extract.filter_facts([fact(evidence="a quote that never appeared anywhere")], TRANSCRIPT)
    assert kept == []
    assert dropped["ungrounded"] == 1


def test_grounded_fact_survives():
    kept, dropped = extract.filter_facts([fact()], TRANSCRIPT)
    assert len(kept) == 1
    assert dropped == {"malformed": 0, "ungrounded": 0, "secret": 0, "denylist": 0}


def test_grounding_tolerates_whitespace_differences():
    kept, _ = extract.filter_facts([fact(evidence="the issue   was a stale\n lock file in /var/run")], TRANSCRIPT)
    assert len(kept) == 1


@pytest.mark.parametrize(
    "secret",
    [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "xoxb-1234567890-abcdefghij",
        "sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "password: hunter2hunter2",
    ],
)
def test_secret_bearing_facts_are_dropped_not_redacted(secret):
    text = TRANSCRIPT + f"\nA: the token is {secret}\n"
    f = fact(content=f"The service authenticates using {secret} which is stored in the deploy config.")
    kept, dropped = extract.filter_facts([f], text)
    assert kept == []
    assert dropped["secret"] == 1


def test_denylisted_terms_are_dropped():
    f = fact(content="The AcmeCorp deploy pipeline failed because a stale lock file blocked the release step.")
    kept, dropped = extract.filter_facts([f], TRANSCRIPT, denylist=["acmecorp"])
    assert kept == []
    assert dropped["denylist"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"content": "too short", "type": "learning", "tags": [], "evidence": "x" * 30},
        fact(type="not-a-type"),
        fact(tags="project:demo"),
        "not even a dict",
    ],
)
def test_malformed_facts_are_dropped(bad):
    kept, dropped = extract.filter_facts([bad], TRANSCRIPT)
    assert kept == []
    assert dropped["malformed"] == 1


def test_to_batch_stamps_provenance():
    batch = extract.to_batch([fact(tags=["tool:git"])], "abcdef1234", "demo", "haiku")
    assert batch[0]["tags"] == ["project:demo", "source:extract", "tool:git"]
    assert batch[0]["metadata"]["evidence"]
    assert batch[0]["metadata"]["extractor"] == "haiku"


# --- Scanner ---


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolate every path the scanner touches."""
    monkeypatch.setattr(extract.Path, "home", classmethod(lambda cls: tmp_path))
    proj = tmp_path / ".claude" / "projects" / extract._encode_cwd(str(tmp_path / "demo"))
    proj.mkdir(parents=True)
    session = proj / "11111111-2222-3333-4444-555555555555.jsonl"
    session.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"type": "user", "message": {"role": "user", "content": "the deploy keeps failing"}},
                {"type": "user", "message": {"role": "user", "content": "still broken"}},
                {"type": "user", "message": {"role": "user", "content": "and staging?"}},
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "the issue was a stale lock file " + "x" * 5000},
                            {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}},
                        ],
                    },
                },
            ]
        )
    )
    import os

    os.utime(session, (0, 0))  # ancient → always past the idle gate
    return tmp_path, session


def test_dry_run_proposes_without_storing(sandbox, store, monkeypatch):
    tmp_path, _ = sandbox
    monkeypatch.setattr(
        extract,
        "run_extractor",
        lambda *a, **k: {"facts": [fact(evidence="the issue was a stale lock file")], "cost_usd": 0.01},
    )
    before = store.health()["total_memories"]
    out = extract.extract_pending(store, cwd=str(tmp_path / "demo"), dry_run=True)

    assert out["dry_run"] is True
    assert out["extracted"] == 1
    assert out["sessions"][0]["batch"][0]["tags"] == ["project:demo", "source:extract"]
    assert store.health()["total_memories"] == before, "dry run must not write"
    assert not (extract.marker_dir()).exists() or not list(extract.marker_dir().glob("*.extract.json"))


def test_stores_and_marks_the_session(sandbox, store, monkeypatch):
    tmp_path, session = sandbox
    monkeypatch.setattr(
        extract,
        "run_extractor",
        lambda *a, **k: {"facts": [fact(evidence="the issue was a stale lock file")], "cost_usd": 0.01},
    )
    out = extract.extract_pending(store, cwd=str(tmp_path / "demo"))
    assert out["stored"] == 1

    marker = extract.marker_dir() / f"{session.stem}.extract.json"
    assert json.loads(marker.read_text())["status"] == "extracted"

    # Second run must skip the marked session — no re-spend.
    again = extract.extract_pending(store, cwd=str(tmp_path / "demo"))
    assert again["scanned"] == 0


def test_extractor_failure_marks_failed_and_stores_nothing(sandbox, store, monkeypatch):
    tmp_path, session = sandbox
    monkeypatch.setattr(extract, "run_extractor", lambda *a, **k: {"error": "unparseable_result"})
    out = extract.extract_pending(store, cwd=str(tmp_path / "demo"))
    assert out["stored"] == 0
    marker = extract.marker_dir() / f"{session.stem}.extract.json"
    assert json.loads(marker.read_text())["status"] == "failed"


def test_gated_session_never_calls_the_model(sandbox, store, monkeypatch):
    import os

    tmp_path, session = sandbox
    session.write_text(json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}))
    os.utime(session, (0, 0))  # rewriting reset the mtime past the idle gate

    def boom(*a, **k):
        raise AssertionError("the model must not be invoked for a gated session")

    monkeypatch.setattr(extract, "run_extractor", boom)
    out = extract.extract_pending(store, cwd=str(tmp_path / "demo"))
    assert out["gated"] == 1
    assert out["extracted"] == 0


def test_daily_budget_stops_the_scan(sandbox, store, monkeypatch):
    tmp_path, _ = sandbox
    monkeypatch.setenv("MEMORY_EXTRACT_DAILY_BUDGET", "0.10")
    extract.record_spend(0.25)
    monkeypatch.setattr(extract, "run_extractor", lambda *a, **k: {"facts": [], "cost_usd": 0.0})
    out = extract.extract_pending(store, cwd=str(tmp_path / "demo"))
    assert "daily budget" in out.get("error", "")
    assert out["extracted"] == 0


def test_explicit_session_ignores_the_idle_gate(sandbox, monkeypatch):
    """SessionEnd knows which transcript just finished; there is nothing to wait
    for and no oldest-first backlog to lose it behind."""
    import os
    import time

    tmp_path, session = sandbox
    os.utime(session, (time.time(), time.time()))  # brand new — would fail an idle gate
    assert extract._pending(str(tmp_path / "demo"), 6.0, 5) == []
    assert extract._pending(str(tmp_path / "demo"), 6.0, 5, session.stem) == [session]


def test_explicit_session_still_respects_its_marker(sandbox):
    tmp_path, session = sandbox
    extract.marker_dir().mkdir(parents=True, exist_ok=True)
    (extract.marker_dir() / f"{session.stem}.extract.json").write_text("{}")
    assert extract._pending(str(tmp_path / "demo"), 0.0, 5, session.stem) == []


def test_unknown_session_id_is_not_an_error(sandbox):
    tmp_path, _ = sandbox
    assert extract._pending(str(tmp_path / "demo"), 0.0, 5, "no-such-session") == []
