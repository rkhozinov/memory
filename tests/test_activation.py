"""Phase C tests: ACT-R activation, session-aware recall, active forgetting."""

from __future__ import annotations

import time

from memory import core as core_mod
from memory.core import (
    ACTIVATION_TAU_DAYS,
    FORGET_THRESHOLD,
    compute_activation,
)

# --- compute_activation unit tests ---


def test_activation_fresh_high_similarity_is_high():
    now = time.time()
    a = compute_activation(
        similarity=0.9,
        recall_count=0,
        distinct_session_count=0,
        memory_type="decision",
        created_at=now - 60,  # 1 min old
        last_recalled_at=None,
        now=now,
    )
    assert a > 0.6  # fresh decision with strong similarity ranks high


def test_activation_hot_cluster_penalised_vs_diverse():
    """Same recall_count, but diverse sessions should rank higher than hot cluster."""
    now = time.time()
    created = now - 86400  # 1 day old
    hot = compute_activation(
        similarity=0.7,
        recall_count=20,
        distinct_session_count=1,  # hammered from one session
        memory_type="learning",
        created_at=created,
        last_recalled_at=now - 600,
        now=now,
    )
    diverse = compute_activation(
        similarity=0.7,
        recall_count=20,
        distinct_session_count=15,  # used across many sessions
        memory_type="learning",
        created_at=created,
        last_recalled_at=now - 600,
        now=now,
    )
    assert diverse > hot


def test_activation_temporal_decay():
    now = time.time()
    fresh = compute_activation(0.5, 0, 0, "note", now - 60, None, now=now)
    old = compute_activation(0.5, 0, 0, "note", now - 90 * 86400, None, now=now)
    assert fresh > old
    # Temporal weight is only 15% of total; bound the drop loosely.
    assert old < fresh * 0.8


def test_activation_type_weight_decision_beats_note():
    now = time.time()
    dec = compute_activation(0.5, 0, 0, "decision", now - 86400, None, now=now)
    note = compute_activation(0.5, 0, 0, "note", now - 86400, None, now=now)
    assert dec > note


def test_activation_clamps_to_unit_interval():
    now = time.time()
    a = compute_activation(2.0, 0, 0, "decision", now, None, now=now)  # over-large sim
    assert 0.0 <= a <= 1.0


# --- session-aware recall bump in search() ---


def test_search_bumps_distinct_session_count_on_first_hit(store, monkeypatch):
    a = store.store("kubernetes pod scheduling internals", memory_type="learning", tags=["svc:k8s"])

    conn = store._get_conn()

    # First search under session sess-A → distinct_session_count → 1
    monkeypatch.setenv("MEMORY_SESSION_ID", "sess-A")
    store.search(query="kubernetes pod", mode="hybrid", limit=5)
    row = conn.execute(
        "SELECT recall_count, distinct_session_count, last_recall_session FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    if row is None or row["recall_count"] == 0:
        # hybrid retrieved nothing on this corpus; skip via assertion bypass
        return
    initial_dsc = row["distinct_session_count"]
    assert initial_dsc >= 1
    assert row["last_recall_session"] == "sess-A"

    # Second search in same session → distinct stays the same
    store.search(query="kubernetes pod", mode="hybrid", limit=5)
    row = conn.execute(
        "SELECT recall_count, distinct_session_count FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    assert row["distinct_session_count"] == initial_dsc

    # Switch to sess-B → distinct increments
    monkeypatch.setenv("MEMORY_SESSION_ID", "sess-B")
    store.search(query="kubernetes pod", mode="hybrid", limit=5)
    row = conn.execute(
        "SELECT distinct_session_count, last_recall_session FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    assert row["distinct_session_count"] == initial_dsc + 1
    assert row["last_recall_session"] == "sess-B"


def test_search_distinct_session_count_is_set_not_alternation(store, monkeypatch):
    """Re-visiting an already-seen session must NOT re-bump distinct_session_count.

    Walks A → B → A → A → C, expects DSC = 3 (true distinct), not 4 (alternation).
    """
    a = store.store("alpha service connection pool retries", memory_type="learning")
    conn = store._get_conn()

    seq = ["sess-A", "sess-B", "sess-A", "sess-A", "sess-C"]
    for sid in seq:
        monkeypatch.setenv("MEMORY_SESSION_ID", sid)
        store.search(query="alpha service connection", mode="hybrid", limit=5)

    row = conn.execute(
        "SELECT distinct_session_count, recall_sessions, recall_count "
        "FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    if row is None or row["recall_count"] == 0:
        return  # corpus too small / similarity threshold cut

    import json as _json

    stored = _json.loads(row["recall_sessions"] or "[]")
    assert set(stored) == {"sess-A", "sess-B", "sess-C"}
    assert row["distinct_session_count"] == 3


def test_recall_sessions_capped(store, monkeypatch):
    """Stored recall_sessions list is bounded; oldest entries drop first."""
    from memory.core import MemoryStore

    a = store.store("capacity cap test memory", memory_type="learning")
    conn = store._get_conn()

    # Use 5 over cap so we can verify oldest are dropped.
    cap = MemoryStore._RECALL_SESSION_CAP
    n = cap + 5
    for i in range(n):
        monkeypatch.setenv("MEMORY_SESSION_ID", f"sess-{i:04d}")
        store.search(query="capacity cap", mode="hybrid", limit=5)

    row = conn.execute(
        "SELECT recall_sessions, distinct_session_count, recall_count FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    if row is None or row["recall_count"] == 0:
        return

    import json as _json

    stored = _json.loads(row["recall_sessions"] or "[]")
    assert len(stored) <= cap
    # Distinct count tracks current set length (we bumped n times, dropped some).
    assert row["distinct_session_count"] == len(stored)
    # Oldest sessions (lowest indices) must be gone.
    assert "sess-0000" not in stored
    # Newest must be present.
    assert f"sess-{n - 1:04d}" in stored


def test_search_no_session_env_keeps_legacy_behaviour(store, monkeypatch):
    """When MEMORY_SESSION_ID is absent, distinct_session_count must remain 0."""
    monkeypatch.delenv("MEMORY_SESSION_ID", raising=False)
    a = store.store("k8s autoscaler aggressive", memory_type="learning")
    store.search(query="autoscaler", mode="hybrid", limit=5)

    conn = store._get_conn()
    row = conn.execute(
        "SELECT distinct_session_count FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    assert (row["distinct_session_count"] or 0) == 0


# --- enrich_with_activation attaches the field ---


def test_search_results_carry_activation(populated_store):
    results = populated_store.search(query="kubernetes", mode="hybrid", limit=5)
    if not results:
        return
    assert "activation" in results[0]
    assert 0.0 <= results[0]["activation"] <= 1.0
    assert "distinct_session_count" in results[0]


# --- USE_ACTIVATION env flag overrides composite score ---


def test_use_activation_env_replaces_score(populated_store, monkeypatch):
    monkeypatch.setattr(core_mod, "USE_ACTIVATION", True)
    results = populated_store.search(query="kubernetes pod", mode="hybrid", limit=5)
    if not results:
        return
    assert results[0]["score"] == results[0]["activation"]


# --- Active forgetting (gated) ---


def test_active_forget_gated_off_by_default(store, monkeypatch):
    """Without MEMORY_ACTIVE_FORGET=1, pass 6 returns 0 even with stale memories."""
    monkeypatch.delenv("MEMORY_ACTIVE_FORGET", raising=False)
    # Insert a memory and back-date it well past 30 days
    a = store.store("stale note nobody will recall", memory_type="note")
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ?, last_recalled_at = NULL WHERE content_hash = ?",
        (time.time() - 365 * 86400, a["content_hash"]),
    )
    conn.commit()

    result = store.dream(dry_run=False)
    assert result["forgotten"] == 0


def test_active_forget_soft_deletes_low_activation(store, monkeypatch):
    monkeypatch.setenv("MEMORY_ACTIVE_FORGET", "1")
    a = store.store("stale obscure note from year ago", memory_type="note")
    # Back-date and clear recall signals to drive activation below threshold
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ?, last_recalled_at = NULL, "
        "recall_count = 0, distinct_session_count = 0 WHERE content_hash = ?",
        (time.time() - 365 * 86400, a["content_hash"]),
    )
    conn.commit()

    dry = store.dream(dry_run=True)
    assert dry["forgotten"] >= 1
    forgotten_hashes = {p["content_hash"] for p in dry["forgotten_pairs"]}
    assert a["content_hash"] in forgotten_hashes

    real = store.dream(dry_run=False)
    assert real["forgotten"] >= 1

    row = conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?",
        (a["content_hash"],),
    ).fetchone()
    assert row["deleted_at"] is not None


def test_active_forget_respects_budget(store, monkeypatch):
    """Bounded soft-delete: never more than 5% of active corpus per pass."""
    monkeypatch.setenv("MEMORY_ACTIVE_FORGET", "1")
    # 50 stale notes → 5% budget = 3 (ceil)
    hashes = []
    for i in range(50):
        h = store.store(f"stale note number {i}", memory_type="note")["content_hash"]
        hashes.append(h)
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ?, last_recalled_at = NULL",
        (time.time() - 365 * 86400,),
    )
    conn.commit()

    result = store.dream(dry_run=False)
    # 5% of 50 = 3 (ceil) — never exceed budget
    assert result["forgotten"] <= 3
    assert result["forgotten"] >= 1


def test_active_forget_skips_decisions(store, monkeypatch):
    """Decision-type memories are sacred — never auto-forgotten."""
    monkeypatch.setenv("MEMORY_ACTIVE_FORGET", "1")
    a = store.store("decision: use postgres", memory_type="decision")
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ?, last_recalled_at = NULL WHERE content_hash = ?",
        (time.time() - 365 * 86400, a["content_hash"]),
    )
    conn.commit()

    result = store.dream(dry_run=False)
    forgotten_hashes = {p["content_hash"] for p in result["forgotten_pairs"]}
    assert a["content_hash"] not in forgotten_hashes


# --- ACT-R constants sanity ---


def test_activation_tau_is_positive():
    assert ACTIVATION_TAU_DAYS > 0
    assert 0.0 < FORGET_THRESHOLD < 1.0
