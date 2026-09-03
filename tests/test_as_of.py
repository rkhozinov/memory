"""`as_of` must answer what the store knew at an instant, in every mode.

Probe P1 measured 33% against a 90% adoption bar: `as_of` reached only graph
traversal, so semantic, FTS and hybrid queries answered a past-tense question
with the present.  Now that an update no longer overwrites what it supersedes
(see test_supersede.py), both facts coexist and the timestamp is what separates
them.

Semantics are transaction time: a memory is visible at T if it was stored at or
before T and had not been deleted by T.
"""

import time

import pytest


def _store_at(store, content, ts, **kw):
    res = store.store(content, **kw)
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ?, updated_at = ? WHERE content_hash = ?",
        (ts, ts, res["content_hash"]),
    )
    conn.commit()
    return res["content_hash"]


@pytest.fixture
def timeline(store):
    """An old fact, then the fact that replaces it a day later."""
    now = time.time()
    old = _store_at(
        store,
        "CI runners are pinned to ubuntu-22.04 for the build workflow",
        now - 86400 * 10,
        memory_type="reference",
        tags=["project:x"],
    )
    new = _store_at(
        store,
        "CI runners are pinned to ubuntu-24.04 for the build workflow",
        now - 86400 * 2,
        memory_type="reference",
        tags=["project:x"],
    )
    return {"old": old, "new": new, "between": now - 86400 * 5, "now": now}


# One query cannot serve every mode: "exact" is substring matching so it needs a
# literal slice of the content, while "semantic" needs to clear
# MIN_SIMILARITY_THRESHOLD (0.45). Pair each mode with a query it can answer.
MODE_QUERIES = [
    ("fts", "ubuntu runners build workflow"),
    ("exact", "for the build workflow"),
    ("semantic", "CI runners pinned for the build workflow"),
    ("hybrid", "CI runners pinned for the build workflow"),
]


@pytest.mark.parametrize(("mode", "query"), MODE_QUERIES)
def test_as_of_before_the_update_returns_only_the_old_fact(timeline, store, mode, query):
    hits = store.search(query, mode=mode, limit=10, as_of=timeline["between"])
    hashes = {h["content_hash"] for h in hits}
    assert timeline["old"] in hashes, f"{mode}: the fact that was true then is missing"
    assert timeline["new"] not in hashes, f"{mode}: answered a past question with the present"


@pytest.mark.parametrize(("mode", "query"), MODE_QUERIES)
def test_without_as_of_the_current_fact_is_visible(timeline, store, mode, query):
    hits = store.search(query, mode=mode, limit=10)
    assert timeline["new"] in {h["content_hash"] for h in hits}


def test_as_of_accepts_an_iso_date(timeline, store):
    from datetime import UTC, datetime

    iso = datetime.fromtimestamp(timeline["between"], tz=UTC).date().isoformat()
    hits = store.search("build workflow", mode="fts", limit=10, as_of=iso)
    assert timeline["new"] not in {h["content_hash"] for h in hits}


def test_as_of_hides_a_memory_deleted_before_the_instant(store):
    now = time.time()
    h = _store_at(store, "temporary note about the staging cluster", now - 86400 * 10)
    conn = store._get_conn()
    conn.execute("UPDATE memories SET deleted_at = ? WHERE content_hash = ?", (now - 86400 * 7, h))
    conn.commit()
    assert h not in {x["content_hash"] for x in store.search("staging cluster", mode="fts", as_of=now - 86400 * 5)}


def test_as_of_shows_a_memory_that_was_alive_at_the_instant(store):
    """Deleted since, but it existed then — that is the point of asking."""
    now = time.time()
    h = _store_at(store, "temporary note about the staging cluster", now - 86400 * 10)
    conn = store._get_conn()
    conn.execute("UPDATE memories SET deleted_at = ? WHERE content_hash = ?", (now - 86400 * 3, h))
    conn.commit()
    hits = store.search("staging cluster", mode="fts", as_of=now - 86400 * 5)
    assert h in {x["content_hash"] for x in hits}
