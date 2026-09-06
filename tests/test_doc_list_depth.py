"""`list_docs(depth=...)` — a listing should not carry every body.

`memory doc list --type session-archive` returned 139 KB for two documents,
because each record ships its full body. Anything that wants to scan the doc
store (a retention prune, a report) has to page through megabytes to read a
handful of fields.
"""

from __future__ import annotations


def test_full_is_the_default_and_keeps_bodies(store):
    store.store_doc(title="T", body="the body text", summary="s", doc_type="spec")

    result = store.list_docs()

    assert result["documents"][0]["body"] == "the body text"


def test_summary_drops_the_body_and_keeps_the_metadata(store):
    store.store_doc(title="T", body="the body text", summary="s", doc_type="spec")

    result = store.list_docs(depth="summary")

    doc = result["documents"][0]
    assert "body" not in doc
    for key in ("content_hash", "title", "doc_type", "created_at"):
        assert key in doc, key


def test_summary_does_not_change_the_count(store):
    store.store_doc(title="A", body="a" * 100, summary="s", doc_type="spec")
    store.store_doc(title="B", body="b" * 100, summary="s", doc_type="spec")

    assert store.list_docs(depth="summary")["total"] == store.list_docs()["total"]
