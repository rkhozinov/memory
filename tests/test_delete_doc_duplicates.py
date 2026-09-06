"""delete_doc must delete what its dry run promised."""

from __future__ import annotations


def _duplicate_row(store, content_hash: str) -> None:
    """Recreate the historical shape: two live rows sharing one content_hash.
    store_doc cannot produce this any more (it dedups inside an IMMEDIATE
    transaction), but rows predating that check still exist."""
    conn = store._get_conn()
    row = conn.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
    cols = [k for k in row.keys() if k != "id"]  # noqa: SIM118 — sqlite3.Row has no __contains__
    conn.execute(
        f"INSERT INTO documents ({','.join(cols)}) SELECT {','.join(cols)} FROM documents WHERE id = ?",  # noqa: S608
        (row["id"],),
    )
    conn.commit()


def test_dry_run_count_matches_what_delete_actually_removes(store):
    h = store.store_doc(title="T", body="b", summary="s", doc_type="spec")["content_hash"]
    _duplicate_row(store, h)

    promised = store.delete_doc(h, dry_run=True)["would_delete"]
    actual = store.delete_doc(h)["deleted"]

    assert promised == actual == 2


def test_deleting_a_duplicated_hash_leaves_nothing_live(store):
    h = store.store_doc(title="T", body="b", summary="s", doc_type="spec")["content_hash"]
    _duplicate_row(store, h)

    store.delete_doc(h)

    live = (
        store._get_conn()
        .execute("SELECT COUNT(*) AS n FROM documents WHERE content_hash = ? AND deleted_at IS NULL", (h,))
        .fetchone()["n"]
    )
    assert live == 0
