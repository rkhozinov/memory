"""Markdown mirror export."""

from __future__ import annotations


def test_writes_one_file_per_memory_grouped_by_type(store, tmp_path):
    store.store("Postgres pooling goes through PgBouncer.", memory_type="reference", tags=["svc:db"])
    store.store("Chose sqlite-vec over faiss for the index.", memory_type="decision")

    result = store.export_markdown(tmp_path / "mirror")

    assert result["memories"] == 2
    files = sorted((tmp_path / "mirror" / "memories").rglob("*.md"))
    assert len(files) == 2
    assert {f.parent.name for f in files} == {"reference", "decision"}


def test_frontmatter_and_body_round_trip(store, tmp_path):
    h = store.store("Chose sqlite-vec over faiss.", memory_type="decision", tags=["project:x"])["content_hash"]

    store.export_markdown(tmp_path / "mirror")
    path = tmp_path / "mirror" / "memories" / "decision" / f"{h[:16]}.md"
    text = path.read_text()

    assert text.startswith("---\n")
    assert f"content_hash: {h}\n" in text
    assert "type: decision\n" in text
    assert "  - project:x\n" in text
    assert text.rstrip().endswith("Chose sqlite-vec over faiss.")


def test_documents_are_mirrored_with_slugged_names(store, tmp_path):
    store.store_doc(
        title="Retention Policy / v2",
        body="Keep archives 30 days.",
        summary="Archive retention.",
        doc_type="spec",
    )

    result = store.export_markdown(tmp_path / "mirror")

    assert result["documents"] == 1
    docs = list((tmp_path / "mirror" / "documents").glob("*.md"))
    assert len(docs) == 1
    assert docs[0].name.startswith("retention-policy-v2-")
    assert "Keep archives 30 days." in docs[0].read_text()


def test_rerun_overwrites_rather_than_duplicating(store, tmp_path):
    store.store("A stable fact.", memory_type="note")
    store.export_markdown(tmp_path / "mirror")
    store.export_markdown(tmp_path / "mirror")

    assert len(list((tmp_path / "mirror" / "memories").rglob("*.md"))) == 1


def test_deleted_memories_are_not_exported(store, tmp_path):
    h = store.store("Transient.", memory_type="note")["content_hash"]
    store.store("Kept.", memory_type="note")
    store.delete(content_hash=h)

    result = store.export_markdown(tmp_path / "mirror")

    assert result["memories"] == 1


def test_reports_filename_collisions_instead_of_hiding_them(store, tmp_path):
    """The documents table has no unique constraint on content_hash: two rows
    can hash identically, and both then map to one filename."""
    conn = store._get_conn()
    store.store_doc(title="Dup", body="Same body.", summary="s", doc_type="spec")
    row = conn.execute("SELECT * FROM documents").fetchone()
    cols = [k for k in row.keys() if k != "id"]  # noqa: SIM118 — sqlite3.Row has no __contains__
    conn.execute(
        f"INSERT INTO documents ({','.join(cols)}) SELECT {','.join(cols)} FROM documents WHERE id = ?",  # noqa: S608
        (row["id"],),
    )
    conn.commit()

    result = store.export_markdown(tmp_path / "mirror")

    assert result["documents"] == 1
    assert result["collisions"] == 1
