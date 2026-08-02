#!/usr/bin/env python3
"""Bulk re-embed all memories + documents with the *current* embedding engine.

Why: vectors are engine-specific. The MLX (macOS) and ONNX (Linux) builds of
modernbert-embed-base produce slightly different vectors for the same text —
enough that cross-engine cosine similarity falls below MIN_SIMILARITY_THRESHOLD
and semantic search returns nothing. When the canonical store moves into the
Linux/ONNX container, run this once so stored document vectors and future query
vectors share one engine.

Idempotent: recomputes embed_doc() for every live row and overwrites the vec0
tables. Run inside the container (MEMORY_DB already points at /data).

    python scripts/reembed.py [--batch 64] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time

from memory.core import MemoryStore, _serialize_f32
from memory.embeddings import get_model


def _reembed_table(conn, model, *, select_sql: str, update_sql: str, batch: int, label: str, dry: bool) -> int:
    rows = conn.execute(select_sql).fetchall()
    total = len(rows)
    print(f"[{label}] {total} rows to re-embed")
    if dry or total == 0:
        return total
    done = 0
    t0 = time.time()
    for i in range(0, total, batch):
        chunk = rows[i : i + batch]
        texts = [r[1] for r in chunk]
        vecs = model.embed_doc_batch(texts)
        conn.executemany(update_sql, [(_serialize_f32(v), r[0]) for r, v in zip(chunk, vecs)])
        conn.commit()
        done += len(chunk)
        if done % (batch * 10) == 0 or done == total:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[{label}] {done}/{total} ({rate:.0f}/s)")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = MemoryStore()
    store.health()  # force lazy connect
    conn = store._conn
    model = get_model()
    print(f"engine backend: {getattr(model, '_backend', '?')}  db: {store.db_path}")

    n_mem = _reembed_table(
        conn,
        model,
        select_sql=(
            "SELECT m.id, m.content FROM memories m "
            "JOIN memory_embeddings e ON e.rowid = m.id WHERE m.deleted_at IS NULL"
        ),
        update_sql="UPDATE memory_embeddings SET content_embedding = ? WHERE rowid = ?",
        batch=args.batch,
        label="memories",
        dry=args.dry_run,
    )

    n_doc = 0
    try:
        n_doc = _reembed_table(
            conn,
            model,
            select_sql=(
                "SELECT d.id, d.summary FROM documents d "
                "JOIN document_embeddings e ON e.rowid = d.id WHERE d.deleted_at IS NULL"
            ),
            update_sql="UPDATE document_embeddings SET summary_embedding = ? WHERE rowid = ?",
            batch=args.batch,
            label="documents",
            dry=args.dry_run,
        )
    except Exception as e:  # documents table/columns may differ across versions
        print(f"[documents] skipped: {e}")

    print(f"done. memories={n_mem} documents={n_doc} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
