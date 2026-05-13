#!/usr/bin/env python3
"""Build an adversarial query corpus from the real production memory DB.

Generates queries that exercise specific Phase A/B/C strengths:

  identifier  — pure FTS / exact-match (e.g. "TICKET-198")
  paraphrase  — semantic-only test: 2 content nouns, no identifiers
  cross_topic — RRF test: 2 distinct tech terms from one memory
  hot_cluster — activation test: high-recall memory's topic, gold is a
                 lower-recall sibling that should outrank it under activation

Each generated (query, expected_hash) pair is validated by running a baseline
hybrid search — if the expected memory isn't in the top-K under baseline, the
query is unreachable and discarded.  Reachable queries form the corpus.

Usage:
  uv run python tests/build_real_corpus.py --out /tmp/real_corpus.json
  uv run python tests/build_real_corpus.py --out /tmp/real_corpus.json --max 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory.core import DB_PATH, MemoryStore  # noqa: E402

# Words to strip from content when building paraphrase queries.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "but", "is", "are",
    "was", "were", "be", "been", "with", "by", "from", "on", "at", "as", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "she", "his", "her", "i", "me", "my", "if", "then",
    "else", "when", "where", "how", "why", "what", "which", "who", "use",
    "uses", "using", "used", "via", "into", "onto", "over", "under", "between",
    "do", "does", "did", "have", "has", "had", "can", "could", "should", "would",
    "must", "may", "might", "will", "shall", "not", "no", "yes", "than", "such",
    "so", "also", "only", "just", "very", "really", "basically", "actually",
}

_IDENTIFIER_RE = re.compile(r"\b([A-Z]{2,10}-\d+|PR\s*#?\d+)\b", re.IGNORECASE)
_BRACKETED_PREFIX = re.compile(r"^\s*\[[^\]]+\]\s*")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _strip_prefix(content: str) -> str:
    return _BRACKETED_PREFIX.sub("", content)


def _tokens(content: str) -> list[str]:
    cleaned = _strip_prefix(content)
    toks = [t.lower() for t in _TOKEN_RE.findall(cleaned)]
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 3]


def _identifiers(content: str) -> list[str]:
    return [m.group(1) for m in _IDENTIFIER_RE.finditer(content)]


def _pick_distinct_terms(content: str, k: int = 2) -> list[str]:
    """Pick `k` reasonably-distinct content tokens (no near-duplicates)."""
    toks = _tokens(content)
    counts = Counter(toks)
    # Bias toward less-common tokens (more specific).
    sorted_tokens = sorted(set(toks), key=lambda t: (counts[t], -len(t)))
    picked: list[str] = []
    for t in sorted_tokens:
        if any(t in p or p in t for p in picked):
            continue
        picked.append(t)
        if len(picked) >= k:
            break
    return picked


def _make_identifier_query(content: str) -> str | None:
    ids = _identifiers(content)
    return ids[0] if ids else None


def _make_paraphrase_query(content: str, rng: random.Random) -> str | None:
    # Drop identifiers + tags; use bare content words.
    cleaned = _IDENTIFIER_RE.sub(" ", content)
    terms = _pick_distinct_terms(cleaned, k=3)
    if len(terms) < 2:
        return None
    rng.shuffle(terms)
    return " ".join(terms[:3])


def _make_cross_topic_query(content: str) -> str | None:
    # Pick 2 long tokens that look like distinct technologies / nouns.
    cleaned = _IDENTIFIER_RE.sub(" ", content)
    terms = _pick_distinct_terms(cleaned, k=2)
    if len(terms) < 2:
        return None
    return f"{terms[0]} {terms[1]}"


def _sample_memories(store: MemoryStore, n_per_type: int = 40) -> list[dict]:
    """Diverse sample weighted by type, biased toward recall-rich memories."""
    conn = store._get_conn()
    rows: list[dict] = []
    for mtype in ("learning", "pattern", "decision", "error", "reference"):
        for r in conn.execute(
            "SELECT content_hash, content, memory_type, recall_count, importance, tags "
            "FROM memories WHERE deleted_at IS NULL AND memory_type = ? "
            "ORDER BY recall_count DESC, RANDOM() LIMIT ?",
            (mtype, n_per_type),
        ).fetchall():
            rows.append(dict(r))
    return rows


def _validate_query(store: MemoryStore, query: str, expected_hash: str, top_k: int = 10) -> int | None:
    """Return rank (1-based) of expected_hash in top-K baseline hybrid, or None."""
    try:
        results = store.search(
            query=query,
            mode="hybrid",
            limit=top_k,
            score_fusion="weighted",
            rerank=False,
            track_recall=False,
        )
    except Exception:  # noqa: BLE001
        return None
    for i, r in enumerate(results):
        if r.get("content_hash") == expected_hash:
            return i + 1
    return None


def build_corpus(out_path: Path, max_queries: int, seed: int, db_src: Path) -> dict:
    rng = random.Random(seed)

    # Operate on a temp copy of the production DB to avoid bumping recall counts
    # or otherwise mutating the user's live store.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        copy_db = td_path / "snapshot.db"
        shutil.copy2(db_src, copy_db)
        store = MemoryStore(db_path=copy_db)

        sample = _sample_memories(store, n_per_type=40)
        print(f"sampled {len(sample)} memories")

        queries: list[dict] = []
        seen: set[tuple[str, str]] = set()

        # Build hot-cluster pool: memories with recall_count >= 20 grouped by tag.
        conn = store._get_conn()
        hot_rows = conn.execute(
            "SELECT content_hash, content, recall_count, tags FROM memories "
            "WHERE deleted_at IS NULL AND recall_count >= 20"
        ).fetchall()
        hot_by_tag: dict[str, list[dict]] = {}
        for r in hot_rows:
            try:
                tags = json.loads(r["tags"] or "[]")
            except Exception:  # noqa: BLE001
                tags = []
            if not isinstance(tags, list):
                continue
            for t in tags:
                hot_by_tag.setdefault(t, []).append(dict(r))

        for mem in sample:
            content = mem["content"]
            ch = mem["content_hash"]

            # 1. identifier query
            iq = _make_identifier_query(content)
            if iq and (iq, ch) not in seen:
                rank = _validate_query(store, iq, ch)
                if rank is not None and rank <= 10:
                    queries.append({"query": iq, "expected_hash": ch, "category": "identifier", "baseline_rank": rank})
                    seen.add((iq, ch))

            # 2. paraphrase
            pq = _make_paraphrase_query(content, rng)
            if pq and (pq, ch) not in seen:
                rank = _validate_query(store, pq, ch)
                if rank is not None and rank <= 10:
                    queries.append({"query": pq, "expected_hash": ch, "category": "paraphrase", "baseline_rank": rank})
                    seen.add((pq, ch))

            # 3. cross-topic
            cq = _make_cross_topic_query(content)
            if cq and cq != pq and (cq, ch) not in seen:
                rank = _validate_query(store, cq, ch)
                if rank is not None and rank <= 10:
                    queries.append({"query": cq, "expected_hash": ch, "category": "cross_topic", "baseline_rank": rank})
                    seen.add((cq, ch))

            if len(queries) >= max_queries:
                break

        # 4. hot-cluster queries: take a tag with multiple recall-rich memories,
        #    expected is the LESS-recalled sibling; the hot one should naturally
        #    win without activation, lose with activation.
        for tag, mems in hot_by_tag.items():
            if len(mems) < 2 or len(queries) >= max_queries:
                continue
            mems_sorted = sorted(mems, key=lambda m: m["recall_count"], reverse=True)
            hot, cold = mems_sorted[0], mems_sorted[-1]
            if hot["content_hash"] == cold["content_hash"]:
                continue
            # Build a query from the cold memory's tokens — it should be findable.
            cq = _make_paraphrase_query(cold["content"], rng)
            if not cq:
                continue
            rank = _validate_query(store, cq, cold["content_hash"])
            if rank is None or rank > 10:
                continue
            queries.append(
                {
                    "query": cq,
                    "expected_hash": cold["content_hash"],
                    "category": "hot_cluster",
                    "baseline_rank": rank,
                    "competing_hot_hash": hot["content_hash"],
                    "competing_recall_count": hot["recall_count"],
                }
            )

        by_cat = Counter(q["category"] for q in queries)
        print(f"built {len(queries)} validated queries: {dict(by_cat)}")

        payload = {
            "source_db": str(db_src),
            "n_sample": len(sample),
            "queries": queries,
            "categories": dict(by_cat),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/real_corpus.json", type=Path)
    parser.add_argument("--max", default=120, type=int, help="cap on number of validated queries")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--db", default=str(DB_PATH), help="source SQLite DB path")
    args = parser.parse_args()

    build_corpus(args.out, max_queries=args.max, seed=args.seed, db_src=Path(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
