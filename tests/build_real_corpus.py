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
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "by",
    "from",
    "on",
    "at",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "his",
    "her",
    "i",
    "me",
    "my",
    "if",
    "then",
    "else",
    "when",
    "where",
    "how",
    "why",
    "what",
    "which",
    "who",
    "use",
    "uses",
    "using",
    "used",
    "via",
    "into",
    "onto",
    "over",
    "under",
    "between",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "should",
    "would",
    "must",
    "may",
    "might",
    "will",
    "shall",
    "not",
    "no",
    "yes",
    "than",
    "such",
    "so",
    "also",
    "only",
    "just",
    "very",
    "really",
    "basically",
    "actually",
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
    except Exception:
        return None
    for i, r in enumerate(results):
        if r.get("content_hash") == expected_hash:
            return i + 1
    return None


def generate_candidates(store: MemoryStore, max_queries: int, rng: random.Random) -> list[dict]:
    """Build candidate (query, expected_hash, category) pairs from memory content.
    Model-agnostic — pure text extraction, NO retrieval/validation.  The same
    candidate list can then be validated independently by any embedder so the
    gold set isn't biased toward whichever model validated it."""
    sample = _sample_memories(store, n_per_type=40)
    print(f"sampled {len(sample)} memories")

    cands: list[dict] = []
    seen: set[tuple[str, str]] = set()

    conn = store._get_conn()
    hot_rows = conn.execute(
        "SELECT content_hash, content, recall_count, tags FROM memories WHERE deleted_at IS NULL AND recall_count >= 20"
    ).fetchall()
    hot_by_tag: dict[str, list[dict]] = {}
    for r in hot_rows:
        try:
            tags = json.loads(r["tags"] or "[]")
        except Exception:
            tags = []
        if not isinstance(tags, list):
            continue
        for t in tags:
            hot_by_tag.setdefault(t, []).append(dict(r))

    def add(query: str | None, ch: str, category: str, **extra) -> None:
        if query and (query, ch) not in seen:
            cands.append({"query": query, "expected_hash": ch, "category": category, **extra})
            seen.add((query, ch))

    for mem in sample:
        content, ch = mem["content"], mem["content_hash"]
        add(_make_identifier_query(content), ch, "identifier")
        pq = _make_paraphrase_query(content, rng)
        add(pq, ch, "paraphrase")
        cq = _make_cross_topic_query(content)
        if cq != pq:
            add(cq, ch, "cross_topic")
        if len(cands) >= max_queries * 3:  # over-generate; validation prunes
            break

    for _tag, mems in hot_by_tag.items():
        if len(mems) < 2:
            continue
        mems_sorted = sorted(mems, key=lambda m: m["recall_count"], reverse=True)
        hot, cold = mems_sorted[0], mems_sorted[-1]
        if hot["content_hash"] == cold["content_hash"]:
            continue
        add(
            _make_paraphrase_query(cold["content"], rng),
            cold["content_hash"],
            "hot_cluster",
            competing_hot_hash=hot["content_hash"],
            competing_recall_count=hot["recall_count"],
        )

    print(f"generated {len(cands)} candidates: {dict(Counter(c['category'] for c in cands))}")
    return cands


def validate_candidates(store: MemoryStore, candidates: list[dict], max_queries: int, top_k: int = 10) -> list[dict]:
    """Keep candidates whose expected memory is in this store's top-K hybrid."""
    out: list[dict] = []
    for c in candidates:
        rank = _validate_query(store, c["query"], c["expected_hash"], top_k)
        if rank is not None:
            out.append({**c, "baseline_rank": rank})
        if len(out) >= max_queries:
            break
    print(f"validated {len(out)}/{len(candidates)}: {dict(Counter(q['category'] for q in out))}")
    return out


def build_corpus(out_path: Path, max_queries: int, seed: int, db_src: Path) -> dict:
    rng = random.Random(seed)  # noqa: S311 - corpus sampling, reproducibility beats unpredictability
    with tempfile.TemporaryDirectory() as td:
        copy_db = Path(td) / "snapshot.db"
        shutil.copy2(db_src, copy_db)
        store = MemoryStore(db_path=copy_db)
        cands = generate_candidates(store, max_queries, rng)
        queries = validate_candidates(store, cands, max_queries)
        by_cat = Counter(q["category"] for q in queries)
        payload = {"source_db": str(db_src), "queries": queries, "categories": dict(by_cat)}
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    # Default lands in the system temp dir rather than a hardcoded /tmp path.
    parser.add_argument("--out", default=Path(tempfile.gettempdir()) / "real_corpus.json", type=Path)
    parser.add_argument("--max", default=120, type=int, help="cap on number of validated queries")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--db", default=str(DB_PATH), help="source SQLite DB path")
    parser.add_argument(
        "--emit-candidates",
        type=Path,
        default=None,
        help="Generate model-agnostic candidate (query,gold) pairs WITHOUT validation and exit. "
        "Validate later per-model with --validate-candidates for an unbiased union corpus.",
    )
    parser.add_argument(
        "--validate-candidates",
        type=Path,
        default=None,
        help="Load candidates from this file and keep those reachable by the current-process "
        "embedder (set via MEMORY_EMBED_MODEL) on --db. Writes reachable subset to --out.",
    )
    args = parser.parse_args()

    if args.emit_candidates:
        rng = random.Random(args.seed)  # noqa: S311 - corpus sampling, not crypto
        with tempfile.TemporaryDirectory() as td:
            copy_db = Path(td) / "snapshot.db"
            shutil.copy2(args.db, copy_db)
            cands = generate_candidates(MemoryStore(db_path=copy_db), args.max, rng)
        args.emit_candidates.write_text(json.dumps({"candidates": cands}, indent=2))
        print(f"wrote {len(cands)} candidates to {args.emit_candidates}")
        return 0

    if args.validate_candidates:
        cands = json.loads(args.validate_candidates.read_text())["candidates"]
        with tempfile.TemporaryDirectory() as td:
            copy_db = Path(td) / "snapshot.db"
            shutil.copy2(args.db, copy_db)
            queries = validate_candidates(MemoryStore(db_path=copy_db), cands, args.max)
        args.out.write_text(json.dumps({"source_db": args.db, "queries": queries}, indent=2))
        print(f"wrote {len(queries)} validated to {args.out}")
        return 0

    build_corpus(args.out, max_queries=args.max, seed=args.seed, db_src=Path(args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
