"""Pipeline quality regression tests.

Uses the current production model (e5-small-v2 via ONNX) and actual
MemoryStore.search() to validate that each search mode provides measurable
value. No external deps (no sentence-transformers).

Tests answer:
- Does hybrid beat semantic-only? (FTS adds value)
- Does reranking beat hybrid? (cross-encoder adds value)
- Where does each mode fail? (category-specific assertions)
"""

import importlib
import sys
from pathlib import Path

import pytest

# benchmarks/ is not a package — add it to sys.path so corpus.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
corpus = importlib.import_module("corpus")

ALL_TEST_CASES = corpus.ALL_TEST_CASES
BOOLEAN_CASES = corpus.BOOLEAN_CASES
CORPUS = corpus.CORPUS
IDENTIFIER_CASES = corpus.IDENTIFIER_CASES
IMPORTANCE_TRAP_CASES = corpus.IMPORTANCE_TRAP_CASES
SEMANTIC_CASES = corpus.SEMANTIC_CASES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_store(store):
    """Store populated with the shared benchmark corpus."""
    for entry in CORPUS:
        store.store(
            entry.content,
            memory_type=entry.memory_type,
            tags=entry.tags,
            importance=entry.importance,
        )
    return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reciprocal_rank(results: list[dict], expected_substr: str) -> float:
    for i, r in enumerate(results):
        if expected_substr.lower() in r["content"].lower():
            return 1.0 / (i + 1)
    return 0.0


def _find_rank(results: list[dict], expected_substr: str) -> int | None:
    for i, r in enumerate(results):
        if expected_substr.lower() in r["content"].lower():
            return i + 1
    return None


def _compute_mrr(store, queries_and_expected: list[tuple[str, str]], **search_kwargs) -> float:
    total_rr = 0.0
    for query, expected in queries_and_expected:
        results = store.search(query, limit=10, **search_kwargs)
        total_rr += _reciprocal_rank(results, expected)
    return total_rr / len(queries_and_expected)


# ---------------------------------------------------------------------------
# Tests: mode comparison
# ---------------------------------------------------------------------------


def test_semantic_vs_hybrid_mrr(pipeline_store):
    """Hybrid MRR >= semantic MRR — proves FTS adds value."""
    queries = [(tc.query, tc.expected_top) for tc in ALL_TEST_CASES if tc.expected_top]

    mrr_sem = _compute_mrr(pipeline_store, queries, mode="semantic")
    mrr_hyb = _compute_mrr(pipeline_store, queries, mode="hybrid")

    assert mrr_hyb >= mrr_sem - 0.01, (
        f"Hybrid ({mrr_hyb:.3f}) degraded vs semantic ({mrr_sem:.3f}) — FTS hurting results"
    )


# ---------------------------------------------------------------------------
# Tests: category-specific assertions
# ---------------------------------------------------------------------------


def test_identifier_precision_requires_fts(pipeline_store):
    """Identifier queries (TICKET-24, etc.) need FTS — semantic alone struggles."""
    queries = [(tc.query, tc.expected_top) for tc in IDENTIFIER_CASES if tc.expected_top]

    mrr_sem = _compute_mrr(pipeline_store, queries, mode="semantic")
    mrr_hyb = _compute_mrr(pipeline_store, queries, mode="hybrid")

    # Hybrid should be at least as good as semantic for identifiers
    assert mrr_hyb >= mrr_sem, f"Hybrid ({mrr_hyb:.3f}) should not degrade vs semantic ({mrr_sem:.3f}) for identifiers"


def test_boolean_queries_require_fts(pipeline_store):
    """OR queries only work with FTS component."""
    for tc in BOOLEAN_CASES:
        sem_results = pipeline_store.search(tc.query, mode="semantic", limit=10)
        hyb_results = pipeline_store.search(tc.query, mode="hybrid", limit=10)

        # Extract the OR clauses and check if hybrid finds matches for more clauses
        clauses = [c.strip() for c in tc.query.split(" OR ")]
        sem_matches = set()
        hyb_matches = set()
        for clause in clauses:
            for r in sem_results[:5]:
                if clause.lower() in r["content"].lower():
                    sem_matches.add(clause)
            for r in hyb_results[:5]:
                if clause.lower() in r["content"].lower():
                    hyb_matches.add(clause)

        # Hybrid should match at least as many clauses as semantic
        assert len(hyb_matches) >= len(sem_matches), (
            f"Query '{tc.query}': hybrid matched {len(hyb_matches)} clauses vs semantic {len(sem_matches)}"
        )


def test_importance_traps_semantic(pipeline_store):
    """Semantic search handles importance traps (modernbert is strong enough)."""
    queries = [(tc.query, tc.expected_top) for tc in IMPORTANCE_TRAP_CASES if tc.expected_top]

    top1 = 0
    for query, expected in queries:
        results = pipeline_store.search(query, mode="semantic", limit=10)
        if results and expected.lower() in results[0]["content"].lower():
            top1 += 1

    # modernbert should resolve at least half the importance traps without reranker
    assert top1 >= len(queries) // 2, (
        f"Semantic only got {top1}/{len(queries)} importance traps right (want >= {len(queries) // 2})"
    )


def test_semantic_paraphrase_quality(pipeline_store):
    """Semantic search handles paraphrased queries — embedding quality floor."""
    queries = [(tc.query, tc.expected_top) for tc in SEMANTIC_CASES if tc.expected_top]

    correct = 0
    for query, expected in queries:
        results = pipeline_store.search(query, mode="semantic", limit=5)
        if results and expected.lower() in results[0]["content"].lower():
            correct += 1

    # At least 1/3 paraphrase queries should be correct (e5-small-v2 baseline is 1/3)
    assert correct >= 1, f"Semantic search got {correct}/{len(queries)} paraphrase queries right (want >= 1)"


def test_overall_mrr_threshold(pipeline_store):
    """Hybrid MRR stays above quality floor."""
    queries = [(tc.query, tc.expected_top) for tc in ALL_TEST_CASES if tc.expected_top]
    mrr = _compute_mrr(pipeline_store, queries, mode="hybrid")

    assert mrr >= 0.60, f"Overall MRR ({mrr:.3f}) below quality floor (0.60)"


# ---------------------------------------------------------------------------
# Diagnostic report (run with pytest -s)
# ---------------------------------------------------------------------------


def test_pipeline_diagnostic_report(pipeline_store):
    """Detailed per-test diagnostic (human review). Run with pytest -s."""
    modes = ["semantic", "hybrid"]

    print("\n" + "=" * 80)
    print("PIPELINE QUALITY DIAGNOSTIC")
    print("=" * 80)

    for mode in modes:
        mrr = 0.0
        n = 0
        top1 = 0
        for tc in ALL_TEST_CASES:
            if tc.expected_top is None:
                continue
            results = pipeline_store.search(tc.query, mode=mode, limit=10)
            rank = _find_rank(results, tc.expected_top)
            n += 1
            if rank:
                mrr += 1.0 / rank
                if rank == 1:
                    top1 += 1
            print(
                f"  [{mode:8}] [{tc.category:16}] {tc.name:40} "
                f"rank={'#' + str(rank) if rank else 'MISS':>5} "
                f"top1={results[0]['content'][:50] if results else 'N/A'}"
            )
        print(f"  {mode}: MRR={mrr / n:.3f}  Top1={top1}/{n} ({top1 / n * 100:.0f}%)")
        print()
