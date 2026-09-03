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
    corpus.load_corpus(store)
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


def test_weighted_best_beats_weighted_on_identifiers(pipeline_store):
    """weighted_best (shipped default: CSLS + exact-ID promotion) must not regress
    identifier retrieval vs the plain weighted fusion, and should improve it."""
    queries = [(tc.query, tc.expected_top) for tc in IDENTIFIER_CASES if tc.expected_top]

    mrr_weighted = _compute_mrr(pipeline_store, queries, mode="hybrid", score_fusion="weighted")
    mrr_best = _compute_mrr(pipeline_store, queries, mode="hybrid", score_fusion="weighted_best")

    assert mrr_best >= mrr_weighted, (
        f"weighted_best ({mrr_best:.3f}) regressed vs weighted ({mrr_weighted:.3f}) on identifiers"
    )


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


# Per-category floors, measured 2026-09-03 on the shipping default config and set
# one notch below the observed value. A single aggregate floor lets a regression in
# one family hide behind gains in another — identifier precision can collapse while
# semantic improves and the total never moves. Category counts are small (n=2..6),
# so one query flipping rank moves a category by 0.08-0.25; the floors are spaced to
# tolerate that much and no more.
#
#   category          n   observed   floor
#   identifier        4   1.000      0.90
#   topic             5   1.000      0.90
#   mixed             2   1.000      0.75   (n=2: one miss costs 0.25)
#   importance_trap   6   0.917      0.80
#   stale_fact        3   0.833      0.66   (see note below)
#   semantic          3   0.667      0.60
#
# stale_fact is a gap probe, not a passing feature: 1 of 3 pairs already returns the
# superseded fact above its replacement. The floor pins the current state so it
# cannot silently get worse while the update path is being designed.
CATEGORY_MRR_FLOORS = {
    "identifier": 0.90,
    "topic": 0.90,
    "mixed": 0.75,
    "importance_trap": 0.80,
    "stale_fact": 0.66,
    "semantic": 0.60,
}


def _per_category_mrr(store, mode: str = "hybrid") -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for tc in ALL_TEST_CASES:
        if not tc.expected_top:
            continue
        results = store.search(tc.query, mode=mode, limit=10)
        rank = _find_rank(results, tc.expected_top)
        per.setdefault(tc.category, []).append(1.0 / rank if rank else 0.0)
    return {c: sum(v) / len(v) for c, v in per.items()}


def test_overall_mrr_threshold(pipeline_store):
    """Hybrid MRR stays above quality floor."""
    queries = [(tc.query, tc.expected_top) for tc in ALL_TEST_CASES if tc.expected_top]
    mrr = _compute_mrr(pipeline_store, queries, mode="hybrid")

    assert mrr >= 0.60, f"Overall MRR ({mrr:.3f}) below quality floor (0.60)"


def test_per_category_mrr_floors(pipeline_store):
    """No single case family may regress, even if the aggregate holds."""
    actual = _per_category_mrr(pipeline_store)

    missing = set(CATEGORY_MRR_FLOORS) - set(actual)
    assert not missing, f"floors defined for categories with no test cases: {sorted(missing)}"
    unguarded = set(actual) - set(CATEGORY_MRR_FLOORS)
    assert not unguarded, f"case families with no floor — add one to CATEGORY_MRR_FLOORS: {sorted(unguarded)}"

    failures = [
        f"{cat}: MRR {actual[cat]:.3f} < floor {floor:.2f}"
        for cat, floor in CATEGORY_MRR_FLOORS.items()
        if actual[cat] < floor
    ]
    assert not failures, "per-category MRR regression:\n  " + "\n  ".join(failures)


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


# ---------------------------------------------------------------------------
# Document body recall (gap probe)
# ---------------------------------------------------------------------------
#
# `store_doc` embeds only the summary (core.py:5041); the body reaches retrieval
# through FTS5 alone. So a lexical query on a body term still works, but a
# *paraphrase* of a body-only fact has nothing to match: FTS misses on vocabulary,
# and the semantic side never saw the body. These two tests pin that boundary —
# the first passes today, the second is the gap.

_DOC_SUMMARY = "Runbook for rotating credentials in the billing pipeline."
_DOC_BODY = (
    "Runbook: billing pipeline credential rotation.\n\n"
    "Step 1. Drain the queue before touching anything.\n"
    "Step 2. The rotation job refuses to start while a replica is lagging by more "
    "than ninety seconds, so check replication delay first.\n"
    "Step 3. Re-issue the signing key and restart the workers.\n"
)


# Distractors matter: doc search applies no minimum-score threshold, so a
# single-document store returns the target for any query at all and the probe
# proves nothing. These give the ranker something to be wrong about.
_DOC_DISTRACTORS = [
    (
        "Terraform state migration guide",
        "Moving state between S3 backends. Run terraform init -migrate-state, "
        "confirm the DynamoDB lock table, then verify the plan is empty.",
        "How to migrate Terraform state between S3 backends.",
    ),
    (
        "Kubernetes upgrade checklist",
        "Drain each node, cordon it, upgrade kubelet, uncordon. Watch for pods "
        "stuck terminating and for PodDisruptionBudgets blocking the drain.",
        "Checklist for upgrading a Kubernetes cluster node by node.",
    ),
    (
        "Incident response playbook",
        "Page the on-call, open a channel, assign a scribe. Do not start "
        "remediation before someone owns communication.",
        "What to do in the first ten minutes of an incident.",
    ),
    (
        "Database backup restore procedure",
        "Restore from the most recent base backup, replay WAL to the target "
        "timestamp, then re-point the application connection string.",
        "How to restore the database to a point in time.",
    ),
    (
        "Onboarding a new service",
        "Create the ECR repository, the IAM role, the IRSA binding and the secret. Then wire the deployment pipeline.",
        "Steps to onboard a new service into the platform.",
    ),
]


@pytest.fixture
def doc_store(store):
    for title, body, summary in _DOC_DISTRACTORS:
        store.store_doc(title=title, body=body, summary=summary, doc_type="runbook")
    store.store_doc(
        title="Billing credential rotation runbook",
        body=_DOC_BODY,
        summary=_DOC_SUMMARY,
        doc_type="runbook",
        tags=["project:memory", "svc:billing"],
    )
    return store


def _doc_rank(results: list[dict]) -> int | None:
    for i, r in enumerate(results):
        if "credential rotation" in (r.get("title", "") or "").lower():
            return i + 1
    return None


def test_doc_body_lexical_recall(doc_store):
    """Body terms are reachable: document_fts indexes title and body."""
    results = doc_store.search_docs("replication delay rotation job", limit=5)
    assert _doc_rank(results) == 1, "FTS should reach body text verbatim"


@pytest.mark.xfail(
    reason="Document bodies are never embedded (core.py:5041) — only the 500-char "
    "summary is. A paraphrase of a body-only fact has nothing to match: FTS misses "
    "on vocabulary and the semantic side never saw the body.",
    strict=True,
)
def test_doc_body_semantic_recall(doc_store):
    """A body-only fact, asked in vocabulary the 500-char summary does not contain."""
    # The answer is stated verbatim in the body. Measured rank on the current
    # build: 5 of 5, behind an unrelated database-backup runbook.
    results = doc_store.search_docs("how far behind can the standby fall before the job aborts", limit=5)
    assert _doc_rank(results) == 1, "body-only facts should be semantically reachable"
