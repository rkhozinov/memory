"""Quality benchmarks proving cross-encoder reranking improves search precision.

Strategy: the bi-encoder + composite scoring formula (0.6*similarity + 0.2*importance
+ 0.2*recency) can be tricked by high-importance memories that are topically adjacent
but not the correct answer. The cross-encoder, which sees (query, candidate) jointly,
should correct these mis-rankings.

Each test defines ground-truth (query → expected best result) pairs and measures
ranking metrics with and without reranking.
"""

import pytest

# ---------------------------------------------------------------------------
# Fixture: corpus with importance traps and topical distractors
# ---------------------------------------------------------------------------

_QUALITY_CORPUS = [
    # --- Database cluster ---
    # TARGET: PgBouncer for "connection pooling" queries
    (
        "PostgreSQL connection pooling with PgBouncer reduces connection overhead by 90%",
        "pattern",
        ["cloud:aws", "svc:database"],
        0.5,  # LOW importance — will rank below distractors
    ),
    # DISTRACTOR: high-importance DB memory that bi-encoder confuses with connection queries
    (
        "CRITICAL: PostgreSQL vacuum must run weekly on tables with heavy write load to prevent bloat",
        "decision",
        ["cloud:aws", "svc:database"],
        0.9,  # HIGH importance — auto-inferred from CRITICAL keyword
    ),
    (
        "RDS instance class upgrade from db.t3.medium to db.r5.large improved query latency by 60%",
        "decision",
        ["cloud:aws", "svc:rds"],
        0.8,
    ),
    (
        "DynamoDB table with LockID partition key required for Terraform state locking",
        "pattern",
        ["cloud:aws", "tool:terraform"],
        0.7,
    ),
    (
        "Redis cluster mode enabled for session caching, reduced API P99 from 200ms to 120ms",
        "learning",
        ["cloud:aws", "svc:redis"],
        0.6,
    ),
    # --- Kubernetes cluster ---
    # TARGET: OOM kills for "pods restarting" queries
    (
        "Kubernetes pod crash loop backoff caused by OOM kills when memory limit set to 256Mi",
        "error",
        ["svc:kubernetes"],
        0.5,  # LOW importance
    ),
    # DISTRACTOR: high-importance K8s memories with overlapping "restart" / "pod" vocabulary
    (
        "IMPORTANT: Kubernetes resource quotas must be set per namespace to prevent noisy-neighbor pod evictions",
        "decision",
        ["svc:kubernetes"],
        0.9,  # HIGH importance — auto-inferred from IMPORTANT keyword
    ),
    (
        "Kubernetes liveness probe on /healthz with 3-second timeout causes false restarts under load",
        "error",
        ["svc:kubernetes"],
        0.8,  # higher importance than OOM target
    ),
    (
        "Kubernetes node autoscaler scales down aggressively during weekend low-traffic windows",
        "learning",
        ["svc:kubernetes"],
        0.5,  # LOW importance
    ),
    (
        "Kubernetes ingress-nginx requires proxy-read-timeout annotation for websocket connections",
        "pattern",
        ["svc:kubernetes"],
        0.7,
    ),
    (
        "Kubernetes DNS resolution fails intermittently when ndots is set to 5 in resolv.conf",
        "error",
        ["svc:kubernetes"],
        0.5,  # LOW importance
    ),
    # Extra K8s distractors to dilute bi-encoder rankings
    (
        "Kubernetes horizontal pod autoscaler targets 70% CPU utilization for worker deployments",
        "pattern",
        ["svc:kubernetes"],
        0.7,
    ),
    (
        "Kubernetes PersistentVolumeClaim stuck in Pending when storage class does not match provisioner",
        "error",
        ["svc:kubernetes"],
        0.7,
    ),
    # --- CI/CD cluster ---
    (
        "GitHub Actions workflow timeout set to 60 minutes causes midnight deployment failures",
        "error",
        ["tool:github-actions"],
        0.7,
    ),
    (
        "Docker multi-stage builds reduced container image size from 1.2GB to 180MB",
        "pattern",
        ["tool:docker"],
        0.5,  # LOW importance — bi-encoder must rely on similarity alone
    ),
    # DISTRACTOR: high-importance Docker memory
    (
        "CRITICAL: Docker BuildKit cache mount requires explicit --mount=type=cache flag or builds are 3x slower",
        "decision",
        ["tool:docker"],
        0.9,  # HIGH importance
    ),
    # --- Networking cluster ---
    (
        "ALB health check path must match the application readiness endpoint exactly or targets go unhealthy",
        "pattern",
        ["cloud:aws", "svc:alb"],
        0.7,
    ),
    (
        "VPC peering between production and staging requires route table updates in both VPCs",
        "learning",
        ["cloud:aws", "svc:vpc"],
        0.7,
    ),
    (
        "DNS CNAME records managed manually in <registrar>, must create after every Kubernetes ALB deployment",
        "learning",
        ["svc:dns"],
        0.5,  # LOW importance
    ),
    # DISTRACTOR: high-importance networking memory
    (
        "IMPORTANT: Always set DNS TTL to 60 seconds during migrations to allow fast rollback",
        "decision",
        ["svc:dns"],
        0.9,  # HIGH importance
    ),
    # --- Terraform cluster ---
    (
        "Terraform S3 backend state locking requires DynamoDB table with LockID partition key",
        "pattern",
        ["tool:terraform", "cloud:aws"],
        0.5,  # LOW importance
    ),
    # DISTRACTOR: high-importance Terraform memory
    (
        "CRITICAL: Terraform plan must always run before apply in CI pipelines to catch drift",
        "decision",
        ["tool:terraform"],
        0.9,  # HIGH importance
    ),
    # --- Noise floor ---
    (
        "Go60 ZMK firmware disabled BLE and RGB underglow to shrink firmware image by 70%",
        "decision",
        ["project:go60"],
        0.8,
    ),
    (
        "Linear GraphQL API requires unquoted enum values in issueRelationCreate mutation",
        "learning",
        ["tool:linear"],
        0.7,
    ),
]


# Ground-truth queries.
# Each query is designed so the correct answer has LOW importance (0.5) while
# a topically adjacent distractor has HIGH importance (0.9). The bi-encoder's
# composite scoring inflates the distractor; the cross-encoder should fix it.
_GROUND_TRUTH = [
    # "connection pooling" → PgBouncer (imp=0.5) vs PostgreSQL vacuum CRITICAL (imp=0.9)
    ("how to reduce database connection overhead", "PgBouncer"),
    # "pods restarting OOM" → OOM kills (imp=0.5) vs resource quotas IMPORTANT (imp=0.9) / liveness probe (imp=0.8)
    ("why are my kubernetes pods restarting and getting killed", "OOM kills"),
    # "ndots DNS" → ndots (imp=0.5) vs DNS TTL IMPORTANT (imp=0.9)
    ("kubernetes DNS resolution intermittent failures", "ndots"),
    # "docker image size" → multi-stage (imp=0.5) vs BuildKit CRITICAL (imp=0.9)
    ("how to make docker images smaller", "multi-stage"),
    # "terraform state locking" → DynamoDB (imp=0.5) vs Terraform plan CRITICAL (imp=0.9)
    ("terraform state file locking mechanism", "DynamoDB"),
    # "DNS after K8s deploy" → CNAME (imp=0.5) vs DNS TTL IMPORTANT (imp=0.9)
    ("DNS records after deploying to kubernetes", "CNAME"),
    # Paraphrase: "nodes disappearing" → autoscaler (imp=0.5) vs resource quotas (imp=0.9)
    ("kubernetes nodes disappearing on weekends", "autoscaler"),
    # "RDS scaling" → RDS upgrade (imp=0.8), but distractors from DB cluster
    ("database instance scaling and sizing", "db.r5.large"),
    # "websocket" → ingress-nginx (imp=0.7), needs cross-encoder to disambiguate from other K8s
    ("websocket connections dropping through kubernetes ingress", "websocket"),
    # "ALB health check" → ALB (imp=0.7), cross-encoder should prefer exact match
    ("load balancer health check failing", "readiness endpoint"),
]


@pytest.fixture
def quality_store(store):
    """Store with importance traps for ranking quality tests."""
    for content, mtype, tags, imp in _QUALITY_CORPUS:
        store.store(content, memory_type=mtype, tags=tags, importance=imp)
    return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reciprocal_rank(results: list[dict], expected_substr: str) -> float:
    """Return 1/rank of the first result containing expected_substr, or 0."""
    for i, r in enumerate(results):
        if expected_substr.lower() in r["content"].lower():
            return 1.0 / (i + 1)
    return 0.0


def _find_rank(results: list[dict], expected_substr: str) -> int | None:
    """Return 1-based rank of the first result containing expected_substr, or None."""
    for i, r in enumerate(results):
        if expected_substr.lower() in r["content"].lower():
            return i + 1
    return None


# ---------------------------------------------------------------------------
# Quality tests
# ---------------------------------------------------------------------------


def test_rerank_improves_mrr(quality_store):
    """Cross-encoder reranking improves Mean Reciprocal Rank over bi-encoder alone.

    MRR measures how high the correct answer ranks on average (1.0 = always #1).
    The corpus has importance traps that inflate distractors in composite scoring;
    the cross-encoder should correct these.
    """
    mrr_base = 0.0
    mrr_reranked = 0.0
    n = len(_GROUND_TRUTH)

    for query, expected in _GROUND_TRUTH:
        base = quality_store.search(query, limit=10)
        reranked = quality_store.search(query, rerank=True, limit=10)
        mrr_base += _reciprocal_rank(base, expected)
        mrr_reranked += _reciprocal_rank(reranked, expected)

    mrr_base /= n
    mrr_reranked /= n

    # Reranking must not degrade MRR
    assert mrr_reranked >= mrr_base, f"Reranking degraded MRR: {mrr_reranked:.3f} < {mrr_base:.3f}"


def test_rerank_improves_top1_accuracy(quality_store):
    """Cross-encoder reranking puts the correct answer at rank 1 more often.

    Top-1 accuracy = fraction of queries where the best result is correct.
    Even a single-position improvement (rank 2 → rank 1) materially improves UX.
    """
    top1_base = 0
    top1_reranked = 0

    for query, expected in _GROUND_TRUTH:
        base = quality_store.search(query, limit=10)
        reranked = quality_store.search(query, rerank=True, limit=10)
        if base and expected.lower() in base[0]["content"].lower():
            top1_base += 1
        if reranked and expected.lower() in reranked[0]["content"].lower():
            top1_reranked += 1

    # Reranking must achieve at least as many top-1 hits
    assert top1_reranked >= top1_base, (
        f"Reranking worsened top-1: {top1_reranked} < {top1_base} out of {len(_GROUND_TRUTH)}"
    )


def test_rerank_score_separation(quality_store):
    """Reranker scores separate relevant from irrelevant better than bi-encoder similarity.

    A good reranker produces a large gap between the relevant result's reranker_score
    and the mean reranker_score of remaining results. We measure this as the ratio
    relevant_score / mean_other_score. Higher = better discrimination.
    """
    separations = []

    for query, expected in _GROUND_TRUTH:
        results = quality_store.search(query, rerank=True, limit=10)

        relevant_score = None
        other_scores = []
        for r in results:
            if expected.lower() in r["content"].lower():
                relevant_score = r["reranker_score"]
            else:
                other_scores.append(r["reranker_score"])

        if relevant_score is not None and other_scores:
            mean_other = sum(other_scores) / len(other_scores)
            if mean_other > 0:
                separations.append(relevant_score / mean_other)
            else:
                # mean_other is 0 → infinite separation (ideal)
                separations.append(float("inf"))

    assert separations, "No separation data collected"
    # Median separation ratio should be at least 2x
    sorted_seps = sorted(separations)
    median_sep = sorted_seps[len(sorted_seps) // 2]
    assert median_sep >= 2.0, f"Median score separation ratio too low: {median_sep:.2f}x (want >= 2x)"


def test_rerank_suppresses_importance_distractors(quality_store):
    """Reranking demotes high-importance distractors that the bi-encoder over-ranks.

    The corpus has CRITICAL/IMPORTANT memories (importance=0.9) that are topically
    adjacent to the correct answer but not actually relevant. The composite scorer
    inflates them; the reranker should push them below the correct answer.
    """
    # Queries where the correct answer has low importance and a distractor has high importance
    importance_trap_queries = [
        # PgBouncer (imp=0.5) vs "CRITICAL: PostgreSQL vacuum" (imp=0.9)
        ("database connection pooling", "PgBouncer"),
        # multi-stage (imp=0.5) vs "CRITICAL: Docker BuildKit cache" (imp=0.9)
        ("reduce docker image size", "multi-stage"),
        # DynamoDB state lock (imp=0.5) vs "CRITICAL: Terraform plan before apply" (imp=0.9)
        ("terraform state locking", "DynamoDB"),
        # CNAME (imp=0.5) vs "IMPORTANT: DNS TTL 60 seconds" (imp=0.9)
        ("DNS CNAME records kubernetes", "CNAME"),
    ]

    wins = 0
    for query, expected in importance_trap_queries:
        reranked = quality_store.search(query, rerank=True, limit=5)
        # The correct answer should be in top 2 after reranking
        top2_content = " ".join(r["content"] for r in reranked[:2]).lower()
        if expected.lower() in top2_content:
            wins += 1

    # At least 3 of 4 importance traps should be resolved
    assert wins >= 3, f"Reranking only resolved {wins}/4 importance trap queries (want >= 3)"


def test_rerank_relevant_scores_high_irrelevant_scores_low(quality_store):
    """Cross-encoder assigns high scores to relevant and near-zero to irrelevant.

    Unlike bi-encoder similarity (which has a high floor ~0.5-0.7 even for unrelated
    content), the cross-encoder should give truly irrelevant content scores near 0.
    """
    # Query about Kubernetes OOM → firmware/GraphQL memories should score near 0
    results = quality_store.search("kubernetes pod out of memory crash", rerank=True, limit=10)

    relevant_found = False
    for r in results:
        if "oom kills" in r["content"].lower():
            # The relevant result should have high reranker score
            assert r["reranker_score"] > 0.5, f"Relevant result scored too low: {r['reranker_score']:.4f}"
            relevant_found = True
        elif any(kw in r["content"].lower() for kw in ["zmk firmware", "graphql", "linear"]):
            # Noise-floor results should score near zero
            assert r["reranker_score"] < 0.1, (
                f"Irrelevant result scored too high: {r['reranker_score']:.4f} — {r['content'][:60]}"
            )

    assert relevant_found, "OOM kills memory not found in results"


def test_rerank_keyword_match_beats_topic_similarity(quality_store):
    """Cross-encoder prefers direct keyword matches over topical neighbors.

    The bi-encoder embeds "ndots" and "DNS TTL" similarly (both are DNS-related),
    but the cross-encoder should clearly prefer the result that actually mentions
    the queried concept.
    """
    results = quality_store.search("ndots setting causing DNS failures", rerank=True, limit=5)
    assert results, "No results returned"
    # The ndots memory must be #1
    assert "ndots" in results[0]["content"].lower(), f"Expected ndots memory at #1, got: {results[0]['content'][:80]}"
    # Its reranker score should dominate
    if len(results) > 1:
        assert results[0]["reranker_score"] > results[1]["reranker_score"], (
            f"#1 reranker_score ({results[0]['reranker_score']:.4f}) should exceed "
            f"#2 ({results[1]['reranker_score']:.4f})"
        )


def test_rerank_detailed_report(quality_store):
    """Detailed per-query ranking report (for human review). Run with pytest -s."""
    print("\n" + "=" * 80)
    print("RERANKING QUALITY REPORT")
    print("=" * 80)

    total_base_rr = 0.0
    total_reranked_rr = 0.0
    improvements = 0
    regressions = 0

    for query, expected in _GROUND_TRUTH:
        base = quality_store.search(query, limit=10)
        reranked = quality_store.search(query, rerank=True, limit=10)
        rank_base = _find_rank(base, expected)
        rank_reranked = _find_rank(reranked, expected)
        rr_base = _reciprocal_rank(base, expected)
        rr_reranked = _reciprocal_rank(reranked, expected)
        total_base_rr += rr_base
        total_reranked_rr += rr_reranked

        if rank_reranked is not None and rank_base is not None:
            if rank_reranked < rank_base:
                delta = "IMPROVED"
                improvements += 1
            elif rank_reranked > rank_base:
                delta = "REGRESSED"
                regressions += 1
            else:
                delta = "unchanged"
        elif rank_base is None and rank_reranked is not None:
            delta = "IMPROVED (found)"
            improvements += 1
        elif rank_base is not None and rank_reranked is None:
            delta = "REGRESSED (lost)"
            regressions += 1
        else:
            delta = "unchanged (not found)"

        print(f"\n  Q: {query}")
        print(f"  Expected: ...{expected}...")
        print(f"  Rank: {rank_base} -> {rank_reranked}  [{delta}]")

        if reranked and rank_reranked:
            r = reranked[rank_reranked - 1]
            print(f"  Reranker score: {r.get('reranker_score', 'N/A')}")

    n = len(_GROUND_TRUTH)
    mrr_base = total_base_rr / n
    mrr_reranked = total_reranked_rr / n

    print(f"\n{'=' * 80}")
    print(f"  MRR:        {mrr_base:.3f} -> {mrr_reranked:.3f}  (delta {mrr_reranked - mrr_base:+.3f})")
    print(f"  Improved:   {improvements}/{n}")
    print(f"  Regressed:  {regressions}/{n}")
    print(f"  Unchanged:  {n - improvements - regressions}/{n}")
    print("=" * 80)
