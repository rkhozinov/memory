"""Shared corpus and test cases for pipeline benchmarks.

Used by both benchmarks/bench_pipeline.py (report script with sentence-transformers)
and tests/test_pipeline_quality.py (pytest regression with ONNX/MemoryStore).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Corpus: production-like memories with importance scores
# ---------------------------------------------------------------------------


@dataclass
class CorpusEntry:
    content: str
    memory_type: str
    tags: list[str]
    importance: float
    # Backdate `created_at` by this many days when loading. Only the stale-fact
    # pairs use it: recency is 10% of the composite score, so an old fact and its
    # replacement are indistinguishable unless they actually differ in age.
    age_days: float = 0.0


CORPUS: list[CorpusEntry] = [
    # --- Ticket identifiers (exact match precision) ---
    CorpusEntry(
        "TICKET-24: RDS instance class upgrade from db.t3.medium to db.r5.large in production",
        "decision",
        ["project:myproject", "cloud:aws", "svc:rds"],
        0.8,
    ),
    CorpusEntry(
        "TICKET-75: DNS records are managed manually in <registrar>. After K8s deployment,"
        " must manually create CNAME pointing to ALB hostname",
        "learning",
        ["project:myproject", "svc:dns", "svc:kubernetes"],
        0.6,
    ),
    CorpusEntry(
        "TICKET-198: Eliminate billing-manager Node.js sidecar. Install frontend-templates at Docker build time",
        "decision",
        ["project:myproject", "tool:docker", "svc:nodejs"],
        0.8,
    ),
    CorpusEntry(
        "TICKET-64: notifications-service PRs: infra repo PR #N covers ECR, IAM policy, SM secret, IRSA",
        "reference",
        ["project:myproject", "cloud:aws", "svc:ecr"],
        0.6,
    ),
    # --- Kubernetes / deployment ---
    CorpusEntry(
        "Kubernetes pod crash loop backoff caused by OOM kills when memory limit set to 256Mi",
        "error",
        ["svc:kubernetes"],
        0.5,
    ),
    CorpusEntry(
        "Kubernetes node autoscaler scales down aggressively during weekend low-traffic windows",
        "learning",
        ["svc:kubernetes"],
        0.5,
    ),
    CorpusEntry(
        "Kubernetes liveness probe on /healthz with 3-second timeout causes false restarts under load",
        "error",
        ["svc:kubernetes"],
        0.7,
    ),
    CorpusEntry(
        "Kubernetes DNS resolution fails intermittently when ndots is set to 5 in resolv.conf",
        "error",
        ["svc:kubernetes"],
        0.5,
    ),
    # --- Terraform / IaC ---
    CorpusEntry(
        "Terraform S3 backend state locking requires DynamoDB table with LockID partition key",
        "pattern",
        ["tool:terraform", "cloud:aws"],
        0.5,
    ),
    CorpusEntry(
        "Terraform module for AWS VPC networking with public and private subnets across 3 AZs",
        "pattern",
        ["tool:terraform", "cloud:aws", "svc:vpc"],
        0.7,
    ),
    # --- Docker / build ---
    CorpusEntry(
        ".NET Dockerfile with BuildKit secrets is incompatible with QEMU cross-compilation",
        "error",
        ["tool:docker", "tool:buildkit"],
        0.7,
    ),
    CorpusEntry(
        "Docker multi-stage builds reduced container image size from 1.2GB to 180MB",
        "pattern",
        ["tool:docker"],
        0.5,
    ),
    # --- Database ---
    CorpusEntry(
        "PostgreSQL connection pooling with PgBouncer reduces connection overhead by 90%",
        "pattern",
        ["cloud:aws", "svc:database"],
        0.5,
    ),
    CorpusEntry(
        "Redis cluster mode enabled for session caching, reduced API P99 from 200ms to 120ms",
        "learning",
        ["cloud:aws", "svc:redis"],
        0.6,
    ),
    # --- Networking ---
    CorpusEntry(
        "ALB health check path must match the application readiness endpoint exactly or targets go unhealthy",
        "pattern",
        ["cloud:aws", "svc:alb"],
        0.7,
    ),
    # --- Noise floor (unrelated topics) ---
    CorpusEntry(
        "Go60 ZMK firmware: disabled BLE and RGB underglow, firmware shrunk 70%",
        "decision",
        ["project:go60"],
        0.8,
    ),
    CorpusEntry(
        "nvim zen-mode on_close: must use vim.schedule() to defer quit command outside WinClosed handler",
        "learning",
        ["tool:nvim"],
        0.6,
    ),
    CorpusEntry(
        "Python asyncio: use asyncio.gather for concurrent IO-bound tasks, avoid mixing with threads",
        "learning",
        ["tool:python"],
        0.6,
    ),
    CorpusEntry(
        "B7 interference confirmed area-wide. Keep B7 excluded,"
        " B1 as PCC gives SINR 8-11dB, much more reliable NR NSA anchor",
        "learning",
        ["project:5g"],
        0.6,
    ),
    # --- Importance traps (high-importance distractors) ---
    CorpusEntry(
        "CRITICAL: Terraform plan must always run before apply in CI pipelines to catch drift",
        "decision",
        ["tool:terraform"],
        0.9,
    ),
    CorpusEntry(
        "IMPORTANT: Kubernetes resource quotas must be set per namespace to prevent noisy-neighbor pod evictions",
        "decision",
        ["svc:kubernetes"],
        0.9,
    ),
    CorpusEntry(
        "CRITICAL: Docker BuildKit cache mount requires explicit --mount=type=cache flag or builds are 3x slower",
        "decision",
        ["tool:docker", "tool:buildkit"],
        0.9,
    ),
    CorpusEntry(
        "IMPORTANT: Always set DNS TTL to 60 seconds during migrations to allow fast rollback",
        "decision",
        ["svc:dns"],
        0.9,
    ),
    CorpusEntry(
        "CRITICAL: PostgreSQL vacuum must run weekly on tables with heavy write load to prevent bloat",
        "decision",
        ["cloud:aws", "svc:database"],
        0.9,
    ),
    CorpusEntry(
        "IMPORTANT: asyncio event loop must not be shared across threads in Python web servers",
        "decision",
        ["tool:python"],
        0.9,
    ),
    # --- Stale-fact pairs: old entry backdated, replacement recent ---
    CorpusEntry(
        "The memory service embeds with all-MiniLM-L6-v2, 384 dimensions, running on ONNX CPU",
        "reference",
        ["project:memory", "svc:embeddings"],
        0.6,
        age_days=180,
    ),
    CorpusEntry(
        "The memory service embeds with modernbert-embed-base at 768 dimensions on the MLX Metal backend",
        "reference",
        ["project:memory", "svc:embeddings"],
        0.6,
        age_days=2,
    ),
    CorpusEntry(
        "CI for this repo builds on GitHub Actions ubuntu-22.04 runners",
        "reference",
        ["project:memory", "tool:github-actions"],
        0.6,
        age_days=120,
    ),
    CorpusEntry(
        "CI for this repo builds on GitHub Actions ubuntu-24.04 runners",
        "reference",
        ["project:memory", "tool:github-actions"],
        0.6,
        age_days=1,
    ),
    CorpusEntry(
        "The API authenticates callers with static API keys stored in the environment",
        "decision",
        ["project:memory", "svc:auth"],
        0.8,
        age_days=200,
    ),
    CorpusEntry(
        "The API authenticates callers with short-lived OIDC tokens instead of static API keys",
        "decision",
        ["project:memory", "svc:auth"],
        0.8,
        age_days=3,
    ),
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    query: str
    # Substring that MUST appear in top-1 result (None = no assertion on top-1)
    expected_top: str | None = None
    category: str = "general"
    # Substrings that should NOT outrank expected_top
    should_not_outrank: list[str] = field(default_factory=list)


# --- Identifier precision: exact ticket match ---
IDENTIFIER_CASES = [
    TestCase(
        "TICKET-24 identifier",
        "TICKET-24",
        "TICKET-24",
        "identifier",
        ["TICKET-75", "TICKET-198", "TICKET-64"],
    ),
    TestCase(
        "TICKET-75 identifier",
        "TICKET-75",
        "TICKET-75",
        "identifier",
        ["TICKET-24", "TICKET-198", "TICKET-64"],
    ),
    TestCase(
        "TICKET-198 identifier",
        "TICKET-198",
        "TICKET-198",
        "identifier",
        ["TICKET-24", "TICKET-75"],
    ),
    TestCase(
        "TICKET-64 identifier",
        "TICKET-64",
        "TICKET-64",
        "identifier",
        ["TICKET-24", "TICKET-75", "TICKET-198"],
    ),
]

# --- Topic retrieval: semantic + keyword ---
TOPIC_CASES = [
    TestCase("terraform state", "terraform state locking", "Terraform S3 backend state locking", "topic"),
    TestCase("k8s crash loop", "kubernetes pod crash loop", "crash loop backoff", "topic"),
    TestCase("DNS CNAME", "DNS CNAME records", "DNS records", "topic"),
    TestCase("Docker BuildKit", "Dockerfile BuildKit cross compilation", "BuildKit secrets", "topic"),
    TestCase("ZMK firmware", "ZMK keyboard firmware BLE", "ZMK firmware", "topic"),
]

# --- Semantic paraphrase: query uses different words than content ---
SEMANTIC_CASES = [
    TestCase("container orchestration → k8s", "container orchestration platform scaling", "Kubernetes", "semantic"),
    TestCase(
        "infrastructure as code → terraform", "infrastructure as code cloud provisioning", "Terraform", "semantic"
    ),
    TestCase("concurrent programming → asyncio", "concurrent programming async await", "asyncio", "semantic"),
]

# --- Importance traps: correct answer (low imp) vs distractor (high imp) ---
IMPORTANCE_TRAP_CASES = [
    TestCase(
        "connection pooling (vs vacuum CRITICAL)",
        "how to reduce database connection overhead",
        "PgBouncer",
        "importance_trap",
    ),
    TestCase(
        "OOM kills (vs quotas IMPORTANT)",
        "why are my kubernetes pods restarting and getting killed",
        "OOM kills",
        "importance_trap",
    ),
    TestCase(
        "docker image size (vs BuildKit CRITICAL)",
        "how to make docker images smaller",
        "multi-stage",
        "importance_trap",
    ),
    TestCase(
        "terraform state lock (vs plan CRITICAL)",
        "terraform state file locking mechanism",
        "DynamoDB",
        "importance_trap",
    ),
    TestCase(
        "DNS after K8s deploy (vs TTL IMPORTANT)",
        "DNS records after deploying to kubernetes",
        "CNAME",
        "importance_trap",
    ),
    TestCase(
        "asyncio gather (vs threads IMPORTANT)",
        "python async await concurrent tasks gather",
        "asyncio.gather",
        "importance_trap",
    ),
]

# --- Boolean queries: OR-separated clauses (FTS-only capability) ---
BOOLEAN_CASES = [
    TestCase(
        "two-clause OR",
        "TICKET-24 OR TICKET-75",
        None,  # either TICKET-24 or TICKET-75 is acceptable top-1
        "boolean",
    ),
    TestCase(
        "three-clause OR",
        "terraform state OR kubernetes crash OR BuildKit",
        None,
        "boolean",
    ),
    TestCase(
        "disjoint OR",
        "ZMK firmware OR asyncio gather",
        None,
        "boolean",
    ),
]

# --- Mixed queries: keyword + semantic meaning ---
MIXED_CASES = [
    TestCase("TICKET-24 + topic", "TICKET-24 RDS upgrade", "TICKET-24", "mixed"),
    TestCase("TICKET-75 + topic", "TICKET-75 DNS kubernetes", "TICKET-75", "mixed"),
]

# --- Noise rejection: should NOT return confident results ---
NOISE_CASES = [
    TestCase("nonexistent ticket", "TICKET-999", None, "noise"),
    TestCase("unrelated topic", "quantum computing qubits entanglement", None, "noise"),
]

# --- Stale facts: a superseded fact and its replacement both present ---
# Probes the missing update operation. `dream`'s supersession pass only fires on
# cos>=0.85 AND >=2 shared tags AND (a contradiction keyword OR type in
# {decision,error}), so the CI-runner pair below is invisible to it: it is a
# `reference` and its replacement contains no contradiction word. Expect these to
# fail on the current build — that failing baseline is the point.
STALE_FACT_CASES = [
    TestCase(
        "embedding model superseded",
        "which embedding model does the memory service use",
        "modernbert-embed-base",
        "stale_fact",
        ["384 dimensions"],
    ),
    TestCase(
        "CI runner image superseded",
        "which CI runner image does the repo build on",
        "ubuntu-24.04",
        "stale_fact",
        ["ubuntu-22.04"],
    ),
    TestCase(
        "auth mechanism superseded",
        "how does the API authenticate callers",
        "short-lived OIDC tokens",
        "stale_fact",
        ["static API keys stored in the environment"],
    ),
]

ALL_TEST_CASES = (
    IDENTIFIER_CASES
    + TOPIC_CASES
    + SEMANTIC_CASES
    + IMPORTANCE_TRAP_CASES
    + BOOLEAN_CASES
    + MIXED_CASES
    + NOISE_CASES
    + STALE_FACT_CASES
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_corpus(store, entries: list[CorpusEntry] | None = None) -> None:
    """Store every corpus entry, backdating `created_at` where `age_days` is set.

    Shared by tests/bench_replay.py and tests/test_pipeline_quality.py so the two
    harnesses can never drift on how the stale-fact pairs are aged.
    """
    import time as _time

    now = _time.time()
    conn = store._get_conn()
    for entry in entries if entries is not None else CORPUS:
        res = store.store(
            entry.content,
            memory_type=entry.memory_type,
            tags=entry.tags,
            importance=entry.importance,
        )
        if not entry.age_days:
            continue
        ts = now - entry.age_days * 86400
        iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(ts))
        conn.execute(
            "UPDATE memories SET created_at = ?, created_at_iso = ? WHERE content_hash = ?",
            (ts, iso, res["content_hash"]),
        )
