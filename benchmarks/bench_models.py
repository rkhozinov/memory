"""Benchmark: embedding model quality comparison.

Compares candidate models on retrieval quality using production-like
memory data. Tests identifier precision, topic retrieval, semantic
similarity, and noise rejection.

Usage:
    # Install benchmark deps (not needed for main package)
    pip install sentence-transformers

    # Run all models
    python benchmarks/bench_models.py

    # Run specific models
    python benchmarks/bench_models.py --models e5-small bge-small mdbr-leaf-mt

    # Include Qwen3 (requires ~1.2GB download)
    python benchmarks/bench_models.py --models all

    # MLX backend (Apple Silicon GPU)
    python benchmarks/bench_models.py --backend mlx
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Corpus: realistic production memories
# ---------------------------------------------------------------------------

CORPUS = [
    # Ticket identifiers — the core precision scenario
    (
        "TICKET-24: RDS instance class upgrade from db.t3.medium to db.r5.large"
        " in production",
        ["TICKET-24", "RDS", "production"],
    ),
    (
        "TICKET-75: DNS records are managed manually in <registrar>. After K8s"
        " deployment, must manually create CNAME pointing to ALB hostname",
        ["TICKET-75", "DNS", "K8s"],
    ),
    (
        "TICKET-198: Eliminate billing-manager Node.js sidecar. Install"
        " frontend-templates at Docker build time",
        ["TICKET-198", "Node.js", "Docker"],
    ),
    (
        "TICKET-64: notifications-service PRs: infra repo PR #N"
        " covers ECR, IAM policy, SM secret, IRSA",
        ["TICKET-64", "ECR", "IAM"],
    ),
    (
        "TICKET-109, TICKET-110, TICKET-111 are unassigned In Progress tickets"
        " that need owner cleanup",
        ["TICKET-109", "tickets", "cleanup"],
    ),
    # Kubernetes/deployment
    (
        "Kubernetes pod crash loop backoff: check container exit code,"
        " OOM kills, and liveness probe misconfiguration",
        ["kubernetes", "crash", "OOM"],
    ),
    (
        "Kubernetes node autoscaler scales down aggressively during"
        " low traffic windows",
        ["kubernetes", "autoscaler", "scaling"],
    ),
    # Terraform
    (
        "Terraform S3 backend state locking requires DynamoDB table"
        " with LockID partition key",
        ["terraform", "state", "DynamoDB"],
    ),
    (
        "Terraform module for AWS VPC networking with public and"
        " private subnets across 3 AZs",
        ["terraform", "VPC", "AWS"],
    ),
    # Docker / build
    (
        ".NET Dockerfile with BuildKit secrets is incompatible with"
        " QEMU cross-compilation",
        ["Docker", ".NET", "BuildKit"],
    ),
    # Unrelated / noise
    (
        "Go60 ZMK firmware: disabled BLE and RGB underglow,"
        " firmware shrunk 70%",
        ["ZMK", "firmware", "BLE"],
    ),
    (
        "nvim zen-mode on_close: must use vim.schedule() to defer"
        " quit command outside WinClosed handler",
        ["nvim", "zen-mode", "lua"],
    ),
    (
        "Python asyncio: use asyncio.gather for concurrent IO-bound"
        " tasks, avoid mixing with threads",
        ["python", "asyncio", "concurrency"],
    ),
    (
        "MediaMTX alerts target only mediamtx-origin pods because"
        " edge nodes always show notReady by design",
        ["MediaMTX", "alerts", "kubernetes"],
    ),
    (
        "B7 interference confirmed area-wide. Keep B7 excluded,"
        " B1 as PCC gives SINR 8-11dB, much more reliable NR NSA anchor",
        ["5G", "interference", "SINR"],
    ),
]

# ---------------------------------------------------------------------------
# Test cases: (query, expected_top_content_substring, test_name)
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    query: str
    # Substring that MUST appear in the top-1 result
    expected_top: str | None = None
    # Substrings that should NOT outrank expected_top
    should_not_outrank: list[str] = field(default_factory=list)
    # Category for grouping in report
    category: str = "general"


TEST_CASES = [
    # --- Identifier precision ---
    TestCase(
        name="TICKET-24 identifier",
        query="TICKET-24",
        expected_top="TICKET-24",
        should_not_outrank=["TICKET-75", "TICKET-198", "TICKET-64", "TICKET-109"],
        category="identifier",
    ),
    TestCase(
        name="TICKET-75 identifier",
        query="TICKET-75",
        expected_top="TICKET-75",
        should_not_outrank=["TICKET-24", "TICKET-198", "TICKET-64"],
        category="identifier",
    ),
    TestCase(
        name="TICKET-198 identifier",
        query="TICKET-198",
        expected_top="TICKET-198",
        should_not_outrank=["TICKET-24", "TICKET-75"],
        category="identifier",
    ),
    TestCase(
        name="TICKET-64 identifier",
        query="TICKET-64",
        expected_top="TICKET-64",
        should_not_outrank=["TICKET-24", "TICKET-75", "TICKET-198"],
        category="identifier",
    ),
    # --- Topic retrieval ---
    TestCase(
        name="terraform state",
        query="terraform state locking",
        expected_top="Terraform S3 backend state locking",
        category="topic",
    ),
    TestCase(
        name="k8s crash loop",
        query="kubernetes pod crash loop",
        expected_top="crash loop backoff",
        category="topic",
    ),
    TestCase(
        name="DNS records",
        query="DNS CNAME records",
        expected_top="DNS records",
        category="topic",
    ),
    TestCase(
        name="Docker BuildKit",
        query="Dockerfile BuildKit cross compilation",
        expected_top="BuildKit secrets",
        category="topic",
    ),
    TestCase(
        name="ZMK firmware",
        query="ZMK keyboard firmware BLE",
        expected_top="ZMK firmware",
        category="topic",
    ),
    # --- Semantic similarity (rephrased queries) ---
    TestCase(
        name="container orchestration → k8s",
        query="container orchestration platform scaling",
        expected_top="Kubernetes",
        category="semantic",
    ),
    TestCase(
        name="infrastructure as code → terraform",
        query="infrastructure as code cloud provisioning",
        expected_top="Terraform",
        category="semantic",
    ),
    TestCase(
        name="concurrent programming → asyncio",
        query="concurrent programming async await",
        expected_top="asyncio",
        category="semantic",
    ),
    # --- Noise rejection ---
    TestCase(
        name="TICKET-999 (nonexistent)",
        query="TICKET-999",
        expected_top=None,  # no specific expected top
        category="noise",
    ),
    TestCase(
        name="unrelated query",
        query="quantum computing qubits entanglement",
        expected_top=None,
        category="noise",
    ),
]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # Current baseline
    "e5-small": {
        "hf_id": "intfloat/e5-small",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "dims": 384,
        "pooling": "mean",
    },
    # Direct upgrade
    "e5-small-v2": {
        "hf_id": "intfloat/e5-small-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "dims": 384,
        "pooling": "mean",
    },
    # Best 12L all-rounder
    "bge-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "cls",
    },
    # LEAF models — best in 6L class
    "mdbr-leaf-mt": {
        "hf_id": "MongoDB/mdbr-leaf-mt",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 384,
        "pooling": "mean",
    },
    "mdbr-leaf-ir": {
        "hf_id": "MongoDB/mdbr-leaf-ir",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 384,
        "pooling": "mean",
    },
    # Ultra-compact
    "arctic-embed-xs": {
        "hf_id": "Snowflake/snowflake-arctic-embed-xs",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "cls",
    },
    # Large but powerful (Qwen3)
    "qwen3-0.6B": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "prefix_query": "Instruct: Retrieve relevant memories for this query\nQuery: ",
        "prefix_doc": "",
        "dims": 1024,
        "truncate_dim": 384,  # MRL truncation to match our schema
        "pooling": "last_token",
    },
}

DEFAULT_MODELS = [
    "e5-small",
    "e5-small-v2",
    "bge-small",
    "mdbr-leaf-mt",
    "mdbr-leaf-ir",
    "arctic-embed-xs",
]


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


def _load_st_model(model_id: str) -> object:
    """Load model via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, trust_remote_code=True)


def _embed_st(model, texts: list[str], is_query: bool, model_cfg: dict) -> np.ndarray:
    """Embed texts using sentence-transformers."""
    prefix = model_cfg["prefix_query"] if is_query else model_cfg["prefix_doc"]
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    # MRL truncation if needed
    truncate_dim = model_cfg.get("truncate_dim")
    if truncate_dim:
        embeddings = embeddings[:, :truncate_dim]
        # Re-normalize after truncation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


def _load_mlx_model(model_id: str) -> tuple:
    """Load model via mlx-embedding-models or mlx-embeddings."""
    try:
        from mlx_embedding_models.embedding import EmbeddingModel

        # Map HF IDs to registry names
        registry_map = {
            "BAAI/bge-small-en-v1.5": "bge-small",
            "Snowflake/snowflake-arctic-embed-xs": "snowflake-xs",
        }
        reg_name = registry_map.get(model_id)
        if reg_name:
            return ("mlx_emb_models", EmbeddingModel.from_registry(reg_name))
    except ImportError:
        pass

    # Fallback: mlx-embeddings (supports Qwen3)
    from mlx_embeddings.utils import load

    mlx_map = {
        "BAAI/bge-small-en-v1.5": "mlx-community/bge-small-en-v1.5-bf16",
        "Qwen/Qwen3-Embedding-0.6B": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    }
    mlx_id = mlx_map.get(model_id, model_id)
    model, tokenizer = load(mlx_id)
    return ("mlx_embeddings", model, tokenizer)


def _embed_mlx(model_tuple, texts: list[str], is_query: bool, model_cfg: dict) -> np.ndarray:
    """Embed texts using MLX backend."""
    import mlx.core as mx

    prefix = model_cfg["prefix_query"] if is_query else model_cfg["prefix_doc"]
    prefixed = [prefix + t for t in texts]

    if model_tuple[0] == "mlx_emb_models":
        _, model = model_tuple
        emb = model.encode(prefixed)
        return np.array(emb)

    _, model, tokenizer = model_tuple
    # mlx-embeddings path
    all_embeddings = []
    for text in prefixed:
        inputs = tokenizer(text, return_tensors="mlx", padding=True, truncation=True)
        outputs = model(**inputs)
        emb = np.array(outputs.text_embeds[0])
        all_embeddings.append(emb)
    embeddings = np.stack(all_embeddings)

    truncate_dim = model_cfg.get("truncate_dim")
    if truncate_dim:
        embeddings = embeddings[:, :truncate_dim]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between a (n, d) and b (m, d)."""
    return a @ b.T


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    top1_content: str
    top1_sim: float
    details: str = ""


def run_tests(
    query_embedder,
    doc_embedder,
    model_cfg: dict,
) -> tuple[list[TestResult], dict]:
    """Run all test cases and return results + metrics."""
    # Embed corpus
    corpus_texts = [c[0] for c in CORPUS]
    corpus_embs = doc_embedder(corpus_texts, is_query=False, model_cfg=model_cfg)

    results = []
    for tc in TEST_CASES:
        query_emb = query_embedder([tc.query], is_query=True, model_cfg=model_cfg)
        sims = cosine_similarity(query_emb, corpus_embs)[0]
        ranked_indices = np.argsort(sims)[::-1]

        top1_idx = ranked_indices[0]
        top1_content = corpus_texts[top1_idx]
        top1_sim = float(sims[top1_idx])

        passed = True
        details = ""

        if tc.expected_top:
            if tc.expected_top not in top1_content:
                passed = False
                details = f"expected '{tc.expected_top}' in top-1, got: {top1_content[:60]}"

        if tc.should_not_outrank and passed:
            for bad in tc.should_not_outrank:
                for i, idx in enumerate(ranked_indices):
                    if bad in corpus_texts[idx]:
                        if i == 0 and tc.expected_top and tc.expected_top not in corpus_texts[idx]:
                            passed = False
                            details = f"'{bad}' outranked expected result"
                        break

        results.append(TestResult(
            name=tc.name,
            category=tc.category,
            passed=passed,
            top1_content=top1_content[:70],
            top1_sim=top1_sim,
            details=details,
        ))

    # Aggregate metrics
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    metrics = {
        "total_pass": sum(1 for r in results if r.passed),
        "total_tests": len(results),
        "by_category": {
            cat: {
                "pass": sum(1 for r in rs if r.passed),
                "total": len(rs),
            }
            for cat, rs in by_cat.items()
        },
    }
    return results, metrics


def benchmark_model(
    model_name: str,
    model_cfg: dict,
    backend: str = "st",
) -> tuple[list[TestResult], dict, float]:
    """Benchmark a single model. Returns (results, metrics, embed_latency_ms)."""
    print(f"\n  Loading {model_cfg['hf_id']}...")
    t0 = time.perf_counter()

    if backend == "mlx":
        model_obj = _load_mlx_model(model_cfg["hf_id"])
        embedder = lambda texts, is_query, model_cfg: _embed_mlx(
            model_obj, texts, is_query, model_cfg
        )
    else:
        model_obj = _load_st_model(model_cfg["hf_id"])
        embedder = lambda texts, is_query, model_cfg: _embed_st(
            model_obj, texts, is_query, model_cfg
        )

    load_time = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {load_time:.0f}ms")

    # Measure embedding latency (10 texts)
    sample_texts = [c[0] for c in CORPUS[:10]]
    t0 = time.perf_counter()
    for _ in range(3):
        embedder(sample_texts, is_query=False, model_cfg=model_cfg)
    embed_latency = (time.perf_counter() - t0) / 3 * 1000  # ms for 10 texts

    # Run quality tests
    test_results, metrics = run_tests(embedder, embedder, model_cfg)

    return test_results, metrics, embed_latency


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_report(
    all_results: dict[str, tuple[list[TestResult], dict, float]],
) -> None:
    """Print comparison report."""
    model_names = list(all_results.keys())

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Model':<20} {'Pass':>6} {'Ident':>7} {'Topic':>7} {'Semant':>7} {'Noise':>7} {'Lat(ms)':>8}"
    print(header)
    print("-" * len(header))

    for name in model_names:
        results, metrics, latency = all_results[name]
        cats = metrics["by_category"]
        total = f"{metrics['total_pass']}/{metrics['total_tests']}"
        ident = f"{cats.get('identifier', {}).get('pass', 0)}/{cats.get('identifier', {}).get('total', 0)}"
        topic = f"{cats.get('topic', {}).get('pass', 0)}/{cats.get('topic', {}).get('total', 0)}"
        sem = f"{cats.get('semantic', {}).get('pass', 0)}/{cats.get('semantic', {}).get('total', 0)}"
        noise = f"{cats.get('noise', {}).get('pass', 0)}/{cats.get('noise', {}).get('total', 0)}"
        print(f"{name:<20} {total:>6} {ident:>7} {topic:>7} {sem:>7} {noise:>7} {latency:>7.0f}")

    # --- Per-test detail ---
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for tc in TEST_CASES:
        print(f"\n  [{tc.category}] {tc.name}: query=\"{tc.query}\"")
        for name in model_names:
            results, _, _ = all_results[name]
            r = next(r for r in results if r.name == tc.name)
            status = "PASS" if r.passed else "FAIL"
            line = f"    {name:<20} {status}  sim={r.top1_sim:.4f}  top1={r.top1_content[:55]}"
            if r.details:
                line += f"  [{r.details}]"
            print(line)

    # --- Similarity distribution for identifier tests ---
    print("\n" + "=" * 80)
    print("IDENTIFIER SIMILARITY GAPS (top-1 sim vs top-2 sim)")
    print("=" * 80)

    for name in model_names:
        results, _, _ = all_results[name]
        gaps = []
        for r in results:
            if r.category == "identifier" and r.passed:
                gaps.append(r.top1_sim)
        if gaps:
            avg_sim = sum(gaps) / len(gaps)
            print(f"  {name:<20} avg top-1 sim: {avg_sim:.4f}")



# ---------------------------------------------------------------------------
# Reranker benchmark
# ---------------------------------------------------------------------------

RERANKER_MODELS_BENCH = {
    "tinybert": "tinybert",
    "minilm6": "minilm6",
}

# Extended corpus with importance traps for reranker evaluation.
# Each cluster has a correct answer (low importance) and a high-importance
# distractor that inflates composite scores but is the wrong answer.
RERANK_CORPUS = CORPUS + [
    # Importance traps — topically adjacent but wrong answers
    (
        "CRITICAL: Terraform plan must always run before apply in CI to catch drift",
        ["terraform", "CI", "CRITICAL"],
    ),
    (
        "IMPORTANT: Kubernetes resource quotas must be set per namespace to prevent noisy-neighbor evictions",
        ["kubernetes", "quotas", "IMPORTANT"],
    ),
    (
        "CRITICAL: Docker BuildKit cache mount requires explicit --mount=type=cache or builds are 3x slower",
        ["Docker", "BuildKit", "CRITICAL"],
    ),
    (
        "IMPORTANT: Always set DNS TTL to 60 seconds during migrations to allow fast rollback",
        ["DNS", "TTL", "IMPORTANT"],
    ),
    (
        "CRITICAL: PostgreSQL vacuum must run weekly on tables with heavy write load to prevent bloat",
        ["PostgreSQL", "vacuum", "CRITICAL"],
    ),
    (
        "IMPORTANT: asyncio event loop must not be shared across threads in Python web servers",
        ["asyncio", "threads", "IMPORTANT"],
    ),
]

# Importance scores: low for correct answers, high for traps.
# Maps corpus text substring → importance weight for composite scoring simulation.
_IMPORTANCE_MAP = {
    "CRITICAL:": 0.9,
    "IMPORTANT:": 0.9,
}
_DEFAULT_IMPORTANCE = 0.5

RERANK_TEST_CASES = [
    TestCase(
        name="tf state (vs plan CRITICAL)",
        query="terraform state locking mechanism",
        expected_top="Terraform S3 backend state locking",
        category="rerank",
    ),
    TestCase(
        name="k8s crash (vs quotas IMPORTANT)",
        query="why are kubernetes pods restarting and getting killed",
        expected_top="crash loop backoff",
        category="rerank",
    ),
    TestCase(
        name="DNS CNAME (vs TTL IMPORTANT)",
        query="DNS records after deploying to kubernetes",
        expected_top="DNS records are managed",
        category="rerank",
    ),
    TestCase(
        name="BuildKit compat (vs cache CRITICAL)",
        query="Docker BuildKit cross compilation incompatibility",
        expected_top="incompatible with",
        category="rerank",
    ),
    TestCase(
        name="autoscaler (paraphrase)",
        query="kubernetes nodes disappearing on weekends",
        expected_top="autoscaler",
        category="rerank",
    ),
    TestCase(
        name="asyncio (vs threads IMPORTANT)",
        query="python async await concurrent tasks gather",
        expected_top="asyncio.gather",
        category="rerank",
    ),
    TestCase(
        name="TICKET-24 identifier",
        query="TICKET-24 RDS upgrade",
        expected_top="TICKET-24",
        category="rerank",
    ),
    TestCase(
        name="5G interference",
        query="cellular band interference signal quality",
        expected_top="interference",
        category="rerank",
    ),
]


@dataclass
class RerankResult:
    name: str
    rank_base: int | None
    rank_reranked: int | None
    base_sim: float
    reranker_score: float
    delta: str  # "improved", "unchanged", "regressed"


def _find_rank(ranked_texts: list[str], expected_substr: str) -> int | None:
    for i, text in enumerate(ranked_texts):
        if expected_substr in text:
            return i + 1
    return None


def _get_importance(text: str) -> float:
    """Get simulated importance for a corpus entry."""
    for prefix, imp in _IMPORTANCE_MAP.items():
        if prefix in text:
            return imp
    return _DEFAULT_IMPORTANCE


def benchmark_rerankers(
    bi_encoder_embedder,
    model_cfg: dict,
    reranker_names: list[str],
) -> dict[str, tuple[list[RerankResult], dict, float]]:
    """Benchmark reranker models on top of a bi-encoder baseline.

    Simulates composite scoring (0.6*sim + 0.2*importance + 0.2*recency)
    to match production behavior where importance traps can mislead ranking.
    """
    from memory.reranker import RerankerModel

    corpus_texts = [c[0] for c in RERANK_CORPUS]
    corpus_embs = bi_encoder_embedder(corpus_texts, is_query=False, model_cfg=model_cfg)
    importance_scores = np.array([_get_importance(t) for t in corpus_texts], dtype=np.float32)

    all_rerank_results = {}

    for reranker_name in reranker_names:
        print(f"\n  Loading reranker: {reranker_name}...")
        reranker = RerankerModel(reranker_name)

        # Warm up
        reranker.score_pairs("warmup", ["warmup text"])

        # Measure latency (score full corpus)
        t0 = time.perf_counter()
        n_runs = 5
        for _ in range(n_runs):
            reranker.score_pairs("benchmark query", corpus_texts)
        latency_ms = (time.perf_counter() - t0) / n_runs * 1000

        results = []
        for tc in RERANK_TEST_CASES:
            if tc.expected_top is None:
                continue

            # Bi-encoder composite ranking (simulates production scoring)
            query_emb = bi_encoder_embedder([tc.query], is_query=True, model_cfg=model_cfg)
            sims = cosine_similarity(query_emb, corpus_embs)[0]
            # Composite: 0.6*similarity + 0.2*importance + 0.2*recency (recency=0.5 for all)
            composite = 0.6 * sims + 0.2 * importance_scores + 0.2 * 0.5
            base_ranked_indices = np.argsort(composite)[::-1]
            base_ranked_texts = [corpus_texts[i] for i in base_ranked_indices]
            rank_base = _find_rank(base_ranked_texts, tc.expected_top)

            # Reranked: blend normalized composite with reranker score
            reranker_scores = reranker.score_pairs(tc.query, corpus_texts)
            max_comp = float(np.max(composite)) or 1.0
            blended = 0.6 * (composite / max_comp) + 0.4 * reranker_scores
            reranked_indices = np.argsort(blended)[::-1]
            reranked_texts = [corpus_texts[i] for i in reranked_indices]
            rank_reranked = _find_rank(reranked_texts, tc.expected_top)

            # Find reranker score for the expected result
            rs = 0.0
            base_sim = 0.0
            for i, text in enumerate(corpus_texts):
                if tc.expected_top in text:
                    rs = float(reranker_scores[i])
                    base_sim = float(composite[i])
                    break

            if rank_reranked is not None and rank_base is not None:
                if rank_reranked < rank_base:
                    delta = "improved"
                elif rank_reranked > rank_base:
                    delta = "regressed"
                else:
                    delta = "unchanged"
            elif rank_base is None and rank_reranked is not None:
                delta = "improved"
            elif rank_base is not None and rank_reranked is None:
                delta = "regressed"
            else:
                delta = "unchanged"

            results.append(RerankResult(
                name=tc.name,
                rank_base=rank_base,
                rank_reranked=rank_reranked,
                base_sim=base_sim,
                reranker_score=rs,
                delta=delta,
            ))

        # Metrics
        mrr_base = 0.0
        mrr_reranked = 0.0
        n = len(results)
        for r in results:
            if r.rank_base:
                mrr_base += 1.0 / r.rank_base
            if r.rank_reranked:
                mrr_reranked += 1.0 / r.rank_reranked
        mrr_base /= n
        mrr_reranked /= n

        metrics = {
            "mrr_base": mrr_base,
            "mrr_reranked": mrr_reranked,
            "mrr_delta": mrr_reranked - mrr_base,
            "improved": sum(1 for r in results if r.delta == "improved"),
            "unchanged": sum(1 for r in results if r.delta == "unchanged"),
            "regressed": sum(1 for r in results if r.delta == "regressed"),
            "total": n,
        }

        all_rerank_results[reranker_name] = (results, metrics, latency_ms)

    return all_rerank_results


def print_rerank_report(
    rerank_results: dict[str, tuple[list[RerankResult], dict, float]],
) -> None:
    """Print reranker comparison report."""
    print("\n" + "=" * 80)
    print("RERANKER COMPARISON")
    print("=" * 80)

    header = f"{'Reranker':<12} {'MRR base':>9} {'MRR rerank':>11} {'Delta':>7} {'Improved':>9} {'Regressed':>10} {'Lat(ms)':>8}"
    print(header)
    print("-" * len(header))

    for name, (results, metrics, latency) in rerank_results.items():
        print(
            f"{name:<12} {metrics['mrr_base']:>9.3f} {metrics['mrr_reranked']:>11.3f} "
            f"{metrics['mrr_delta']:>+7.3f} {metrics['improved']:>5}/{metrics['total']}"
            f"   {metrics['regressed']:>5}/{metrics['total']}  {latency:>7.1f}"
        )

    # Per-query details
    print("\n" + "-" * 80)
    print("PER-QUERY DETAIL")
    print("-" * 80)

    reranker_names = list(rerank_results.keys())
    for i, tc in enumerate(RERANK_TEST_CASES):
        if tc.expected_top is None:
            continue
        print(f"\n  [{tc.name}] query=\"{tc.query}\"")
        for rname in reranker_names:
            results, _, _ = rerank_results[rname]
            r = results[i]
            arrow = {"improved": "UP", "regressed": "DOWN", "unchanged": "=="}[r.delta]
            print(
                f"    {rname:<12} rank: {r.rank_base} -> {r.rank_reranked}  "
                f"[{arrow}]  sim={r.base_sim:.4f}  rs={r.reranker_score:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models and rerankers")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test (default: all BERT-based). Use 'all' to include Qwen3.",
    )
    parser.add_argument(
        "--backend",
        choices=["st", "mlx"],
        default="st",
        help="Backend: st (sentence-transformers, default) or mlx (Apple Silicon GPU)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Also benchmark cross-encoder rerankers",
    )
    parser.add_argument(
        "--rerank-only",
        action="store_true",
        default=False,
        help="Only benchmark rerankers (uses e5-small-v2 as bi-encoder)",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=None,
        help="Reranker models to test (default: all). Options: tinybert, minilm6",
    )
    args = parser.parse_args()

    reranker_names = args.rerankers or list(RERANKER_MODELS_BENCH.keys())
    for name in reranker_names:
        if name not in RERANKER_MODELS_BENCH:
            print(f"Unknown reranker: {name}. Available: {', '.join(RERANKER_MODELS_BENCH.keys())}")
            return

    if args.rerank_only:
        # Only reranker benchmark — use our ONNX e5-small-v2 embedder (no extra deps)
        from memory.embeddings import get_model as get_embedding_model

        print("=" * 80)
        print("memory RERANKER benchmark")
        print("=" * 80)
        print("Bi-encoder: e5-small-v2 (ONNX, built-in)")
        print(f"Rerankers: {', '.join(reranker_names)}")
        print(f"Corpus: {len(RERANK_CORPUS)} memories ({len(RERANK_CORPUS) - len(CORPUS)} importance traps)")
        print(f"Tests: {len(RERANK_TEST_CASES)}")

        emb_model = get_embedding_model()
        # Wrap in the same interface as sentence-transformers embedder
        cfg = MODELS["e5-small-v2"]

        def onnx_embedder(texts, is_query, model_cfg):
            prefix = model_cfg["prefix_query"] if is_query else model_cfg["prefix_doc"]
            prefixed = [prefix + t for t in texts]
            return emb_model.embed_batch(prefixed)

        rerank_results = benchmark_rerankers(onnx_embedder, cfg, reranker_names)
        print_rerank_report(rerank_results)
        return

    if args.models and "all" in args.models:
        model_names = list(MODELS.keys())
    elif args.models:
        model_names = args.models
    else:
        model_names = DEFAULT_MODELS

    # Validate model names
    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")
            return

    print("=" * 80)
    print("memory embedding model benchmark")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Backend: {args.backend}")
    print(f"Corpus: {len(CORPUS)} memories")
    print(f"Tests: {len(TEST_CASES)} ({len([t for t in TEST_CASES if t.category == 'identifier'])} identifier, "
          f"{len([t for t in TEST_CASES if t.category == 'topic'])} topic, "
          f"{len([t for t in TEST_CASES if t.category == 'semantic'])} semantic, "
          f"{len([t for t in TEST_CASES if t.category == 'noise'])} noise)")

    all_results = {}
    for name in model_names:
        cfg = MODELS[name]
        print(f"\n--- {name} ({cfg['hf_id']}) ---")
        try:
            test_results, metrics, latency = benchmark_model(name, cfg, args.backend)
            all_results[name] = (test_results, metrics, latency)
            print(f"  Score: {metrics['total_pass']}/{metrics['total_tests']}  Latency: {latency:.0f}ms/10texts")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_report(all_results)

    # Reranker benchmarks
    if args.rerank and all_results:
        print("\n\n")
        print("=" * 80)
        print("RERANKER BENCHMARK")
        print("=" * 80)

        # Use the first model as bi-encoder baseline
        bi_encoder_name = model_names[0]
        bi_encoder_cfg = MODELS[bi_encoder_name]
        print(f"Bi-encoder baseline: {bi_encoder_name}")

        if args.backend == "mlx":
            model_obj = _load_mlx_model(bi_encoder_cfg["hf_id"])
            embedder = lambda texts, is_query, model_cfg: _embed_mlx(
                model_obj, texts, is_query, model_cfg
            )
        else:
            model_obj = _load_st_model(bi_encoder_cfg["hf_id"])
            embedder = lambda texts, is_query, model_cfg: _embed_st(
                model_obj, texts, is_query, model_cfg
            )

        rerank_results = benchmark_rerankers(embedder, bi_encoder_cfg, reranker_names)
        print_rerank_report(rerank_results)


if __name__ == "__main__":
    main()
