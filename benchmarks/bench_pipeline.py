"""Benchmark: search pipeline comparison (model × search-mode × reranker).

Answers: "Can a better embedding model alone replace the hybrid pipeline?"

Compares semantic-only, hybrid (semantic+FTS), and hybrid+rerank modes across
multiple embedding models. Measures MRR, top-1 accuracy, per-category breakdown,
and embedding latency.

Usage:
    # Install benchmark deps
    pip install sentence-transformers

    # Validate all models load and embed correctly on this hardware
    python benchmarks/bench_pipeline.py --validate

    # Quick run: 2 models × 3 modes
    python benchmarks/bench_pipeline.py --models e5-small-v2 bge-small

    # Full comparison
    python benchmarks/bench_pipeline.py --models all

    # 384-dim only (no 768-dim models)
    python benchmarks/bench_pipeline.py --models 384

    # Include reranker
    python benchmarks/bench_pipeline.py --models e5-small-v2 bge-base --rerank
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from corpus import ALL_TEST_CASES, CORPUS, TestCase

# ---------------------------------------------------------------------------
# Model registry (mirrors bench_models.py but adds helper metadata)
# ---------------------------------------------------------------------------

MODELS = {
    # --- 384-dim models ---
    "e5-small-v2": {
        "hf_id": "intfloat/e5-small-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "dims": 384,
        "pooling": "mean",
    },
    "bge-small": {
        "hf_id": "BAAI/bge-small-en-v1.5",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "cls",
    },
    "gte-small": {
        "hf_id": "thenlper/gte-small",
        "prefix_query": "",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "mean",
    },
    "mdbr-leaf-mt": {
        "hf_id": "MongoDB/mdbr-leaf-mt",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 1024,
        "truncate_dim": 384,
        "pooling": "mean",
    },
    "arctic-embed-xs": {
        "hf_id": "Snowflake/snowflake-arctic-embed-xs",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "cls",
    },
    # --- 768-dim models (native) ---
    "e5-base-v2": {
        "hf_id": "intfloat/e5-base-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "dims": 768,
        "pooling": "mean",
    },
    "bge-base": {
        "hf_id": "BAAI/bge-base-en-v1.5",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 768,
        "pooling": "cls",
    },
    "nomic-embed": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 768,
        "pooling": "mean",
    },
    # --- 768-dim models (MRL truncated to 384) ---
    "e5-base-v2-384": {
        "hf_id": "intfloat/e5-base-v2",
        "prefix_query": "query: ",
        "prefix_doc": "passage: ",
        "dims": 768,
        "truncate_dim": 384,
        "pooling": "mean",
    },
    "bge-base-384": {
        "hf_id": "BAAI/bge-base-en-v1.5",
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_doc": "",
        "dims": 768,
        "truncate_dim": 384,
        "pooling": "cls",
    },
    "nomic-embed-384": {
        "hf_id": "nomic-ai/nomic-embed-text-v1.5",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 768,
        "truncate_dim": 384,
        "pooling": "mean",
    },
    # --- Additional models from bench_models.py ---
    "mdbr-leaf-ir": {
        "hf_id": "MongoDB/mdbr-leaf-ir",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 1024,
        "truncate_dim": 384,
        "pooling": "mean",
    },
    "qwen3-0.6B": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "prefix_query": "Instruct: Retrieve relevant memories for this query\nQuery: ",
        "prefix_doc": "",
        "dims": 1024,
        "truncate_dim": 384,
        "pooling": "last_token",
    },
    # --- ModernBERT / new 2026 models ---
    "granite-small-r2": {
        "hf_id": "ibm-granite/granite-embedding-small-english-r2",
        "prefix_query": "",
        "prefix_doc": "",
        "dims": 384,
        "pooling": "mean",
    },
    "modernbert-embed-base": {
        "hf_id": "nomic-ai/modernbert-embed-base",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 768,
        "pooling": "mean",
    },
    "modernbert-embed-base-384": {
        "hf_id": "nomic-ai/modernbert-embed-base",
        "prefix_query": "search_query: ",
        "prefix_doc": "search_document: ",
        "dims": 768,
        "truncate_dim": 384,
        "pooling": "mean",
    },
    # --- EmbeddingGemma (Google, 300M, ONNX-native) ---
    "embeddinggemma": {
        "hf_id": "onnx-community/embeddinggemma-300m-ONNX",
        "hf_id_pytorch": "google/embeddinggemma-300m",
        "prefix_query": "",
        "prefix_doc": "",
        "dims": 768,
        "pooling": "mean",
    },
    "embeddinggemma-384": {
        "hf_id": "onnx-community/embeddinggemma-300m-ONNX",
        "hf_id_pytorch": "google/embeddinggemma-300m",
        "prefix_query": "",
        "prefix_doc": "",
        "dims": 768,
        "truncate_dim": 384,
        "pooling": "mean",
    },
}

MODELS_384 = [k for k, v in MODELS.items() if v.get("truncate_dim", v["dims"]) == 384]
MODELS_768 = [k for k, v in MODELS.items() if v.get("truncate_dim", v["dims"]) == 768]
DEFAULT_MODELS = ["e5-small-v2", "bge-small", "gte-small", "e5-base-v2-384", "bge-base-384", "embeddinggemma-384"]

# Scoring weights matching production (core.py DEFAULT_SCORING_WEIGHTS)
W_SIM, W_IMP, W_REC = 0.6, 0.2, 0.2
DEFAULT_RECENCY = 0.5  # all corpus entries are equally "recent" for fairness


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

# Global backend setting (set from CLI args)
_BACKEND = "onnx"


def _load_model(hf_id: str, cfg: dict) -> tuple:
    """Load model using the configured backend. Returns (model, actual_backend)."""
    from sentence_transformers import SentenceTransformer

    if _BACKEND == "onnx":
        try:
            model_kwargs = {"provider": "CPUExecutionProvider"}
            model = SentenceTransformer(hf_id, backend="onnx", model_kwargs=model_kwargs, trust_remote_code=True)
            return model, "onnx"
        except Exception as e:
            print(f"\n    ONNX failed ({e}), falling back to pytorch...", end=" ", flush=True)
            model = SentenceTransformer(hf_id, trust_remote_code=True)
            return model, "pytorch"
    else:
        return SentenceTransformer(hf_id, trust_remote_code=True), "pytorch"


def _embed(model, texts: list[str], is_query: bool, cfg: dict) -> np.ndarray:
    prefix = cfg["prefix_query"] if is_query else cfg["prefix_doc"]
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    truncate_dim = cfg.get("truncate_dim")
    if truncate_dim:
        embeddings = embeddings[:, :truncate_dim]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings


# ---------------------------------------------------------------------------
# FTS5 simulation (temp SQLite for keyword search)
# ---------------------------------------------------------------------------


def _build_fts_index(texts: list[str]) -> tuple[sqlite3.Connection, dict[int, int]]:
    """Build a temp FTS5 index. Returns (conn, {rowid: corpus_index})."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE fts USING fts5(content, tokenize='porter ascii')")
    idx_map = {}
    for i, text in enumerate(texts):
        conn.execute("INSERT INTO fts(rowid, content) VALUES (?, ?)", (i + 1, text))
        idx_map[i + 1] = i
    return conn, idx_map


def _fts_search(conn: sqlite3.Connection, query: str, idx_map: dict[int, int]) -> dict[int, float]:
    """BM25 search, returns {corpus_index: normalized_similarity}."""
    # Import sanitizer from our codebase
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from memory.core import _sanitize_fts_query

    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return {}

    rows = conn.execute(
        "SELECT rowid, rank FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 50",
        (safe_query,),
    ).fetchall()

    if not rows:
        return {}

    ranks = {r[0]: r[1] for r in rows}
    rank_values = list(ranks.values())
    min_rank = min(rank_values)
    max_rank = max(rank_values)
    rank_range = max_rank - min_rank

    result = {}
    for rowid, rank in ranks.items():
        sim = 1.0 if rank_range == 0 else (max_rank - rank) / rank_range
        result[idx_map[rowid]] = sim
    return result


# ---------------------------------------------------------------------------
# Search modes
# ---------------------------------------------------------------------------


def _rank_semantic(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    importance_scores: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank by composite: w_sim*similarity + w_imp*importance + w_rec*recency."""
    sims = (query_emb @ corpus_embs.T)[0]
    scores = W_SIM * sims + W_IMP * importance_scores + W_REC * DEFAULT_RECENCY
    indices = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in indices]


def _rank_hybrid(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    importance_scores: np.ndarray,
    fts_conn: sqlite3.Connection,
    fts_idx_map: dict[int, int],
    query: str,
) -> list[tuple[int, float]]:
    """Hybrid: merge semantic + FTS scores (mirrors core.py _search_hybrid)."""
    # Semantic scores
    sims = (query_emb @ corpus_embs.T)[0]
    sem_scores = W_SIM * sims + W_IMP * importance_scores + W_REC * DEFAULT_RECENCY

    # FTS scores
    fts_sims = _fts_search(fts_conn, query, fts_idx_map)
    fts_scores = np.zeros(len(sims))
    for idx, sim in fts_sims.items():
        fts_scores[idx] = W_SIM * sim + W_IMP * importance_scores[idx] + W_REC * DEFAULT_RECENCY

    # Merge: for entries found by both, add scores (matches _search_hybrid behavior)
    merged = sem_scores.copy()
    for idx in fts_sims:
        merged[idx] = sem_scores[idx] + fts_scores[idx]

    indices = np.argsort(merged)[::-1]
    return [(int(i), float(merged[i])) for i in indices]


def _rank_hybrid_rerank(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    importance_scores: np.ndarray,
    fts_conn: sqlite3.Connection,
    fts_idx_map: dict[int, int],
    query: str,
    corpus_texts: list[str],
    reranker,
    rerank_weight: float = 0.4,
) -> list[tuple[int, float]]:
    """Hybrid + cross-encoder reranking (mirrors core.py _apply_rerank)."""
    # Get hybrid ranking first (overfetch)
    hybrid_ranked = _rank_hybrid(query_emb, corpus_embs, importance_scores, fts_conn, fts_idx_map, query)

    # Take top candidates for reranking
    top_indices = [idx for idx, _ in hybrid_ranked[:30]]
    top_texts = [corpus_texts[i] for i in top_indices]
    top_composites = np.array([score for _, score in hybrid_ranked[:30]])

    # Reranker scores
    reranker_scores = reranker.score_pairs(query, top_texts)

    # Blend: (1 - w) * normalized_composite + w * reranker_score
    max_comp = float(np.max(top_composites)) or 1.0
    blended = (1 - rerank_weight) * (top_composites / max_comp) + rerank_weight * reranker_scores

    sorted_indices = np.argsort(blended)[::-1]
    return [(top_indices[int(i)], float(blended[int(i)])) for i in sorted_indices]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    rank: int | None  # 1-based rank of expected result, None if not found
    top1_content: str
    top1_score: float


def _find_rank(ranked: list[tuple[int, float]], corpus_texts: list[str], substr: str) -> int | None:
    for i, (idx, _) in enumerate(ranked):
        if substr in corpus_texts[idx]:
            return i + 1
    return None


def _run_test(
    tc: TestCase,
    ranked: list[tuple[int, float]],
    corpus_texts: list[str],
) -> TestResult:
    """Evaluate a single test case against ranked results."""
    top1_idx, top1_score = ranked[0] if ranked else (-1, 0.0)
    top1_content = corpus_texts[top1_idx][:70] if top1_idx >= 0 else ""

    if tc.expected_top is None:
        # Noise / boolean — just record what we got
        return TestResult(tc.name, tc.category, True, None, top1_content, top1_score)

    rank = _find_rank(ranked, corpus_texts, tc.expected_top)
    passed = rank == 1

    # Also check should_not_outrank
    if passed and tc.should_not_outrank:
        for bad in tc.should_not_outrank:
            bad_rank = _find_rank(ranked, corpus_texts, bad)
            if bad_rank is not None and bad_rank < (rank or 999):
                passed = False
                break

    return TestResult(tc.name, tc.category, passed, rank, top1_content, top1_score)


@dataclass
class LatencyInfo:
    load_ms: float
    embed_corpus_ms: float
    query_5_ms: float  # latency to embed 5 queries
    search_per_query_ms: dict[str, float]  # mode → avg ms per search


def run_pipeline(
    model_name: str,
    model_cfg: dict,
    modes: list[str],
    reranker=None,
) -> tuple[dict[str, list[TestResult]], LatencyInfo]:
    """Run all test cases across specified modes for one model.

    Returns (results_by_mode, latency_info).
    """
    # Some models have separate ONNX HF repos (e.g., embeddinggemma)
    hf_id = model_cfg["hf_id"]
    if _BACKEND == "pytorch" and "hf_id_pytorch" in model_cfg:
        hf_id = model_cfg["hf_id_pytorch"]
    print(f"\n  Loading {hf_id} (backend={_BACKEND})...")
    t0 = time.perf_counter()
    model, actual_backend = _load_model(hf_id, model_cfg)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {load_ms:.0f}ms ({actual_backend})")

    corpus_texts = [e.content for e in CORPUS]
    importance_scores = np.array([e.importance for e in CORPUS], dtype=np.float32)

    # Embed corpus
    t0 = time.perf_counter()
    corpus_embs = _embed(model, corpus_texts, is_query=False, cfg=model_cfg)
    embed_corpus_ms = (time.perf_counter() - t0) * 1000
    print(f"  Embedded {len(corpus_texts)} docs in {embed_corpus_ms:.0f}ms")

    # Build FTS index
    fts_conn, fts_idx_map = _build_fts_index(corpus_texts)

    # Measure query embedding latency (warm, 3 runs of 5 queries)
    sample_queries = [tc.query for tc in ALL_TEST_CASES[:5]]
    t0 = time.perf_counter()
    for _ in range(3):
        _embed(model, sample_queries, is_query=True, cfg=model_cfg)
    query_5_ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"  Query latency: {query_5_ms:.0f}ms / 5 queries")

    results_by_mode: dict[str, list[TestResult]] = {}
    search_latency: dict[str, float] = {}

    for mode in modes:
        results = []
        mode_t0 = time.perf_counter()
        for tc in ALL_TEST_CASES:
            query_emb = _embed(model, [tc.query], is_query=True, cfg=model_cfg)

            if mode == "semantic":
                ranked = _rank_semantic(query_emb, corpus_embs, importance_scores)
            elif mode == "hybrid":
                ranked = _rank_hybrid(
                    query_emb,
                    corpus_embs,
                    importance_scores,
                    fts_conn,
                    fts_idx_map,
                    tc.query,
                )
            elif mode == "hybrid+rerank":
                if reranker is None:
                    raise ValueError("Reranker required for hybrid+rerank mode")
                ranked = _rank_hybrid_rerank(
                    query_emb,
                    corpus_embs,
                    importance_scores,
                    fts_conn,
                    fts_idx_map,
                    tc.query,
                    corpus_texts,
                    reranker,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            results.append(_run_test(tc, ranked, corpus_texts))
        mode_elapsed = (time.perf_counter() - mode_t0) * 1000
        search_latency[mode] = mode_elapsed / len(ALL_TEST_CASES)
        results_by_mode[mode] = results

    fts_conn.close()
    latency = LatencyInfo(
        load_ms=load_ms,
        embed_corpus_ms=embed_corpus_ms,
        query_5_ms=query_5_ms,
        search_per_query_ms=search_latency,
    )
    return results_by_mode, latency


# ---------------------------------------------------------------------------
# Validation: check all models load and embed correctly
# ---------------------------------------------------------------------------


def validate_models(model_names: list[str]) -> dict[str, str]:
    """Validate each model loads and produces correct-shape embeddings on M4 Max."""
    print("=" * 70)
    print("MODEL VALIDATION (load + embed + cosine sanity)")
    print("=" * 70)

    sample_texts = [
        "Kubernetes pod crash loop backoff",
        "container orchestration platform scaling issues",
        "quantum computing qubits entanglement",
    ]

    results = {}
    for name in model_names:
        cfg = MODELS[name]
        effective_dims = cfg.get("truncate_dim", cfg["dims"])
        hf_id = cfg["hf_id"]
        if _BACKEND == "pytorch" and "hf_id_pytorch" in cfg:
            hf_id = cfg["hf_id_pytorch"]
        print(f"\n  {name} ({hf_id}, {effective_dims}d, {_BACKEND})...", end=" ", flush=True)

        try:
            t0 = time.perf_counter()
            model, actual_backend = _load_model(hf_id, cfg)
            load_ms = (time.perf_counter() - t0) * 1000

            # Embed and check shape
            embs = _embed(model, sample_texts, is_query=True, cfg=cfg)
            assert embs.shape == (3, effective_dims), f"Shape mismatch: {embs.shape} != (3, {effective_dims})"

            # Check L2 normalization
            norms = np.linalg.norm(embs, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-3), f"Not normalized: norms={norms}"

            # Cosine sanity: k8s query should be closer to k8s doc than quantum
            sim_k8s = float(embs[0] @ embs[1])
            sim_quantum = float(embs[0] @ embs[2])
            sane = sim_k8s > sim_quantum

            # Latency: 10 embeddings
            t0 = time.perf_counter()
            for _ in range(3):
                _embed(model, sample_texts, is_query=False, cfg=cfg)
            latency_ms = (time.perf_counter() - t0) / 3 * 1000

            status = "OK" if sane else "WARN (cosine sanity failed)"
            print(
                f"{status}  [{actual_backend}]  load={load_ms:.0f}ms  lat={latency_ms:.0f}ms/3  "
                f"sim_k8s={sim_k8s:.3f}  sim_unrel={sim_quantum:.3f}"
            )
            results[name] = f"{status} [{actual_backend}]"

        except Exception as e:
            print(f"FAIL: {e}")
            results[name] = f"FAIL: {e}"

    # Summary
    print("\n" + "-" * 70)
    ok = sum(1 for v in results.values() if v.startswith("OK"))
    warn = sum(1 for v in results.values() if v.startswith("WARN"))
    fail = sum(1 for v in results.values() if v.startswith("FAIL"))
    print(f"  {ok} OK, {warn} WARN, {fail} FAIL out of {len(results)} models")

    if fail:
        print("\n  Failed models (exclude from benchmarks):")
        for name, status in results.items():
            if status.startswith("FAIL"):
                print(f"    {name}: {status}")

    return results


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------


def _compute_metrics(results: list[TestResult]) -> dict:
    by_cat: dict[str, list[TestResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    mrr = 0.0
    top1 = 0
    n_ranked = 0  # only count tests with expected_top

    for r in results:
        if r.rank is not None:
            mrr += 1.0 / r.rank
            n_ranked += 1
            if r.rank == 1:
                top1 += 1

    return {
        "mrr": mrr / n_ranked if n_ranked else 0.0,
        "top1": top1,
        "top1_pct": top1 / n_ranked * 100 if n_ranked else 0.0,
        "n_ranked": n_ranked,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "by_category": {
            cat: {
                "passed": sum(1 for r in rs if r.passed),
                "total": len(rs),
                "mrr": (
                    sum(1.0 / r.rank for r in rs if r.rank is not None) / sum(1 for r in rs if r.rank is not None)
                    if any(r.rank is not None for r in rs)
                    else 0.0
                ),
            }
            for cat, rs in by_cat.items()
        },
    }


def print_report(
    all_results: dict[str, dict[str, list[TestResult]]],
    all_latencies: dict[str, LatencyInfo] | None = None,
) -> None:
    """Print the model × mode comparison report."""
    model_names = list(all_results.keys())
    modes = list(next(iter(all_results.values())).keys())

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("MODEL × MODE COMPARISON")
    print("=" * 90)

    # Header
    mode_headers = "  ".join(f"{m:>18}" for m in modes)
    print(f"{'Model':<22} {mode_headers}")
    print(f"{'':22} " + "  ".join(f"{'MRR  Top1':>18}" for _ in modes))
    print("-" * (22 + 20 * len(modes)))

    for model_name in model_names:
        row = f"{model_name:<22}"
        for mode in modes:
            metrics = _compute_metrics(all_results[model_name][mode])
            row += f"  {metrics['mrr']:.3f}  {metrics['top1_pct']:4.0f}%  "
        print(row)

    # --- Latency table ---
    if all_latencies:
        print("\n" + "=" * 90)
        print("LATENCY (ms)")
        print("=" * 90)

        lat_modes = list(next(iter(all_latencies.values())).search_per_query_ms.keys())
        search_headers = "  ".join(f"{m:>10}" for m in lat_modes)
        print(
            f"{'Model':<22} {'Load':>7} {'Corpus':>7} {'5-Query':>8}  "
            f"{'--- per-query search ---':>{10 * len(lat_modes)}}"
        )
        print(f"{'':22} {'':>7} {'embed':>7} {'embed':>8}  {search_headers}")
        print("-" * (50 + 12 * len(lat_modes)))

        for model_name in model_names:
            lat = all_latencies[model_name]
            row = f"{model_name:<22} {lat.load_ms:>6.0f}  {lat.embed_corpus_ms:>6.0f}  {lat.query_5_ms:>7.0f}  "
            for m in lat_modes:
                row += f"  {lat.search_per_query_ms.get(m, 0):>9.1f}"
            print(row)

    # --- Per-category breakdown ---
    categories = ["identifier", "topic", "semantic", "importance_trap", "boolean", "mixed", "noise"]
    print("\n" + "=" * 90)
    print("PER-CATEGORY MRR")
    print("=" * 90)

    cat_headers = "  ".join(f"{c[:8]:>8}" for c in categories)
    print(f"{'Model + Mode':<30} {cat_headers}")
    print("-" * (30 + 10 * len(categories)))

    for model_name in model_names:
        for mode in modes:
            label = f"{model_name} / {mode}"
            if len(label) > 28:
                label = label[:28]
            metrics = _compute_metrics(all_results[model_name][mode])
            row = f"{label:<30}"
            for cat in categories:
                cat_m = metrics["by_category"].get(cat, {})
                mrr = cat_m.get("mrr", 0.0)
                row += f"  {mrr:>7.3f} "
            print(row)

    # --- Detailed per-test results ---
    print("\n" + "=" * 90)
    print("DETAILED RESULTS (per test case)")
    print("=" * 90)

    for tc in ALL_TEST_CASES:
        print(f'\n  [{tc.category}] {tc.name}: query="{tc.query}"')
        if tc.expected_top:
            print(f"  Expected: ...{tc.expected_top}...")
        for model_name in model_names:
            for mode in modes:
                results = all_results[model_name][mode]
                r = next(r for r in results if r.name == tc.name)
                status = "PASS" if r.passed else "FAIL"
                rank_str = f"#{r.rank}" if r.rank else "N/A"
                print(
                    f"    {model_name}/{mode:<15} {status}  rank={rank_str:<4} "
                    f"score={r.top1_score:.4f}  top1={r.top1_content[:50]}"
                )

    # --- Key findings ---
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    # Find best semantic-only model
    best_sem = max(model_names, key=lambda m: _compute_metrics(all_results[m]["semantic"])["mrr"])
    best_sem_mrr = _compute_metrics(all_results[best_sem]["semantic"])["mrr"]

    # Find best overall hybrid
    if "hybrid" in modes:
        best_hyb = max(model_names, key=lambda m: _compute_metrics(all_results[m]["hybrid"])["mrr"])
        best_hyb_mrr = _compute_metrics(all_results[best_hyb]["hybrid"])["mrr"]
        print(f"  Best semantic-only:  {best_sem} (MRR={best_sem_mrr:.3f})")
        print(f"  Best hybrid:         {best_hyb} (MRR={best_hyb_mrr:.3f})")
        delta = best_hyb_mrr - best_sem_mrr
        if delta > 0.01:
            print(f"  -> Hybrid adds +{delta:.3f} MRR over best semantic-only")
        else:
            print(f"  -> Hybrid provides negligible benefit ({delta:+.3f} MRR)")

    if "hybrid+rerank" in modes:
        best_rr = max(model_names, key=lambda m: _compute_metrics(all_results[m]["hybrid+rerank"])["mrr"])
        best_rr_mrr = _compute_metrics(all_results[best_rr]["hybrid+rerank"])["mrr"]
        print(f"  Best hybrid+rerank:  {best_rr} (MRR={best_rr_mrr:.3f})")

    # Check: can best semantic-only beat current hybrid?
    if "hybrid" in modes:
        current_hyb_mrr = (
            _compute_metrics(all_results.get("e5-small-v2", {}).get("hybrid", []))["mrr"]
            if "e5-small-v2" in all_results
            else 0
        )
        if current_hyb_mrr > 0:
            print(f"\n  Current production (e5-small-v2 hybrid): MRR={current_hyb_mrr:.3f}")
            if best_sem_mrr >= current_hyb_mrr:
                print(f"  ** {best_sem} semantic-only MATCHES OR BEATS current hybrid **")
                print("     -> FTS can potentially be removed with this model")
            else:
                print(f"  Best semantic-only ({best_sem_mrr:.3f}) still trails current hybrid ({current_hyb_mrr:.3f})")
                print("     -> FTS still adds value, keep hybrid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Search pipeline benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test. 'all' for everything, '384' for 384-dim only.",
    )
    parser.add_argument("--rerank", action="store_true", help="Include hybrid+rerank mode")
    parser.add_argument("--validate", action="store_true", help="Only validate models (no benchmark)")
    parser.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="Inference backend (default: onnx). ONNX matches production latency.",
    )
    args = parser.parse_args()

    global _BACKEND
    _BACKEND = args.backend

    if args.models and "all" in args.models:
        model_names = list(MODELS.keys())
    elif args.models and "384" in args.models:
        model_names = MODELS_384
    elif args.models:
        model_names = args.models
    else:
        model_names = DEFAULT_MODELS

    # Validate model names
    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")
            return

    if args.validate:
        validate_models(model_names)
        return

    modes = ["semantic", "hybrid"]
    if args.rerank:
        modes.append("hybrid+rerank")

    reranker = None
    if args.rerank:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from memory.reranker import RerankerModel

        print("Loading reranker...")
        reranker = RerankerModel("tinybert")
        reranker.score_pairs("warmup", ["warmup"])

    print("=" * 90)
    print("SEARCH PIPELINE BENCHMARK")
    print("=" * 90)
    print(f"Backend: {_BACKEND}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Corpus: {len(CORPUS)} memories")
    print(f"Tests: {len(ALL_TEST_CASES)}")

    all_results: dict[str, dict[str, list[TestResult]]] = {}
    all_latencies: dict[str, LatencyInfo] = {}
    for name in model_names:
        cfg = MODELS[name]
        print(f"\n--- {name} ({cfg['hf_id']}) ---")
        try:
            results_by_mode, latency = run_pipeline(name, cfg, modes, reranker=reranker)
            all_results[name] = results_by_mode
            all_latencies[name] = latency
            for mode, results in results_by_mode.items():
                metrics = _compute_metrics(results)
                per_q = latency.search_per_query_ms.get(mode, 0)
                print(f"  {mode}: MRR={metrics['mrr']:.3f} Top1={metrics['top1_pct']:.0f}% ({per_q:.1f}ms/query)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    if all_results:
        print_report(all_results, all_latencies)


if __name__ == "__main__":
    main()
