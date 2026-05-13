"""Cross-encoder reranker — ONNX CPU, lazy singleton with L1/L2 cache.

Wraps a sequence-classification cross-encoder that scores (query, document) pairs.
Default model: BAAI/bge-reranker-v2-m3 (cross-encoder, 568M, BEIR ~55, multilingual).
Override via MEMORY_RERANK_REPO / MEMORY_RERANK_FILE / MEMORY_RERANK_TOKENIZER env vars
(e.g. mixedbread-ai/mxbai-rerank-base-v2 when its ONNX export stabilises).

Why default to bge-reranker-v2-m3 instead of mxbai-rerank-base-v2 as referenced in
the upgrade plan: bge has reliable `onnx/model.onnx` artefacts on HF and is a true
cross-encoder; mxbai-base-v2 is a hybrid generative reranker whose ONNX export path
is less stable. Both fit the same calling convention here.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "repos" / "memory" / "data"
MODEL_DIR = DATA_DIR / "models" / "rerank"
CACHE_DB_PATH = DATA_DIR / "rerank_cache.db"

DEFAULT_REPO = "BAAI/bge-reranker-v2-m3"
DEFAULT_ONNX_FILE = "onnx/model.onnx"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"

MAX_SEQ_LENGTH = 512
_L1_CACHE_MAX = 512


def _pair_hash(query: str, doc: str) -> str:
    return hashlib.sha256(f"{query}\x00{doc}".encode()).hexdigest()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    """Lazy-loaded ONNX cross-encoder with disk-backed pair-score cache.

    Public API:
        score(query, docs) -> np.ndarray[float]   # scores in [0, 1]
        rerank(query, results, top_n, content_key="content") -> list[dict]
            # writes 'rerank_score' on each result, returns top_n sorted desc
    """

    def __init__(
        self,
        repo: str | None = None,
        onnx_file: str | None = None,
        tokenizer_file: str | None = None,
        cache_db: str | Path | None = None,
        scorer: Callable[[str, list[str]], np.ndarray] | None = None,
    ) -> None:
        self._repo = repo or os.environ.get("MEMORY_RERANK_REPO", DEFAULT_REPO)
        self._onnx_file = onnx_file or os.environ.get("MEMORY_RERANK_FILE", DEFAULT_ONNX_FILE)
        self._tokenizer_file = tokenizer_file or os.environ.get("MEMORY_RERANK_TOKENIZER", DEFAULT_TOKENIZER_FILE)
        self._session = None
        self._tokenizer = None
        self._scorer = scorer  # test hook: inject a deterministic fake
        self._l1: dict[str, float] = {}
        self._cache_db_path = Path(cache_db) if cache_db else CACHE_DB_PATH
        self._cache_conn: sqlite3.Connection | None = None
        self._store_count = 0

    # --- Model loading ---

    def _load(self) -> None:
        if self._session is not None or self._scorer is not None:
            return

        onnx_path = MODEL_DIR / self._onnx_file
        tokenizer_path = MODEL_DIR / self._tokenizer_file
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

        if not onnx_path.exists():
            url = f"https://huggingface.co/{self._repo}/resolve/main/{self._onnx_file}"
            urllib.request.urlretrieve(url, str(onnx_path))  # nosec B310  # noqa: S310
        if not tokenizer_path.exists():
            url = f"https://huggingface.co/{self._repo}/resolve/main/{self._tokenizer_file}"
            urllib.request.urlretrieve(url, str(tokenizer_path))  # nosec B310  # noqa: S310

        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 8
        opts.inter_op_num_threads = 2
        self._session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

    # --- Cache layer ---

    def _get_cache_conn(self) -> sqlite3.Connection:
        if self._cache_conn is None:
            self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_conn = sqlite3.connect(str(self._cache_db_path))
            self._cache_conn.execute("PRAGMA journal_mode=WAL")
            self._cache_conn.execute("PRAGMA synchronous=NORMAL")
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS rerank_cache "
                "(pair_hash TEXT PRIMARY KEY, score REAL, last_accessed_at REAL)"
            )
        return self._cache_conn

    def _l2_get_many(self, pair_hashes: list[str]) -> dict[str, float]:
        if not pair_hashes:
            return {}
        conn = self._get_cache_conn()
        placeholders = ",".join("?" * len(pair_hashes))
        rows = conn.execute(
            f"SELECT pair_hash, score FROM rerank_cache WHERE pair_hash IN ({placeholders})",  # noqa: S608
            pair_hashes,
        ).fetchall()
        result = {h: float(s) for h, s in rows}
        if result:
            now = time.time()
            hit = list(result.keys())
            hit_ph = ",".join("?" * len(hit))
            conn.execute(
                f"UPDATE rerank_cache SET last_accessed_at = ? WHERE pair_hash IN ({hit_ph})",  # noqa: S608
                [now, *hit],
            )
            conn.commit()
        return result

    def _l2_put_many(self, items: list[tuple[str, float]]) -> None:
        if not items:
            return
        conn = self._get_cache_conn()
        now = time.time()
        conn.executemany(
            "INSERT OR REPLACE INTO rerank_cache (pair_hash, score, last_accessed_at) VALUES (?, ?, ?)",
            [(h, s, now) for h, s in items],
        )
        conn.commit()
        self._store_count += 1
        if self._store_count % 200 == 0:
            self.evict_cache()

    def evict_cache(self, max_entries: int = 50000) -> int:
        conn = self._get_cache_conn()
        count = conn.execute("SELECT COUNT(*) FROM rerank_cache").fetchone()[0]
        if count <= max_entries:
            return 0
        to_delete = count - max_entries
        conn.execute(
            "DELETE FROM rerank_cache WHERE pair_hash IN ("
            "  SELECT pair_hash FROM rerank_cache "
            "  ORDER BY COALESCE(last_accessed_at, 0) ASC LIMIT ?)",
            (to_delete,),
        )
        conn.commit()
        return to_delete

    def _l1_put(self, key: str, score: float) -> None:
        if len(self._l1) >= _L1_CACHE_MAX:
            oldest = next(iter(self._l1))
            del self._l1[oldest]
        self._l1[key] = score

    # --- Public API ---

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Score each (query, doc) pair. Returns float scores in [0, 1]."""
        if not docs:
            return np.array([], dtype=np.float32)

        results: list[float | None] = [None] * len(docs)
        need_l2: list[tuple[int, str]] = []

        for i, d in enumerate(docs):
            key = _pair_hash(query, d)
            cached = self._l1.get(key)
            if cached is not None:
                results[i] = cached
            else:
                need_l2.append((i, key))

        if need_l2:
            disk = self._l2_get_many([k for _, k in need_l2])
            need_compute: list[tuple[int, str]] = []
            for i, k in need_l2:
                if k in disk:
                    results[i] = disk[k]
                    self._l1_put(k, disk[k])
                else:
                    need_compute.append((i, k))

            if need_compute:
                compute_docs = [docs[i] for i, _ in need_compute]
                computed = self._compute(query, compute_docs)
                puts: list[tuple[str, float]] = []
                for (i, k), s in zip(need_compute, computed, strict=True):
                    sf = float(s)
                    results[i] = sf
                    self._l1_put(k, sf)
                    puts.append((k, sf))
                self._l2_put_many(puts)

        return np.array(results, dtype=np.float32)

    def rerank(
        self,
        query: str,
        items: list[dict],
        top_n: int | None = None,
        content_key: str = "content",
        score_key: str = "rerank_score",
    ) -> list[dict]:
        """Rerank items by cross-encoder score. Writes score_key on each item in place.

        Sorts descending; truncates to top_n if provided.
        """
        if not items:
            return items
        docs = [item.get(content_key, "") for item in items]
        scores = self.score(query, docs)
        for item, s in zip(items, scores, strict=True):
            item[score_key] = round(float(s), 4)
        items.sort(key=lambda m: m.get(score_key, 0.0), reverse=True)
        return items[:top_n] if top_n else items

    # --- Inference ---

    def _compute(self, query: str, docs: list[str]) -> np.ndarray:
        if self._scorer is not None:
            return np.asarray(self._scorer(query, docs), dtype=np.float32)

        self._load()
        # Cross-encoder: tokenize (query, doc) pairs together
        encodings = self._tokenizer.encode_batch([(query, d) for d in docs])
        n = len(encodings)
        input_ids = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int64)
        attention_mask = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int64)
        for i, e in enumerate(encodings):
            input_ids[i, : len(e.ids)] = e.ids
            attention_mask[i, : len(e.attention_mask)] = e.attention_mask

        feeds: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Some ONNX exports include token_type_ids
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, feeds)
        logits = outputs[0].astype(np.float32)
        # Shape can be (n, 1) or (n,) for binary cross-encoder
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] == 2:
            # Some models output (n, 2); take positive class
            logits = logits[:, 1] - logits[:, 0]
        return _sigmoid(logits)


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
