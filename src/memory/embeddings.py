"""Embedding model (nomic-ai/modernbert-embed-base) — MLX Metal GPU on macOS, ONNX CPU fallback."""

from __future__ import annotations

import contextlib
import hashlib
import platform
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".claude" / "tools" / "memory" / "data" / "models"
DATA_DIR = Path.home() / ".claude" / "tools" / "memory" / "data"
CACHE_DB_PATH = DATA_DIR / "embedding_cache.db"
MODEL_NAME = "modernbert-embed-base"
HF_REPO = f"nomic-ai/{MODEL_NAME}"

# ONNX fallback (Linux/Windows, or if MLX unavailable)
ONNX_MODEL_FILE = "onnx/model_uint8.onnx"
ONNX_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{ONNX_MODEL_FILE}"
TOKENIZER_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/tokenizer.json"

EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512
PREFIX_QUERY = "search_query: "
PREFIX_DOC = "search_document: "
_L1_CACHE_MAX = 256

_IS_MACOS = platform.system() == "Darwin"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingModel:
    """Lazy-loaded embedding model with L1 (memory) and L2 (disk) cache.

    Uses MLX (Metal GPU) on macOS for ~5ms/query inference.
    Falls back to ONNX Runtime CPU on other platforms (~67ms/query).
    """

    def __init__(self, cache_db: str | Path | None = None) -> None:
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._onnx_session = None
        self._tokenizer = None  # tokenizers.Tokenizer (for ONNX path)
        self._backend: str | None = None
        self._l1: dict[str, np.ndarray] = {}
        self._cache_db_path = Path(cache_db) if cache_db else CACHE_DB_PATH
        self._cache_conn: sqlite3.Connection | None = None
        self._store_count: int = 0

    def _load(self) -> None:
        """Load model — tries MLX on macOS first, falls back to ONNX."""
        if self._backend is not None:
            return

        if _IS_MACOS:
            try:
                self._load_mlx()
                return
            except (ImportError, RuntimeError, OSError):
                pass

        self._load_onnx()

    def _load_mlx(self) -> None:
        """Load model via MLX (Metal GPU)."""
        from mlx_embeddings.utils import load

        self._mlx_model, self._mlx_tokenizer = load(HF_REPO)
        self._backend = "mlx"

    def _load_onnx(self) -> None:
        """Load model via ONNX Runtime (CPU fallback)."""
        model_dir = MODEL_DIR / MODEL_NAME
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = model_dir / ONNX_MODEL_FILE
        tokenizer_path = model_dir / "tokenizer.json"

        for path, url in [(onnx_path, ONNX_URL), (tokenizer_path, TOKENIZER_URL)]:
            if not path.exists():
                print(f"Downloading {path.name}...")
                urllib.request.urlretrieve(url, str(path))  # nosec B310

        import onnxruntime as ort
        from tokenizers import Tokenizer

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 2

        self._onnx_session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(length=MAX_SEQ_LENGTH)
        self._backend = "onnx"

    # --- Cache layer ---

    def _get_cache_conn(self) -> sqlite3.Connection:
        if self._cache_conn is None:
            self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_conn = sqlite3.connect(str(self._cache_db_path))
            self._cache_conn.execute("PRAGMA journal_mode=WAL")
            self._cache_conn.execute("PRAGMA synchronous=NORMAL")
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (text_hash TEXT PRIMARY KEY, embedding BLOB, last_accessed_at REAL)"
            )
            with contextlib.suppress(sqlite3.OperationalError):
                self._cache_conn.execute("ALTER TABLE cache ADD COLUMN last_accessed_at REAL")
        return self._cache_conn

    def _l2_get_many(self, text_hashes: list[str]) -> dict[str, np.ndarray]:
        if not text_hashes:
            return {}
        conn = self._get_cache_conn()
        placeholders = ",".join("?" * len(text_hashes))
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM cache WHERE text_hash IN ({placeholders})",
            text_hashes,
        ).fetchall()
        result = {row[0]: np.frombuffer(row[1], dtype=np.float32).copy() for row in rows}
        if result:
            now = time.time()
            hit_hashes = list(result.keys())
            hit_ph = ",".join("?" * len(hit_hashes))
            conn.execute(
                f"UPDATE cache SET last_accessed_at = ? WHERE text_hash IN ({hit_ph})",
                [now, *hit_hashes],
            )
            conn.commit()
        return result

    def _l2_put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        conn = self._get_cache_conn()
        now = time.time()
        conn.executemany(
            "INSERT OR REPLACE INTO cache (text_hash, embedding, last_accessed_at) VALUES (?, ?, ?)",
            [(h, emb.tobytes(), now) for h, emb in items],
        )
        conn.commit()
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        self._store_count += 1
        if self._store_count % 100 == 0:
            self.evict_cache()

    def evict_cache(self, max_entries: int = 10000) -> int:
        conn = self._get_cache_conn()
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count <= max_entries:
            return 0
        to_delete = count - max_entries
        conn.execute(
            "DELETE FROM cache WHERE text_hash IN ("
            "  SELECT text_hash FROM cache ORDER BY COALESCE(last_accessed_at, 0) ASC LIMIT ?"
            ")",
            (to_delete,),
        )
        conn.commit()
        return to_delete

    # --- Public API ---

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query (adds search_query: prefix)."""
        return self.embed(PREFIX_QUERY + text)

    def embed_doc(self, text: str) -> np.ndarray:
        """Embed a document/memory for storage (adds search_document: prefix)."""
        return self.embed(PREFIX_DOC + text)

    def embed_query_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple search queries."""
        return self.embed_batch([PREFIX_QUERY + t for t in texts])

    def embed_doc_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple documents for storage."""
        return self.embed_batch([PREFIX_DOC + t for t in texts])

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text. Returns shape (EMBEDDING_DIM,)."""
        cached = self._l1.get(text)
        if cached is not None:
            return cached
        th = _text_hash(text)
        disk = self._l2_get_many([th])
        if th in disk:
            self._l1_put(text, disk[th])
            return disk[th]
        result = self._compute_embeddings([text])[0]
        self._l1_put(text, result)
        self._l2_put_many([(th, result)])
        return result

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        results: list[np.ndarray | None] = [None] * len(texts)
        need_l2: list[tuple[int, str, str]] = []

        for i, t in enumerate(texts):
            cached = self._l1.get(t)
            if cached is not None:
                results[i] = cached
            else:
                need_l2.append((i, t, _text_hash(t)))

        if not need_l2:
            return np.stack(results)

        l2_hashes = [th for _, _, th in need_l2]
        disk = self._l2_get_many(l2_hashes)
        need_compute: list[tuple[int, str, str]] = []
        for i, t, th in need_l2:
            if th in disk:
                results[i] = disk[th]
                self._l1_put(t, disk[th])
            else:
                need_compute.append((i, t, th))

        if need_compute:
            compute_texts = [t for _, t, _ in need_compute]
            computed = self._compute_embeddings(compute_texts)
            disk_inserts = []
            for j, (idx, t, th) in enumerate(need_compute):
                results[idx] = computed[j]
                self._l1_put(t, computed[j])
                disk_inserts.append((th, computed[j]))
            self._l2_put_many(disk_inserts)

        return np.stack(results)

    def _l1_put(self, text: str, embedding: np.ndarray) -> None:
        if len(self._l1) >= _L1_CACHE_MAX:
            oldest = next(iter(self._l1))
            del self._l1[oldest]
        self._l1[text] = embedding

    # --- Inference backends ---

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Run inference via daemon, MLX, or ONNX."""
        from .daemon import daemon_available, daemon_embed

        if daemon_available():
            result = daemon_embed(texts)
            if result is not None:
                return np.array(result, dtype=np.float32)

        self._load()

        if self._backend == "mlx":
            return self._infer_mlx(texts)
        return self._infer_onnx(texts)

    def _infer_mlx(self, texts: list[str]) -> np.ndarray:
        """MLX inference — runs on Metal GPU."""
        import mlx.core as mx

        inputs = self._mlx_tokenizer(
            texts, return_tensors="mlx", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        )
        outputs = self._mlx_model(**inputs)
        mx.eval(outputs.text_embeds)
        return np.array(outputs.text_embeds, dtype=np.float32)

    def _infer_onnx(self, texts: list[str]) -> np.ndarray:
        """ONNX Runtime inference — CPU fallback."""
        encodings = self._tokenizer.encode_batch(texts)
        n = len(encodings)
        input_ids = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int64)
        attention_mask = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int64)
        for i, e in enumerate(encodings):
            input_ids[i, : len(e.ids)] = e.ids
            attention_mask[i, : len(e.attention_mask)] = e.attention_mask

        outputs = self._onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        token_embeddings = outputs[0].astype(np.float32)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.maximum(mask_expanded.sum(axis=1), 1e-9)
        mean_pooled = summed / counts
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        return (mean_pooled / np.maximum(norms, 1e-9)).astype(np.float32)


_model: EmbeddingModel | None = None


def get_model() -> EmbeddingModel:
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model
