"""ONNX-based embedding model (BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".claude" / "tools" / "memory" / "data" / "models"
DATA_DIR = Path.home() / ".claude" / "tools" / "memory" / "data"
CACHE_DB_PATH = DATA_DIR / "embedding_cache.db"
MODEL_NAME = "bge-small-en-v1.5"
HF_REPO = f"BAAI/{MODEL_NAME}"
ONNX_MODEL_FILE = "onnx/model.onnx"
ONNX_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{ONNX_MODEL_FILE}"
TOKENIZER_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/tokenizer.json"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256
POOLING = "cls"  # bge uses CLS token pooling (e5 used mean pooling)
_L1_CACHE_MAX = 256


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingModel:
    """Lazy-loaded ONNX embedding model with L1 (memory) and L2 (disk) cache."""

    def __init__(self, cache_db: str | Path | None = None) -> None:
        self._session = None
        self._tokenizer = None
        self._l1: dict[str, np.ndarray] = {}
        self._cache_db_path = Path(cache_db) if cache_db else CACHE_DB_PATH
        self._cache_conn: sqlite3.Connection | None = None
        self._store_count: int = 0

    def _ensure_model(self) -> Path:
        """Download model files if not present."""
        model_dir = MODEL_DIR / MODEL_NAME
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / ONNX_MODEL_FILE
        tokenizer_path = model_dir / "tokenizer.json"

        for path, url in [(model_path, ONNX_URL), (tokenizer_path, TOKENIZER_URL)]:
            if not path.exists():
                print(f"Downloading {path.name}...")
                urllib.request.urlretrieve(url, str(path))  # nosec B310

        return model_dir

    def _load(self) -> None:
        """Load ONNX model and tokenizer."""
        if self._session is not None:
            return

        model_dir = self._ensure_model()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 2

        self._session = ort.InferenceSession(
            str(model_dir / ONNX_MODEL_FILE),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

    def _get_cache_conn(self) -> sqlite3.Connection:
        if self._cache_conn is None:
            self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_conn = sqlite3.connect(str(self._cache_db_path))
            self._cache_conn.execute("PRAGMA journal_mode=WAL")
            self._cache_conn.execute("PRAGMA synchronous=NORMAL")
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (text_hash TEXT PRIMARY KEY, embedding BLOB, last_accessed_at REAL)"
            )
            # Idempotent migration for existing DBs
            with contextlib.suppress(sqlite3.OperationalError):
                self._cache_conn.execute("ALTER TABLE cache ADD COLUMN last_accessed_at REAL")
        return self._cache_conn

    def _l2_get_many(self, text_hashes: list[str]) -> dict[str, np.ndarray]:
        """Fetch multiple embeddings from disk cache by hash."""
        if not text_hashes:
            return {}
        conn = self._get_cache_conn()
        placeholders = ",".join("?" * len(text_hashes))
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM cache WHERE text_hash IN ({placeholders})",
            text_hashes,
        ).fetchall()
        result = {row[0]: np.frombuffer(row[1], dtype=np.float32).copy() for row in rows}
        # Update last_accessed_at for cache hits
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
        """Write multiple embeddings to disk cache."""
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
        """Trigger eviction every 100 stores."""
        self._store_count += 1
        if self._store_count % 100 == 0:
            self.evict_cache()

    def evict_cache(self, max_entries: int = 10000) -> int:
        """Evict oldest cache entries by last_accessed_at when count exceeds max."""
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

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text. Returns shape (EMBEDDING_DIM,)."""
        # L1: in-memory
        cached = self._l1.get(text)
        if cached is not None:
            return cached
        # L2: disk
        th = _text_hash(text)
        disk = self._l2_get_many([th])
        if th in disk:
            self._l1_put(text, disk[th])
            return disk[th]
        # L3: ONNX
        result = self._compute_embeddings([text])[0]
        self._l1_put(text, result)
        self._l2_put_many([(th, result)])
        return result

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        results: list[np.ndarray | None] = [None] * len(texts)
        need_l2: list[tuple[int, str, str]] = []  # (index, text, text_hash)

        # L1 check
        for i, t in enumerate(texts):
            cached = self._l1.get(t)
            if cached is not None:
                results[i] = cached
            else:
                need_l2.append((i, t, _text_hash(t)))

        if not need_l2:
            return np.stack(results)

        # L2 check
        l2_hashes = [th for _, _, th in need_l2]
        disk = self._l2_get_many(l2_hashes)
        need_onnx: list[tuple[int, str, str]] = []
        for i, t, th in need_l2:
            if th in disk:
                results[i] = disk[th]
                self._l1_put(t, disk[th])
            else:
                need_onnx.append((i, t, th))

        # L3: ONNX for remaining
        if need_onnx:
            onnx_texts = [t for _, t, _ in need_onnx]
            computed = self._compute_embeddings(onnx_texts)
            disk_inserts = []
            for j, (idx, t, th) in enumerate(need_onnx):
                results[idx] = computed[j]
                self._l1_put(t, computed[j])
                disk_inserts.append((th, computed[j]))
            self._l2_put_many(disk_inserts)

        return np.stack(results)

    def _l1_put(self, text: str, embedding: np.ndarray) -> None:
        """Add entry to L1 in-memory cache with FIFO eviction."""
        if len(self._l1) >= _L1_CACHE_MAX:
            oldest = next(iter(self._l1))
            del self._l1[oldest]
        self._l1[text] = embedding

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Run ONNX inference + pooling for a list of texts."""
        self._load()

        encodings = self._tokenizer.encode_batch(texts)
        n = len(encodings)
        input_ids = np.empty((n, MAX_SEQ_LENGTH), dtype=np.int64)
        attention_mask = np.empty((n, MAX_SEQ_LENGTH), dtype=np.int64)
        for i, e in enumerate(encodings):
            input_ids[i] = e.ids
            attention_mask[i] = e.attention_mask
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        token_embeddings = outputs[0].astype(np.float32)

        # CLS pooling: take the first token's embedding
        pooled = token_embeddings[:, 0, :]

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normalized = (pooled / norms).astype(np.float32)

        return normalized


# Module-level singleton
_model: EmbeddingModel | None = None


def get_model() -> EmbeddingModel:
    """Get or create the singleton embedding model."""
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model
