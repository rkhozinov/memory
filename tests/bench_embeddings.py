#!/usr/bin/env python3
"""Embedding inference benchmark for Apple Silicon (M4 Max).

Tests combinations of runtimes and models to find the optimal setup.
Run directly: python tests/bench_embeddings.py

Runtimes tested:
  1. ONNX CPU FP32 single-threaded (current baseline)
  2. ONNX CPU FP32 multi-threaded (intra=8, inter=2)
  3. ONNX CPU INT8 multi-threaded
  4. ONNX CoreML FP32
  5. MLX (if installed)

Models tested:
  1. all-MiniLM-L6-v2 (current — 22M params, 384-dim)
  2. snowflake-arctic-embed-xs (22M params, 384-dim, modern training)
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / "repos" / "memory" / "data" / "models"
RESULTS_PATH = Path(__file__).parent / "bench_results.json"

MODELS = {
    "all-MiniLM-L6-v2": {
        "onnx_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "tokenizer_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
        "dim": 384,
        "max_seq": 256,
        "has_token_type_ids": True,
    },
    "snowflake-arctic-embed-xs": {
        "onnx_url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model.onnx",
        "tokenizer_url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/tokenizer.json",
        "dim": 384,
        "max_seq": 512,
        "has_token_type_ids": False,
    },
}

# INT8 quantized model — only available for snowflake
INT8_ONNX_URLS = {
    "snowflake-arctic-embed-xs": "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model_quantized.onnx",
}

WARM_RUNS = 100
BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Test corpus — 50 representative memory texts
# ---------------------------------------------------------------------------

CORPUS = [
    # Short tags / labels (10)
    "terraform aws vpc module",
    "python asyncio pattern",
    "docker compose networking",
    "git rebase strategy",
    "sqlite WAL mode",
    "ONNX runtime optimization",
    "MCP server stdio",
    "Claude Code hooks",
    "numpy vectorization",
    "API rate limiting",
    # Medium sentences (20)
    "The terraform state file should be stored in S3 with DynamoDB locking enabled.",
    "Use content_hash as the primary key for deduplication across memory entries.",
    "ONNX Runtime CoreML provider routes computation to Apple Neural Engine on M-series chips.",
    "Soft delete pattern: set deleted_at timestamp instead of removing rows from the database.",
    "The embedding model uses mean pooling over token embeddings masked by attention weights.",
    "SQLite busy timeout of 15 seconds prevents lock contention in multi-agent scenarios.",
    "Claude Code MCP tools send all parameters as strings requiring server-side type coercion.",
    "The all-MiniLM-L6-v2 model produces 384-dimensional vectors with L2 normalization.",
    "Use IMMEDIATE transactions in SQLite to prevent TOCTOU race conditions on write paths.",
    "Natural language time expressions like 'last week' are parsed for search and delete filters.",
    "The singleton pattern with lazy loading avoids model initialization overhead on import.",
    "HuggingFace model files are cached locally to avoid repeated downloads on cold starts.",
    "Vector similarity search uses cosine distance via the sqlite-vec virtual table extension.",
    "Click CLI framework provides automatic help generation and argument validation.",
    "AWS profile configuration should use named profiles instead of default credentials.",
    "Memory recall tracking increments recall_count and updates last_recalled_at timestamp.",
    "The operation_events table logs analytics for store, search, and delete operations.",
    "Graph edges between memories use source_hash and target_hash for relationship mapping.",
    "Batch embedding computes all vectors in a single ONNX session.run() call for efficiency.",
    "Token type IDs are zero-filled for single-segment models like MiniLM and Arctic.",
    # Long paragraphs (20)
    "When deploying the memory service in a multi-agent environment, it is critical to use "
    "IMMEDIATE transaction mode in SQLite. This prevents a common race condition where two agents "
    "both check for duplicate content, find none, and then both insert the same memory. The "
    "IMMEDIATE lock is acquired at BEGIN rather than at first write, serializing the check-and-insert.",
    "The embedding pipeline processes text in three stages: tokenization using the HuggingFace "
    "tokenizers library with truncation at max_seq_length, ONNX inference producing per-token "
    "hidden states, and post-processing with mean pooling and L2 normalization. This matches "
    "the sentence-transformers library output exactly.",
    "SQLite WAL mode allows concurrent readers while a single writer holds the lock. Combined "
    "with a 15-second busy timeout, this configuration handles the typical Claude Code usage "
    "pattern where one agent writes memories while others search. The busy timeout prevents "
    "immediate SQLITE_BUSY errors during brief write contention.",
    "The deduplication strategy uses a two-tier approach: exact dedup via SHA256 content hash "
    "catches identical texts, while semantic dedup via cosine similarity with a configurable "
    "threshold (default 0.85) catches paraphrased or near-duplicate content. The semantic check "
    "only runs when explicitly requested via the --dedup flag.",
    "CoreML execution provider in ONNX Runtime routes compatible operations to Apple Silicon's "
    "Neural Engine, which can perform matrix multiplications significantly faster than CPU for "
    "certain model architectures. However, there is a cold-start penalty as the model must be "
    "compiled to CoreML format on first inference.",
    "The MCP server uses stdio transport for communication with Claude Code. All tool parameters "
    "arrive as strings regardless of their declared type, so the server implements explicit type "
    "coercion for integers, booleans, and JSON objects. Input validation is intentionally disabled "
    "to reduce latency for trusted local-only communication.",
    "Memory search supports three modes: semantic (vector similarity), exact (SQL LIKE pattern), "
    "and hybrid (both combined). Semantic search computes the query embedding and finds nearest "
    "neighbors in the sqlite-vec virtual table. Results are ranked by cosine similarity with "
    "optional filtering by tags, time range, and minimum similarity threshold.",
    "The CLI supports multiple output formats to serve different consumers: json for programmatic "
    "access, text for human-readable terminal output, and hook format for Claude Code's hook "
    "system which expects a specific JSON structure with type and content fields.",
    "Model download happens lazily on first inference call. Files are stored under "
    "<data-dir>/models/<model-name>/ (MEMORY_DATA_DIR, default ~/.local/share/memory, "
    "legacy ~/repos/memory/data). The download uses urllib.request for "
    "simplicity and prints progress to stdout. Once cached, subsequent loads skip the download "
    "and only initialize the ONNX session and tokenizer.",
    "The memory graph feature stores directed edges between related memories, identified by their "
    "content hashes. This enables traversal queries like 'find all memories related to X' by "
    "following edges from a source hash. The graph is stored in a simple SQLite table with "
    "source_hash, target_hash, and relationship_type columns.",
    "Tracemalloc provides Python-level memory tracking that captures allocation snapshots. While "
    "it does not capture native allocations from ONNX Runtime's C++ internals, it gives a useful "
    "lower bound on Python-side memory usage including numpy arrays and tokenizer data structures.",
    "The benchmark measures four key metrics for each runtime-model combination: cold start latency "
    "including model load and first inference, warm single-item latency as the median of 100 runs, "
    "batch throughput for 50 texts processed together, and peak memory usage tracked via tracemalloc.",
    "INT8 quantization reduces model weights from 32-bit floating point to 8-bit integers, "
    "typically achieving 3-4x speedup on CPU with minimal accuracy loss. The quantized model "
    "file is about 4x smaller on disk. ONNX Runtime handles dequantization transparently "
    "during inference.",
    "Apple Silicon's unified memory architecture means the GPU, Neural Engine, and CPU all share "
    "the same physical RAM. This eliminates data transfer overhead between processors, which is "
    "a significant advantage for inference workloads that would otherwise need to copy tensors "
    "between CPU and GPU memory.",
    "The MLX framework from Apple Research is designed specifically for Apple Silicon, leveraging "
    "unified memory and lazy evaluation. Operations are only computed when results are needed, "
    "and the framework can automatically fuse operations for better performance. It provides "
    "a numpy-like API that feels familiar to Python ML practitioners.",
    "Sentence-transformers wraps HuggingFace transformers with a focus on producing fixed-length "
    "sentence embeddings. It handles tokenization, inference, and pooling in a single encode() "
    "call. However, it pulls in the full PyTorch dependency which adds significant install size "
    "and startup time compared to direct ONNX inference.",
    "The benchmark adapters are intentionally minimal: each implements load() and embed_batch() "
    "with no shared base class logic beyond the interface. This keeps the benchmark code simple "
    "and self-contained, avoiding the complexity of a production abstraction layer.",
    "Resource.getrusage tracks maximum resident set size at the OS level, capturing all memory "
    "including native allocations from ONNX Runtime. Combined with tracemalloc for Python-level "
    "tracking, this gives both a ceiling (RSS) and a floor (Python allocations) for memory usage.",
    "The snowflake-arctic-embed-xs model uses the same 384-dimensional output as MiniLM but "
    "was trained more recently with improved techniques including hard negative mining and "
    "knowledge distillation. Published benchmarks show it outperforms MiniLM on retrieval "
    "tasks despite having a similar parameter count.",
    "When comparing inference runtimes, it is important to separate cold start (one-time cost) "
    "from warm latency (per-request cost). A runtime with a 2-second cold start but 1ms warm "
    "latency is preferable to one with 100ms cold start and 5ms warm latency for a long-running "
    "service that processes many requests.",
]

assert len(CORPUS) == BATCH_SIZE, f"Corpus has {len(CORPUS)} texts, expected {BATCH_SIZE}"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    runtime: str
    model: str
    cold_start_ms: float = 0.0
    warm_median_ms: float = 0.0
    warm_p95_ms: float = 0.0
    batch_total_ms: float = 0.0
    batch_per_item_ms: float = 0.0
    peak_memory_mb: float = 0.0
    embedding_dim: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Model file downloader
# ---------------------------------------------------------------------------


def ensure_model_files(model_name: str, need_int8: bool = False) -> Path:
    """Download ONNX model and tokenizer if not present. Returns model directory."""
    cfg = MODELS[model_name]
    model_dir = MODEL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (model_dir / "model.onnx", cfg["onnx_url"]),
        (model_dir / "tokenizer.json", cfg["tokenizer_url"]),
    ]
    if need_int8 and model_name in INT8_ONNX_URLS:
        files.append((model_dir / "model_quantized.onnx", INT8_ONNX_URLS[model_name]))

    for path, url in files:
        if not path.exists():
            print(f"  Downloading {model_name}/{path.name} ...")
            urllib.request.urlretrieve(url, str(path))

    return model_dir


# ---------------------------------------------------------------------------
# Runtime adapters
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """Minimal interface for benchmark runtime adapters."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cfg = MODELS[model_name]

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Produce normalized embeddings for a batch of texts."""


class OnnxAdapter(BaseAdapter):
    """ONNX Runtime CPU adapter with configurable threads and model variant."""

    def __init__(
        self,
        model_name: str,
        *,
        intra_threads: int = 1,
        inter_threads: int = 1,
        use_int8: bool = False,
        use_coreml: bool = False,
        label: str = "",
    ):
        super().__init__(model_name)
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads
        self.use_int8 = use_int8
        self.use_coreml = use_coreml
        self.label = label
        self._session = None
        self._tokenizer = None

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = ensure_model_files(self.model_name, need_int8=self.use_int8)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.intra_threads
        sess_options.inter_op_num_threads = self.inter_threads

        model_file = "model_quantized.onnx" if self.use_int8 else "model.onnx"

        if self.use_coreml:
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "MLComputeUnits": "ALL",
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(model_dir / model_file),
            sess_options,
            providers=providers,
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.cfg["max_seq"])
        self._tokenizer.enable_padding(length=self.cfg["max_seq"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Some models expect token_type_ids, others don't
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, feeds)

        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / counts

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = mean_pooled / norms

        return [row.tolist() for row in normalized]


class MlxAdapter(BaseAdapter):
    """MLX-based embedding adapter (Apple Silicon optimized)."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._weights = None
        self._tokenizer = None
        self._mx = None
        self._num_layers = 0
        self._num_heads = 0
        self._hidden_size = 0

    def load(self) -> None:
        import mlx.core as mx
        import safetensors.numpy
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        hf_ids = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "snowflake-arctic-embed-xs": "Snowflake/snowflake-arctic-embed-xs",
        }
        hf_id = hf_ids[self.model_name]
        model_path = Path(snapshot_download(hf_id))

        # Load tokenizer (same lib as ONNX adapter)
        self._tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.cfg["max_seq"])
        self._tokenizer.enable_padding(length=self.cfg["max_seq"])

        # Parse config.json directly instead of using transformers
        with open(model_path / "config.json") as f:
            config = json.load(f)
        self._num_layers = config["num_hidden_layers"]
        self._num_heads = config["num_attention_heads"]
        self._hidden_size = config["hidden_size"]

        # Load weights from safetensors
        safetensors_file = model_path / "model.safetensors"
        if not safetensors_file.exists():
            raise FileNotFoundError(f"No model.safetensors in {model_path} (torch weights not supported)")
        state_dict = safetensors.numpy.load_file(str(safetensors_file))

        self._weights = {k: mx.array(v) for k, v in state_dict.items()}
        self._mx = mx

    def _forward(self, input_ids, attention_mask, token_type_ids):
        """Manual BERT forward pass in MLX."""
        mx = self._mx

        # Embeddings
        word_emb = self._weights["embeddings.word_embeddings.weight"]
        pos_emb = self._weights["embeddings.position_embeddings.weight"]
        tok_emb = self._weights.get(
            "embeddings.token_type_embeddings.weight",
            mx.zeros((2, word_emb.shape[1])),
        )
        ln_w = self._weights["embeddings.LayerNorm.weight"]
        ln_b = self._weights["embeddings.LayerNorm.bias"]

        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)

        hidden = word_emb[input_ids] + pos_emb[position_ids] + tok_emb[token_type_ids]

        # Layer norm
        mean = mx.mean(hidden, axis=-1, keepdims=True)
        var = mx.var(hidden, axis=-1, keepdims=True)
        hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

        # Transformer layers
        num_layers = self._num_layers
        num_heads = self._num_heads
        head_dim = self._hidden_size // num_heads

        for i in range(num_layers):
            prefix = f"encoder.layer.{i}"

            q_w = self._weights[f"{prefix}.attention.self.query.weight"]
            q_b = self._weights[f"{prefix}.attention.self.query.bias"]
            k_w = self._weights[f"{prefix}.attention.self.key.weight"]
            k_b = self._weights[f"{prefix}.attention.self.key.bias"]
            v_w = self._weights[f"{prefix}.attention.self.value.weight"]
            v_b = self._weights[f"{prefix}.attention.self.value.bias"]

            q = hidden @ q_w.T + q_b
            k = hidden @ k_w.T + k_b
            v = hidden @ v_w.T + v_b

            batch_size = q.shape[0]
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(float(head_dim)))
            mask = attention_mask[:, None, None, :].astype(mx.float32)
            scores = scores + (1.0 - mask) * (-1e9)
            attn_weights = mx.softmax(scores, axis=-1)
            attn_out = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

            out_w = self._weights[f"{prefix}.attention.output.dense.weight"]
            out_b = self._weights[f"{prefix}.attention.output.dense.bias"]
            attn_out = attn_out @ out_w.T + out_b

            hidden = hidden + attn_out
            ln_w = self._weights[f"{prefix}.attention.output.LayerNorm.weight"]
            ln_b = self._weights[f"{prefix}.attention.output.LayerNorm.bias"]
            mean = mx.mean(hidden, axis=-1, keepdims=True)
            var = mx.var(hidden, axis=-1, keepdims=True)
            hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

            ff1_w = self._weights[f"{prefix}.intermediate.dense.weight"]
            ff1_b = self._weights[f"{prefix}.intermediate.dense.bias"]
            ff2_w = self._weights[f"{prefix}.output.dense.weight"]
            ff2_b = self._weights[f"{prefix}.output.dense.bias"]

            ff_out = hidden @ ff1_w.T + ff1_b
            ff_out = ff_out * 0.5 * (1.0 + mx.tanh(mx.sqrt(mx.array(2.0 / np.pi)) * (ff_out + 0.044715 * ff_out**3)))
            ff_out = ff_out @ ff2_w.T + ff2_b

            hidden = hidden + ff_out
            ln_w = self._weights[f"{prefix}.output.LayerNorm.weight"]
            ln_b = self._weights[f"{prefix}.output.LayerNorm.bias"]
            mean = mx.mean(hidden, axis=-1, keepdims=True)
            var = mx.var(hidden, axis=-1, keepdims=True)
            hidden = (hidden - mean) / mx.sqrt(var + 1e-12) * ln_w + ln_b

        return hidden

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        mx = self._mx

        encodings = self._tokenizer.encode_batch(texts)
        ids_np = np.array([e.ids for e in encodings], dtype=np.int32)
        mask_np = np.array([e.attention_mask for e in encodings], dtype=np.int32)

        input_ids = mx.array(ids_np)
        attention_mask = mx.array(mask_np)
        token_type_ids = mx.array(np.zeros_like(ids_np, dtype=np.int32))

        hidden = self._forward(input_ids, attention_mask, token_type_ids)

        # Force computation
        mx.synchronize()

        # Mean pooling (in numpy for simplicity)
        hidden_np = np.array(hidden)
        mask_np = np.array(attention_mask)[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(hidden_np * mask_np, axis=1)
        counts = np.clip(mask_np.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / counts

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = mean_pooled / norms

        return [row.tolist() for row in normalized]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(adapter: BaseAdapter, label: str) -> BenchResult:
    """Run the full benchmark suite for a single adapter."""
    result = BenchResult(runtime=label, model=adapter.model_name)

    try:
        # -- Cold start (load + first inference) --
        tracemalloc.start()
        t0 = time.perf_counter()
        adapter.load()
        _ = adapter.embed_batch([CORPUS[0]])
        t1 = time.perf_counter()
        result.cold_start_ms = (t1 - t0) * 1000

        # Check embedding dimension
        test_emb = adapter.embed_batch(["test"])
        result.embedding_dim = len(test_emb[0])

        # -- Warm single-item latency (median of N runs) --
        latencies = []
        for i in range(WARM_RUNS):
            text = CORPUS[i % len(CORPUS)]
            t0 = time.perf_counter()
            _ = adapter.embed_batch([text])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        latencies.sort()
        result.warm_median_ms = statistics.median(latencies)
        result.warm_p95_ms = latencies[int(len(latencies) * 0.95)]

        # -- Batch throughput (all 50 texts at once) --
        t0 = time.perf_counter()
        _ = adapter.embed_batch(CORPUS)
        t1 = time.perf_counter()
        result.batch_total_ms = (t1 - t0) * 1000
        result.batch_per_item_ms = result.batch_total_ms / len(CORPUS)

        # -- Peak memory --
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_memory_mb = peak / (1024 * 1024)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    return result


def check_coreml_available() -> bool:
    """Check if CoreML execution provider is available."""
    try:
        import onnxruntime as ort

        return "CoreMLExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def check_mlx_available() -> bool:
    """Check if MLX is installed."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def build_adapters() -> list[tuple[str, BaseAdapter]]:
    """Build all available runtime+model combinations."""
    adapters = []
    has_coreml = check_coreml_available()
    has_mlx = check_mlx_available()

    for model_name in MODELS:
        # 1. ONNX CPU FP32 single-threaded (baseline)
        adapters.append(
            (
                f"onnx-cpu-fp32-1t | {model_name}",
                OnnxAdapter(model_name, intra_threads=1, inter_threads=1),
            )
        )

        # 2. ONNX CPU FP32 multi-threaded
        adapters.append(
            (
                f"onnx-cpu-fp32-8t | {model_name}",
                OnnxAdapter(model_name, intra_threads=8, inter_threads=2),
            )
        )

        # 3. ONNX CPU INT8 multi-threaded (only if quantized model exists)
        if model_name in INT8_ONNX_URLS:
            adapters.append(
                (
                    f"onnx-cpu-int8-8t | {model_name}",
                    OnnxAdapter(model_name, intra_threads=8, inter_threads=2, use_int8=True),
                )
            )

        # 4. ONNX CoreML
        if has_coreml:
            adapters.append(
                (
                    f"onnx-coreml-fp32 | {model_name}",
                    OnnxAdapter(model_name, use_coreml=True),
                )
            )

        # 5. MLX
        if has_mlx:
            adapters.append(
                (
                    f"mlx | {model_name}",
                    MlxAdapter(model_name),
                )
            )

    return adapters


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_table(results: list[BenchResult]) -> None:
    """Print results as a formatted table."""
    headers = [
        "Runtime + Model",
        "Cold (ms)",
        "Warm med (ms)",
        "Warm p95 (ms)",
        "Batch 50 (ms)",
        "Per-item (ms)",
        "Mem (MB)",
        "Dim",
    ]

    rows = []
    for r in results:
        if r.error:
            rows.append(
                [
                    f"{r.runtime} | {r.model}" if "|" not in r.runtime else r.runtime,
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    r.error[:40],
                ]
            )
        else:
            rows.append(
                [
                    f"{r.runtime} | {r.model}" if "|" not in r.runtime else r.runtime,
                    f"{r.cold_start_ms:.1f}",
                    f"{r.warm_median_ms:.2f}",
                    f"{r.warm_p95_ms:.2f}",
                    f"{r.batch_total_ms:.1f}",
                    f"{r.batch_per_item_ms:.2f}",
                    f"{r.peak_memory_mb:.1f}",
                    str(r.embedding_dim),
                ]
            )

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Print
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True)) + " |"

    print()
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths, strict=True)) + " |"
        print(line)
    print(sep)
    print()


def save_results(results: list[BenchResult]) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warm_runs": WARM_RUNS,
        "batch_size": BATCH_SIZE,
        "results": [asdict(r) for r in results],
    }
    RESULTS_PATH.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Embedding Inference Benchmark")
    print("=" * 70)

    has_coreml = check_coreml_available()
    has_mlx = check_mlx_available()

    print("\nRuntimes available:")
    print("  ONNX CPU:    yes")
    print(f"  ONNX CoreML: {'yes' if has_coreml else 'no (CoreMLExecutionProvider not found)'}")
    print(f"  MLX:         {'yes' if has_mlx else 'no (mlx not installed)'}")
    print(f"\nModels: {', '.join(MODELS.keys())}")
    print(f"Warm runs: {WARM_RUNS}, Batch size: {BATCH_SIZE}")
    print()

    adapters = build_adapters()
    results = []

    for label, adapter in adapters:
        print(f"[{len(results) + 1}/{len(adapters)}] Benchmarking: {label}")
        result = run_benchmark(adapter, label)
        results.append(result)

        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(
                f"  Cold: {result.cold_start_ms:.1f}ms | "
                f"Warm: {result.warm_median_ms:.2f}ms | "
                f"Batch: {result.batch_total_ms:.1f}ms | "
                f"Mem: {result.peak_memory_mb:.1f}MB"
            )

    print_table(results)
    save_results(results)

    # Find the best warm latency (excluding errors)
    valid = [r for r in results if r.error is None]
    if valid:
        best_warm = min(valid, key=lambda r: r.warm_median_ms)
        best_batch = min(valid, key=lambda r: r.batch_total_ms)
        print(f"Best warm latency:     {best_warm.runtime} ({best_warm.warm_median_ms:.2f}ms)")
        print(f"Best batch throughput: {best_batch.runtime} ({best_batch.batch_total_ms:.1f}ms)")


if __name__ == "__main__":
    main()
