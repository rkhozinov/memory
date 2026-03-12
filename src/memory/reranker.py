"""ONNX-based cross-encoder reranker for search result re-scoring."""

from __future__ import annotations

import platform
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".claude" / "tools" / "memory" / "data" / "models"

RERANKER_MODELS = {
    "tinybert": {
        "hf_repo": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "max_seq_length": 512,
    },
    "minilm6": {
        "hf_repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_seq_length": 512,
    },
}


def _quantized_model_filename() -> str:
    """Pick the right quantized ONNX variant for the current platform."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "model_qint8_arm64.onnx"
    return "model_quint8_avx2.onnx"


class RerankerModel:
    """Lazy-loaded ONNX cross-encoder for reranking search results."""

    def __init__(self, model_name: str = "tinybert") -> None:
        if model_name not in RERANKER_MODELS:
            raise ValueError(f"Unknown reranker model: {model_name}. Choose from: {list(RERANKER_MODELS)}")
        self.model_name = model_name
        self._config = RERANKER_MODELS[model_name]
        self._session = None
        self._tokenizer = None

    def _ensure_model(self) -> Path:
        """Download ONNX model and tokenizer if not present."""
        model_dir = MODEL_DIR / f"reranker-{self.model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        hf_repo = self._config["hf_repo"]
        onnx_filename = _quantized_model_filename()
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        files = [
            (model_path, f"https://huggingface.co/{hf_repo}/resolve/main/onnx/{onnx_filename}"),  # nosec B310
            (tokenizer_path, f"https://huggingface.co/{hf_repo}/resolve/main/tokenizer.json"),  # nosec B310
        ]

        for path, url in files:
            if not path.exists():
                print(f"Downloading {self.model_name} reranker: {path.name}...")
                urllib.request.urlretrieve(url, str(path))  # nosec B310

        return model_dir

    def _load(self) -> None:
        """Load ONNX session and tokenizer lazily."""
        if self._session is not None:
            return

        model_dir = self._ensure_model()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self._config["max_seq_length"])

    def score_pairs(self, query: str, candidates: list[str]) -> np.ndarray:
        """Score (query, candidate) pairs. Returns sigmoid scores in [0, 1]."""
        if not candidates:
            return np.array([], dtype=np.float32)

        self._load()

        # Tokenize as sentence pairs
        max_len = self._config["max_seq_length"]
        encodings = self._tokenizer.encode_batch([(query, c) for c in candidates])

        n = len(encodings)
        input_ids = np.zeros((n, max_len), dtype=np.int64)
        attention_mask = np.zeros((n, max_len), dtype=np.int64)
        token_type_ids = np.zeros((n, max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            length = min(len(enc.ids), max_len)
            input_ids[i, :length] = enc.ids[:length]
            attention_mask[i, :length] = enc.attention_mask[:length]
            token_type_ids[i, :length] = enc.type_ids[:length]

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        logits = outputs[0].flatten().astype(np.float32)

        # Numerically stable sigmoid
        scores = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )

        return scores


# Module-level singleton
_reranker: RerankerModel | None = None


def get_reranker(model_name: str | None = None) -> RerankerModel:
    """Get or create the singleton reranker model. Switches model if name changes."""
    global _reranker
    name = model_name or "tinybert"
    if _reranker is None or _reranker.model_name != name:
        _reranker = RerankerModel(name)
    return _reranker
