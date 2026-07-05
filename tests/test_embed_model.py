"""Win #1: pluggable embedding model (nomic modernbert vs gte-modernbert).

gte-modernbert-base beats nomic on code/identifier retrieval but needs CLS
pooling (not mean) and NO instruction prefix — the opposite of nomic.  A small
registry resolves per-model repo/prefixes/pooling so the swap is env-driven.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from memory.embeddings import _resolve_embed_model


def test_resolve_default_is_nomic_mean_with_prefixes():
    m = _resolve_embed_model("modernbert-embed-base")
    assert m["repo"] == "nomic-ai/modernbert-embed-base"
    assert m["pooling"] == "mean"
    assert m["query_prefix"] == "search_query: "
    assert m["doc_prefix"] == "search_document: "


def test_resolve_gte_is_cls_no_prefix():
    m = _resolve_embed_model("gte-modernbert-base")
    assert m["repo"] == "Alibaba-NLP/gte-modernbert-base"
    assert m["pooling"] == "cls"
    assert m["query_prefix"] == ""
    assert m["doc_prefix"] == ""


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        _resolve_embed_model("no-such-model")


@pytest.mark.slow
def test_gte_batch_mixed_lengths_is_finite():
    """gte batched with a long (512-token) item forces short items to be padded
    to 512 — those fully-masked pad rows must not NaN.  Regression guard for the
    float16 mask bug (-1e9 -> -inf in fp16 -> softmax NaN on masked rows)."""
    code = (
        "import numpy as np; from memory.embeddings import EmbeddingModel;"
        "m=EmbeddingModel(cache_db='/tmp/gte_batch_probe.db');"
        "texts=['TICKET-194 short note', 'word '*4000, 'another short one'];"
        "vs=m.embed_doc_batch(texts);"
        "print(all(not np.isnan(v).any() and abs(np.linalg.norm(v)-1.0)<0.05 for v in vs))"
    )
    env = {**os.environ, "MEMORY_EMBED_MODEL": "gte-modernbert-base"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "True", f"gte batch produced NaN/non-unit vectors: {out.stdout}"


@pytest.mark.slow
def test_gte_produces_finite_unit_embeddings():
    """gte-modernbert CLS pooling must yield finite unit vectors through the full
    plugin path.  Runs in a subprocess because the active model resolves at
    import time from $MEMORY_EMBED_MODEL.  Regression guard for the fixed-512
    padding bug that made gte's masked-softmax rows NaN."""
    code = (
        "import numpy as np; from memory.embeddings import EmbeddingModel;"
        "v = EmbeddingModel().embed_query('cameras-server TICKET-194 migration');"
        "print(float(np.linalg.norm(v)), bool(np.isnan(v).any()))"
    )
    env = {**os.environ, "MEMORY_EMBED_MODEL": "gte-modernbert-base"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    norm_s, hasnan_s = out.stdout.strip().split()
    assert hasnan_s == "False", f"gte embedding has NaN: {out.stdout}"
    assert abs(float(norm_s) - 1.0) < 0.05, f"gte embedding not unit-norm: {norm_s}"
