"""Tests for cluster-based deduplication (Phase D follow-up)."""

from __future__ import annotations

import io
import json
from unittest.mock import patch


def _seed_similar_cluster(store, contents: list[str], project: str | None = None) -> list[str]:
    """Store memories and force identical embeddings so they cluster reliably."""
    tags = [f"project:{project}"] if project else []
    hashes = [store.store(c, memory_type="learning", tags=tags)["content_hash"] for c in contents]
    if len(hashes) < 2:
        return hashes
    conn = store._get_conn()
    base_emb = conn.execute(
        "SELECT content_embedding FROM memory_embeddings WHERE rowid = "
        "(SELECT id FROM memories WHERE content_hash = ?)",
        (hashes[0],),
    ).fetchone()["content_embedding"]
    for h in hashes[1:]:
        conn.execute(
            "UPDATE memory_embeddings SET content_embedding = ? "
            "WHERE rowid = (SELECT id FROM memories WHERE content_hash = ?)",
            (base_emb, h),
        )
    conn.commit()
    return hashes


def test_find_clusters_basic(store):
    seed = _seed_similar_cluster(
        store,
        ["k8s pod scheduling alpha", "k8s pod scheduling beta", "k8s pod scheduling gamma"],
    )
    out = store.find_clusters(threshold=0.85, project_scoped=False)
    assert out["n_clusters"] == 1
    cluster = out["clusters"][0]
    assert cluster["size"] == 3
    assert {m["content_hash"] for m in cluster["members"]} == set(seed)
    assert cluster["survivor_hash"] in seed


def test_find_clusters_respects_project_scope(store):
    _seed_similar_cluster(
        store,
        ["postgres tuning alpha", "postgres tuning beta"],
        project="proj-a",
    )
    _seed_similar_cluster(
        store,
        ["postgres tuning alpha2", "postgres tuning beta2"],
        project="proj-b",
    )
    out_scoped = store.find_clusters(threshold=0.85, project_scoped=True)
    sizes_scoped = sorted(c["size"] for c in out_scoped["clusters"])
    assert sizes_scoped == [2, 2]

    out_open = store.find_clusters(threshold=0.85, project_scoped=False)
    assert out_open["n_clusters"] == 1
    assert out_open["clusters"][0]["size"] == 4


def test_consolidate_cluster_keep_higher_recall(store):
    seed = _seed_similar_cluster(store, ["alpha topic one", "alpha topic two", "alpha topic three"])
    conn = store._get_conn()
    conn.execute("UPDATE memories SET recall_count = 50 WHERE content_hash = ?", (seed[1],))
    conn.commit()

    result = store.consolidate(threshold=0.85, cluster=True, project_scoped=False)
    assert result["consolidated"] == 2
    assert result["n_clusters"] == 1

    survivor = result["clusters"][0]["survivor_hash"]
    assert survivor == seed[1]

    loser_count = conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE content_hash IN (?, ?) AND deleted_at IS NOT NULL",
        (seed[0], seed[2]),
    ).fetchone()["n"]
    assert loser_count == 2

    edges = conn.execute(
        "SELECT target_hash, relationship_type FROM memory_graph "
        "WHERE relationship_type = 'merged_into' AND target_hash = ?",
        (seed[1],),
    ).fetchall()
    assert len(edges) >= 2


def test_consolidate_cluster_concat_appends_related(store):
    seed = _seed_similar_cluster(
        store,
        [
            "OnCall API: SA token cannot resolve alert groups; returns 403",
            "Grafana SA token cannot silence/ack/resolve OnCall alerts via API",
            "Service account tokens lack permission for OnCall alert group resolve",
        ],
    )
    conn = store._get_conn()
    conn.execute("UPDATE memories SET recall_count = 5 WHERE content_hash = ?", (seed[0],))
    conn.commit()

    result = store.consolidate(
        threshold=0.85,
        cluster=True,
        content_strategy="concat",
        project_scoped=False,
    )
    assert result["consolidated"] == 2

    survivor_row = conn.execute("SELECT content FROM memories WHERE content_hash = ?", (seed[0],)).fetchone()
    content = survivor_row["content"]
    assert "Related (merged):" in content
    assert "Grafana" in content or "Service account" in content


def test_consolidate_cluster_keep_longer(store):
    short_text = "kafka short note"
    long_text = "kafka topic partitions throughput per-broker tuning long detailed analysis with examples"
    hashes = _seed_similar_cluster(store, [short_text, long_text])
    result = store.consolidate(
        threshold=0.85,
        cluster=True,
        content_strategy="keep_longer",
        project_scoped=False,
    )
    assert result["consolidated"] == 1

    conn = store._get_conn()
    surviving = conn.execute(
        "SELECT content FROM memories WHERE content_hash IN (?, ?) AND deleted_at IS NULL",
        tuple(hashes),
    ).fetchall()
    assert len(surviving) == 1
    assert surviving[0]["content"] == long_text


def test_find_clusters_splits_oversized_groups(store):
    contents = [f"kubernetes pod variant {i}" for i in range(12)]
    _seed_similar_cluster(store, contents)
    out = store.find_clusters(threshold=0.85, project_scoped=False, max_cluster_size=10, min_cluster_size=2)
    sizes = sorted(c["size"] for c in out["clusters"])
    assert sizes == [2, 10]


def test_consolidate_cluster_mmr_union_preserves_unique_facts(store):
    """MMR strategy must keep unique-fact sentences from all members."""
    seed = _seed_similar_cluster(
        store,
        [
            "OnCall API: SA token cannot resolve alert groups; returns 403. "
            "Affects oncall provider in Terraform.",
            "Grafana SA token cannot silence OnCall alerts via API. "
            "Use admin role instead. Workaround: rotate token monthly.",
            "Service account tokens lack permission for resolve. "
            "Critical for SRE on-call rotation runbook.",
        ],
    )
    conn = store._get_conn()
    conn.execute("UPDATE memories SET recall_count = 5 WHERE content_hash = ?", (seed[0],))
    conn.commit()

    result = store.consolidate(
        threshold=0.85, cluster=True, content_strategy="mmr_union", project_scoped=False
    )
    assert result["consolidated"] == 2

    survivor = conn.execute(
        "SELECT content FROM memories WHERE content_hash = ?", (seed[0],)
    ).fetchone()
    content = survivor["content"].lower()
    # Unique facts from each member must survive (matched by keyword).
    assert "403" in content or "resolve" in content  # member 1 fact
    # At least one of the unique elements from member 2 or 3 must appear.
    member_2_fragments = ["silence", "admin role", "rotate"]
    member_3_fragments = ["sre", "runbook", "rotation"]
    found_other = any(f in content for f in member_2_fragments + member_3_fragments)
    assert found_other, f"No unique fact from members 2/3 survived. Content: {content[:300]}"


def test_rrsb_fuse_returns_score():
    from memory.core import _rrsb_fuse

    sem = [{"content_hash": "a", "score": 0.9}, {"content_hash": "b", "score": 0.5}]
    fts = [{"content_hash": "b", "score": 10.0}, {"content_hash": "c", "score": 5.0}]
    out = _rrsb_fuse(sem, fts, limit=3)
    assert len(out) == 3
    for r in out:
        assert "rrsb_score" in r
        assert r["score"] == r["rrsb_score"]
    assert out[0]["content_hash"] == "b"


# --- CLI: memory admin clusters --depth {summary,full} ---


def _run_admin_clusters_cli(store, extra_args: list[str] | None = None) -> str:
    """Invoke `memory admin clusters` against the given store's DB. Returns stdout."""
    from memory import cli as memory_cli

    args = ["admin", "clusters", "--threshold", "0.85"]
    if extra_args:
        args.extend(extra_args)
    buf = io.StringIO()
    with (
        patch("memory.core.DB_PATH", store.db_path),
        patch("memory.cli.MemoryStore", lambda: store),
        patch("sys.stdout", buf),
    ):
        memory_cli.main(args)
    return buf.getvalue()


def test_cli_clusters_depth_full_returns_json(store):
    """--depth full preserves the original JSON dump shape from find_clusters()."""
    _seed_similar_cluster(
        store,
        ["alpha topic one", "alpha topic two", "alpha topic three"],
    )
    raw = _run_admin_clusters_cli(store, ["--depth", "full"])
    data = json.loads(raw)
    assert data["threshold"] == 0.85
    assert data["n_clusters"] == 1
    cluster = data["clusters"][0]
    assert cluster["size"] == 3
    assert "survivor_hash" in cluster
    assert "members" in cluster
    member = cluster["members"][0]
    assert "content_hash" in member
    assert "content_preview" in member
    assert "recall_count" in member


def test_cli_clusters_depth_summary_one_line_per_cluster(store):
    """--depth summary emits one plain-text line per cluster, not JSON."""
    _seed_similar_cluster(
        store,
        ["beta topic one", "beta topic two", "beta topic three"],
    )
    _seed_similar_cluster(
        store,
        ["gamma topic one", "gamma topic two"],
    )
    raw = _run_admin_clusters_cli(store, ["--depth", "summary"])
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    assert len(lines) == 2
    # Lines must NOT be JSON.
    for ln in lines:
        assert not ln.lstrip().startswith("{")
        assert not ln.lstrip().startswith("[")
    # Each line: <cluster_id> size=<N> survivor=<hash12> "..."
    line_ids = []
    for ln in lines:
        parts = ln.split(" ", 3)
        assert len(parts) >= 3
        cluster_id, size_field, survivor_field = parts[0], parts[1], parts[2]
        line_ids.append(cluster_id)
        assert size_field.startswith("size=")
        assert survivor_field.startswith("survivor=")
        survivor_prefix = survivor_field.split("=", 1)[1]
        assert len(survivor_prefix) == 12
        # size value must parse as int >= 2
        size_val = int(size_field.split("=", 1)[1])
        assert size_val >= 2
    # Cluster ids are distinct and the first line starts with "1"
    assert line_ids[0] == "1"
    assert len(set(line_ids)) == len(line_ids)


def test_cli_clusters_default_depth_is_summary(store):
    """No --depth flag falls back to summary output (plain text, not JSON)."""
    _seed_similar_cluster(
        store,
        ["delta topic one", "delta topic two"],
    )
    raw = _run_admin_clusters_cli(store)
    stripped = raw.strip()
    assert stripped, "expected non-empty summary output"
    # Must not be JSON.
    assert not stripped.startswith("{")
    assert not stripped.startswith("[")
    first_line = stripped.splitlines()[0]
    parts = first_line.split(" ", 3)
    assert parts[0] == "1"
    assert parts[1].startswith("size=")
    assert parts[2].startswith("survivor=")
