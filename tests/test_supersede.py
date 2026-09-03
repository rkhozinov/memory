"""An update must not be swallowed as a duplicate.

Semantic dedup at 0.90 rejected 624 stores in production, 519 of them in the
0.90-0.95 band.  A fact that supersedes another is near-identical to it by
construction — same subject, same tags, one value changed — so it clears the
cosine bar and is discarded.  The discriminator is value-token divergence: a
rephrasing preserves versions, settings and symbol names; an update changes one.
"""

from memory.core import _value_tokens, differs_by_value


def test_value_tokens_picks_up_versions_and_symbols():
    assert _value_tokens("CI runners are on ubuntu-22.04") == {"ubuntu-22.04"}
    assert "mmr_union" in _value_tokens("the default strategy is mmr_union")
    assert "0.92" in _value_tokens("dream runs at threshold 0.92")


def test_pure_rephrasing_is_not_an_update():
    a = "Kubernetes pods run in clusters"
    b = "Kubernetes pods run in clusters for container orchestration"
    assert differs_by_value(a, b) is False


def test_changed_version_is_an_update():
    a = "CI runners are on ubuntu-22.04"
    b = "CI runners are on ubuntu-24.04"
    assert differs_by_value(a, b) is True


def test_changed_setting_name_is_an_update():
    a = "consolidation default is keep_higher_recall"
    b = "consolidation default is mmr_union"
    assert differs_by_value(a, b) is True


def test_no_value_tokens_either_side_is_not_an_update():
    assert differs_by_value("we prefer boring code", "boring code is preferred") is False


def test_store_keeps_the_updated_fact_instead_of_dropping_it(store):
    """The regression this whole change exists for."""
    first = store.store(
        "CI runners are pinned to ubuntu-22.04 for the build workflow",
        memory_type="reference",
        tags=["project:x"],
    )
    assert first["status"] == "stored"

    second = store.store(
        "CI runners are pinned to ubuntu-24.04 for the build workflow",
        memory_type="reference",
        tags=["project:x"],
        dedup_threshold=0.90,
    )
    assert second["status"] == "stored", "the newer fact must not be discarded"
    assert second.get("supersedes_candidate") == first["content_hash"]

    hits = store.search("ubuntu-24.04 runners", mode="fts", limit=10)
    assert any("24.04" in h["content"] for h in hits)


def test_true_restatement_is_still_deduped(store):
    store.store(
        "The nightly dream cycle consolidates near-duplicate memories",
        memory_type="reference",
        tags=["project:x"],
    )
    again = store.store(
        "The nightly dream cycle consolidates near-duplicate memories",
        memory_type="reference",
        tags=["project:x"],
        dedup_threshold=0.90,
    )
    assert again["status"] == "duplicate"


def test_exact_hash_duplicate_is_untouched_by_the_discriminator(store):
    """Byte-identical content stays a duplicate even though value tokens match
    trivially — 171 production rejections came through this path and all of them
    are correct."""
    content = "Release 1.2.3 shipped on 2026-01-01 with 4 fixes"
    store.store(content, memory_type="reference")
    dup = store.store(content, memory_type="reference", dedup_threshold=0.90)
    assert dup["status"] == "duplicate"
