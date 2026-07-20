"""Tests for recall-time staleness flagging."""

import subprocess

import pytest

from memory.provenance import (
    annotate,
    deleted_tree,
    extract_paths,
    head_tree,
    project_of,
    repo_context,
    stale_refs,
)


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _clear_caches():
    head_tree.cache_clear()
    deleted_tree.cache_clear()
    repo_context.cache_clear()


@pytest.fixture
def repo(tmp_path):
    """A repo exercising all three ways a cited path can go missing.

    At HEAD:  terraform/live/keep.tf, scripts/keep.py
    Deleted:  terraform/live/main.tf      (plain delete)
              terraform/gone/old.tf       (delete taking its directory with it)
    Renamed:  scripts/moved.py -> scripts/renamed.py
    """
    r = tmp_path / "myproject"
    (r / "terraform" / "live").mkdir(parents=True)
    (r / "terraform" / "gone").mkdir(parents=True)
    (r / "scripts").mkdir()
    _git(r.parent, "init", "myproject")
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")

    (r / "terraform" / "live" / "main.tf").write_text("resource {}\n")
    (r / "terraform" / "live" / "keep.tf").write_text("resource {}\n")
    (r / "terraform" / "gone" / "old.tf").write_text("resource {}\n")
    (r / "scripts" / "keep.py").write_text("x = 1\n")
    (r / "scripts" / "moved.py").write_text("y = 2\n" * 40)
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "init")

    (r / "terraform" / "live" / "main.tf").unlink()
    (r / "terraform" / "gone" / "old.tf").unlink()
    (r / "scripts" / "moved.py").rename(r / "scripts" / "renamed.py")
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "delete and rename")

    _clear_caches()
    yield r
    _clear_caches()


@pytest.fixture
def sets(repo):
    return head_tree(str(repo)), deleted_tree(str(repo))


# --- extraction ------------------------------------------------------------


def test_extracts_plain_and_backticked_paths():
    got = extract_paths("see `terraform/live/main.tf` and scripts/keep.py now")
    assert "terraform/live/main.tf" in got
    assert "scripts/keep.py" in got


@pytest.mark.parametrize(
    "text,forbidden",
    [
        ("https://github.com/foo/bar", "https://github.com/foo/bar"),
        ("ratio was 3/4 overall", "3/4"),
        ("upgraded to v1.2 today", "v1.2"),
        ("costs $5/mo at rest", "5/mo"),
        ("throughput 10/day sustained", "10/day"),
        ("elided as terraform/.../monitoring", "terraform/.../monitoring"),
        ("relative ../sibling/file.tf", "../sibling/file.tf"),
    ],
)
def test_rejects_non_paths(text, forbidden):
    """Precision guard: a flag that cries wolf trains agents to ignore every flag."""
    assert forbidden not in extract_paths(text)


def test_repo_qualified_reference_is_not_read_as_ours():
    """`acme/frontend-2.0:.github/workflows/main.yml` names another repo's file.

    Without the leading-colon guard the tail parses as a path in *our* repo and
    gets flagged as missing — 3 real false positives in the corpus came from this.
    """
    assert ".github/workflows/main.yml" not in extract_paths(
        "deploy via acme/frontend-2.0:.github/workflows/main.yml today"
    )


def test_extract_handles_empty():
    assert extract_paths("") == set()
    assert extract_paths(None) == set()


# --- staleness -------------------------------------------------------------


def test_flags_deleted_path(sets):
    tree, deleted = sets
    assert stale_refs("edit terraform/live/main.tf to fix it", tree, deleted) == ["terraform/live/main.tf"]


def test_flags_renamed_away_path(sets):
    """The old name of a renamed file is exactly the rot worth catching.

    Guards `--no-renames` in deleted_tree(): with git's default rename detection
    the move is classified R, not D, and this path silently stops being flagged.
    Removing that flag drops corpus recall from 0.96 to 0.71.
    """
    tree, deleted = sets
    assert stale_refs("run scripts/moved.py nightly", tree, deleted) == ["scripts/moved.py"]


def test_flags_directory_that_lost_all_its_files(sets):
    """git records file deletions only; a cited directory needs prefix expansion."""
    tree, deleted = sets
    assert stale_refs("the layer at terraform/gone is where", tree, deleted) == ["terraform/gone"]


def test_does_not_flag_live_path(sets):
    tree, deleted = sets
    assert stale_refs("run scripts/keep.py first", tree, deleted) == []


def test_does_not_flag_live_directory(sets):
    tree, deleted = sets
    assert stale_refs("everything under terraform/live is fine", tree, deleted) == []


def test_does_not_flag_path_git_never_tracked(sets):
    """The property that took corpus precision from 0.70 to 1.00.

    Memories legitimately cite other repos' files and paths that were only ever
    proposed in a design note. Absent-from-HEAD is not rot; absent-from-HEAD
    *and* once-tracked is.
    """
    tree, deleted = sets
    assert stale_refs("see dist/index.js in the app repo", tree, deleted) == []
    assert stale_refs("propose terraform/future/new.tf", tree, deleted) == []


def test_empty_tree_flags_nothing():
    assert stale_refs("terraform/live/main.tf", frozenset(), frozenset()) == []


# --- repo context ----------------------------------------------------------


def test_repo_context_resolves_project_name(repo):
    ctx = repo_context(str(repo))
    assert ctx is not None
    assert ctx[0] == "myproject"


def test_repo_context_outside_git_returns_none(tmp_path):
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    _clear_caches()
    assert repo_context(str(outside)) is None


def test_head_tree_on_non_repo_is_empty(tmp_path):
    outside = tmp_path / "bare"
    outside.mkdir()
    _clear_caches()
    assert head_tree(str(outside)) == frozenset()


def test_deleted_tree_on_non_repo_is_empty(tmp_path):
    outside = tmp_path / "bare2"
    outside.mkdir()
    _clear_caches()
    assert deleted_tree(str(outside)) == frozenset()


def test_project_of():
    assert project_of(["a", "project:foo", "b"]) == "foo"
    assert project_of(["a", "b"]) is None
    assert project_of(None) is None
    assert project_of([None, 3, "project:x"]) == "x"


# --- annotate --------------------------------------------------------------


def test_annotate_flags_only_matching_project(repo):
    results = [
        {"content": "fix terraform/live/main.tf", "tags": ["project:myproject"]},
        {"content": "fix terraform/live/main.tf", "tags": ["project:other"]},
        {"content": "run scripts/keep.py", "tags": ["project:myproject"]},
    ]
    out = annotate(results, cwd=str(repo))
    assert out[0]["stale_refs"] == ["terraform/live/main.tf"]
    assert "stale_refs" not in out[1], "must not judge another repo's paths"
    assert "stale_refs" not in out[2], "clean hits stay byte-identical"


def test_annotate_outside_repo_is_noop(tmp_path):
    outside = tmp_path / "nowhere"
    outside.mkdir()
    _clear_caches()
    results = [{"content": "terraform/live/main.tf", "tags": ["project:x"]}]
    assert annotate(results, cwd=str(outside)) == results


def test_annotate_tolerates_odd_shapes(repo):
    assert annotate({"error": "nope"}, cwd=str(repo)) == {"error": "nope"}
    assert annotate([], cwd=str(repo)) == []
    annotate(["junk", {"content": "x", "tags": []}], cwd=str(repo))


def test_annotate_untagged_memory_is_untouched(repo):
    results = [{"content": "fix terraform/live/main.tf", "tags": []}]
    assert "stale_refs" not in annotate(results, cwd=str(repo))[0]
