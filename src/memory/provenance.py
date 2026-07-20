"""Recall-time staleness flagging for memories that cite repo paths.

A stored memory is a snapshot of a moment. The repo it describes keeps moving.
Measured on a real 3153-memory corpus: 66 memories cite a path that git once had
and HEAD no longer does, and every one of them had been recalled at least once —
including one at 65 recalls pointing at a directory that no longer exists.

This module annotates search hits with the paths they cite that are gone from
HEAD. It flags, it never deletes: a path can vanish to a rename while the insight
attached to it survives intact.

Validated against git-verified ground truth on that corpus: precision 1.00,
recall 0.96 (65 true positives, 0 false positives, 3 misses). The misses are
paths that only ever existed on unmerged branches — deliberately not flagged,
since precision is what keeps the annotation worth reading.

Everything here is best-effort. No git repo, no git binary, a bare repo, a
detached HEAD with no tree — all degrade to "nothing flagged". A staleness hint
is never worth failing a search over.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache

# A path-ish token: two or more slash-separated segments. No extension list is
# needed — dots are already legal inside a segment, so `a/b/main.tf` matches as
# plainly as `a/b`. Whether a token is really a repo path is settled downstream
# by git, not by guessing from its spelling.
#
# The leading `:` exclusion is load-bearing: memories cite other repos as
# `acme/frontend-2.0:.github/workflows/main.yml`, and without it the tail
# parses as a path in *our* repo.
PATH_RE = re.compile(r"(?<![\w./:-])((?:[\w.-]+/){1,}[\w.-]*)(?![\w/])")

# Tokens the regex catches that are never repo paths.
REJECT_RE = re.compile(
    r"^(?:https?|s3|gs|git@|ssh|ftp)"  # URLs / remotes
    r"|^\d+/\d+$"  # ratios: 3/4, 76/1411
    r"|^v?\d+\.\d+"  # versions: v1.2, 1.0.14
    r"|/(?:day|days|week|weeks|mo|month|months|yr|year|s|sec|secs|min|mins|hr|hrs|h|m)$"  # rates: 10/day, $5/mo
    r"|^\d+$",
    re.IGNORECASE,
)


def extract_paths(content: str) -> set[str]:
    """Pull candidate repo paths out of free-form memory text."""
    if not content:
        return set()
    out = set()
    for raw in PATH_RE.findall(content):
        cand = raw.strip().rstrip(".,;:)]}'\"").rstrip("/")
        if not cand or "/" not in cand:
            continue
        if REJECT_RE.search(cand):
            continue
        # Elided paths ("terraform/.../monitoring") are prose shorthand, not a
        # claim that this exact path exists. Nothing can be verified about them.
        if any(seg == "..." or seg == ".." for seg in cand.split("/")):
            continue
        out.add(cand)
    return out


def _run_git(repo_root: str, *args: str) -> str | None:
    """Run a git command, returning stdout, or None on any failure."""
    try:
        r = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


@lru_cache(maxsize=8)
def head_tree(repo_root: str) -> frozenset[str]:
    """Every path tracked at HEAD, plus every directory prefix of each.

    Cached per process: a single search must not shell out repeatedly, and a
    repo does not change underneath one CLI invocation.
    """
    out = _run_git(repo_root, "ls-tree", "-r", "--name-only", "HEAD")
    if not out:
        return frozenset()
    paths: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        paths.add(line)
        parts = line.split("/")
        for i in range(1, len(parts)):
            paths.add("/".join(parts[:i]))
    return frozenset(paths)


@lru_cache(maxsize=8)
def repo_context(cwd: str) -> tuple[str, str] | None:
    """Resolve cwd to (project_name, repo_root), or None outside a git repo.

    Zero config by design: the project tag we match against is just the repo
    directory name, which is the same convention `/remember` already uses when
    it stamps `project:$(basename $(pwd))`.
    """
    out = _run_git(cwd, "rev-parse", "--show-toplevel")
    if not out:
        return None
    root = out.strip()
    if not root:
        return None
    return os.path.basename(root), root


@lru_cache(maxsize=8)
def deleted_tree(repo_root: str) -> frozenset[str]:
    """Every path git ever deleted, plus every directory prefix of each.

    One history walk, cached per process. Measured at ~80ms across the full
    history of a large, high-churn repo — cheap enough to consult on every
    search, which is what makes the authoritative check affordable here.

    Directory prefixes matter because git only records file deletions: a memory
    citing a removed *directory* (`terraform/aws/acme-main/org`) would never
    match a raw deletion record.
    """
    # --no-renames is load-bearing. With rename detection on (the default when no
    # pathspec is given), a file moved to a new location is classified R, not D,
    # so its old path never appears here — and a memory citing that old path is
    # exactly the kind of rot worth flagging. Measured: recall 0.71 -> 0.96.
    out = _run_git(
        repo_root,
        "log",
        "--all",
        "--no-renames",
        "--diff-filter=D",
        "--name-only",
        "--format=",
    )
    if not out:
        return frozenset()
    paths: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        paths.add(line)
        parts = line.split("/")
        for i in range(1, len(parts)):
            paths.add("/".join(parts[:i]))
    return frozenset(paths)


def stale_refs(content: str, tree: frozenset[str], deleted: frozenset[str]) -> list[str]:
    """Paths cited by `content` that git once tracked and HEAD no longer has.

    Both conditions are required. "Absent from HEAD" alone is not rot — memories
    legitimately cite other repos' files, and agent reports cite paths that were
    only ever proposed. Requiring git to have actually tracked the path is what
    separates a decayed fact from a reference to somewhere else.
    """
    if not tree:
        return []
    return sorted(c for c in extract_paths(content) if c not in tree and c in deleted)


def project_of(tags) -> str | None:
    """The `project:<name>` tag on a memory, if it carries one."""
    for t in tags or []:
        if isinstance(t, str) and t.startswith("project:"):
            return t.split(":", 1)[1]
    return None


def annotate(results, cwd: str | None = None):
    """Add `stale_refs` to each hit whose cited paths are gone from HEAD.

    Only memories tagged for the *current* repo are checked — we can only verify
    paths against a working tree we actually have. Hits with nothing stale are
    returned untouched, so clean output is byte-identical to before.
    """
    if not isinstance(results, list):
        return results
    ctx = repo_context(cwd or os.getcwd())
    if not ctx:
        return results
    project, root = ctx
    tree = head_tree(root)
    if not tree:
        return results
    deleted = deleted_tree(root)
    if not deleted:
        return results

    for hit in results:
        if not isinstance(hit, dict):
            continue
        if project_of(hit.get("tags")) != project:
            continue
        stale = stale_refs(hit.get("content") or "", tree, deleted)
        if stale:
            hit["stale_refs"] = stale
    return results
