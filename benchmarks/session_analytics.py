#!/usr/bin/env python3
"""Does the memory plugin actually help in real sessions?

Every quality number in this repo up to now came from a replay harness: a query
set, a gold answer, an MRR. That measures the ranker. It does not measure whether
the thing the ranker surfaced was of any use to the agent that received it.

This script measures the second thing, from two read-only sources:

  operation_events   in the memory DB. Every store/search/dream call since
                     2026-02, with duration, result count and dedup flags. Needs
                     no transcripts and covers the full history.
  ~/.claude/projects Claude Code session transcripts. The UserPromptSubmit hook's
                     stdout is recorded verbatim, so each injected memory is
                     recoverable with its hash and its retrieval score.

Five measures:

  1. mechanical      store/search rates, zero-result rate, latency (DB only)
  2. injection use   did the assistant's later text reuse the injected content?
  3. calibration     reuse rate bucketed by injected score — is --min-score 0.5
                     cutting in the right place?
  4. coverage        what gets written that is never read back
  5. surface         how the plugin is actually invoked (CLI vs skill vs hook)

Read-only throughout: the DB is copied to a temp file before it is opened, and
transcripts are only ever read.

Usage:
  uv run python benchmarks/session_analytics.py
  uv run python benchmarks/session_analytics.py --json /tmp/analytics.json
  uv run python benchmarks/session_analytics.py --max-files 50   # smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# One injected memory line: "<16 hex> [type] score=0.57 <preview>"
INJECTED_LINE_RE = re.compile(r"^([0-9a-f]{16}) \[(\w+)\] score=([0-9.]+) (.*)$")

# A token distinctive enough that its reappearance is unlikely to be chance.
# Eight characters excludes almost all ordinary English words while keeping
# identifiers, paths, flags and compound technical terms.
TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]{8,}")

# Words that clear the length bar but carry no attribution signal — they show up
# in any technical conversation regardless of what was injected.
STOPWORDS = frozenset(
    [
        "actually",
        "additional",
        "available",
        "basically",
        "configuration",
        "containing",
        "currently",
        "different",
        "essentially",
        "everything",
        "following",
        "implementation",
        "important",
        "including",
        "information",
        "interesting",
        "otherwise",
        "particular",
        "probably",
        "regarding",
        "representing",
        "something",
        "specifically",
        "therefore",
        "understand",
        "understanding",
    ]
)

MEMORY_CLI_RE = re.compile(r"\bmemory\s+(store|add|search|find|get|delete|rm|forget|update|doc|admin|health)\b")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. Mechanical rates — DB only, full history
# ---------------------------------------------------------------------------


def mechanical_rates(db: Path) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    ops = [
        dict(r)
        for r in conn.execute(
            """
            SELECT operation,
                   COUNT(*)                                   AS n,
                   AVG(result_count)                          AS avg_results,
                   AVG(duration_ms)                           AS avg_ms,
                   SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_results,
                   AVG(top_similarity)                        AS avg_top_sim
            FROM operation_events
            GROUP BY operation
            ORDER BY n DESC
            """
        )
    ]
    span = conn.execute(
        "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi, COUNT(*) AS n FROM operation_events"
    ).fetchone()

    searches = conn.execute(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero "
        "FROM operation_events WHERE operation IN ('search', 'doc_search')"
    ).fetchone()

    # The raw zero-result rate is not a defect rate, and reporting it as one is
    # actively misleading. It decomposes into four things with nothing in common:
    #
    #   empty query    the `query` column is blank. 100% of these return zero by
    #                  construction. Almost all are from a March-May 2026 window
    #                  that has since stopped; a logging or caller artifact, not
    #                  retrieval.
    #   exact mode     a lookup that missed. Returning nothing is the correct
    #                  answer, not a failure.
    #   slash command  a `/command` that reached search at all. The hook skips
    #                  these, so these come from some other caller. Wasted work,
    #                  and the one genuinely actionable slice here.
    #   real miss      a substantive query that found nothing. This is the only
    #                  number that behaves like a defect rate.
    zero_split = conn.execute(
        """
        SELECT
          COUNT(*)                                                          AS total_zero,
          SUM(CASE WHEN query IS NULL OR query = '' THEN 1 ELSE 0 END)      AS empty_query,
          SUM(CASE WHEN query LIKE '/%' THEN 1 ELSE 0 END)                  AS slash_command,
          SUM(CASE WHEN search_mode = 'exact' AND query NOT LIKE '/%'
                   AND query IS NOT NULL AND query <> '' THEN 1 ELSE 0 END) AS exact_miss
        FROM operation_events
        WHERE operation IN ('search', 'doc_search') AND result_count = 0
        """
    ).fetchone()

    by_mode = [
        dict(r)
        for r in conn.execute(
            """
            SELECT search_mode,
                   COUNT(*) AS n,
                   SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero
            FROM operation_events WHERE operation IN ('search', 'doc_search')
            GROUP BY search_mode ORDER BY n DESC
            """
        )
    ]
    for row in by_mode:
        row["zero_pct"] = row["zero"] / row["n"] if row["n"] else 0.0

    # Repeated identical queries are eval-harness fixtures, not user traffic.
    # Reporting the ratio keeps them from being mistaken for a real miss rate.
    repeats = conn.execute(
        """
        SELECT COUNT(*) AS n, COUNT(DISTINCT query) AS distinct_q
        FROM operation_events
        WHERE operation IN ('search', 'doc_search') AND result_count = 0
          AND query IS NOT NULL AND query <> ''
        """
    ).fetchone()

    dedup = conn.execute(
        "SELECT SUM(CASE WHEN dedup_used = 1 THEN 1 ELSE 0 END) AS used, "
        "       SUM(CASE WHEN duplicate_detected = 1 THEN 1 ELSE 0 END) AS caught, "
        "       COUNT(*) AS n FROM operation_events WHERE operation IN ('store', 'doc_store')"
    ).fetchone()

    conn.close()

    def _ts(v):
        try:
            return datetime.fromtimestamp(float(v), UTC).date().isoformat()
        except (TypeError, ValueError):
            return str(v)

    return {
        "events_total": span["n"],
        "span": {"from": _ts(span["lo"]), "to": _ts(span["hi"])},
        "by_operation": ops,
        "search_zero_result_rate": (searches["zero"] or 0) / searches["n"] if searches["n"] else None,
        "search_total": searches["n"],
        "zero_result_split": dict(zero_split),
        "zero_result_by_mode": by_mode,
        "zero_result_query_repetition": {
            "n": repeats["n"],
            "distinct": repeats["distinct_q"],
            "repeat_ratio": 1 - (repeats["distinct_q"] / repeats["n"]) if repeats["n"] else 0.0,
        },
        "store_total": dedup["n"],
        "store_dedup_used": dedup["used"] or 0,
        "store_duplicate_caught": dedup["caught"] or 0,
    }


# ---------------------------------------------------------------------------
# 4. Write/read coverage — DB only
# ---------------------------------------------------------------------------


def coverage(db: Path) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    live = "deleted_at IS NULL"

    total = conn.execute(f"SELECT COUNT(*) AS n FROM memories WHERE {live}").fetchone()["n"]
    never = conn.execute(
        f"SELECT COUNT(*) AS n FROM memories WHERE {live} AND COALESCE(recall_count,0) = 0"
    ).fetchone()["n"]

    by_type = [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT memory_type,
                   COUNT(*) AS n,
                   SUM(CASE WHEN COALESCE(recall_count,0) = 0 THEN 1 ELSE 0 END) AS never_recalled,
                   AVG(COALESCE(recall_count,0)) AS avg_recalls
            FROM memories WHERE {live}
            GROUP BY memory_type ORDER BY n DESC
            """
        )
    ]
    for row in by_type:
        row["never_recalled_pct"] = row["never_recalled"] / row["n"] if row["n"] else 0.0

    # Age is the obvious confounder: a memory written yesterday has had no chance
    # to be recalled. Split it out rather than letting it contaminate the total.
    by_age = [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT CASE
                     WHEN (strftime('%s','now') - created_at) < 604800  THEN '0-7d'
                     WHEN (strftime('%s','now') - created_at) < 2592000 THEN '7-30d'
                     WHEN (strftime('%s','now') - created_at) < 7776000 THEN '30-90d'
                     ELSE '90d+'
                   END AS age_bucket,
                   COUNT(*) AS n,
                   SUM(CASE WHEN COALESCE(recall_count,0) = 0 THEN 1 ELSE 0 END) AS never_recalled
            FROM memories WHERE {live}
            GROUP BY age_bucket
            """
        )
    ]
    for row in by_age:
        row["never_recalled_pct"] = row["never_recalled"] / row["n"] if row["n"] else 0.0

    # Tag prefixes on never-recalled memories: what class of thing is dead weight?
    dead_tags: Counter = Counter()
    for (tags_json,) in conn.execute(f"SELECT tags FROM memories WHERE {live} AND COALESCE(recall_count,0) = 0"):
        try:
            for t in json.loads(tags_json or "[]"):
                dead_tags[t.split(":")[0] if ":" in t else t] += 1
        except (json.JSONDecodeError, AttributeError):
            continue

    # recall_count is only trustworthy if it accumulated organically. A single day
    # holding a large share of all last_recalled_at values means something swept the
    # corpus with tracking on — a migration, a briefing, a benchmark run — and every
    # memory it touched now looks "recalled" whether or not anyone read it.
    bulk = [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT date(last_recalled_at, 'unixepoch') AS day, COUNT(*) AS n
            FROM memories WHERE {live} AND last_recalled_at IS NOT NULL
            GROUP BY day HAVING n >= 100 ORDER BY n DESC
            """
        )
    ]

    conn.close()
    return {
        "bulk_recall_days": bulk,
        "bulk_recall_contaminated": sum(b["n"] for b in bulk),
        "memories_live": total,
        "never_recalled": never,
        "never_recalled_pct": never / total if total else 0.0,
        "by_type": by_type,
        "by_age": by_age,
        "never_recalled_tag_prefixes": dead_tags.most_common(15),
    }


# ---------------------------------------------------------------------------
# Transcript scan (feeds measures 2, 3, 5)
# ---------------------------------------------------------------------------


@dataclass
class Injection:
    session: str
    project: str
    timestamp: str
    duration_ms: int | None
    memories: list[dict] = field(default_factory=list)  # hash, type, score, preview
    # The prompt that triggered this injection. Needed to reconstruct the
    # candidate pool the hook chose from — the transcript records only the five
    # lines it picked, never the pool.
    prompt: str = ""


@dataclass
class SessionScan:
    session: str
    project: str
    injections: list[Injection] = field(default_factory=list)
    # (timestamp, tokens) for everything the agent read from a source that is not
    # the injection: the user's own words and tool results.
    context_turns: list[tuple[str, set[str]]] = field(default_factory=list)
    # (timestamp, tokens) for assistant text, so we can gate on ordering
    assistant_turns: list[tuple[str, set[str]]] = field(default_factory=list)
    cli_calls: Counter = field(default_factory=Counter)
    skill_calls: Counter = field(default_factory=Counter)


def _tokens(text: str) -> set[str]:
    out = set()
    for m in TOKEN_RE.finditer(text or ""):
        t = m.group(0).lower().strip("./-_")
        if len(t) < 8 or t.isdigit() or t in STOPWORDS:
            continue
        out.add(t)
    return out


def _parse_injection(stdout: str) -> list[dict]:
    mems = []
    for line in stdout.splitlines():
        m = INJECTED_LINE_RE.match(line.strip())
        if m:
            mems.append(
                {
                    "hash": m.group(1),
                    "type": m.group(2),
                    "score": float(m.group(3)),
                    "preview": m.group(4),
                }
            )
    return mems


def scan_transcript(path: Path) -> SessionScan | None:
    """One pass over a session file. Returns None if it holds nothing of interest."""
    project = path.parent.name
    scan = SessionScan(session=path.stem, project=project)
    saw_anything = False
    last_prompt = ""

    with path.open(errors="replace") as fh:
        for line in fh:
            # Cheap prefilter: most lines are irrelevant and json.loads is not free.
            if not (
                "memory_context" in line
                or '"assistant"' in line
                or '"user"' in line
                or "memory:" in line
                or "memory " in line
            ):
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue

            rtype = rec.get("type")
            ts = rec.get("timestamp", "")

            if rtype == "attachment":
                _att = rec.get("attachment") or {}
                if _att.get("prompt"):
                    last_prompt = str(_att["prompt"])
                att = rec.get("attachment") or {}
                stdout = att.get("stdout") or ""
                if str(att.get("type", "")).startswith("hook") and "<memory_context" in stdout:
                    mems = _parse_injection(stdout)
                    if mems:
                        scan.injections.append(
                            Injection(
                                session=scan.session,
                                project=project,
                                timestamp=ts,
                                duration_ms=att.get("durationMs"),
                                memories=mems,
                                prompt=last_prompt,
                            )
                        )
                        saw_anything = True
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")

            if rtype == "user":
                # Everything the user typed, and every tool result the agent read,
                # is a legitimate alternative source for any token. Subtracting it
                # is what separates "used the memory" from "coincidence". Keep the
                # timestamp: whether a source counts depends on whether it arrived
                # before or after the injection.
                toks: set[str] = set()
                if isinstance(content, str):
                    last_prompt = content
                    toks |= _tokens(content)
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text":
                            toks |= _tokens(b.get("text", ""))
                        elif b.get("type") == "tool_result":
                            c = b.get("content")
                            if isinstance(c, str):
                                toks |= _tokens(c)
                            elif isinstance(c, list):
                                for sub in c:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        toks |= _tokens(sub.get("text", ""))
                if toks:
                    scan.context_turns.append((ts, toks))
                continue

            if rtype == "assistant" and isinstance(content, list):
                text_parts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type")
                    if btype == "text":
                        text_parts.append(b.get("text", ""))
                    elif btype == "tool_use":
                        name = b.get("name", "")
                        inp = b.get("input") or {}
                        if name == "Skill" and str(inp.get("skill", "")).startswith("memory:"):
                            scan.skill_calls[inp["skill"]] += 1
                            saw_anything = True
                        elif name == "Bash":
                            cmd = str(inp.get("command", ""))
                            for hit in MEMORY_CLI_RE.finditer(cmd):
                                scan.cli_calls[hit.group(1)] += 1
                                saw_anything = True
                if text_parts:
                    scan.assistant_turns.append((ts, _tokens("\n".join(text_parts))))

    return scan if saw_anything else None


# ---------------------------------------------------------------------------
# 2 + 3. Injection usefulness and score calibration
# ---------------------------------------------------------------------------


def _reuse_for_injection(scan: SessionScan, inj: Injection) -> list[dict]:
    """Per injected memory: how much of its distinctive vocabulary reappears in
    assistant text written *after* the injection and not attributable elsewhere.

    Two controls, deliberately reported side by side because they bracket the
    truth rather than pin it:

      ordered  subtract only what the agent had already seen when the injection
               landed. A file it read afterwards may well have been read
               *because* of the memory, so subtracting that would erase real
               credit. This is the upper bound.
      strict   subtract everything the agent ever saw in the session from any
               other source. Erases genuine credit whenever the memory and a
               later file happen to share vocabulary. This is the lower bound.

    The honest number is somewhere between them, and how far apart they are is
    itself worth knowing.
    """
    later: set[str] = set()
    for ts, toks in scan.assistant_turns:
        if ts > inj.timestamp:
            later |= toks

    before: set[str] = set()
    all_context: set[str] = set()
    for ts, toks in scan.context_turns:
        all_context |= toks
        if ts <= inj.timestamp:
            before |= toks

    out = []
    for mem in inj.memories:
        raw = _tokens(mem["preview"])
        ordered = raw - before
        strict = raw - all_context
        # A memory counts as "used" only if at least two of its distinctive tokens
        # reappear. One is too easy to hit by chance on a shared technical
        # vocabulary; two co-occurring is meaningfully harder.
        out.append(
            {
                "hash": mem["hash"],
                "type": mem["type"],
                "score": mem["score"],
                "tokens": len(ordered),
                "tokens_strict": len(strict),
                "reused": len(ordered & later),
                "reused_strict": len(strict & later),
                "used": len(ordered & later) >= 2,
                "used_strict": len(strict & later) >= 2,
                "attributable": len(ordered) > 0,
            }
        )
    return out


def _bucket(score: float) -> str:
    for lo in (0.7, 0.65, 0.6, 0.55, 0.5):
        if score >= lo:
            return f">={lo}"
    return "<0.5"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyse(projects_dir: Path, db: Path, max_files: int | None = None) -> dict:
    files = sorted(projects_dir.rglob("*.jsonl"))
    if max_files:
        files = files[:max_files]

    scans: list[SessionScan] = []
    for f in files:
        try:
            s = scan_transcript(f)
        except OSError:
            continue
        if s:
            scans.append(s)

    mtimes = [f.stat().st_mtime for f in files if f.exists()]
    window = {
        "from": datetime.fromtimestamp(min(mtimes), UTC).date().isoformat() if mtimes else None,
        "to": datetime.fromtimestamp(max(mtimes), UTC).date().isoformat() if mtimes else None,
    }

    per_memory: list[dict] = []
    injections = 0
    for scan in scans:
        for inj in scan.injections:
            injections += 1
            per_memory.extend(_reuse_for_injection(scan, inj))

    scored = [m for m in per_memory if m["attributable"]]
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for m in scored:
        by_bucket[_bucket(m["score"])].append(m)

    calibration = [
        {
            "bucket": b,
            "n": len(v),
            "used_rate": sum(x["used"] for x in v) / len(v),
            "used_rate_strict": sum(x["used_strict"] for x in v) / len(v),
            "mean_reused_tokens": sum(x["reused"] for x in v) / len(v),
        }
        for b, v in sorted(by_bucket.items(), reverse=True)
    ]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for m in scored:
        by_type[m["type"]].append(m)

    cli = Counter()
    skill = Counter()
    for s in scans:
        cli.update(s.cli_calls)
        skill.update(s.skill_calls)

    return {
        "_meta": {
            "generated": _now_iso(),
            "projects_dir": str(projects_dir),
            "transcript_files_scanned": len(files),
            "transcript_files_with_signal": len(scans),
            "transcript_mtime_window": window,
            "caveats": [
                "Transcripts are a retention window, not full history: the memory DB's "
                "operation_events go back further. Only measures 1 and 4 (DB-only) cover "
                "the whole history; 2, 3 and 5 cover the transcript window.",
                "The UserPromptSubmit hook fires once per session, so n is roughly one "
                "injection per session — enough to calibrate a global score threshold, "
                "too thin to slice per project.",
                "Reuse is token overlap, not comprehension. It cannot see a memory that "
                "correctly told the agent NOT to do something, which is a real and "
                "systematically undercounted kind of usefulness.",
                "Two controls bracket the reuse rate rather than pin it. 'ordered' "
                "subtracts only what the agent had already seen when the injection landed "
                "(upper bound); 'strict' subtracts everything it saw from any other source "
                "at any point in the session (lower bound). Neither is the true rate.",
                "The headline zero-result rate is not a defect rate. Most of it is empty "
                "query text from a March-May 2026 window that has since stopped, exact-mode "
                "lookups where returning nothing is correct, and repeated eval-harness "
                "fixtures. Read the split, never the headline.",
                "memories.recall_count counts CLI and skill recalls too, so it is not a "
                "measure of hook-injection effectiveness on its own.",
                "The hook injects --depth summary, so what the transcript records is a "
                "~200-char preview, not the full memory. Reuse is measured against the "
                "preview; a memory whose useful part was truncated away scores zero here "
                "through no fault of its own. Mean distinctive tokens per memory is only "
                "~5, so single-token noise moves this metric easily.",
                "The highest score bucket having the lowest reuse is not necessarily a "
                "ranking failure. A memory scoring >=0.7 is by construction very close to "
                "the prompt, so its vocabulary is mostly already in the prompt and gets "
                "subtracted as pre-existing. Read it as: high similarity is not the same "
                "as high informativeness. Not as: the ranker is broken.",
            ],
        },
        "mechanical": mechanical_rates(db),
        "coverage": coverage(db),
        "injection": {
            "injections": injections,
            "memories_injected": len(per_memory),
            "memories_scorable": len(scored),
            "used_rate": sum(m["used"] for m in scored) / len(scored) if scored else None,
            "used_rate_strict": sum(m["used_strict"] for m in scored) / len(scored) if scored else None,
            "mean_tokens": sum(m["tokens"] for m in scored) / len(scored) if scored else None,
            "mean_tokens_strict": sum(m["tokens_strict"] for m in scored) / len(scored) if scored else None,
            "mean_reused": sum(m["reused"] for m in scored) / len(scored) if scored else None,
            "by_type": [
                {
                    "type": t,
                    "n": len(v),
                    "used_rate": sum(x["used"] for x in v) / len(v),
                    "used_rate_strict": sum(x["used_strict"] for x in v) / len(v),
                }
                for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
            ],
        },
        "calibration": calibration,
        "surface": {
            "cli_calls": cli.most_common(),
            "skill_calls": skill.most_common(),
            "sessions_with_cli": sum(1 for s in scans if s.cli_calls),
            "sessions_with_skill": sum(1 for s in scans if s.skill_calls),
        },
    }


def render(r: dict) -> str:
    m, cov, inj, cal, sur = r["mechanical"], r["coverage"], r["injection"], r["calibration"], r["surface"]
    meta = r["_meta"]
    L = []
    a = L.append

    a("=" * 74)
    a("MEMORY SERVICE — REAL USAGE")
    a("=" * 74)
    a(
        f"transcripts   {meta['transcript_files_scanned']} files, "
        f"{meta['transcript_files_with_signal']} with memory signal, "
        f"{meta['transcript_mtime_window']['from']} → {meta['transcript_mtime_window']['to']}"
    )
    a(f"db events     {m['events_total']} events, {m['span']['from']} → {m['span']['to']}")

    a("\n1. MECHANICAL RATES  (DB only — full history)")
    a(f"  {'operation':<14} {'n':>7} {'avg results':>12} {'avg ms':>9}")
    for op in m["by_operation"]:
        ar = f"{op['avg_results']:.2f}" if op["avg_results"] is not None else "—"
        am = f"{op['avg_ms']:.0f}" if op["avg_ms"] is not None else "—"
        a(f"  {op['operation']:<14} {op['n']:>7} {ar:>12} {am:>9}")
    if m["search_zero_result_rate"] is not None:
        zs = m["zero_result_split"]
        a(
            f"  zero-result searches: {m['search_zero_result_rate']:.1%} of {m['search_total']} "
            f"— but that is NOT a defect rate. It splits into:"
        )
        tz = zs["total_zero"] or 1
        a(
            f"    empty query      {zs['empty_query']:>5} ({zs['empty_query'] / tz:.0%})  "
            "blank query column; zero by construction, not retrieval"
        )
        a(
            f"    exact-mode miss  {zs['exact_miss']:>5} ({zs['exact_miss'] / tz:.0%})  "
            "a lookup that missed; nothing IS the right answer"
        )
        a(
            f"    slash command    {zs['slash_command']:>5} ({zs['slash_command'] / tz:.0%})  "
            "reached search at all; the one actionable slice"
        )
        real = tz - (zs["empty_query"] or 0) - (zs["exact_miss"] or 0) - (zs["slash_command"] or 0)
        a(f"    remaining        {real:>5} ({real / tz:.0%})  substantive queries that found nothing")
        rep = m["zero_result_query_repetition"]
        a(
            f"    of those with text, {rep['repeat_ratio']:.0%} are repeats of an earlier query "
            "— eval fixtures, not user traffic"
        )
        a("  by mode:")
        for row in m["zero_result_by_mode"]:
            a(f"    {row['search_mode'] or '—':<10} {row['n']:>6} searches, {row['zero_pct']:>6.1%} zero")
    a(
        f"  stores: {m['store_total']}, dedup ran on {m['store_dedup_used']}, "
        f"caught {m['store_duplicate_caught']} duplicates"
    )

    a("\n2. INJECTION USEFULNESS  (transcript window)")
    if inj["memories_scorable"]:
        a(
            f"  {inj['injections']} injections → {inj['memories_injected']} memories "
            f"({inj['memories_scorable']} scorable)"
        )
        a(f"  used rate: {inj['used_rate']:.1%} (ordered control) .. {inj['used_rate_strict']:.1%} (strict control)")
        a(
            f"  mean distinctive tokens {inj['mean_tokens']:.1f} ordered / "
            f"{inj['mean_tokens_strict']:.1f} strict, mean reused {inj['mean_reused']:.1f}"
        )
        a(f"  {'type':<14} {'n':>6} {'used':>8} {'strict':>8}")
        for t in inj["by_type"]:
            a(f"  {t['type']:<14} {t['n']:>6} {t['used_rate']:>7.1%} {t['used_rate_strict']:>7.1%}")
    else:
        a("  no scorable injections found")

    a("\n3. SCORE CALIBRATION  (is --min-score 0.5 in the right place?)")
    a(f"  {'bucket':<10} {'n':>6} {'used':>8} {'strict':>8} {'mean reused':>13}")
    for c in cal:
        a(
            f"  {c['bucket']:<10} {c['n']:>6} {c['used_rate']:>7.1%} "
            f"{c['used_rate_strict']:>7.1%} {c['mean_reused_tokens']:>13.2f}"
        )

    a("\n4. WRITE/READ COVERAGE  (DB only — full history)")
    a(
        f"  {cov['never_recalled']}/{cov['memories_live']} live memories never recalled "
        f"({cov['never_recalled_pct']:.1%})"
    )
    if cov["bulk_recall_days"]:
        a("  !! recall_count is contaminated. These days each bumped >=100 memories at once,")
        a("     which is a sweep, not reading. Treat the rate above as an UNDERCOUNT of")
        a("     what is really never read:")
        for b in cov["bulk_recall_days"]:
            a(f"       {b['day']}  {b['n']} memories")
        a(f"     {cov['bulk_recall_contaminated']} memories affected in total.")
    a(f"  {'type':<14} {'n':>6} {'never recalled':>16} {'avg recalls':>12}")
    for t in cov["by_type"]:
        a(f"  {t['memory_type'] or '—':<14} {t['n']:>6} {t['never_recalled_pct']:>15.1%} {t['avg_recalls']:>12.2f}")
    a("  by age (a memory written yesterday has had no chance to be recalled):")
    for b in sorted(cov["by_age"], key=lambda x: x["age_bucket"]):
        a(f"    {b['age_bucket']:<8} {b['n']:>6} never {b['never_recalled_pct']:>6.1%}")
    a("  never-recalled tag prefixes: " + ", ".join(f"{k}={v}" for k, v in cov["never_recalled_tag_prefixes"][:8]))

    a("\n5. INVOCATION SURFACE  (transcript window)")
    a(f"  CLI:   {', '.join(f'{k}={v}' for k, v in sur['cli_calls']) or 'none'}")
    a(f"  Skill: {', '.join(f'{k}={v}' for k, v in sur['skill_calls']) or 'none'}")
    a(f"  sessions using CLI {sur['sessions_with_cli']}, using a skill {sur['sessions_with_skill']}")

    a("\nCAVEATS")
    for c in meta["caveats"]:
        a(f"  - {c}")
    a("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR))
    ap.add_argument("--db", default=None, help="Defaults to the live DB (copied before opening).")
    ap.add_argument("--max-files", type=int, default=None, help="Cap transcripts scanned (smoke runs).")
    ap.add_argument("--json", default=None, help="Write the full result as JSON.")
    args = ap.parse_args()

    from memory.core import DB_PATH

    src = Path(args.db) if args.db else Path(DB_PATH)
    if not src.exists():
        print(f"DB not found: {src}", file=sys.stderr)
        return 2

    with TemporaryDirectory() as td:
        # Never open the live DB, even read-only: a stray write path would be
        # unrecoverable and this script has no business touching it.
        snap = Path(td) / "snapshot.db"
        shutil.copy2(src, snap)
        result = analyse(Path(args.projects_dir), snap, max_files=args.max_files)

    print(render(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
