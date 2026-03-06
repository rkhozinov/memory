"""CLI tool for memory operations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Callable

from .core import MemoryStore


def _json_out(data: dict | list) -> None:
    print(json.dumps(data, indent=2, default=str))


def _out(fmt: str, data: Any, text_fn: Callable[[Any], str]) -> None:
    """Output data in the requested format."""
    if fmt == "text":
        print(text_fn(data))
    else:
        _json_out(data)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _short_hash(h: str) -> str:
    return h[:16]


_TYPE_PREFIX_RE = re.compile(
    r"^\[(Pattern|Observation|Decision|Learning|Error|Note|Reference)\]\s*"
)


def _fmt_health(d: dict) -> str:
    types = d.get("memory_types", {})
    parts = [f"{v} {k}" for k, v in types.items()]
    return (
        f"{d['status']} | {d['total_memories']} memories"
        f" ({', '.join(parts)})"
        f" | {_human_size(d['database_size_bytes'])}"
        f" | {d['database']}"
    )


def _fmt_memory_line(m: dict) -> str:
    h = _short_hash(m["content_hash"])
    mtype = m.get("memory_type", "note")
    sim = m.get("similarity")
    score = m.get("score")
    conf = m.get("confidence")
    imp = m.get("importance")
    header = f"{h} [{mtype}]"
    if score is not None:
        header += f" score={score:.2f}"
    elif sim is not None:
        header += f" {sim:.2f}"
    extras = []
    if conf is not None and conf < 1.0:
        extras.append(f"conf={conf:.2f}")
    if imp is not None:
        extras.append(f"imp={imp:.2f}")
    if extras:
        header += f" ({', '.join(extras)})"
    content = _TYPE_PREFIX_RE.sub("", m.get("content", ""))
    indented = content.replace("\n", "\n  ")
    return f"{header}\n  {indented}"


def _fmt_search_titles(results: list) -> str:
    if not results:
        return "no results"
    lines = []
    for m in results:
        h = _short_hash(m["content_hash"])
        mtype = m.get("memory_type", "note")
        score = m.get("score")
        content = _TYPE_PREFIX_RE.sub("", m.get("content", ""))
        first_line = content.split("\n")[0]
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        score_str = f" score={score:.2f}" if score is not None else ""
        lines.append(f"{h} [{mtype}]{score_str} {first_line}")
    return "\n".join(lines)


def _fmt_search(results: list) -> str:
    if not results:
        return "no results"
    return "\n\n".join(_fmt_memory_line(m) for m in results)


def _fmt_memory_full(m: dict) -> str:
    """Format a single memory with all metadata."""
    lines = [_fmt_memory_line(m)]
    lines.append(f"  tags: {', '.join(m.get('tags', []))}")
    lines.append(f"  created: {m.get('created_at', 'n/a')}")
    lines.append(f"  updated: {m.get('updated_at', 'n/a')}")
    rc = m.get("recall_count", 0)
    lr = m.get("last_recalled_at")
    lines.append(f"  recall_count: {rc}")
    lines.append(f"  last_recalled_at: {lr if lr else 'never'}")
    conf = m.get("confidence")
    imp = m.get("importance")
    lines.append(f"  confidence: {conf}")
    lines.append(f"  importance: {imp}")
    meta = m.get("metadata", {})
    if meta:
        lines.append(f"  metadata: {json.dumps(meta)}")
    return "\n".join(lines)


def _fmt_search_full(results: list) -> str:
    if not results:
        return "no results"
    return "\n\n".join(_fmt_memory_full(m) for m in results)


def _fmt_search_hook(results: list) -> str:
    if not results:
        return ""
    lines = []
    for m in results:
        content = _TYPE_PREFIX_RE.sub("", m.get("content", ""))
        lines.append(f"- {content}")
    payload = {"additionalContext": "Relevant memories from previous sessions:\n" + "\n".join(lines)}
    return json.dumps(payload)


def _fmt_list(d: dict, depth: str = "summary") -> str:
    memories = d.get("memories", [])
    total = d.get("total", 0)
    page = d.get("page", 1)
    header = f"{total} memories (page {page})"
    if not memories:
        return header
    if depth == "titles":
        body = _fmt_search_titles(memories)
    elif depth == "full":
        body = "\n\n".join(_fmt_memory_full(m) for m in memories)
    else:
        body = "\n\n".join(_fmt_memory_line(m) for m in memories)
    return header + "\n\n" + body


def _fmt_store(d: dict) -> str:
    if d.get("similar_hash"):
        return f"duplicate {_short_hash(d['content_hash'])} (similar to {_short_hash(d['similar_hash'])} {d['message']})"
    return f"{d['status']} {_short_hash(d['content_hash'])}"


def _fmt_store_batch(results: list) -> str:
    stored = sum(1 for r in results if r.get("status") == "stored")
    header = f"{stored}/{len(results)} stored"
    lines = [_fmt_store(r) for r in results]
    return header + "\n" + "\n".join(lines)


def _fmt_get(d: dict, depth: str = "summary") -> str:
    if "error" in d:
        return f"error: {d['error']}"
    if depth == "titles":
        return _fmt_search_titles([d])
    if depth == "full":
        return _fmt_memory_full(d)
    return _fmt_memory_line(d)


def _fmt_delete(d: dict) -> str:
    if "error" in d:
        return f"error: {d['error']}"
    if d.get("dry_run"):
        hashes = ", ".join(_short_hash(h) for h in d.get("hashes", []))
        return f"would delete {d['would_delete']}: {hashes}"
    hashes = ", ".join(_short_hash(h) for h in d.get("deleted_hashes", []))
    count = d.get("deleted", 0)
    return f"deleted {count}: {hashes}" if hashes else f"deleted {count}"


def _fmt_update(d: dict) -> str:
    if "error" in d:
        return f"error: {d['error']}"
    return f"updated {_short_hash(d['content_hash'])}"


def _fmt_cleanup(d: dict) -> str:
    return f"{d.get('duplicates_removed', 0)} duplicates removed"


def _fmt_list_tags(d: dict) -> str:
    total = d.get("total_unique", 0)
    tags = d.get("tags", [])
    if not tags:
        return "no tags"
    lines = [f"{total} unique tags:"]
    for entry in tags:
        lines.append(f"  {entry['tag']}: {entry['count']}")
    return "\n".join(lines)


def _fmt_purge(d: dict) -> str:
    if d.get("dry_run"):
        hashes = ", ".join(_short_hash(h) for h in d.get("hashes", []))
        return f"would purge {d['would_purge']} (retention: {d['retention_days']}d): {hashes}"
    count = d.get("purged", 0)
    hashes = ", ".join(_short_hash(h) for h in d.get("purged_hashes", []))
    return f"purged {count} (retention: {d['retention_days']}d): {hashes}" if hashes else f"purged {count}"


def _fmt_consolidate(d: dict) -> str:
    if d.get("dry_run"):
        lines = [f"would consolidate {d.get('would_consolidate', 0)} pairs:"]
        for p in d.get("pairs", []):
            lines.append(f"  keep {_short_hash(p['keep_hash'])} ← remove {_short_hash(p['remove_hash'])} (sim={p['similarity']:.2f})")
        return "\n".join(lines)
    count = d.get("consolidated", 0)
    if not count:
        return "0 pairs consolidated"
    lines = [f"{count} pairs consolidated:"]
    for p in d.get("pairs", []):
        lines.append(f"  keep {_short_hash(p['keep_hash'])} ← remove {_short_hash(p['remove_hash'])} (sim={p['similarity']:.2f})")
    return "\n".join(lines)


def _fmt_decay(d: dict) -> str:
    return f"{d.get('updated', 0)} updated, {d.get('pruned', 0)} pruned"


def _fmt_briefing(d: dict) -> str:
    return d.get("markdown", "No memories stored.")


def _fmt_stats(d: dict) -> str:
    total = d.get("total_memories", 0)
    recalled = d.get("recalled_at_least_once", 0)
    never = d.get("never_recalled", 0)
    recalled_pct = (recalled / total * 100) if total else 0
    never_pct = (never / total * 100) if total else 0

    lines = [
        "=== Memory Statistics ===",
        "",
        f"Memories: {total} total, {recalled} recalled ({recalled_pct:.1f}%), {never} never recalled ({never_pct:.1f}%)",
        "",
        "Operations:",
    ]

    stores = d.get("stores_total", 0)
    spd = d.get("stores_per_day", 0)
    dups = d.get("duplicates_total", 0)
    dedup_rate = d.get("dedup_rate")
    dedup_str = f"{dedup_rate * 100:.1f}%" if dedup_rate is not None else "n/a"
    lines.append(f"  Stores: {stores} ({spd}/day) | Duplicates: {dups} ({dedup_str})")

    searches = d.get("searches_total", 0)
    hit_rate = d.get("search_hit_rate")
    hit_str = f"{hit_rate * 100:.1f}%" if hit_rate is not None else "n/a"
    avg_sim = d.get("avg_top_similarity")
    sim_str = f"{avg_sim:.2f}" if avg_sim is not None else "n/a"
    avg_res = d.get("avg_results_per_search")
    res_str = f"{avg_res:.1f}" if avg_res is not None else "n/a"
    lines.append(f"  Searches: {searches} | Hit rate: {hit_str} | Avg similarity: {sim_str} | Avg results: {res_str}")

    deletes = d.get("deletes_total", 0)
    lines.append(f"  Deletes: {deletes}")

    lines.append("")
    lines.append("Performance:")
    avg_store = d.get("avg_store_ms")
    avg_search = d.get("avg_search_ms")
    store_str = f"{avg_store:.0f}ms" if avg_store is not None else "n/a"
    search_str = f"{avg_search:.0f}ms" if avg_search is not None else "n/a"
    lines.append(f"  Avg store: {store_str} | Avg search: {search_str}")

    total_chars = d.get("total_chars_returned", 0)
    if total_chars >= 1_000_000:
        chars_str = f"{total_chars / 1_000_000:.1f}M"
    elif total_chars >= 1_000:
        chars_str = f"{total_chars / 1_000:.1f}K"
    else:
        chars_str = str(total_chars)
    est_tokens = total_chars // 3
    if est_tokens >= 1_000_000:
        tok_str = f"~{est_tokens / 1_000_000:.0f}M"
    elif est_tokens >= 1_000:
        tok_str = f"~{est_tokens / 1_000:.0f}K"
    else:
        tok_str = f"~{est_tokens}"
    lines.append(f"\nOutput: {chars_str} chars returned ({tok_str} tokens est.)")

    # Top tags
    top_tags = d.get("top_tags", [])
    if top_tags:
        lines.append("\nTop tags (recalled memories):")
        for tag, count in top_tags[:10]:
            lines.append(f"  {tag}: {count}")

    # Top recalled
    top_recalled = d.get("top_recalled", [])
    if top_recalled:
        lines.append("\nTop recalled:")
        for i, m in enumerate(top_recalled, 1):
            h = _short_hash(m["content_hash"])
            mtype = m.get("memory_type", "note")
            rc = m.get("recall_count", 0)
            preview = m.get("content_preview", "")[:80]
            lines.append(f"  {i}. {h} [{mtype}] (recalled {rc}x) {preview}")

    # Never recalled
    never_list = d.get("never_recalled_list", [])
    if never_list:
        lines.append("\nNever recalled:")
        for m in never_list:
            h = _short_hash(m["content_hash"])
            mtype = m.get("memory_type", "note")
            dt = m.get("created_at_iso", "")[:10]
            preview = m.get("content_preview", "")[:80]
            lines.append(f"  {h} [{mtype}] {dt} {preview}")

    # Stale
    stale_list = d.get("stale_memories", [])
    if stale_list:
        lines.append("\nStale (oldest, never recalled — cleanup candidates):")
        for m in stale_list:
            h = _short_hash(m["content_hash"])
            mtype = m.get("memory_type", "note")
            dt = m.get("created_at_iso", "")[:10]
            preview = m.get("content_preview", "")[:80]
            lines.append(f"  {h} [{mtype}] {dt} {preview}")

    return "\n".join(lines)


def _fmt_doc_line(d: dict) -> str:
    """Format a single document as a summary line."""
    h = _short_hash(d["content_hash"])
    dtype = d.get("doc_type", "document")
    version = d.get("version", 1)
    title = d.get("title", "")
    header = f"{h} [{dtype}] v{version} {title}"
    sim = d.get("similarity")
    score = d.get("score")
    if score is not None:
        header += f" score={score:.2f}"
    elif sim is not None:
        header += f" sim={sim:.2f}"
    summary = d.get("summary", "")
    if summary:
        if len(summary) > 120:
            summary = summary[:117] + "..."
        return f"{header}\n  {summary}"
    return header


def _fmt_doc_full(d: dict) -> str:
    """Format a single document with all metadata."""
    lines = [_fmt_doc_line(d)]
    lines.append(f"  tags: {', '.join(d.get('tags', []))}")
    lines.append(f"  created: {d.get('created_at', 'n/a')}")
    lines.append(f"  updated: {d.get('updated_at', 'n/a')}")
    rc = d.get("recall_count", 0)
    lr = d.get("last_recalled_at")
    lines.append(f"  recall_count: {rc}")
    lines.append(f"  last_recalled_at: {lr if lr else 'never'}")
    meta = d.get("metadata", {})
    if meta:
        lines.append(f"  metadata: {json.dumps(meta)}")
    body = d.get("body", "")
    if body:
        lines.append(f"  body:\n    {body.replace(chr(10), chr(10) + '    ')}")
    return "\n".join(lines)


def _fmt_doc_store(d: dict) -> str:
    if d.get("status") == "duplicate":
        return f"duplicate {_short_hash(d['content_hash'])} ({d['message']})"
    return f"{d['status']} {_short_hash(d['content_hash'])}"


def _fmt_doc_search(results: list) -> str:
    if not results:
        return "no results"
    return "\n\n".join(_fmt_doc_line(d) for d in results)


def _fmt_doc_list(d: dict) -> str:
    docs = d.get("documents", [])
    total = d.get("total", 0)
    page = d.get("page", 1)
    header = f"{total} documents (page {page})"
    if not docs:
        return header
    body = "\n\n".join(_fmt_doc_line(doc) for doc in docs)
    return header + "\n\n" + body


def _fmt_doc_update(d: dict) -> str:
    if "error" in d:
        return f"error: {d['error']}"
    v = d.get("version", "?")
    return f"updated {_short_hash(d['content_hash'])} (v{v})"


def _fmt_doc_delete(d: dict) -> str:
    if "error" in d:
        return f"error: {d['error']}"
    if d.get("dry_run"):
        return f"would delete {_short_hash(d['content_hash'])}"
    return f"deleted {_short_hash(d['content_hash'])}"


def _fmt_doc_get(d: dict) -> str:
    if "error" in d:
        return f"error: {d['error']}"
    return _fmt_doc_full(d)


# --- Command handlers ---

def cmd_store(args, store: MemoryStore, fmt: str) -> None:
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    try:
        meta = json.loads(args.metadata)
    except json.JSONDecodeError:
        meta = {}
    dedup = None if args.force else args.dedup_threshold
    result = store.store(
        args.content, tags=tag_list, memory_type=args.memory_type,
        metadata=meta, dedup_threshold=dedup,
        importance=args.importance,
    )
    _out(fmt, result, _fmt_store)


def cmd_store_batch(args, store: MemoryStore, fmt: str) -> None:
    try:
        if args.file_path == "-":
            data = json.load(sys.stdin)
        else:
            with open(args.file_path) as f:
                data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}", file=sys.stderr)
        if args.file_path == "-":
            print("\nHint: Use a heredoc to avoid shell quoting issues:", file=sys.stderr)
            print("  cat <<'ENDJSON' | memory -f text store-batch --dedup 0.85", file=sys.stderr)
            print('  [{"content": "...", "tags": ["t"], "memory_type": "decision"}]', file=sys.stderr)
            print("  ENDJSON", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("Error: expected a JSON array", file=sys.stderr)
        sys.exit(1)
    results = store.store_batch(data, dedup_threshold=args.dedup_threshold)
    _out(fmt, results, _fmt_store_batch)


def cmd_search(args, store: MemoryStore, fmt: str) -> None:
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    results = store.search(
        query=args.query, mode=args.mode, limit=args.limit, tags=tag_list,
        time_expr=args.time_expr, after=args.after, before=args.before,
    )
    if args.min_similarity is not None:
        results = [m for m in results if m.get("similarity", 0) >= args.min_similarity]
    if fmt == "hook":
        output = _fmt_search_hook(results)
        if output:
            print(output)
    else:
        depth = getattr(args, "depth", "summary")
        if depth == "titles":
            text_fn = _fmt_search_titles
        elif depth == "full":
            text_fn = _fmt_search_full
        else:
            text_fn = _fmt_search
        _out(fmt, results, text_fn)


def cmd_list(args, store: MemoryStore, fmt: str) -> None:
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    result = store.list(page=args.page, page_size=args.page_size, tags=tag_list, memory_type=args.memory_type)
    depth = getattr(args, "depth", "summary")
    _out(fmt, result, lambda d: _fmt_list(d, depth=depth))


def cmd_delete(args, store: MemoryStore, fmt: str) -> None:
    content_hash = args.hash_arg or args.content_hash
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    result = store.delete(
        content_hash=content_hash, tags=tag_list,
        before=args.before, after=args.after, dry_run=args.dry_run,
    )
    _out(fmt, result, _fmt_delete)


def cmd_get(args, store: MemoryStore, fmt: str) -> None:
    result = store.get(content_hash=args.content_hash)
    depth = getattr(args, "depth", "summary")
    _out(fmt, result, lambda d: _fmt_get(d, depth=depth))


def cmd_update(args, store: MemoryStore, fmt: str) -> None:
    updates: dict = {}
    if args.content is not None:
        updates["content"] = args.content
    if args.tags is not None:
        updates["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.memory_type is not None:
        updates["memory_type"] = args.memory_type
    if args.metadata is not None:
        try:
            updates["metadata"] = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: --metadata must be valid JSON", file=sys.stderr)
            sys.exit(1)
    if args.importance is not None:
        updates["importance"] = args.importance
    if args.confidence is not None:
        updates["confidence"] = args.confidence
    if not updates:
        print("Error: no updates specified", file=sys.stderr)
        sys.exit(1)
    result = store.update(content_hash=args.content_hash, updates=updates)
    _out(fmt, result, _fmt_update)


def cmd_health(args, store: MemoryStore, fmt: str) -> None:
    _out(fmt, store.health(), _fmt_health)


def cmd_cleanup(args, store: MemoryStore, fmt: str) -> None:
    _out(fmt, store.cleanup(), _fmt_cleanup)


def cmd_list_tags(args, store: MemoryStore, fmt: str) -> None:
    _out(fmt, store.list_tags(), _fmt_list_tags)


def cmd_purge(args, store: MemoryStore, fmt: str) -> None:
    result = store.purge(retention_days=args.retention_days, dry_run=args.dry_run)
    _out(fmt, result, _fmt_purge)


def cmd_consolidate(args, store: MemoryStore, fmt: str) -> None:
    exclude = [t.strip() for t in args.exclude_types.split(",") if t.strip()] if args.exclude_types else []
    result = store.consolidate(threshold=args.threshold, dry_run=args.dry_run, exclude_types=exclude)
    _out(fmt, result, _fmt_consolidate)


def cmd_decay(args, store: MemoryStore, fmt: str) -> None:
    result = store.apply_decay(min_confidence=args.min_confidence)
    _out(fmt, result, _fmt_decay)


def cmd_briefing(args, store: MemoryStore, fmt: str) -> None:
    result = store.briefing(budget=args.budget)
    _out(fmt, result, _fmt_briefing)


def cmd_stats(args, store: MemoryStore, fmt: str) -> None:
    result = store.stats(
        after=args.after, before=args.before,
        top_recalled=args.top_recalled, never_recalled=args.never_recalled,
        stale=args.stale,
    )
    _out(fmt, result, _fmt_stats)


def cmd_doc_store(args, store: MemoryStore, fmt: str) -> None:
    # Read body from file or stdin
    if args.body_file:
        if args.body_file == "-":
            body = sys.stdin.read()
        else:
            with open(args.body_file) as f:
                body = f.read()
    elif args.body:
        body = args.body
    else:
        print("Error: --body or --body-file is required", file=sys.stderr)
        sys.exit(1)
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    try:
        meta = json.loads(args.metadata)
    except json.JSONDecodeError:
        meta = {}
    result = store.store_doc(
        title=args.title, body=body, summary=args.summary,
        doc_type=args.doc_type, tags=tag_list, metadata=meta,
    )
    _out(fmt, result, _fmt_doc_store)


def cmd_doc_get(args, store: MemoryStore, fmt: str) -> None:
    result = store.get_doc(content_hash=args.content_hash)
    _out(fmt, result, _fmt_doc_get)


def cmd_doc_search(args, store: MemoryStore, fmt: str) -> None:
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    results = store.search_docs(
        query=args.query, mode=args.mode, limit=args.limit,
        tags=tag_list, doc_type=args.doc_type,
    )
    _out(fmt, results, _fmt_doc_search)


def cmd_doc_list(args, store: MemoryStore, fmt: str) -> None:
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    result = store.list_docs(
        page=args.page, page_size=args.page_size,
        tags=tag_list, doc_type=args.doc_type,
    )
    _out(fmt, result, _fmt_doc_list)


def cmd_doc_update(args, store: MemoryStore, fmt: str) -> None:
    kwargs = {}
    if args.title is not None:
        kwargs["title"] = args.title
    if args.body_file:
        if args.body_file == "-":
            kwargs["body"] = sys.stdin.read()
        else:
            with open(args.body_file) as f:
                kwargs["body"] = f.read()
    if args.summary is not None:
        kwargs["summary"] = args.summary
    if args.doc_type is not None:
        kwargs["doc_type"] = args.doc_type
    if args.tags is not None:
        kwargs["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.metadata is not None:
        try:
            kwargs["metadata"] = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: --metadata must be valid JSON", file=sys.stderr)
            sys.exit(1)
    if not kwargs:
        print("Error: no updates specified", file=sys.stderr)
        sys.exit(1)
    result = store.update_doc(content_hash=args.content_hash, **kwargs)
    _out(fmt, result, _fmt_doc_update)


def cmd_doc_delete(args, store: MemoryStore, fmt: str) -> None:
    content_hash = args.hash_arg or args.content_hash
    if not content_hash:
        print("Error: content hash required", file=sys.stderr)
        sys.exit(1)
    result = store.delete_doc(content_hash=content_hash, dry_run=args.dry_run)
    _out(fmt, result, _fmt_doc_delete)


_DOC_DISPATCH = {
    "store": cmd_doc_store,
    "get": cmd_doc_get,
    "search": cmd_doc_search,
    "list": cmd_doc_list,
    "update": cmd_doc_update,
    "delete": cmd_doc_delete,
}


def cmd_doc(args, store: MemoryStore, fmt: str) -> None:
    """Dispatch doc subcommands."""
    doc_cmd = getattr(args, "doc_command", None)
    if not doc_cmd:
        print("Usage: memory doc {store,get,search,list,update,delete}", file=sys.stderr)
        sys.exit(1)
    handler = _DOC_DISPATCH.get(doc_cmd)
    if not handler:
        print(f"Unknown doc subcommand: {doc_cmd}", file=sys.stderr)
        sys.exit(1)
    handler(args, store, fmt)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Memory CLI — store, search, and manage memories.",
    )
    parser.add_argument(
        "--format", "-f", dest="fmt",
        choices=["json", "text", "hook"], default="json",
        help="Output format",
    )
    sub = parser.add_subparsers(dest="command")

    # store
    p = sub.add_parser("store", help="Store a single memory")
    p.add_argument("content")
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--type", dest="memory_type", default="note", help="Memory type")
    p.add_argument("--metadata", "-m", default="{}", help="JSON metadata")
    p.add_argument("--dedup", dest="dedup_threshold", default=None, type=float,
                   help="Skip if existing memory has similarity >= threshold (0.0-1.0)")
    p.add_argument("--importance", default=None, type=float,
                   help="Importance score (0.0-1.0). Auto-inferred if not set.")
    p.add_argument("--force", action="store_true", default=False,
                   help="Store even if dedup detects a similar memory")

    # store-batch
    p = sub.add_parser("store-batch", help="Store multiple memories from a JSON array")
    p.add_argument("--file", "-f", dest="file_path", default="-",
                   help="JSON file with items (- for stdin)")
    p.add_argument("--dedup", dest="dedup_threshold", default=None, type=float,
                   help="Skip if existing memory has similarity >= threshold (0.0-1.0)")

    # get
    p = sub.add_parser("get", help="Get a single memory by content hash")
    p.add_argument("content_hash", help="Full or prefix content hash")
    p.add_argument("--depth", default="summary", choices=["titles", "summary", "full"],
                   help="Output depth: titles (one-line), summary (default), full (all metadata)")

    # search
    p = sub.add_parser("search", help="Search memories")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--mode", default="semantic", choices=["semantic", "exact", "hybrid", "fts"])
    p.add_argument("--limit", "-n", default=10, type=int)
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--time-expr", default=None, help="Natural language time filter")
    p.add_argument("--after", default=None, help="ISO date (YYYY-MM-DD)")
    p.add_argument("--before", default=None, help="ISO date (YYYY-MM-DD)")
    p.add_argument("--min-similarity", default=None, type=float,
                   help="Filter results below this similarity threshold")
    p.add_argument("--depth", default="summary", choices=["titles", "summary", "full"],
                   help="Output depth: titles (one-line), summary (default), full (all metadata)")

    # list
    p = sub.add_parser("list", help="List memories with pagination")
    p.add_argument("--page", default=1, type=int)
    p.add_argument("--page-size", default=20, type=int)
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--type", dest="memory_type", default=None, help="Filter by memory type")
    p.add_argument("--depth", default="summary", choices=["titles", "summary", "full"],
                   help="Output depth: titles (one-line), summary (default), full (all metadata)")

    # delete
    p = sub.add_parser("delete", help="Delete memories by hash, tags, or time range")
    p.add_argument("hash_arg", nargs="?", default=None, metavar="HASH")
    p.add_argument("--hash", dest="content_hash", default=None, help="Delete by content hash")
    p.add_argument("--tags", "-t", default="", help="Delete by tags (comma-separated)")
    p.add_argument("--before", default=None, help="Delete before date (YYYY-MM-DD)")
    p.add_argument("--after", default=None, help="Delete after date (YYYY-MM-DD)")
    p.add_argument("--dry-run", action="store_true", help="Preview deletions without executing")

    # update
    p = sub.add_parser("update", help="Update memory metadata or content")
    p.add_argument("content_hash")
    p.add_argument("--content", default=None, help="New content (rehashes and re-embeds)")
    p.add_argument("--tags", "-t", default=None, help="New tags (comma-separated)")
    p.add_argument("--type", dest="memory_type", default=None, help="New memory type")
    p.add_argument("--metadata", "-m", default=None, help="JSON metadata to merge")
    p.add_argument("--importance", default=None, type=float, help="New importance (0.0-1.0)")
    p.add_argument("--confidence", default=None, type=float, help="New confidence (0.0-1.0)")

    # health
    sub.add_parser("health", help="Check database health and stats")

    # cleanup
    sub.add_parser("cleanup", help="Remove duplicate entries")

    # list-tags
    sub.add_parser("list-tags", help="List all unique tags with frequency counts")

    # purge
    p = sub.add_parser("purge", help="Hard-delete soft-deleted memories older than retention period")
    p.add_argument("--retention-days", default=30, type=int,
                   help="Only purge entries deleted more than N days ago (default: 30)")
    p.add_argument("--dry-run", action="store_true", help="Preview without executing")

    # consolidate
    p = sub.add_parser("consolidate", help="Merge near-duplicate memories (deterministic)")
    p.add_argument("--threshold", default=0.92, type=float,
                   help="Cosine similarity threshold for merging (default 0.92)")
    p.add_argument("--dry-run", action="store_true", help="Preview without executing")
    p.add_argument("--exclude-types", default="reference",
                   help="Comma-separated memory types to skip (default: reference). Use '' to include all.")

    # decay
    p = sub.add_parser("decay", help="Apply confidence decay and optionally prune low-confidence memories")
    p.add_argument("--min-confidence", default=0.0, type=float,
                   help="Prune memories below this confidence (default 0.0 = no pruning)")

    # briefing
    p = sub.add_parser("briefing", help="Generate a compact session briefing of top memories")
    p.add_argument("--budget", default=150, type=int,
                   help="Total line budget for the briefing (default 150)")

    # stats
    p = sub.add_parser("stats", help="Show usage statistics and analytics")
    p.add_argument("--after", default=None, help="Filter events after date (YYYY-MM-DD)")
    p.add_argument("--before", default=None, help="Filter events before date (YYYY-MM-DD)")
    p.add_argument("--top-recalled", type=int, default=None, help="Show N most recalled memories")
    p.add_argument("--never-recalled", action="store_true", help="Show memories never recalled")
    p.add_argument("--stale", action="store_true", help="Show stale memories (old, never recalled)")

    # doc (subcommand group)
    doc_parser = sub.add_parser("doc", help="Document management")
    doc_sub = doc_parser.add_subparsers(dest="doc_command")

    # doc store
    p = doc_sub.add_parser("store", help="Store a document")
    p.add_argument("--title", required=True, help="Document title")
    p.add_argument("--summary", required=True, help="Short summary for embedding (~100-200 tokens)")
    p.add_argument("--body", default=None, help="Document body text (use --body-file for files)")
    p.add_argument("--body-file", default=None, help="Read body from file (- for stdin)")
    p.add_argument("--type", dest="doc_type", default="document",
                   help="Document type: document, plan, spec, runbook, session, reference")
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--metadata", "-m", default="{}", help="JSON metadata")

    # doc get
    p = doc_sub.add_parser("get", help="Get a document by content hash")
    p.add_argument("content_hash", help="Full or prefix content hash")

    # doc search
    p = doc_sub.add_parser("search", help="Search documents")
    p.add_argument("query", help="Search query")
    p.add_argument("--mode", default="auto", choices=["semantic", "fts", "auto"],
                   help="Search mode (default: auto = semantic + FTS merged)")
    p.add_argument("--limit", "-n", default=5, type=int)
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--type", dest="doc_type", default=None, help="Filter by document type")

    # doc list
    p = doc_sub.add_parser("list", help="List documents")
    p.add_argument("--page", default=1, type=int)
    p.add_argument("--page-size", default=20, type=int)
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--type", dest="doc_type", default=None, help="Filter by document type")

    # doc update
    p = doc_sub.add_parser("update", help="Update a document")
    p.add_argument("content_hash", help="Full or prefix content hash")
    p.add_argument("--title", default=None, help="New title")
    p.add_argument("--summary", default=None, help="New summary (re-embeds)")
    p.add_argument("--body-file", default=None, help="New body from file (- for stdin)")
    p.add_argument("--type", dest="doc_type", default=None, help="New document type")
    p.add_argument("--tags", "-t", default=None, help="New tags (comma-separated)")
    p.add_argument("--metadata", "-m", default=None, help="JSON metadata to merge")

    # doc delete
    p = doc_sub.add_parser("delete", help="Delete a document")
    p.add_argument("hash_arg", nargs="?", default=None, metavar="HASH")
    p.add_argument("--hash", dest="content_hash", default=None, help="Delete by content hash")
    p.add_argument("--dry-run", action="store_true", help="Preview deletion without executing")

    return parser


_DISPATCH = {
    "store": cmd_store,
    "store-batch": cmd_store_batch,
    "get": cmd_get,
    "search": cmd_search,
    "list": cmd_list,
    "delete": cmd_delete,
    "update": cmd_update,
    "health": cmd_health,
    "cleanup": cmd_cleanup,
    "list-tags": cmd_list_tags,
    "purge": cmd_purge,
    "consolidate": cmd_consolidate,
    "decay": cmd_decay,
    "briefing": cmd_briefing,
    "stats": cmd_stats,
    "doc": cmd_doc,
}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    store = MemoryStore()
    try:
        _DISPATCH[args.command](args, store, args.fmt)
    finally:
        store.close()


if __name__ == "__main__":
    main()
