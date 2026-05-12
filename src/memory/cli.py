"""CLI tool for memory operations. JSON-only output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import MemoryStore


def _read_file(path: str) -> str:
    return Path(path).read_text()


def _json_out(data: dict | list) -> None:
    print(json.dumps(data, indent=2, default=str))


def _parse_tags(raw: str | None) -> list[str] | None:
    """Parse comma-separated tag string into list, or None."""
    if not raw:
        return None
    result = [t.strip() for t in raw.split(",") if t.strip()]
    return result or None


def _parse_json_or_none(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Error: invalid JSON: {raw[:80]}", file=sys.stderr)
        sys.exit(1)


def _read_stdin() -> str:
    return sys.stdin.read()


def _resolve_hash(store: MemoryStore, prefix: str) -> str | None:
    """Resolve a partial hash to full hash. Returns full hash or None."""
    if len(prefix) >= 64:
        return prefix
    conn = store._get_conn()
    matches = conn.execute(
        "SELECT content_hash FROM memories WHERE content_hash LIKE ? AND deleted_at IS NULL",
        (prefix + "%",),
    ).fetchall()
    if len(matches) == 1:
        return matches[0]["content_hash"]
    if len(matches) > 1:
        hashes = [m["content_hash"] for m in matches]
        _json_out({"error": f"Ambiguous hash prefix '{prefix}'", "matches": hashes})
        sys.exit(1)
    return None


# --- Command handlers ---


def _check_rejected(result: dict) -> None:
    """If store returned a rejected status, print the error JSON and exit 1."""
    if isinstance(result, dict) and result.get("status") == "rejected":
        _json_out(result)
        sys.exit(1)


def cmd_store(args, store: MemoryStore) -> None:
    content = args.content
    reject_injection: bool | None = args.reject_injection or None  # False → None (use env default)

    # Try parsing content as JSON (single object or array)
    if content == "-":
        content = _read_stdin()

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, list):
        # Batch mode
        results = store.store_batch(parsed, dedup_threshold=args.dedup_threshold, reject_injection=reject_injection)
        _json_out(results)
        # Exit 1 if any item was rejected
        if any(isinstance(r, dict) and r.get("status") == "rejected" for r in results):
            sys.exit(1)
        return

    if isinstance(parsed, dict):
        # Single JSON object
        result = store.store(
            parsed.get("content", ""),
            tags=parsed.get("tags", []),
            memory_type=parsed.get("type", parsed.get("memory_type", "note")),
            metadata={
                k: v for k, v in parsed.items() if k not in ("content", "tags", "type", "memory_type", "importance")
            },
            dedup_threshold=parsed.get("dedup", args.dedup_threshold),
            importance=parsed.get("importance"),
            reject_injection=reject_injection,
        )
        _check_rejected(result)
        _json_out(result)
        return

    # Plain text content with flags
    tag_list = _parse_tags(args.tags) or []
    meta = _parse_json_or_none(args.metadata) or {}
    dedup = None if args.force else args.dedup_threshold
    result = store.store(
        content,
        tags=tag_list,
        memory_type=args.memory_type,
        metadata=meta,
        dedup_threshold=dedup,
        importance=args.importance,
        reject_injection=reject_injection,
    )
    _check_rejected(result)
    _json_out(result)


def cmd_search(args, store: MemoryStore) -> None:
    results = store.search(
        query=args.query,
        mode=args.mode,
        limit=args.limit,
        tags=_parse_tags(args.tags),
        memory_types=_parse_tags(args.types),
        max_hops=args.hops,
        track_recall=not args.no_track_recall,
    )
    _json_out(results)


def cmd_get(args, store: MemoryStore) -> None:
    full_hash = _resolve_hash(store, args.content_hash)
    if not full_hash:
        _json_out({"error": f"Not found: {args.content_hash}"})
        sys.exit(1)
    result = store.get(content_hash=full_hash)
    _json_out(result)


def cmd_delete(args, store: MemoryStore) -> None:
    content_hash = args.hash_arg
    tag_list = _parse_tags(args.tags)

    # Resolve partial hash
    if content_hash and len(content_hash) < 64:
        full = _resolve_hash(store, content_hash)
        if not full:
            _json_out({"error": f"Not found: {content_hash}", "deleted": 0})
            return
        content_hash = full

    result = store.delete(
        content_hash=content_hash,
        tags=tag_list,
        before=args.before,
        after=args.after,
        dry_run=args.dry_run,
    )
    _json_out(result)


def cmd_update(args, store: MemoryStore) -> None:
    full_hash = _resolve_hash(store, args.content_hash)
    if not full_hash:
        _json_out({"error": f"Not found: {args.content_hash}"})
        sys.exit(1)

    updates: dict = {}
    if args.content is not None:
        updates["content"] = args.content
    if args.tags is not None:
        updates["tags"] = _parse_tags(args.tags) or []
    if args.memory_type is not None:
        updates["memory_type"] = args.memory_type
    if args.metadata is not None:
        updates["metadata"] = _parse_json_or_none(args.metadata)
    if args.importance is not None:
        updates["importance"] = args.importance

    if not updates:
        print(
            "Error: provide at least one field to update (--content, --tags, --type, --metadata, --importance).",
            file=sys.stderr,
        )
        sys.exit(1)

    result = store.update(content_hash=full_hash, updates=updates)
    _json_out(result)


def cmd_health(args, store: MemoryStore) -> None:
    _json_out(store.health())


# --- Doc subcommands ---


def cmd_doc_store(args, store: MemoryStore) -> None:
    body = args.body
    if args.body_file:
        body = _read_stdin() if args.body_file == "-" else _read_file(args.body_file)
    if not body:
        print("Error: --body or --body-file required.", file=sys.stderr)
        sys.exit(1)
    result = store.store_doc(
        title=args.title,
        body=body,
        summary=args.summary,
        doc_type=args.doc_type,
        tags=_parse_tags(args.tags) or [],
        metadata=_parse_json_or_none(args.metadata) or {},
    )
    _json_out(result)


def cmd_doc_get(args, store: MemoryStore) -> None:
    result = store.get_doc(content_hash=args.content_hash)
    _json_out(result)


def cmd_doc_search(args, store: MemoryStore) -> None:
    results = store.search_docs(
        query=args.query,
        mode=args.mode,
        limit=args.limit,
        tags=_parse_tags(args.tags),
        doc_type=args.doc_type,
    )
    _json_out(results)


def cmd_doc_list(args, store: MemoryStore) -> None:
    result = store.list_docs(
        page=args.page,
        page_size=args.page_size,
        tags=_parse_tags(args.tags),
        doc_type=args.doc_type,
    )
    _json_out(result)


def cmd_doc_update(args, store: MemoryStore) -> None:
    kwargs: dict = {}
    if args.title is not None:
        kwargs["title"] = args.title
    if args.summary is not None:
        kwargs["summary"] = args.summary
    if args.body_file is not None:
        kwargs["body"] = _read_stdin() if args.body_file == "-" else _read_file(args.body_file)
    if args.doc_type is not None:
        kwargs["doc_type"] = args.doc_type
    if args.tags is not None:
        kwargs["tags"] = _parse_tags(args.tags) or []
    if args.metadata is not None:
        kwargs["metadata"] = _parse_json_or_none(args.metadata)
    if not kwargs:
        print("Error: provide at least one field to update.", file=sys.stderr)
        sys.exit(1)
    result = store.update_doc(content_hash=args.content_hash, **kwargs)
    _json_out(result)


def cmd_doc_delete(args, store: MemoryStore) -> None:
    content_hash = args.content_hash
    result = store.delete_doc(content_hash=content_hash, dry_run=args.dry_run)
    _json_out(result)


def cmd_doc(args, store: MemoryStore) -> None:
    doc_cmd = getattr(args, "doc_command", None)
    if not doc_cmd:
        print("Usage: memory doc {store|get|search|list|update|delete}", file=sys.stderr)
        sys.exit(1)
    _DOC_DISPATCH[doc_cmd](args, store)


# --- Admin subcommands ---


def cmd_admin_cleanup(args, store: MemoryStore) -> None:
    _json_out(store.cleanup())


def cmd_admin_consolidate(args, store: MemoryStore) -> None:
    exclude = _parse_tags(args.exclude_types) or []
    result = store.consolidate(threshold=args.threshold, dry_run=args.dry_run, exclude_types=exclude)
    _json_out(result)


def cmd_admin_decay(args, store: MemoryStore) -> None:
    _json_out(store.apply_decay(min_confidence=args.min_confidence))


def cmd_admin_auto_extract(args, store: MemoryStore) -> None:
    """Extract memories from a brief file and optionally insert them."""
    from .auto_extract import extract_memories

    brief_text = Path(args.brief_file).read_text(encoding="utf-8")
    entries = extract_memories(
        brief_text,
        model=args.model,
        max_entries=args.max_entries,
    )

    if args.dry_run:
        _json_out({"dry_run": True, "extracted": entries, "count": len(entries)})
        return

    result = store.store_auto_extracted(entries)
    _json_out(result)


def cmd_admin_dream(args, store: MemoryStore) -> None:
    _json_out(
        store.dream(
            dry_run=args.dry_run,
            threshold_new=args.threshold_new,
            threshold_age_hours=args.threshold_age_hours,
        )
    )


def cmd_admin_demoted(args, store: MemoryStore) -> None:
    _json_out(store.demoted(limit=args.limit))


def cmd_admin_purge(args, store: MemoryStore) -> None:
    _json_out(store.purge(retention_days=args.retention_days, dry_run=args.dry_run))


def cmd_admin_export(args, store: MemoryStore) -> None:
    result = store.export_all(include_documents=not args.no_documents)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps({"status": "exported", "file": args.output}))
    else:
        _json_out(result)


def cmd_admin_import(args, store: MemoryStore) -> None:
    raw = _read_stdin() if args.file_path == "-" else _read_file(args.file_path)
    data = json.loads(raw)
    result = store.import_all(data, force=args.force)
    _json_out(result)


def cmd_admin_tags_list(args, store: MemoryStore) -> None:
    _json_out(store.list_tags())


def cmd_admin_tags_rename(args, store: MemoryStore) -> None:
    _json_out(store.rename_tag(args.old_tag, args.new_tag))


def cmd_admin_tags_merge(args, store: MemoryStore) -> None:
    sources = [t.strip() for t in args.source_tags.split(",") if t.strip()]
    _json_out(store.merge_tags(sources, args.target_tag))


def cmd_admin_stats(args, store: MemoryStore) -> None:
    _json_out(
        store.stats(
            after=args.after,
            before=args.before,
            top_recalled=args.top_recalled,
            never_recalled=args.never_recalled,
            stale=args.stale,
        )
    )


def cmd_admin_briefing(args, store: MemoryStore) -> None:
    _json_out(store.briefing(budget=args.budget))


def cmd_admin_graph_build(args, store: MemoryStore) -> None:
    _json_out(store.build_graph(dry_run=args.dry_run))


def cmd_admin_graph_entities(args, store: MemoryStore) -> None:
    _json_out(store.list_entities(entity_type=args.entity_type, limit=args.limit))


def cmd_admin_graph_context(args, store: MemoryStore) -> None:
    _json_out(store.entity_context(args.entity, limit=args.limit))


def cmd_admin_graph_search(args, store: MemoryStore) -> None:
    results = store.search(query=args.query, mode="graph", limit=args.limit, max_hops=args.hops)
    _json_out(results)


def cmd_admin(args, store: MemoryStore) -> None:
    admin_cmd = getattr(args, "admin_command", None)
    if not admin_cmd:
        print(
            "Usage: memory admin {cleanup|consolidate|decay|dream|demoted|purge|export|import|tags|stats|briefing|graph|auto-extract}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle nested subcommands (tags, graph)
    if admin_cmd == "tags":
        tags_cmd = getattr(args, "tags_command", None)
        if not tags_cmd:
            print("Usage: memory admin tags {list|rename|merge}", file=sys.stderr)
            sys.exit(1)
        _ADMIN_TAGS_DISPATCH[tags_cmd](args, store)
        return

    if admin_cmd == "graph":
        graph_cmd = getattr(args, "graph_command", None)
        if not graph_cmd:
            print("Usage: memory admin graph {build|entities|context|search}", file=sys.stderr)
            sys.exit(1)
        _ADMIN_GRAPH_DISPATCH[graph_cmd](args, store)
        return

    _ADMIN_DISPATCH[admin_cmd](args, store)


# --- Dispatch tables ---

_DISPATCH = {
    "store": cmd_store,
    "search": cmd_search,
    "get": cmd_get,
    "delete": cmd_delete,
    "update": cmd_update,
    "health": cmd_health,
    "doc": cmd_doc,
    "admin": cmd_admin,
}

_DOC_DISPATCH = {
    "store": cmd_doc_store,
    "get": cmd_doc_get,
    "search": cmd_doc_search,
    "list": cmd_doc_list,
    "update": cmd_doc_update,
    "delete": cmd_doc_delete,
}

_ADMIN_DISPATCH = {
    "cleanup": cmd_admin_cleanup,
    "consolidate": cmd_admin_consolidate,
    "decay": cmd_admin_decay,
    "dream": cmd_admin_dream,
    "demoted": cmd_admin_demoted,
    "purge": cmd_admin_purge,
    "export": cmd_admin_export,
    "import": cmd_admin_import,
    "stats": cmd_admin_stats,
    "briefing": cmd_admin_briefing,
    "auto-extract": cmd_admin_auto_extract,
}

_ADMIN_TAGS_DISPATCH = {
    "list": cmd_admin_tags_list,
    "rename": cmd_admin_tags_rename,
    "merge": cmd_admin_tags_merge,
}

_ADMIN_GRAPH_DISPATCH = {
    "build": cmd_admin_graph_build,
    "entities": cmd_admin_graph_entities,
    "context": cmd_admin_graph_context,
    "search": cmd_admin_graph_search,
}


# --- Parser ---


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory", description="Memory CLI — JSON output only.")
    sub = parser.add_subparsers(dest="command")

    # store
    p = sub.add_parser("store", help="Store memory (plain text, JSON object, or JSON array)")
    p.add_argument("content", help="Content string, JSON object, JSON array, or - for stdin")
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--type", dest="memory_type", default="note", help="Memory type")
    p.add_argument("--metadata", "-m", default=None, help="JSON metadata")
    p.add_argument("--dedup", dest="dedup_threshold", default=None, type=float, help="Dedup similarity threshold")
    p.add_argument("--importance", default=None, type=float, help="Importance (0-1)")
    p.add_argument("--force", action="store_true", help="Skip dedup check")
    p.add_argument(
        "--reject-injection",
        action="store_true",
        default=False,
        help="Refuse to store content that matches a prompt-injection pattern (exit 1)",
    )

    # search
    p = sub.add_parser("search", help="Search memories")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--mode", default="hybrid", choices=["semantic", "exact", "hybrid", "fts", "graph"])
    p.add_argument("--limit", "-n", default=10, type=int)
    p.add_argument("--tags", "-t", default="", help="Comma-separated tags")
    p.add_argument("--types", default="", help="Comma-separated memory types")
    p.add_argument("--hops", type=int, default=2, help="Graph mode hops")
    p.add_argument(
        "--no-track-recall",
        action="store_true",
        help="Don't bump recall_count/last_recalled_at/confidence. Use for automated/hook searches.",
    )

    # get
    p = sub.add_parser("get", help="Get memory by hash (prefix supported)")
    p.add_argument("content_hash")

    # delete
    p = sub.add_parser("delete", help="Delete memory by hash, tags, or date")
    p.add_argument("hash_arg", nargs="?", default=None, metavar="HASH")
    p.add_argument("--tags", "-t", default="", help="Delete by tags")
    p.add_argument("--before", default=None)
    p.add_argument("--after", default=None)
    p.add_argument("--dry-run", action="store_true")

    # update
    p = sub.add_parser("update", help="Update memory fields")
    p.add_argument("content_hash")
    p.add_argument("--content", default=None)
    p.add_argument("--tags", "-t", default=None)
    p.add_argument("--type", dest="memory_type", default=None)
    p.add_argument("--metadata", "-m", default=None)
    p.add_argument("--importance", default=None, type=float)

    # health
    sub.add_parser("health", help="Database health check")

    # doc (subcommand group)
    doc_parser = sub.add_parser("doc", help="Document operations")
    doc_sub = doc_parser.add_subparsers(dest="doc_command")

    p = doc_sub.add_parser("store")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--body", default=None)
    p.add_argument("--body-file", default=None)
    p.add_argument("--type", dest="doc_type", default="document")
    p.add_argument("--tags", "-t", default="")
    p.add_argument("--metadata", "-m", default=None)

    p = doc_sub.add_parser("get")
    p.add_argument("content_hash")

    p = doc_sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="auto", choices=["semantic", "fts", "auto"])
    p.add_argument("--limit", "-n", default=5, type=int)
    p.add_argument("--tags", "-t", default="")
    p.add_argument("--type", dest="doc_type", default=None)

    p = doc_sub.add_parser("list")
    p.add_argument("--page", default=1, type=int)
    p.add_argument("--page-size", default=20, type=int)
    p.add_argument("--tags", "-t", default="")
    p.add_argument("--type", dest="doc_type", default=None)

    p = doc_sub.add_parser("update")
    p.add_argument("content_hash")
    p.add_argument("--title", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--body-file", default=None)
    p.add_argument("--type", dest="doc_type", default=None)
    p.add_argument("--tags", "-t", default=None)
    p.add_argument("--metadata", "-m", default=None)

    p = doc_sub.add_parser("delete")
    p.add_argument("content_hash")
    p.add_argument("--dry-run", action="store_true")

    # admin (subcommand group)
    admin_parser = sub.add_parser("admin", help="Maintenance and analytics")
    admin_sub = admin_parser.add_subparsers(dest="admin_command")

    admin_sub.add_parser("cleanup", help="Remove exact duplicates")

    p = admin_sub.add_parser("consolidate", help="Merge near-duplicates")
    p.add_argument("--threshold", default=0.92, type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--exclude-types", default="reference")

    p = admin_sub.add_parser("decay", help="Apply confidence decay")
    p.add_argument("--min-confidence", default=0.0, type=float)

    p = admin_sub.add_parser("purge", help="Hard-delete old soft-deletes")
    p.add_argument("--retention-days", default=30, type=int)
    p.add_argument("--dry-run", action="store_true")

    p = admin_sub.add_parser("dream", help="Composite maintenance pass")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--threshold-new", default=50, type=int)
    p.add_argument("--threshold-age-hours", default=24, type=int)

    p = admin_sub.add_parser("demoted", help="List memories most penalised by demotion ranker")
    p.add_argument("--limit", "-n", default=50, type=int)

    p = admin_sub.add_parser("export", help="Export to JSON")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--no-documents", action="store_true")

    p = admin_sub.add_parser("import", help="Import from JSON")
    p.add_argument("--file", "-f", dest="file_path", default="-")
    p.add_argument("--force", action="store_true")

    p = admin_sub.add_parser("stats", help="Usage statistics")
    p.add_argument("--after", default=None)
    p.add_argument("--before", default=None)
    p.add_argument("--top-recalled", type=int, default=None)
    p.add_argument("--never-recalled", action="store_true")
    p.add_argument("--stale", action="store_true")

    p = admin_sub.add_parser("briefing", help="Session briefing")
    p.add_argument("--budget", default=150, type=int)

    p = admin_sub.add_parser("auto-extract", help="Extract memories from a brief file using LLM")
    p.add_argument("--brief-file", required=True, help="Path to session brief text file")
    p.add_argument("--dry-run", action="store_true", help="Extract but do not insert")
    p.add_argument("--model", default="claude-haiku-4-5", help="Anthropic model to use")
    p.add_argument("--max-entries", default=20, type=int, help="Max entries to extract")

    # admin tags
    tags_parser = admin_sub.add_parser("tags", help="Tag management")
    tags_sub = tags_parser.add_subparsers(dest="tags_command")
    tags_sub.add_parser("list", help="List all tags")
    p = tags_sub.add_parser("rename")
    p.add_argument("old_tag")
    p.add_argument("new_tag")
    p = tags_sub.add_parser("merge")
    p.add_argument("source_tags", help="Comma-separated source tags")
    p.add_argument("target_tag")

    # admin graph
    graph_parser = admin_sub.add_parser("graph", help="Knowledge graph")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")
    p = graph_sub.add_parser("build")
    p.add_argument("--dry-run", action="store_true")
    p = graph_sub.add_parser("entities")
    p.add_argument("--type", dest="entity_type", default=None)
    p.add_argument("--limit", "-n", default=50, type=int)
    p = graph_sub.add_parser("context")
    p.add_argument("entity")
    p.add_argument("--limit", "-n", default=20, type=int)
    p = graph_sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--hops", default=2, type=int)
    p.add_argument("--limit", "-n", default=10, type=int)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    store = MemoryStore()
    try:
        _DISPATCH[args.command](args, store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
