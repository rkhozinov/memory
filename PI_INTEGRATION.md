# Memory Service Integration with Pi

This document describes how the memory service integrates with the pi agent via a custom extension.

## Quick Start

1. **Restart pi** to load the memory extension:
   - If pi is running, run `/reload` in the pi interface
   - The extension is auto-discovered from `~/.pi/agent/extensions/memory.ts`

2. **Verify the extension is loaded**:
   - Look for the notification: `🧠 Memory service extension loaded`
   - Run `/memory-status` to check service health

## Available Tools

The pi extension provides the following tools that the LLM can call:

### Memory Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `memory_store` | Store new memories with tags, type, and importance | Store facts, decisions, patterns, learnings |
| `memory_search` | Search memories by semantic similarity or exact match | Find relevant context from past sessions |
| `memory_get` | Retrieve a specific memory by hash | Get full details of a memory |
| `memory_delete` | Delete memories by hash, tags, or date | Clean up outdated information |
| `memory_update` | Update memory metadata (tags, type) | Modify categorization without recreating |
| `memory_health` | Check memory service health | Verify service is running |
| `memory_list` | List memories with pagination | Browse stored memories |

### Document Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `document_store` | Store long-form documents (plans, specs, runbooks) | Store session summaries, architecture docs |
| `document_search` | Search documents by topic | Find relevant documents |
| `document_get` | Retrieve a full document by hash | Get complete document content |
| `document_list` | List documents with pagination | Browse stored documents |
| `document_delete` | Soft-delete a document | Remove outdated documents |

### Graph Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `graph_build` | Build/rebuild knowledge graph from memories | Update entity relationships |
| `graph_search` | Traverse graph to find connected memories | Discover relationships via shared entities |

## Commands

User-accessible commands:

- `/memory-status` - Check memory service health and get statistics
- `/memory-briefing [budget]` - Generate compact briefing of top memories (default budget: 150 lines)

## Memory Types

The memory service automatically classifies content based on keywords:

- `decision` - "decided", "approved", "chose"
- `pattern` - "pattern", "convention", "always do"
- `error` - "error", "root cause", "bug"
- `learning` - "discovered", "learned", "surprising"
- `reference` - "link", "pointer", "see", "reference"
- `observation` - "status", "today", "currently"
- `note` - Default type for general information

## Example Usage

### Storing a Memory

When the LLM identifies important information, it can call:

```
memory_store(
  content: "Decided to use Terraform for AWS infrastructure",
  memory_type: "decision",
  tags: ["aws", "terraform", "infrastructure"],
  importance: 0.8
)
```

### Searching for Context

When starting a new session on a topic:

```
memory_search(
  query: "AWS infrastructure setup",
  mode: "hybrid",
  limit: 10
)
```

### Storing a Session Summary

Store important session summaries for future reference:

```
document_store(
  title: "Kubernetes Cluster Migration Plan",
  body: "# Migration Plan\n\n...",
  summary: "Plan for migrating Kubernetes clusters from EKS to self-hosted",
  doc_type: "plan",
  tags: ["kubernetes", "migration", "cluster"]
)
```

### Graph Search

Find related memories through entity relationships:

```
graph_search(
  query: "kubernetes",
  max_hops: 2,
  limit: 15
)
```

## Memory Storage

Memory data is stored by the memory service at:
- Default: `~/repos/memory/data/memory.db`

## Skills

The memory service includes skills that work with Claude Code:
- `~/repos/memory/skills/recall/` - `/recall` command for retrieval
- `~/repos/memory/skills/remember/` - `/remember` command for storage
- `~/repos/memory/skills/forget/` - `/forget` command for deletion
- `~/repos/memory/skills/memory-status/` - `/memory-status` command

These skills provide an alternative interface to the same functionality and can be symlinked into your project's `.claude/skills/` directory.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Pi Agent                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Memory Extension (memory.ts)               │   │
│  │  - Wraps memory CLI commands                          │   │
│  │  - Registers tools for LLM to call                     │   │
│  │  - Formats results for display                        │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ node:child_process                    │
└─────────────────────┼───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Memory CLI (~/.local/bin/memory)           │
│  - JSON input/output                                        │
│  - Semantic search (embeddings)                             │
│  - Document storage                                         │
│  - Knowledge graph                                          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Memory Store (SQLite + Embeddings)              │
│  - ~/repos/memory/data/memory.db                    │
│  - MLX-based local embeddings                               │
│  - Full-text search                                          │
└─────────────────────────────────────────────────────────────┘
```

## MCP Server Alternative

The memory service also provides an MCP server (`memory-mcp-server`) that can be used directly by MCP-compatible clients. The pi extension uses the CLI for simplicity, but you could alternatively:

1. Run the MCP server: `memory-mcp-server`
2. Create an extension that connects via stdio
3. Or integrate MCP directly if pi adds native MCP client support

## Troubleshooting

### Extension not loading

- Check syntax: `bun build ~/.pi/agent/extensions/memory.ts --no-bundle`
- Verify path: `ls -la ~/.pi/agent/extensions/memory.ts`
- Restart pi and run `/reload`

### Memory CLI not found

- Check installation: `which memory`
- Ensure it's in PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Service health errors

- Run `/memory-status` to diagnose
- Check if memory daemon is running: `memory health`
- Verify database permissions: `ls -la ~/repos/memory/data/`

## Future Enhancements

Potential improvements to consider:

1. **Automatic memory extraction** - Extension could automatically extract and store important facts from each session
2. **Session summaries** - Auto-generate document summaries at session end
3. **Graph auto-build** - Rebuild graph after significant memory additions
4. **Skill integration** - Create pi skills that call the memory tools
5. **MCP client** - Direct MCP server connection if pi adds support
