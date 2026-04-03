---
name: remember
description: Extract and store key facts, decisions, and patterns from conversation into persistent memory. Automatically classifies content type and route (memory vs document). Run immediately.
argument-hint: "[what to remember] [--global]"
allowed-tools: Bash
---

# /remember

Reads `$ARGUMENTS` and stores it to memory immediately with correct classification.

Set `PROJECT=$(basename "$(pwd)")` before commands.

## Step 1: Classify Content

**Route**:
- ≤800 chars, atomic fact → `memory store` (single memory)
- >800 chars, multi-section → `memory doc store` (document)

**Type** (from content keywords):
- "decided", "approved", "chose" → `decision`
- "pattern", "convention", "always do" → `pattern`
- "error", "root cause", "bug" → `error`
- "discovered", "learned", "surprising" → `learning`
- "link", "pointer", "see", "reference" → `reference`
- "status", "today", "currently" → `observation`
- Otherwise → `note`

**Scope**:
- `$ARGUMENTS` contains "--global" or "global" → tag `scope:global`
- Otherwise → tag `project:$PROJECT`

## Step 2: Execute NOW

### Single memory (≤800 chars):
```bash
PROJECT=$(basename "$(pwd)")
memory store "<YOUR_CONTENT_HERE>" --tags "project:$PROJECT,<inferred-topic>" --type <inferred-type> --dedup 0.90
```

### Batch memories (2+ facts from same prompt):
If the message contains multiple distinct facts, use JSON batch format (single efficient call):
```bash
PROJECT=$(basename "$(pwd)")
memory store '[
  {"content":"<fact 1>","type":"<type1>","tags":["project:'$PROJECT'"]},
  {"content":"<fact 2>","type":"<type2>","tags":["project:'$PROJECT'"]}
]' --dedup 0.90
```

### Long-form document (>800 chars):
```bash
PROJECT=$(basename "$(pwd)")
memory doc store --title "<auto-detected or inferred title>" \
  --summary "<one-line summary for search>" \
  --body "<full content from $ARGUMENTS>" \
  --tags "project:$PROJECT" --type plan
```

## Step 3: Output

Run the command and print the returned hash as confirmation:
```json
{
  "content_hash": "abc123...",
  "status": "stored",
  "message": "Memory stored successfully"
}
```

---

## Memory Types Reference

| Type | Use for | Example |
|------|---------|---------|
| `decision` | Architecture choices, approved approaches | "We decided to use modernbert-embed-base for embeddings" |
| `pattern` | Conventions, recurring solutions | "Always validate at system boundaries, trust internal code" |
| `reference` | Pointers to external resources | "Grafana dashboard: grafana.internal/d/api-latency" |
| `error` | Root causes, fix recipes | "MLX bfloat16 requires .astype(mx.float32) before numpy conversion" |
| `learning` | Non-obvious discoveries | "MTEB leaderboard scores don't correlate with exact-match precision" |
| `observation` | Status snapshots, context | "As of Apr 3: oMLX has Qwen3.5-35B-A3B, 38 TPS, 100% tools" |
| `note` | General (default) | Any other factoid |

## Tags

Standard taxonomy: `project:<name>`, `scope:global`, `cloud:<provider>`, `svc:<service>`, `tool:<tool>`.

## Important

- Always use `--dedup 0.90` (skips storage if near-duplicate exists with same type+tags)
- Importance is auto-inferred from keywords (CRITICAL→0.9, IMPORTANT→0.8) and type
- For batch: use JSON array format for 2+ facts in one prompt (one round-trip, more efficient)
