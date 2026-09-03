# Consolidation and lifecycle redesign — proposal

Proposal only. Nothing here is implemented. Each item states the measurement that
justifies it, and items without a measurement behind them are marked as such.

Evidence base:

- `benchmarks/results/README.md` — retrieval A/B, 200 production queries
- `benchmarks/results/consolidate-baseline.json` — fact retention per strategy
- `docs/research/real-usage-report.md` — 202 real injections, 29 240 DB events
- `docs/research/2026-memory-landscape.md` — what the field does differently

Ordered cheapest-first within each tier. The tiers are about evidence strength,
not effort.

---

## Tier 1 — measured problem, obvious fix

### 1.1 Change the default `content_strategy` to `mmr_union` — done, and it matters less than it looked

**Measured** (`tests/bench_consolidate.py`, `benchmarks/results/consolidate-baseline.json`):

| strategy | fact retention |
|---|---|
| **pairwise — dream's actual path** | **62%** |
| `keep_higher_recall` (was the default) | 62% |
| `keep_longer` | 62% |
| `concat` | 100% |
| `mmr_union` (now the default) | 100% |

The default kept the survivor's text verbatim, so every other member's
distinguishing detail was soft-deleted with the row that held it.

**But the pre-checks changed the conclusion**
(`benchmarks/results/consolidate-threshold-safety.json`):

**Consolidation is inert on the current corpus.** `dream` calls
`consolidate()` with every default (`core.py:4666`), which is *pairwise at
0.92* — and pairwise never reads `content_strategy` at all. A dry-run sweep on
the production DB:

| threshold | pairs that would merge |
|---|---|
| **0.92 (shipping default)** | **0** |
| 0.90 | 16 |
| 0.88 | 51 |
| 0.85 | 127 |

It has already eaten the available duplicates — 304 historical merges via
`consolidate`, 348 via `dream`. So the 62% gap is real and currently fires on
nothing. The default is now `mmr_union` as latent safety: correct when it next
matters, no effect today.

**And lowering the threshold to make it fire would be unsafe.** At 0.85 there
are 74 clusters covering 178 memories, with a **median pairwise token overlap
of 0.19** — 69 of the 74 sit below 0.35. Cosine ≥0.85 here is driven by shared
topic and shared template, not shared content. The largest cluster is 13
distinct agent reports that happen to open with the same boilerplate line, at
cosine 0.94. Merging at that threshold would destroy 149 distinct memories.

**A lexical-overlap guard was built for exactly that and reverted.** The idea:
require Jaccard over distinctive (≥6 char) tokens before merging. It is
backwards for the case it was built for. The template-driven clusters score
*highest* on raw lexical overlap — 0.61, 0.72, 0.86 — and pass it untouched.
What it blocks instead is legitimate paraphrase merging: an existing test
fixture of three genuine restatements of one fact scores ~0.13 and was vetoed.

The right discriminator is **IDF-weighted** overlap. Tokens appearing across
many memories — which is what boilerplate is — get near-zero weight, and rare
tokens carry the signal. Raw Jaccard weights both the same, and that is the
bug. Filed separately; not worth building until consolidation has something to
consolidate.

**Retrieval is unaffected either way.** Consolidating a production snapshot
under each strategy and replaying the same 200-query corpus at `+best` gives
MRR 0.970 in all three cases (unmutated, pairwise-merged, mmr_union-merged).
Weak evidence — only ~1.5% of memories merged and no gold hash was among them —
but there is no sign that a stitched-together survivor retrieves worse.


### 1.2 ~~Attack the zero-result rate~~ — done, and it was not what it looked like

**This item is closed.** It was written as "13.1% of searches return nothing —
the largest single failure in the system", and it was wrong. Reading the queries
rather than the count decomposed it into four unrelated things:

| slice | n | what it is |
|---|---|---|
| empty query text | 1 403 (53%) | blank `query` column, zero by construction. 1 249 in April 2026; stopped in May. |
| exact-mode miss | 657 (25%) | `exact` is 83.1% zero *by design*. Nothing is the right answer. |
| slash command | 48 (2%) | a `/command` reaching search. Actionable. |
| remaining | 545 (21%) | 62% verbatim repeats — eval fixtures, not user traffic. |

Excluding empty-query events, hybrid has run 5.5-9.0% zero since May, against
30.8% in April. And the substantive remainder is queries like "Thank you
@Ruslan", "add milk, eggs and coffee to my shopping list", and "what articles
are trending on Hacker News". Returning nothing there is correct behaviour.

`benchmarks/session_analytics.py` now prints this split automatically, with the
headline explicitly labelled as not a defect rate, so it cannot be misread the
same way twice.

**What survives**: one small fix — 48 slash-command queries reached `search`
since May. `hooks/memory-topic-recall.sh` skips slash commands, so another
caller does not. Find it and skip there too.

**What this cost**: a top-priority item that turned out to be noise. The lesson
is cheap and worth writing down — a rate computed over a column nobody had read
is not a measurement, it is a hypothesis. Read the rows.

### 1.3 Stop trusting `recall_count` before 2026-07

**Measured**: 2 152 memories share one of three `last_recalled_at` days — 1 874 on
2026-06-23 alone. Of 2 458 memories older than 90 days, exactly **one** has
`recall_count = 0`. That is a sweep with tracking on, not reading.

`recall_count` feeds the ranking demotion factor, dream's demotion pass, the index
tier-2 score, and active-forget eligibility. All four are reading contaminated
data for the pre-July corpus. Options, in order of preference:

1. Add a `recall_count_since` epoch and score on recalls after it.
2. Reset `recall_count` for memories whose only recall falls on a detected bulk
   day. Destructive and irreversible; needs explicit sign-off.
3. Do nothing but document it. Cheapest, and honest, if lifecycle work is not
   imminent.

`benchmarks/session_analytics.py` now detects and prints bulk-recall days, so at
minimum the contamination will not be silently rediscovered.

---

## Tier 2 — measured gap, needs design

### 2.1 Chunk and embed document bodies — selectively

**Measured** (P4, `benchmarks/results/probe-p4-brevity.json`):

- **Memories are fine.** Only 149 of 5 882 (2.5%) exceed the ~512-token embedding
  ceiling. Chunking the memories table is not justified; leave the truncation.
- **Documents are not.** 56.5 MB of body text, 366 KB of summary — **0.65% of
  document text is embedded**. 554 of 1 081 summaries sit at the 500-char cap,
  so they were themselves truncated. `test_doc_body_semantic_recall` (strict
  xfail) is the behavioural proof: a fact stated verbatim in a body ranks 5th
  of 5 behind an unrelated runbook.

**But do not chunk everything, and this is the part the original proposal got
wrong.** 683 of the 1 081 documents are raw conversation dumps:

| doc_type | n | avg body | avg summary | embedded |
|---|---|---|---|---|
| `session-archive` | 600 | 64 688 | 474 | 0.7% |
| `plan` | 368 | 3 635 | 184 | 5.1% |
| `transcript` | 46 | **351 035** | **68** | 0.02% |
| `session` | 37 | 2 791 | 179 | 6.4% |

`session-archive` and `transcript` carry roughly 40 MB of the 56.5 MB. And
`auto_archive_pending` was **deliberately unwired** from SessionEnd
(`hooks/memory-session-end.sh:66-72`) because it produced a searchable pile of
raw conversation rather than facts. Chunking them would embed precisely what
that decision rejected, at roughly 80 000 vectors — and would push the corpus
straight through the CSLS scaling wall (3.3) as a side effect.

**Revised scope**: chunk the ~398 non-conversation documents — `plan`, `spec`,
`runbook`, `reference`, `decision`, `pattern`. Small bodies (avg 3.6 KB), real
knowledge, roughly 3 000 chunks. Leave the conversation dumps alone pending a
separate decision about whether they should exist at all.

Shape: 512-token windows with overlap, chunk rows keyed to the parent document,
a chunk-level sibling to `document_embeddings`, retrieval returning the parent
doc deduplicated across its chunks. Flip the xfail when done.

### 2.2 An explicit update path

**Measured**: `stale_fact` category MRR is 0.833 on the synthetic corpus — 1 of 3
superseded facts already outranks its replacement. `dream`'s supersession pass
fires only on cosine ≥0.85 **and** ≥2 shared tags **and** (a contradiction keyword
**or** type ∈ {decision, error}). A `reference` whose replacement happens not to
contain the word "instead" is invisible to it, permanently.

Two halves:

- **`memory update --supersedes <hash>`** — an explicit, human-driven path. Small.
- **Conflict detection on store** — when a new memory is ≥0.90 to an existing one
  of the same type with shared tags, *surface the conflict* rather than storing
  both silently. Note the store path already computes exactly this comparison for
  dedup (`core.py:1477`), so the signal is in hand; what is missing is a third
  outcome besides "duplicate, rejected" and "distinct, stored".

Then populate `memory_graph.valid_from` / `valid_to`. **The bi-temporal columns
already exist and are unused** (`core.py:888`, `1065`); `search(as_of=...)` already
threads a timestamp into graph traversal. This is wiring, not schema work.

### 2.3 Rebalance what the injection slots hold

**Measured**: `decision` memories are reused **28.8%** of the time versus
`reference` at **3.6%** — an 8× gap — yet `reference` occupies 110 of 657 injected
slots against `decision`'s 66. The hook injects five lines per session; the
composition of those five is a pure-upside change.

Also measured: the ≥0.70 score band has a **0.0%** reuse rate while ≥0.60 has
22.2%. Read `real-usage-report.md` §3 for why that is partly a measurement
artifact and partly a real finding — the short version is that a memory very close
to the prompt tells the agent what it just read. The implication is a novelty or
diversity term on the *injection* decision, not a higher score floor. Raising
`--min-score` would remove the useful 0.55–0.65 band and keep the useless top.

**Do not implement this from these numbers alone.** n=66 for `decision`, and the
reuse metric brackets rather than pins (15.3% ordered vs 2.0% strict). Confirm on
a second window before changing weights.

---

## Tier 3 — known risk, not yet biting

### 3.1 Entity resolution at write time

`extract_entities` is ~85 hardcoded technology keywords plus ticket/PR/service
regexes (`core.py:726`). Surface forms are never canonicalised, so "the auth
service", "auth-svc" and "authentication service" are three entities. 2 293
entities exist to seed an alias table.

No measurement justifies this yet — the graph is barely used. Which is itself the
point: see 3.2.

### 3.2 Typed edges, or drop the graph

All 19 886 `entity_relations` are `co_occurrence`. `memory_graph` supports
`supersedes` / `contradicts` / `refines` / `references` / `merged_into` and holds
308 rows. Co-occurrence edges largely re-derive what cosine already found, so
graph mode is paying O(k²) write cost per memory (`_link_entities`, `core.py:1285`)
for retrieval that overlaps heavily with the semantic ranker.

Two honest options, and the current state is neither:

- Add typed edges and make traversal earn its cost. gbrain reports +31.4 P@5 from
  its typed graph over vector-only (its own benchmark, directional only).
- Measure graph mode against hybrid on the production corpus and, if it adds
  nothing, retire it and reclaim the write cost.

**Measure before building.** `bench_replay` does not currently cover graph mode.

### 3.3 Bound the CSLS hubness computation

`compute_hubness` is an O(n²) full-corpus matmul, cached per process, on the
**default** retrieval path (`core.py:322`, `2380`). The code's own comment puts it
at ~130 ms for 3k documents and calls it unsuitable past ~10k. The corpus is at
6 767 and growing roughly 1 200/month, so this becomes a problem in about six
months — sooner if 2.1 adds chunk vectors to the same table.

CSLS is worth keeping: it contributes **+3.7pp MRR** on its own and is most of
`weighted_best`'s +4.4pp. The fix is to bound it, not remove it — sample-based
hubness over a fixed subset, or an ANN index. Either way, re-run
`benchmarks/results/replay-proddb-baseline.json` to confirm the approximation
does not eat the gain it exists to provide.

### 3.4 Markdown export mirror

gbrain's structural advantage is that markdown in git is the system of record and
the database is a rebuildable index: you can `git diff` what the agent learned
overnight, review writes line by line, and rebuild if the DB is lost. Here, the
SQLite file *is* the source of truth — no diff, no review surface, no rebuild.

The lazy version of that benefit is an **export**, not a migration: `memory admin
export --markdown` into a directory, committed on a schedule. Keeps SQLite as the
engine, adds a human-reviewable audit trail, costs nothing at query time. An
`export` command already exists (`cli.py`); this is a formatter on top of it.

---

## Method probes — test the borrowed ideas before adopting them

Section 5 of `2026-memory-landscape.md` lists ideas worth stealing. None of them
should be built on the strength of someone else's benchmark. Three times already
in this codebase the field's default turned out to be the wrong choice here — RRF
(-16.1pp), the cross-encoder (+1.3pp for 120x latency), and LLM-assisted merge
(unnecessary; the extractive path already retains 100%). Each of those was caught
by a measurement that took under an hour.

So each borrowed method gets a probe first: a fixture, a metric, and a stated
threshold that would justify adoption. A probe that cannot fail is not a probe.
Every one below is designed to fail on the current build — that failing baseline
is the deliverable.

### P1 — Temporal retrieval and write-time conflict detection

*Tests: Zep/Graphiti bi-temporal facts, Mem0 conflict detection.*

Two fixtures, both extending `benchmarks/corpus.py`:

- **`as_of` queries.** Store a fact, supersede it, then ask for the state at a
  timestamp between the two. Correct answer is the *old* fact. Today `as_of` only
  reaches graph traversal (`core.py:2492`); the semantic and FTS rankers ignore it
  entirely, so this returns the current fact regardless.
- **Conflict pairs.** Store a memory, then store a conflicting one of the same type
  with shared tags at cosine >=0.90. Assert the store result reports a conflict.
  Today it reports `stored` and both coexist forever.

**Adoption threshold**: `as_of` accuracy 0.9+ on the fixture, and conflict
detection with a false-positive rate under 5% measured against the *existing*
near-duplicate clusters in the production DB — a conflict detector that fires on
ordinary duplicates is worse than none, because it will train the operator to
ignore it.

**Cheap because**: the ≥0.90 comparison already runs on every store for dedup
(`core.py:1477`), and `valid_from`/`valid_to` already exist unused.

### P2 — Does write-time linking pay? A multi-hop probe

*Tests: A-MEM's memory-evolves-on-write, HippoRAG's graph traversal argument.*

A-MEM's distinctive claim is that a new note updates the notes it links to, so the
network refines itself continuously rather than in a batch. Before building that,
establish whether **multi-hop retrieval is a real need here at all**.

Fixture: a `multi_hop` case family where the answer requires joining two memories
that share no query vocabulary — "which service did the incident that caused the
rollback belong to", answerable only by chaining incident → rollback → service.

Run it three ways: hybrid (current default), graph mode, and hybrid+graph.
`bench_replay` currently hardcodes `mode="hybrid"` and cannot express this — add
mode to the config dict.

**Adoption threshold**: if graph mode does not beat hybrid on `multi_hop` by a
clear margin, neither A-MEM-style linking nor typed edges are worth building, and
the honest move is to retire graph mode and reclaim the O(k²) per-write cost. If
it does, this is the evidence that justifies the typed-edge work.

This probe decides the graph's fate. Do it before any entity-resolution work.

### P3 — Index churn across rebuilds

*Tests: ACE's context-collapse claim, against `build_index`.*

ACE's argument is that rewriting a context wholesale on every update erodes
detail, and that incremental itemized updates avoid it. `build_index`
(`core.py:5637`) rebuilds wholesale every time, at 60 lines against ~5000 eligible
memories — under 2% of the corpus.

Probe: snapshot the DB, rebuild the index N times while adding memories between
rebuilds, and measure **retention** — what fraction of tier-1 entries present in
build *k* survive into build *k+1*. Also record the churn rate for entries that
have not changed.

**Adoption threshold**: if retention is high, the wholesale rebuild is fine and
ACE's concern does not apply at this budget — record that and close it. If tier-1
entries are churning out because tier-2 scores fluctuate, that is context collapse
in the literal sense and incremental updates are justified.

**Note the footer bug this will surface**: the index's "Demoted" count is
`excluded_count + demoted_from_cap` (`core.py:5819`) — mostly budget overflow, not
a lifecycle state. It reads as ~4500 demoted memories and is nothing of the kind.
Fix the label while you are in there.

### P4 — Brevity bias: does the summary carry the answer?

*Tests: ACE's brevity-bias failure mode, on three surfaces at once.*

`tests/bench_consolidate.py` already measures fact retention through a merge. The
same question applies to every other place this service compresses:

| Surface | Budget | Probe |
|---|---|---|
| index line | 80-char preview (`core.py:5765`) | can the entry be identified from the preview alone? |
| hook injection | ~200-char summary line | is the actionable part inside the truncation? |
| document summary | 500 chars, the only embedded text | already proven lossy — `test_doc_body_semantic_recall` |
| embedding input | 512 tokens, hard truncation | what fraction of stored memories exceed it? |

The last one is a one-line query and worth running immediately: if a meaningful
share of memories are being silently clipped at 512 tokens, that is a live recall
bug, not a design tradeoff.

**Adoption threshold**: none — this is diagnosis, not a method to adopt. It sizes
the problem the other probes assume.

**RUN, 2026-09-03** (`benchmarks/results/probe-p4-brevity.json`). Result: memory
truncation is a non-issue at 2.5%; document truncation is severe at 0.65% of
56.5 MB embedded, but 40 MB of that is raw conversation that was deliberately
excluded from ingest. Rewrote 2.1 from "chunk document bodies" to "chunk the 398
non-conversation documents", and opened a separate question about whether the
683 conversation dumps should exist. The index-line and injection-line surfaces
remain unmeasured — they need a can-you-identify-it judgement, not a length
query.

### P5 — Injection diversity, simulated offline

*Tests: the finding that retrieval score inverts as a usefulness predictor.*

The measurement in `real-usage-report.md` §3 says a memory scoring ≥0.70 has a
0.0% observed reuse rate because it mostly restates the prompt. The proposed fix
is a novelty term on the injection decision — but that is one measurement on
n=42, so simulate before shipping.

Probe, entirely offline against data already on disk: for each of the 202 recorded
real injections, recompute the five-line selection under (a) the current top-5 by
score and (b) an MMR-style selection trading score against dissimilarity to the
prompt and to already-selected lines. Then compare, on the same transcripts:

- mean pairwise similarity among the selected five (redundancy)
- mean similarity to the prompt (novelty)
- projected reuse, using the per-memory reuse already computed by
  `session_analytics.py`

**Adoption threshold**: MMR selection must raise projected reuse on the recorded
injections. If it only reduces redundancy without moving reuse, it is a metric
that improved and a system that did not.

**Honest limit**: this reuses the same 202 injections that produced the
hypothesis, so it can confirm internal consistency but cannot validate. A second
measurement window is still required before changing the hook.

### Probe summary

| Probe | Tests | Decides |
|---|---|---|
| P1 | bi-temporal, conflict detection | whether the update path is built as proposed |
| P2 | write-time linking, graph traversal | whether the graph lives or is retired |
| P3 | wholesale index rebuild | whether incremental index updates are needed |
| P4 | compression budgets | sizes the problem; no adoption decision |
| P5 | injection diversity | whether a novelty term ships |

All five write committed JSON into `benchmarks/results/`, same convention as the
existing baselines, so the decision and the evidence stay together.

## Explicitly not proposed

| | Why |
|---|---|
| Postgres / Neo4j migration | Forfeits local-first — the property no hosted competitor matches — to solve nothing currently measured. |
| Mem0 / Zep as a backend | Same, plus a network round trip inside a hook with a 4 s timeout. |
| `weighted_best` → RRF | RRF measures **16.1pp worse MRR** here (0.765 vs 0.970, n=200). |
| Cross-encoder rerank on by default | +1.3pp for 714 ms p50 cold, against +4.4pp at 6 ms from `weighted_best`. Retiring it is the more defensible change. |
| **LLM-assisted cluster merge** | The stated gate was "only if extractive merge loses facts." Measured: `mmr_union` retains 100%. The gate did not open. Revisit only if 1.1's coherence check fails. |
| Porting LoCoMo / LongMemEval | Conversational shape, and increasingly measures context length rather than memory quality. |

## Suggested order

~~`1.2`~~ (closed, see above) → `1.1 (default strategy)` → `1.3 (recall_count)`
→ `2.1 (chunking)` → `2.2 (update path)` → `3.3 (bound CSLS)` → `2.3 (injection
mix, after a second measurement window)` → `3.2 (decide the graph's fate)`.

1.2 ran first precisely because it could have reordered the rest. It did — by
removing itself. 1.1 is now the highest-value item, and unlike 1.2 its evidence
comes from a harness that reads the actual merged text rather than a count.
