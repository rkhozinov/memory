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

**What survived, and then closed itself**: slash-command queries reaching
`search`. Investigated — it is already 85% fixed and the rest is not worth
touching.

`hooks/memory-topic-recall.sh` gained a skip on 2026-08-03 (`ce59936`). Monthly
counts either side: May 89, June 60, July 86, then **14 across the rest of
August**. The hook was the main source and the fix worked.

The residual trickle (~1-3/day) is not from this repo. The hook's skip is
correct, and nothing under `~/repos/handoff` or the plugin cache shells out to
`memory search`. Most likely an agent calling the CLI or the MCP tool directly
with a prompt-shaped string. At ~14 events/month and ~300 ms each, that is
about four seconds of CPU a month — a defensive guard in `core` would cost more
to maintain than it saves. **Won't fix.**

**What this cost**: a top-priority item that turned out to be noise. The lesson
is cheap and worth writing down — a rate computed over a column nobody had read
is not a measurement, it is a hypothesis. Read the rows.

### 1.3 ~~Stop trusting `recall_count`~~ — measured, and it is inert

**The contamination is real.** Three days each bumped ≥100 memories at once —
2026-06-23 (1 874), 2026-08-30 (147), 2026-08-25 (131). Of 2 458 memories older
than 90 days, exactly one has `recall_count = 0`. That is a sweep with tracking
on, not reading.

**The 2 152 figure over-counts, though.** Most of those memories have other
recalls too, so their count is not purely an artifact. Only **477** have a
single recall that lands on a sweep day.

**And zeroing those 477 changes nothing measurable.** Cloning the production DB,
zeroing them, and comparing every consumer:

| consumer | contaminated | cleaned | difference |
|---|---|---|---|
| index selection | 32 entries | 32 entries | **byte-identical** |
| retrieval MRR (`+best`, 200 queries) | 0.970 | 0.970 | none |
| `dream` demote pass | 0 | 35 | 35 of 5 869 (0.6%) |

The demote pass is the only consumer that moves, and demotion's effect is
exclusion from the index — which is identical between the two. So the
difference produces no observable change.

**Closed. A `recall_count_since` epoch column was scoped and would have been
built for nothing.** `benchmarks/session_analytics.py` detects and prints
bulk-recall days, which is the right amount of effort to spend here.

**Revisit if active forgetting is ever enabled** (`MEMORY_ACTIVE_FORGET=1`).
That path soft-deletes on an activation score that reads `recall_count` and
`distinct_session_count`, and unlike demotion it is destructive. Re-run this
comparison before turning it on.


## Tier 2 — measured gap, needs design

### 2.1 Chunk and embed document bodies — selectively ✅ DONE

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

**Shipped.** `document_chunks` + `document_chunk_embeddings`, 2000-char windows
with 200-char overlap seeking a paragraph/line/sentence boundary, capped at 40
chunks per document. `store_doc` and `update_doc` keep them current;
`memory admin reindex-chunks` backfills.

Backfill on the production corpus: **398 documents, 986 chunks, 683 conversation
dumps correctly skipped**, 18 seconds, and no measurable growth in a 262 MB DB.

Measured on 60 body-only probes (a sentence from the second half of a body,
discarded if the summary already covers it):

| | MRR@10 | Recall@10 | Top1 |
|---|---|---|---|
| before (summary only) | 0.830 | 0.900 | 0.800 |
| after (body chunks) | **0.886** | **0.950** | **0.850** |

The baseline is high because `document_fts` already indexes the body, so a
*verbatim* body query was reachable lexically all along. This probe uses verbatim
sentences and therefore measures the smaller half of the gain. The half FTS
structurally cannot serve is paraphrase — `test_doc_body_semantic_recall` went
from a strict xfail at rank 5 of 5 to passing at rank 1, and is kept as a
regression test rather than deleted.

Memory retrieval is unchanged at MRR 0.970: chunks live in their own table and
never enter the memory path.

One design note worth keeping. The first cut skipped documents whose body fits
in a single chunk, reasoning that the summary already covered them. That is
precisely the assumption that created the gap — the summary is a separate
hand-written text, not a prefix of the body — and it left the probe failing.
Every chunked document now gets at least one chunk vector.

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

**RUN, 2026-09-03** (`benchmarks/results/probe-p1-temporal-conflict.json`,
`tests/bench_probe_p1.py`). Three results:

1. **`as_of` accuracy 33%** against the 90% threshold. `as_of` only reaches
   graph traversal; the semantic and FTS rankers ignore it, so a past-tense
   query returns the present. The one apparent pass is not evidence — it is the
   case where the current-tense control also missed, so the old fact won by
   accident.
2. **Conflicts detected: 0%**, and worse than silence. One case returned status
   `duplicate` — the replacement fact was **rejected and lost**. `store()` has
   two outcomes, `duplicate` and `stored`; an update that resembles what it
   replaces takes the first one. That is a live bug, filed separately, and it
   blocks the update path: building one is pointless while updates are being
   discarded.
3. **False-positive sweep** over 179 700 real pairs, on the candidate rule
   (cosine ≥ t, same type, ≥1 shared tag):

   | threshold | flags per 100 memories |
   |---|---|
   | 0.90 | 0.2 |
   | 0.85 | 1.5 |
   | **0.80** | **2.0** |
   | 0.75 | 3.7 |
   | 0.70 | 8.3 |

   At 0.90 the rule fires on essentially nothing — the same saturation that makes
   consolidation inert at 0.92. **0.80 is the operating point**: ~2 flags per 100
   memories, about 24 prompts a month at the current write rate, tolerable when
   the action is *surface this* rather than *merge these*.

   Honest limit: this bounds **noise, not precision**. The corpus has no labelled
   conflicts, so 2 flags per 100 could be 2 real conflicts or 2 false alarms. It
   says a detector would not drown the user. It does not say it would find
   anything.

**Adoption**: build both — `as_of` as wiring against a 33% baseline, conflict
detection as an advisory signal at 0.80 that never auto-merges. Fix the
silent-rejection bug first.

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
the honest move is to retire graph mode and reclaim the O(k²) per-write cost.

**RUN, 2026-09-03** (`benchmarks/results/probe-p2-multihop.json`). Population:
30 memory pairs sharing ≥2 entities but semantically distant (cosine < 0.45).

| config | MRR | Recall@10 | median |
|---|---|---|---|
| hybrid | 0.000 | 0.000 | 8 ms |
| graph hops=1 | 0.007 | 0.033 | 26 ms |
| graph hops=2 | 0.005 | 0.033 | 690 ms |
| graph + similarity rerank (pool 300) | 0.003 | 0.033 | 35 ms |
| *ceiling — gold present in a 300-deep pool* | | *0.500* | |

**Threshold not met.** 3.3% against 0.0% is a real difference — the graph does
reach things hybrid cannot — but it is not a capability, and hops=2 costs 690 ms
against hybrid's 8 ms.

The diagnosis is more useful than the verdict:

- **Traversal works.** The gold sits in a 300-deep 2-hop pool in 15 of 30 cases.
- **Ranking is the bottleneck.** It reaches the top 10 in 3.3%. The graph score is
  `0.5/(1+hops) + 0.3·importance + 0.2·recency` (`core.py:2596`) — **no query term
  at all**. A hub entity links hundreds of memories, all at hops=1, all scored
  identically bar importance and recency.
- **The obvious fix is structurally wrong.** A similarity rerank does not help and
  slightly hurts. It cannot: this population is *defined* by being semantically
  distant from the query, so reranking by cosine pushes the right answer down. The
  graph's value is precisely the targets similarity cannot find, so its ranking
  signal must come from structure, not the embedding.
- **What is missing is entity specificity.** A shared `terraform` is near-zero
  evidence; a shared rare entity is strong evidence. Seeding and scoring weight
  both the same. This is the *same* IDF insight as the consolidation guard: the
  discriminator is how **common** a shared feature is, and raw counts do not
  encode that.

**Do not build** typed edges, A-MEM-style write-time linking, or write-time
entity resolution. One bounded experiment first: IDF-weighted entity seeding and
path scoring, against a measured 50% recall ceiling. If that moves 3.3%
appreciably toward it, typed edges become the obvious next step. If not, retire
graph mode and reclaim the per-write cost.

**Probe-design note, recorded because it nearly produced a wrong verdict.** The
first cut used a bare sentence from memory A as the query and scored 0.000
everywhere. That was a bug in the probe, not a fact about the graph: the entities
A and B share come from their *full* content, and one sentence of A need not
contain the bridge. The corrected probe names the shared entity so the traversal
is seeded with the bridge it is meant to cross.

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

**RUN, 2026-09-03** (`benchmarks/results/probe-p3-index-churn.json`). Eight
rebuilds, 25 memories added between each:

| | |
|---|---|
| mean retention | **100.0%** |
| worst round | 100.0% |
| mean position shift | 0.00 places |

So context collapse does not happen. **But that perfect stability is not health,
it is ossification** — and the follow-up is the actual finding. Storing a fresh,
high-importance, correctly-typed, correctly-tagged `decision` and rebuilding does
**not** put it in the index. The index is a fixed set of 32 entries that new
knowledge cannot enter.

Two causes, both fixable and neither is what ACE describes:

1. **The token cap binds, not the line cap.** The hook passes `--max-lines 60
   --max-tokens 1200` and gets 32 lines. At ~150 chars per line the token budget
   is exhausted at 32 entries, so the 60 never binds. It reads like the
   constraint and is not.
2. **Tier 1 has no ranking.** 1 447 `decision`/`reference` memories are tier-1
   eligible and compete for ~32 slots in *unordered curation order*
   (`core.py:5718-5728`). `--tags` stable-sorts in-scope entries to the front but
   does not rank within them. A new important decision queues behind 1 447 others
   with no mechanism to get ahead of any of them. Tier 2 *does* have a score;
   tier 1, the higher-priority tier, does not.

**Revised conclusion.** ACE's argument lands, but not as stated. The failure is
not collapse through repeated rewriting — it is ossification through unranked
wholesale selection. Incremental itemized updates would not fix it. Ranking
tier 1 does.

**FIXED, same day** (`benchmarks/results/probe-p3-index-churn-after-fix.json`).
Tier 1 now scores `auto_mult · recency · importance · (1 + log1p(recall_count))`
and sorts by it, mirroring tier 2.

| | before | after |
|---|---|---|
| mean retention | 100.0% | 80.2% |
| mean position shift | 0.00 | 6.88 |
| new high-importance decision enters | **no** | **yes, at rank 1** |
| low-value observation enters | no | no |

Retention *falling* is the fix working: 100% was ossification. A fresh decision
now displaces exactly one entry rather than being locked out. Watch this number —
if it collapses toward zero the tier-1 score has become too volatile and the
index will thrash between sessions.

Two smaller things fixed alongside. The `--max-lines 60 / --max-tokens 1200`
pair is now documented in `hooks/lib/hooklib.sh` as what it actually is: only the
token cap binds, at ~32 lines, and the line cap is a safety valve against
pathologically short entries. And the footer that read `## Demoted (search-only)`
with a count near 4 500 — which is budget overflow, not a lifecycle state, and
which this research misread as one — now reads `## Not shown here (search-only)`
and says the entries did not fit the budget.

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

**RUN, 2026-09-03 — REFUTED** (`benchmarks/results/probe-p5-injection-diversity.json`).
185 real injections, candidate pools of 20:

| selector | useful-line rate | redundancy |
|---|---|---|
| **current: top-5 by score** | **22.3%** | 0.053 |
| MMR λ=0.9 | 20.9% | 0.023 |
| MMR λ=0.7 | 16.8% | 0.008 |
| MMR λ=0.5 | 15.8% | 0.007 |
| MMR λ=0.3 | 15.1% | 0.006 |
| *oracle (pick what turned out useful)* | *40.5%* | *0.040* |

MMR is **monotonically worse**, and the more diversity is weighted the worse it
gets. Redundancy fell exactly as designed while the useful rate fell with it —
precisely the failure the threshold was written to catch.

**This also corrects the finding that motivated it.** `real-usage-report.md` §3
showed the ≥0.70 score *band* had 0% reuse and read that as "score does not
predict usefulness". P5 shows that within a candidate *pool*, score predicts
usefulness better than novelty does. Those are different questions and the
earlier write-up conflated them: across injections a high band score means the
memory restated the prompt; within one injection the highest-scoring candidates
are still the best available.

**Headroom is real but the lever is not this one.** The oracle reaches 40.5%
against the current 22.3%. A better selector exists. It is not MMR over lexical
overlap, and 2.3 below should not be built on the novelty argument.

Note the first cut of this probe returned every selector tied at 100%, oracle
included. The transcript records only the five lines injected, never the pool it
chose from — with five candidates and five slots there is nothing to select. The
pools here are reconstructed by re-running retrieval with the original prompt,
which makes them directional rather than a replay.

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

## Decision: the cross-encoder reranker is retired

Removed: `src/memory/rerank.py`, `tests/test_rerank.py`, the `--rerank` /
`--rerank-top-n` CLI flags, the `rerank` / `rerank_top_n` MCP arguments, the
`rerank` parameter on `MemoryStore.search()`, and the `MEMORY_AUTO_RERANK`,
`MEMORY_RERANK_REPO`, `MEMORY_RERANK_FILE`, `MEMORY_RERANK_TOKENIZER` env vars.

The reasoning is not just cost. On the 200-query production corpus the
cross-encoder loses to the shipping `weighted_best` fusion in **every** query
category:

| category | `+best` | `+rerank` |
|---|---|---|
| paraphrase | **0.989** | 0.955 |
| cross_topic | **0.978** | 0.952 |
| identifier | **0.812** | 0.771 |
| overall MRR@10 | **0.970** | 0.939 |

And stacking it on the shipping config (`+all` = `weighted` + rerank +
activation) scores 0.939 — it *removes* 3.1pp from what already ships. There is
no measured population where it helps, so "keep it opt-in" was preserving a
switch whose only documented effect is to make results worse, at 714 ms p50 cold
against 6 ms.

What removal reclaims: 279 lines, an ~80 MB ONNX model download on first use, a
second SQLite cache DB with its own eviction path, and four env vars. No
dependency is dropped — `onnxruntime` and `tokenizers` are shared with
`embeddings.py`.

**The limit on this evidence, stated plainly.** `build_real_corpus.py` discards
candidates that baseline retrieval cannot already reach, so the 200-query corpus
contains only queries the bi-encoder already answers (baseline R@1 = 0.875).
That is precisely the population with the least headroom for a reranker, and it
excludes the hard cases a cross-encoder exists to rescue. The corpus therefore
*understates* the reranker's ceiling. It does not rescue the decision — on
everything measurable the reranker is worse, and it drags the shipping config
down — but if a hard-case corpus is ever built (see the `hot_cluster` family,
absent from these 200 queries because all 75 candidates failed validation), this
is the decision to revisit first. The code is one `git revert` away.

## Explicitly not proposed

| | Why |
|---|---|
| Postgres / Neo4j migration | Forfeits local-first — the property no hosted competitor matches — to solve nothing currently measured. |
| Mem0 / Zep as a backend | Same, plus a network round trip inside a hook with a 4 s timeout. |
| `weighted_best` → RRF | RRF measures **16.1pp worse MRR** here (0.765 vs 0.970, n=200). |
| Cross-encoder rerank (any setting) | **Retired**, not merely left off — see the decision below. |
| **LLM-assisted cluster merge** | The stated gate was "only if extractive merge loses facts." Measured: `mmr_union` retains 100%. The gate did not open. Revisit only if 1.1's coherence check fails. |
| Porting LoCoMo / LongMemEval | Conversational shape, and increasingly measures context length rather than memory quality. |

## Suggested order

All of Tier 1 is now closed. What survived it:

| item | outcome |
|---|---|
| 1.1 default strategy | changed to `mmr_union` — but consolidation is inert at 0.92, so it is latent safety |
| ~~1.2 zero-result rate~~ | not a defect rate. One small fix survives (slash commands reaching search). |
| ~~1.3 recall_count~~ | contaminated, and inert. Zeroing it changes no output. |

Three items entered Tier 1 as measured problems. One turned out to be a
mislabelled statistic, one turned out to have no consumer that cares, and the
third fires on nothing today. **That is the tier working as intended** — it cost
a few hours of measurement to avoid building three things, one of which
(a `recall_count_since` epoch column) was fully scoped before the check.

Remaining order: `2.1 (chunk the ~398 non-conversation docs)` →
`2.2 (update path)` → `3.3 (bound CSLS)` → `2.3 (injection mix, after a second
measurement window)` → `3.2 (decide the graph's fate)`.

2.1 leads because P4 measured it as the one Tier-2 item with an unambiguous,
currently-active cost: 0.65% of document text is embedded, and a fact stated
verbatim in a body ranks 5th of 5.
