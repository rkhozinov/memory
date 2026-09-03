# LLM agent memory in 2026 — what the field does, and what this service does

A practitioner's survey of agent-memory research and shipping systems as of
September 2026, read specifically against this repository. The goal is not
completeness; it is a keep / steal / skip decision on each idea, backed where
possible by numbers measured on this codebase rather than published by a vendor.

Companion documents:

- `docs/research/real-usage-report.md` — how this service performs in real sessions
- `benchmarks/results/README.md` — retrieval A/B baselines on the production corpus

---

## 1. The taxonomy everyone converged on

Two 2026 surveys — *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and
Emerging Frontiers* (arXiv:2603.07670) and *LLM Agent Memory: A Survey from a
Unified Representation Perspective* — independently land on the same frame, and
it is a genuinely useful one.

**Memory is a write → manage → read loop**, not a store and a search box.
Formally the surveys cast memory as the agent's belief state over a partially
observable world; practically, it means five operations:

| Operation | What it means | In this repo |
|---|---|---|
| **Store** | admit new information | `store()`, dedup at SHA256 + semantic ≥0.90 |
| **Retrieve** | surface it when relevant | `search()`, hybrid + `weighted_best` fusion |
| **Update** | supersede what changed | ⚠️ keyword-heuristic only, in `dream` |
| **Compress** | consolidate the redundant | `consolidate()`, similarity-only, **loses 38% of facts** |
| **Forget** | drop what is wrong or dead | `_dream_active_forget`, env-gated, off |

The surveys' recurring observation is that teams build the first two and skip the
last three, and that the skipped three are where quality rots. **This repo matches
that pattern exactly.** Store and retrieve are well-built and now well-measured;
update is a heuristic, compress is lossy, forget is disabled.

The other common axis is **episodic / semantic / procedural**. This repo has all
three but does not name them: `session-archive` documents are episodic, typed
memories are semantic, and `CLAUDE.md` is procedural memory in the sense the
literature means — a declarative playbook loaded every session. The 2026 consensus
is that declarative injection via `CLAUDE.md` / `AGENTS.md` is a legitimate and
underrated form of procedural memory, not a stopgap.

## 2. Papers worth knowing

**Surveys and frames**

- **arXiv:2603.07670** — *Memory for Autonomous LLM Agents*. The write→manage→read
  loop, a three-dimensional taxonomy (temporal scope × representational substrate ×
  control policy), and an honest open-problems section: continual consolidation,
  causally grounded retrieval, learned forgetting. Single-author preprint, not
  peer-reviewed; cite it as a frame, not as evidence.
- **arXiv:2404.13501** — *A Survey on the Memory Mechanism of LLM-based Agents*.
  The earlier reference point; introduces the direct/indirect evaluation split.

**Architectures**

- **MemGPT / Letta** — the OS analogy: tiered context with the agent paging facts
  in and out through tool calls. The idea that stuck is *the agent manages its own
  memory*; the cost is adopting a whole runtime.
- **A-MEM** (arXiv:2502.12110, NeurIPS 2025) — Zettelkasten-style agentic memory.
  Each new note is written with structured attributes (context, keywords, tags),
  linked to relevant existing notes, and **triggers updates to the notes it links
  to**. That last part is the interesting one: memory that evolves on write rather
  than accumulating. Reported ~2× over MemGPT/MemoryBank/ReadAgent on multi-hop.
- **Mem0** (arXiv:2504.19413) — two-phase pipeline: LLM extraction, then **conflict
  detection and graph update**. Three scopes (user/session/agent), hybrid vector +
  graph backend. The conflict-detection step is the piece this repo lacks entirely.
- **Zep / Graphiti** — a **bi-temporal** knowledge graph: every fact carries both a
  valid time (when it was true) and a transaction time (when we learned it).
  Superseded facts are marked, not deleted, so "what was true on date X" is a real
  query. The single best answer in the field to the staleness problem.
- **ACE** (arXiv:2510.04618) — *Agentic Context Engineering*. Generator / Reflector
  / Curator roles evolve a structured "playbook" instead of rewriting a context
  wholesale. Names two failure modes precisely: **brevity bias** (compressing
  domain insight into uninformative summaries) and **context collapse** (iterative
  rewriting eroding detail). Reports +10.6% on agent tasks with no weight updates.
  Directly relevant to `build_index` and to `consolidate`.
- **HippoRAG** — hippocampus-inspired indexing; personalized PageRank over a
  knowledge graph for multi-hop retrieval. The argument for graph traversal over
  pure similarity, stated in its strongest form.

**The through-line**: what to store, when to retrieve, how to update and what to
discard are *architecture decisions*, and they do not get made correctly by
default. A store that only ever appends will quietly serve stale facts forever.

## 3. Benchmarks, and why none of them fit this repo

| Benchmark | Tests | Status in 2026 |
|---|---|---|
| **LoCoMo** (ACL 2024, arXiv:2402.17753) | multi-session conversational recall | Widely reported, increasingly saturated; no explicit knowledge-update scoring |
| **LongMemEval** (ICLR 2025, arXiv:2410.10813) | long-term chat memory, incl. updates | The de facto standard for conversational memory |
| **LongMemEval-V2** (arXiv:2605.12493) | **web-agent trajectories**, up to 115M tokens | Moves the format from chat to agent experience |
| **BEAM** (arXiv:2510.27246) | architecture comparison at 1M–10M tokens | Deliberately unsaturated; scores sit well below LoCoMo headlines |
| **MemoryArena** (arXiv:2602.16313) | interdependent multi-session *tasks* | Models near-perfect on LoCoMo drop to ~40–60% |
| **STATE-Bench** (Microsoft, May 2026) | state tracking on stateful enterprise tasks | Relatively neutral source — Microsoft sells no memory product |

Three cautions that matter more than the numbers:

1. **Vendor self-reports are not cross-comparable.** Mem0, Zep, Supermemory,
   Hindsight, Emergence and OMEGA all publish LongMemEval or LoCoMo figures on
   their own harnesses, on different dates, with different judges. Zep's founder
   publicly disputed Mem0's LoCoMo methodology. Treat any single-vendor table as a
   claim, not a measurement.
2. **The benchmarks were designed for 32k context windows.** With million-token
   windows, "dump everything into context" now scores competitively on LoCoMo and
   LongMemEval-S — so they increasingly measure whether the model can read, not
   whether the memory system retrieves. MemoryArena and BEAM exist because of this.
3. **None of them is about coding agents.** All are conversational or web-agent.
   A coding-agent memory service is retrieving decisions, error causes and repo
   conventions against a codebase that is itself ground truth. There is no public
   benchmark for that shape.

**Consequence for this repo, and it is the load-bearing one:** porting LoCoMo or
LongMemEval here would buy a comparable-looking number and very little
information. The right substitutes were built instead — a golden set on the
production corpus (`tests/build_real_corpus.py`, `benchmarks/results/`) and
real-session measurement (`benchmarks/session_analytics.py`). Keep it that way.

## 4. Systems matrix

| System | Storage | Retrieval | Update / conflict | Consolidation | Local-only | Notes |
|---|---|---|---|---|---|---|
| **this repo** | SQLite + sqlite-vec (brute force) + FTS5 | hybrid, weighted sum + CSLS | ⚠️ keyword heuristic | similarity-only, no LLM | ✅ fully | 6.7k memories, 6 ms p50 |
| **gbrain** | markdown in git → Postgres/pgvector (PGLite local) | HNSW + tsvector + RRF + reranker | timeline + compiled "truth" header per page | nightly dream cycle | ❌ needs OpenAI embeddings, hosted reranker | typed graph edges, ~29k★ |
| **Mem0** | vector + light graph + KV | hybrid, semantic + BM25 + entity | ✅ LLM conflict detection | extraction-time | partial (self-host option) | widest adoption |
| **Zep / Graphiti** | temporal knowledge graph (Neo4j/Kuzu/Neptune) | embeddings + BM25 + traversal | ✅ **bi-temporal**, best in class | episode-based | Graphiti yes, Zep Cloud no | Apache-2.0 core |
| **Letta (MemGPT)** | Postgres/SQLite, agent-editable blocks | agent-driven paging | agent rewrites its own memory | agent-driven | ✅ | you adopt the runtime |
| **cognee** | knowledge graph + vector | multiple modes | ✅ re-weights on correction | ECL pipeline | ✅ embedded stores | claims 79% BEAM@100k |
| **Hindsight** | Postgres + pgvector | multi-strategy + cross-encoder | entity resolution at write time | LLM consolidation | ✅ Docker | most explicit about consolidation |
| **claude-mem** | SQLite + FTS5 + Chroma | hybrid, causal-graph walk | — | AI-compressed observations | ✅ | closest peer in the Claude Code niche |
| **Supermemory** | managed context engine | proprietary | — | — | ❌ (local binary, closed engine) | SOTA claims on LongMemEval-S |
| **Neural Memory** | graph, spreading activation | traversal, no embeddings in free tier | decay + reinforcement | neuroscience-inspired | ✅ | FTS5-only until paid tier |

### gbrain, in more detail

It is the closest philosophical neighbour to this repo and the most useful single
comparison, so it is worth being precise about where the two agree and differ.

**Convergent, independently:** a nightly "dream cycle" that enriches and
consolidates; entity extraction and link generation at write time **without LLM
calls**; hybrid keyword + vector retrieval; a health/doctor command; an MCP
surface generated alongside a CLI. Two designs arriving at the same shape from
different directions is reasonable evidence the shape is right.

**Where gbrain is ahead:**

- **Markdown in git is the system of record**, with Postgres as a rebuildable
  index. You can `git diff` what the agent learned overnight, branch the brain,
  review writes line by line, and rebuild the DB if it is lost. This repo's SQLite
  file *is* the source of truth — there is no diff, no branch, no human review
  surface, and no rebuild path. This is the single biggest structural difference
  and it is not in this repo's favour.
- **Typed graph edges** (`works_at`, `founded`, `invested_in`, `attended`) with
  multi-hop traversal, versus 19 886 `co_occurrence` edges here — one edge type
  that largely re-derives what cosine already found. gbrain reports +31.4 P@5 from
  its graph layer over vector-only (its own benchmark, so treat as directional).
- **Real chunking** of markdown before embedding. This repo truncates at 512
  tokens and never embeds document bodies at all.
- A compiled "truth" header plus an **append-only timeline** per page — a
  lightweight bi-temporal model that keeps the current answer and its history
  distinct.

**Where this repo is ahead:**

- **Genuinely local.** gbrain needs OpenAI embeddings and a hosted reranker
  (Voyage `rerank-2.5`, with a deprecated ZeroEntropy fallback whose API sunsets
  2026-09-04 — an external dependency with an expiry date). This repo runs MLX
  INT8 embeddings on-device with no API key and no network.
- **Measured fusion.** gbrain uses RRF. On this repo's production corpus RRF
  scores **16.1pp worse MRR** than the shipping additive-weighted + CSLS path
  (0.765 vs 0.970, n=200, `benchmarks/results/README.md`). RRF is the textbook
  default; here it is measurably the wrong one. Corpus-specific, but measured.
- **Injection safety.** Nonced fence tags, an `injection_suspicious` filter on
  every read path, and verbatim-evidence grounding in the extractor. A 2026 study
  found >90% of tested agents vulnerable to memory poisoning, with a 100% relapse
  rate when teams tried to fix it by correcting the agent in conversation. This is
  a real and under-served axis, and this repo is ahead of most of the field on it.

## 5. Verdict — keep, steal, skip

### Keep

| Thing | Why |
|---|---|
| `weighted_best` fusion (additive + CSLS) | 0.970 MRR vs RRF's 0.765 on 200 production queries. The field's default is worse here. |
| Local-first, no API key | The one property no hosted competitor can match, and gbrain's hosted-reranker sunset shows the cost of giving it up. |
| Untrusted-content fencing | Ahead of the field on an axis the field mostly ignores. |
| Deterministic `dream` | 2.6 s, backgrounded, no spend. LLM-based consolidation would be slower, costlier and — per §6 below — is not currently needed for fact retention. |
| No LoCoMo/LongMemEval port | They measure a different shape of problem. The corpus-specific golden set is the right substitute. |

### Steal

| Idea | From | Why now |
|---|---|---|
| **Bi-temporal facts** | Zep/Graphiti | `memory_graph` already has `valid_from`/`valid_to` and they are unused. This is wiring, not schema work. |
| **Conflict detection on write** | Mem0, cognee | Today a changed fact silently coexists with its replacement unless it happens to contain the word "instead". `stale_fact` cases in `benchmarks/corpus.py` pin this. |
| **Entity resolution at write time** | Hindsight | Resolving surface forms at query time is expensive and unreliable; 2 293 entities already exist to seed an alias table. |
| **Typed edges** | gbrain, Neural Memory | One `co_occurrence` type mostly re-derives cosine. Typed edges are what make traversal earn its cost. |
| **Chunking** | everyone | 1 104 documents are reachable only via a 500-char summary. `test_doc_body_semantic_recall` is an `xfail` proving the gap. |
| **Markdown export / git mirror** | gbrain | Not a migration — an *export*. A diffable mirror of what the agent wrote would make review possible without giving up SQLite. |
| **Incremental itemized index updates** | ACE | `build_index` rebuilds wholesale; ACE's argument against exactly that is context collapse. |

### Skip

| Idea | Why not |
|---|---|
| Migrating to Postgres / Neo4j | Forfeits local-first, the main advantage, to solve a problem not yet measured. |
| Adopting Mem0 / Zep as a backend | Same, plus a network dependency in a hook with a 4 s timeout. |
| Replacing `weighted_best` with RRF | Contradicted by a measurement in this repo. |
| Porting LoCoMo / LongMemEval | Wrong shape (conversational), and increasingly measures context length rather than memory. |
| Cross-encoder rerank on by default | +1.3pp MRR for a 714 ms p50 cold, versus `weighted_best`'s +4.4pp at 6 ms. Retiring it is more defensible than enabling it. |
| LLM-based cluster merge | Gated on evidence that extractive merge loses facts. It does not — see §6. |

## 6. Where the literature and this repo's own measurements disagree

Worth stating plainly, because these are the places where following the field
would have made things worse.

1. **RRF is the field default and is 16pp worse here.** Reciprocal Rank Fusion is
   what gbrain, Hindsight and most hybrid stacks use. On this corpus the additive
   weighted sum plus CSLS hubness correction wins decisively. Measure before
   adopting a default.
2. **Cross-encoder reranking is near-universal and does not pay here.** The
   literature treats a reranker as a straightforward quality win. Measured:
   +1.3pp for 120× the latency, against a cheaper path that gives +4.4pp.
3. **The consolidation literature argues for LLM-assisted merging. The measured
   problem is the default, not the method.** `tests/bench_consolidate.py`: the
   shipping `keep_higher_recall` strategy retains 62% of unique facts, while the
   existing extractive `mmr_union` retains 100% at the same surviving-memory
   count. There is no measured case for an LLM here — there is a case for
   changing one default.
4. **Retrieval score does not predict usefulness, and inverts at the top.**
   Measured over 648 injected memories: the ≥0.70 score band has a 0.0% observed
   reuse rate while the ≥0.60 band has 22.2%. High similarity to the prompt means
   low informativeness — the agent already knew it. No paper surveyed here frames
   the injection decision this way; they optimise retrieval score and stop.
   See `docs/research/real-usage-report.md` §3 for the method and its limits.

---

## Sources

Surveys and papers: arXiv:2603.07670; arXiv:2404.13501; A-MEM arXiv:2502.12110
(NeurIPS 2025); Mem0 arXiv:2504.19413; ACE arXiv:2510.04618; LoCoMo
arXiv:2402.17753 (ACL 2024); LongMemEval arXiv:2410.10813 (ICLR 2025);
LongMemEval-V2 arXiv:2605.12493; BEAM arXiv:2510.27246; MemoryArena
arXiv:2602.16313; MemoryAgentBench arXiv:2507.05257; STATE-Bench (Microsoft,
May 2026); ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems.

Systems: github.com/garrytan/gbrain; mem0.ai; getzep.com and the Graphiti repo;
letta.com; cognee.ai; hindsight.vectorize.io; supermemory.ai;
redis.github.io/agent-memory-server; thedotmack/claude-mem.

All web sources reviewed 2026-09-03. Vendor benchmark figures are quoted as
vendor claims and are not comparable across harnesses.
