# Decision log

Format for every entry:

```
Decision / Why / Alternatives considered / Why rejected / Tradeoffs
```

The point of this file is that six months from now, in an interview, you can explain
*why* each choice exists rather than "that's what got generated."

---

## D-001 — DuckDB is the query engine; Polars is not used

**Decision.** All analytical query execution goes through DuckDB. Polars is not a
dependency of the MVP.

**Why.** The agent has to emit the analysis step. If it emits **SQL**, that is text we
can parse into an AST, validate against an allowlist, execute in a sandbox, and store
verbatim as evidence in the execution trace. If it emitted **Polars code**, we would be
`exec()`-ing model-written Python — which the roadmap forbids in §1.3, correctly.
The choice is therefore structural, not a preference: SQL is the only one of the two
that gives an untrusted producer a safe, inspectable surface.

**Alternatives considered.**
1. *Polars for ingestion/profiling, DuckDB for queries.* The initially proposed split.
2. *Both, used interchangeably.* As written in the original roadmap.
3. *pandas.* Not seriously considered — slower and looser typing than either.

**Why rejected.** (1) was rejected after checking whether the ingestion/profiling job
actually needs Polars. It does not: DuckDB's `SUMMARIZE` returns types, null counts,
distinct counts, min/max/mean/stddev and quartiles in a single statement, which is the
entire M2 dataset profile, and DuckDB's CSV sniffer handles messy real-world files at
least as well. A dependency with no remaining job is a dependency that gets cut.
(2) was rejected because "we use two dataframe engines" has no good answer when the
honest one is "both looked good on a CV."

**Tradeoffs.** We lose Polars' more ergonomic Python API for per-column branching logic
during profiling, and we lose a CV keyword. If a concrete ingestion problem later
appears that DuckDB handles badly, we add Polars *then* and record what forced it —
which is a better story than having had it from the start.

---

## D-002 — Postgres in Docker from day one; the app is not containerized until M6

**Decision.** `docker-compose.yml` contains Postgres and nothing else. Backend, worker
and frontend run natively during development.

**Why.** Installing Postgres natively on Windows is the single most annoying setup step
in a project like this, and a container removes it for the cost of an 8-line file. But
containerizing the *app* while its code is churning slows the edit-reload loop for no
benefit, and this laptop has 15.8 GB RAM total — spending it on containers we do not
need is a real cost, not a theoretical one.

**Alternatives considered.** (1) Everything containerized from day one, as the original
roadmap's Phase 20 implies but does not do. (2) Native Postgres install. (3) SQLite
until later.

**Why rejected.** (1) slows development and competes for RAM with the model. (2) is
painful on Windows and not reproducible for anyone cloning the repo. (3) would hide the
concurrency work — `SELECT ... FOR UPDATE SKIP LOCKED` is the whole point of the M4 job
queue and SQLite cannot express it.

**Tradeoffs.** Dev and prod environments differ until M6, so container-only bugs surface
late. Mitigated by keeping the app's contact with Postgres behind config, so the only
thing that changes at M6 is the host name.

---

## D-003 — Host port 5433, not 5432

**Decision.** Postgres is published on host port `5433`.

**Why.** Avoids silently colliding with any Postgres already installed on the host, and
makes it unambiguous which database a connection string is pointing at. A dev who
connects to `5432` out of habit gets a clean connection refusal instead of quietly
reading the wrong database.

**Tradeoffs.** One more thing to remember. Documented in `.env.example`.

---

## D-004 — Alembic reads its database URL from `app.config`, not `alembic.ini`

**Decision.** `alembic.ini` has a blank `sqlalchemy.url`; `app/db/migrations/env.py`
injects it at runtime from `get_settings().database_url`.

**Why.** Two reasons. First, the generated template hardcodes a URL *with the password*
into a committed file — an immediate secret leak. Second, and more subtly: if the app
and the migrations read their connection details from different places, they can drift
onto different databases, and you get the genuinely confusing failure mode where
migrations "succeed" against a database the app never touches.

**Alternatives considered.** Environment-variable interpolation inside `alembic.ini`.

**Why rejected.** Works, but splits config into two systems. One source of truth is
worth more than the small amount of magic.

**Tradeoffs.** `env.py` now imports the app package, so migrations only run from the
project root with the venv active. Acceptable — that is how they are always run.

---

## D-005 — Migrations use the sync Alembic template, not async

**Decision.** Sync template, even though the app itself will be async.

**Why.** A migration is a short-lived, single-connection, offline script. Async buys it
nothing — there is no concurrency to overlap — and costs an extra `asyncio` layer in
`env.py` to read and explain. The app being async does not require its migration tool to be.

**Tradeoffs.** None material. The SQLAlchemy `psycopg` dialect supports both, so this
can be revisited without changing the driver.

---

## D-006 — Model choice benchmarked, not assumed

**Decision.** `scripts/bench_model.py` measures latency, structured-output validity,
tool-calling reliability and SQL correctness for any Ollama model. The model choice
follows the numbers. See `docs/benchmarking.md` for results.

**Why.** The original roadmap named "Qwen3 8B" as the target based on VRAM arithmetic.
On measured hardware (6141 MiB VRAM, 15.8 GB RAM) an 8B Q4_K_M is ~5.2 GB of weights,
leaving roughly 200 MB for KV cache after the Windows desktop's own VRAM use — so it
either runs at a uselessly short context or spills to CPU. And with 15.8 GB of system
RAM shared with Docker, a Vite dev server and a browser, there is no headroom to absorb
that spill. But this is still *arithmetic*. The benchmark turns it into evidence.

**Alternatives considered.** Pick 8B and hope; pick 4B and hope.

**Why rejected.** Both leave you unable to answer "why this model?" — and the
benchmark's byproduct (real tokens/sec, real tool-call rates) is exactly the material
the CV bullets need.

**Tradeoffs.** A day of work before any application code exists. Bought back many times
over during M5, when "did that prompt change help?" becomes a measurable question.

### The outcome: `qwen3:4b`, with reasoning enabled

Measured on the reference hardware (RTX 4050 Laptop, 6141 MiB VRAM, 15.8 GB RAM):

```
                         qwen3:4b        qwen3:8b       ratio
tokens/sec (median)         53.8             9.4        5.7× slower
wall per call (median)     15.07 s         31.0 s       2.1× slower
cold load                   2.25 s         11.07 s      4.9× slower
GPU placement            100% GPU     38% CPU / 62% GPU
resident size               3.5 GB          6.0 GB

JSON valid (free-form)      10/10           10/10       equal
JSON valid (constrained)    10/10           10/10       equal
tool call + valid args      15/15           15/15       equal
SQL usable end-to-end       10/10           10/10       equal
SQL emitted cleanly         10/10           10/10       equal

benchmark wall time         1243 s          3097 s      2.5× longer
```

**Every capability metric is a tie.** That was predicted before 8B's sections C and D ran
(on the grounds that 4B was already at 100% and nothing can exceed it), and it held.

**The decisive fact is not speed, it is that there is nothing to buy with the slowness.**
`qwen3:4b` scores 100% on every capability this architecture requires. A larger model
has no quality deficit to fix, so its 5.7× throughput penalty purchases nothing.

**Why 8B is so much slower than its size suggests.** Not 2× slower for 2× the parameters
— 5.7×. `ollama ps` reports `38%/62% CPU/GPU`: 5.2 GB of weights against ~5.4 GB of free
VRAM leaves nothing for the KV cache, so a third of the model is evicted to system RAM
and read over PCIe at roughly a quarter of VRAM bandwidth. From first principles
(`learning-notes.md` §4b):

```
0.62 × (5.2 GB / 192 GB/s)  +  0.38 × (5.2 GB / ~50 GB/s)
   =  16.8 ms  +  39.5 ms   =  56 ms/token  →  ~18 tok/s ceiling
measured: 9.4 tok/s   (remainder: PCIe transfer overhead)
```

The minority of the model living on the CPU dominates the cost. **Spilling a third of a
model to system RAM does not cost a third of your speed; it costs most of it.** This is
why "pick the biggest model that fits" is bad advice — the biggest model that *fits* is
the one that fits with nothing to spare, which is exactly where this cliff is.

This prediction (`< 37 tok/s`, "well below rather than near it", caused by CPU spill)
was written into `learning-notes.md` §4b **before** 8B was run. Recording predictions
before measuring is what makes the measurement worth anything.

**What this decision does NOT establish.** Every measurement above is single-turn against
a 10-row, 6-column table. Multi-step tool sequences, recovery from tool errors, knowing
when to stop, and realistic 40-column schemas are all unmeasured — and they are where
agents actually fail. M3's eval set exists to cover exactly that gap, and the M1 result
raises its priority rather than lowering it: a 100% single-turn score means a single-turn
eval would be useless as a development instrument.

**Revisit this decision if** M3's multi-turn eval shows `qwen3:4b` failing on reasoning
depth rather than mechanics. In that case the next step is a larger *quantization* of 4B
or a 6–7B model that still fits entirely in VRAM — not 8B, whose problem is placement,
not capability.

---

## D-007 — The benchmark grades SQL by executing it

**Decision.** Generated SQL is run against a real DuckDB fixture table and its returned
number is compared to a reference value computed from hand-written SQL. No regex
matching, no similarity scoring.

**Why.** This is the project's whole philosophy compressed into one function. A query
that reads beautifully and returns the wrong number is a failure, and any grader that
cannot tell the difference is measuring the wrong thing.

**Tradeoffs.** Only works for questions with a single scalar answer, so the fixture
tasks are all scalar by construction. The M3 eval set will need richer comparison
(row sets, ordering) — a known extension, not an oversight.

---

## D-008 — The benchmark uses Ollama's native API; the app will use the OpenAI-compatible one

**Decision.** `scripts/bench_model.py` calls `/api/chat`. `app/llm/` (M5) will call
`/v1/chat/completions`.

**Why.** Different jobs. The native endpoint returns `load_duration`,
`prompt_eval_count/duration` and `eval_count/duration` straight from the inference
engine in nanoseconds — exact tokens/sec, not a wall-clock estimate polluted by HTTP
and Python overhead. The OpenAI-compatible endpoint returns none of that, but it makes
swapping in any other provider a base-URL change. Measure with the precise instrument;
ship with the portable one.

**Tradeoffs.** Two code paths talk to Ollama. Contained: the benchmark is standalone
and deliberately shares no code with the app, so it stays runnable while the app churns.

**Verified, 2026-08-27.** This decision carried a risk worth checking before M5 depended
on it: if the OpenAI-compatible endpoint merged the model's reasoning into `content`, we
would lose the separation that makes responses parseable (see `learning-notes.md` §4c)
and portability would have cost us correctness. Tested directly — it does separate them,
but **under a different field name**:

```
native  /api/chat             →  message.thinking
OpenAI  /v1/chat/completions  →  message.reasoning     content: "SELECT SUM(revenue) FROM sales;"
```

The `LLMProvider` interface must normalise this. A provider abstraction that hardcodes
one field name would return an empty reasoning trace and nobody would notice until an
execution trace turned up blank.

---

## D-009 — Reasoning (thinking) stays enabled

**Decision.** The agent runs Qwen3 with reasoning ON. `think: false` is not used.

**Why.** It is the opposite of the obvious optimisation, and it is measured. `think: false`
does not skip deliberation; it relocates it from a dedicated field into `content`, mixed
with the answer. Full benchmark, `qwen3:4b`, same tasks, temperature 0:

```
                              reasoning ON     think: false
SQL usable end-to-end            10/10            7/10
SQL emitted cleanly              10/10            0/10
JSON valid (free-form)           10/10           10/10
tool call + valid args           15/15           15/15
wall per call (median)           15.07 s         15.14 s
tokens/sec (median)              53.8            54.9
```

Two things to read precisely here:

- **"SQL usable end-to-end", not "SQL correct".** The three `think: false` failures were
  all `ParserException` — the extractor could not recover a statement from the prose. We
  cannot tell whether the underlying query was right, only that the response was
  unusable. That is the metric that matters for an agent, but it is not a claim about
  the model's SQL knowledge.
- **Latency is a wash.** 15.07 s vs 15.14 s, and tokens/sec is within noise. An earlier
  reading of this decision claimed reasoning ON was ~30% faster; that came from a
  two-prompt comparison and did not survive the full run. Corrected: there is no
  meaningful latency difference, and the case rests entirely on output structure.

So the decision is not a tradeoff between speed and structure. It is free: the same
latency, with machine-readable output instead of prose.

**Alternatives considered.** Disable reasoning to cut latency — the plan's original
assumption, carried for most of M1.

**Why rejected.** It does not cut latency (the model deliberates either way, and answers
more verbosely when it cannot deliberate separately), and it destroys output structure.

**Tradeoffs.** Responses carry a reasoning block we mostly discard, which consumes
context in a multi-turn loop. Watch this in M5: reasoning traces from earlier turns
should NOT be fed back into subsequent requests, or context will grow far faster than
necessary.

---

## D-010 — SQL is executed in a four-layer sandbox, verified empirically

**Decision.** Every query runs through: (L1) a sqlglot AST allowlist, (L2) a DuckDB
connection with filesystem and network access disabled and its configuration locked,
(L3) exactly one dataset registered as a view, (L4) row/byte/time limits.

**Why.** In M5 this SQL is written by a language model that can be steered by text
inside the dataset itself. "The model would not write that" is not a security control.

The layers are not redundant — each covers what the others miss:

```
L1  structure    rejects DROP, multi-statement injection, CTE-wrapped writes,
                 table functions. Produces a CLEAR message, which matters because
                 that message is fed back to a model for repair.
L2  capability   the engine cannot reach the filesystem at all. THIS is what makes
                 it safe; L1 and L3 make it explainable.
L3  scope        the model supplies a dataset_id; no path ever appears in SQL.
L4  resource     a legal query cannot hang a worker or exhaust memory.
```

**The recipe, and why the order matters.** Verified against DuckDB 1.5.5:

```sql
SET memory_limit / threads          -- FIRST: frozen by the lock below
SET allowed_paths = ['<parquet>']   -- the one file that stays readable
SET enable_external_access = false  -- everything else off
SET lock_configuration = true       -- the query cannot undo any of the above
```

The naive approach — `enable_external_access=false` alone — also blocks reading our own
Parquet file. `allowed_paths` is documented as files "ALWAYS allowed to be queried, even
when external access is disabled", which is exactly the exception needed.

Measured after lockdown: reading any other path, `read_text`/`read_blob`/`glob`,
`COPY` out, `ATTACH` of a file database, `INSTALL`, http(s) URLs, and re-enabling
external access **all fail**. Only harmless introspection survives, and L1 rejects that
anyway.

**Alternatives considered.** (1) Keyword blocklist. (2) Materialise the dataset into
memory, then disable all file access. (3) Trust the model.

**Why rejected.** (1) loses to comments, nesting and string literals —
`SELECT 1; /*x*/ DROP TABLE t` defeats it. (2) would work but forfeits the reason for
using Parquet: streaming multi-million-row files without loading them. (3) is not a
control.

**Tradeoffs.** A fresh connection and view per query costs a few milliseconds. Measured
at ~11 ms for a group-by over 5,000 rows end to end, which is irrelevant next to a
15-second model call.

---

## D-011 — Table functions are rejected structurally, not by name

**Decision.** L1 requires every table reference in the AST to be a plain identifier.
Any table-valued function in `FROM` is refused.

**Why.** The first implementation blocklisted function names (`read_csv`,
`read_parquet`, `glob`, ...) by looking for sqlglot `Anonymous` nodes. It silently
missed the two most important ones: sqlglot gives `read_csv` and `read_parquet`
**dedicated AST classes** (`ReadCSV`, `ReadParquet`) which are not `Anonymous` at all.
L2 blocked the query anyway — defence in depth working — but a check that misses the
obvious cases is not a check.

Adding two names would have fixed the symptom. The real problem is that a name list
only blocks functions someone thought of, and DuckDB ships many table functions with
extensions adding more. Since this sandbox exposes exactly ONE table, the correct rule
is structural: a table reference must be an identifier. That rejects every
table-valued function, including ones that do not exist yet.

**Tradeoffs.** Legitimate generators like `range()` and `generate_series()` are also
refused. Acceptable: no analytical question about a dataset needs them, and the M1
benchmark showed the model reaching for plain `SELECT ... FROM dataset` anyway. If a
real need appears, the fix is an explicit allowlist of specific safe functions — the
opposite direction from a blocklist.

---

## D-012 — Profile statistics are computed exactly, never estimated

**Decision.** Null counts and distinct counts use `count(*) FILTER (...)` and
`count(DISTINCT ...)`, not SUMMARIZE's `null_percentage` and `approx_unique`.

**Why.** `approx_unique` is a HyperLogLog estimate. Measured on a 30-row fixture with
30 genuinely distinct values, it returned **27** — a 10% error, which was enough to
flip a high-cardinality threshold during development. `null_percentage` is
pre-rounded: 0.4% of 2.4M rows rounds to a figure off by thousands.

These numbers are shown to users and, in M5, reasoned over by the agent before it
answers. An estimate stored in a field named `distinct_count` is a small lie that
propagates into every conclusion drawn from it.

**Tradeoffs.** `count(DISTINCT ...)` across every column is more expensive than
HyperLogLog. Paid once per ingest, never per query — 5,000 rows × 7 columns profiles in
well under a second, and the cost scales with ingest, which is already the slow path.

---

## D-013 — PostgreSQL for metadata, Parquet + DuckDB for data

**Decision.** Postgres stores what datasets exist, their versions and their profiles.
The rows themselves live in Parquet and are read only by DuckDB. Postgres never touches
a data row.

**Why.** They are built for opposite workloads:

```
                 PostgreSQL                 DuckDB
shape            row-oriented               column-oriented
built for        many small reads/writes    few huge scans
concurrency      many writers, ACID         one process, effectively read-only
our use          the catalogue              the compute
```

The decisive requirement is M4's job queue: several workers must each claim the next
analysis with no two claiming the same one, which needs `SELECT ... FOR UPDATE SKIP
LOCKED`, row locks and real transactions. DuckDB has no equivalent — it is an
analytical engine for one process, not a coordination point for many. Conversely, a
`GROUP BY` over millions of rows in Postgres would be far slower than DuckDB reading
Parquet.

A 5,000-row dataset produces 8 rows of Postgres metadata. The ratio only improves with
size.

**Tradeoffs.** Two stores that cannot be committed atomically. Handled in D-014.

---

## D-014 — Write the file first, commit metadata second, roll back the file on failure

**Decision.** `create_dataset` writes Parquet, then commits the database transaction,
and deletes the Parquet file if the commit fails.

**Why.** Two stores, no distributed transaction. One of the two inconsistent states has
to be chosen as the one to defend against:

```
file without a row   invisible to every listing, occupies disk forever,
                     findable only by walking directories
row without a file   appears in listings, looks healthy, fails only when
                     someone finally queries it
```

The second is worse: it is a broken record that advertises itself as working. So the
file is written first and removed on failure, and a crash between the two leaves at
worst an orphaned file — recoverable, and never surfaced to a user as a real dataset.

Deletion runs in the opposite order for the same reason: database first, then files.

**Tradeoffs.** A hard process kill between the file write and the commit still leaves an
orphan. A future sweeper can reconcile directories against the database; not worth
building until it happens.

---

## D-015 — One general tool plus five guard rails, not a catalogue of canned analyses

**Decision.** The registry holds six tools. `execute_sql` accepts any read-only SELECT;
`inspect_schema`, `profile_column`, `compare_groups`, `correlation` and `create_chart`
exist only where they enforce something SQL cannot.

**Why.** The obvious objection to a fixed tool set is that it caps what the agent can
analyse. It does not, because one of the tools is general:

```
arbitrary Python   unbounded    cannot be validated without running it
arbitrary SQL      large        parses to an AST we can allowlist, on a
                                connection with the filesystem switched off
canned analyses    tiny         safe, and useless for real questions
```

Anything expressible in SQL — window functions, CTEs, self-joins, cohorts, percentiles
— is reachable. The other five do not add power; they add refusals that arrive *before*
execution, in language a model can act on:

| tool | what it enforces that SQL does not |
|---|---|
| `compare_groups` | metric must be numeric; grouping column must not be an identifier; returns row counts and shares beside each aggregate |
| `correlation` | both columns numeric; reports Pearson **and** Spearman, so a curved relationship is not read as "no relationship" |
| `create_chart` | axis types match the chart type; category count is readable; returns a spec, never pixels |
| `inspect_schema` / `profile_column` | one call for what a model would otherwise infer by eyeballing sample rows |

The genuine ceiling is what SQL cannot express — regression, clustering, time-series
decomposition. Those arrive in M6 with SciPy and scikit-learn, as more tools.

**Tradeoffs.** Six entries in every prompt. Selection accuracy on small models degrades
as the list grows, which is why `detect_anomalies` waits for M6 rather than being added
speculatively, and why `ToolRegistry.specs(only=[...])` can narrow the list per run.

---

## D-016 — The model supplies a column *name*; the SQL gets the *canonical* name

**Decision.** Every tool that accepts a column name resolves it against the live schema
before use. The model's string is never interpolated into SQL.

**Why.** `execute_sql` is safe because the whole statement passes the sandbox's AST
allowlist. `compare_groups(group_column="...")` is different: it *assembles* SQL, and
string assembly plus untrusted input is the shape of every SQL injection ever written.

Quoting is not the defence. Substitution is:

```
model says  "Reveune"
                |
                v  resolve_column  -- looked up in the REAL schema
                |
SQL gets    "revenue"      <- the canonical name from DuckDB, not the model's text
```

A name that matches nothing never reaches SQL at all; it becomes a `ToolError` naming
the columns that do exist. Quoting is still applied, because a real column name can
contain a space or a quote — but by then the value is ours.

Matching is case-insensitive on purpose. Small models get casing wrong constantly
("Revenue" for `revenue`), and since the *canonical* name is what proceeds, accepting a
case variant costs nothing in safety and removes a whole class of pointless failure.

**Tradeoffs.** One extra schema read per tool call (~1 ms, shared across arguments
within a call).

---

## D-017 — A failed tool call is a value, not an exception

**Decision.** `ToolRegistry.call()` never raises. Every failure returns a `ToolResult`
with `ok=False` and a message written to be read by a model.

**Why.** In M5 the agent loop must be able to hand the model back:

```
column 'reveune' does not exist. Did you mean: revenue?
Available columns: order_id, order_date, region, category, revenue, cost, units
```

…and let it try again. An exception aborts the run; a result object is a repair prompt.
This is why the error messages name the valid alternatives rather than only stating
what was wrong — the message *is* the interface.

Three failure kinds stay distinct, because collapsing them is how an agent ends up
retrying a call that can never work:

```
unknown tool      -> the agent is not working from the action space it was given
bad arguments     -> a repairable slip
runtime failure   -> either repairable (ToolError) or a bug in us (anything else)
```

**Tradeoffs.** Callers must check `.ok` rather than relying on exceptions to propagate.
Enforced by the shape: `ToolResult` carries no data when `ok` is false.

---

## D-018 — The evaluation set is built *before* the agent, and its expected values are computed, never typed

**Decision.** M3 ships 50 golden questions across three datasets. Each carries reference
SQL; the expected values are produced by executing that SQL and checked into
`eval/answers/*.json`.

**Why.** Two separate arguments.

*Order.* In M1, five of six apparent "model failures" were bugs in the measuring
harness. Built after the agent, every one of them would have presented as "the agent is
broken", and the fix would have been prompt-tuning against a broken ruler.

*Derivation.* A benchmark with hand-entered answers has two sources of truth that drift
apart silently — someone tunes a generator, the data moves, and forty constants keep
asserting what used to be true. Here the reference SQL is the only authored artefact.
`python -m eval.build --check` recomputes and exits non-zero if anything moved, so a
generator change surfaces as a reviewable diff of exactly which answers changed.

Reference SQL runs through the **same sandbox** the agent uses, so a question answerable
only outside the sandbox fails at build time rather than producing a score nobody can
reach.

**Tradeoffs.** Requires Postgres to build (the datasets are registered first). Once
built, the answers are read from JSON and need nothing.

---

## D-019 — Evaluation datasets are generated from seeded code, not downloaded

**Decision.** Three datasets — `ecommerce` (clean), `marketing` (44 columns, messy),
`sensors` (hourly time series) — built by committed generators with fixed seeds. The
CSVs are gitignored; the generators are not.

**Why.** A benchmark question needs an answer that exists independently of the query
used to find it. For a downloaded CSV, "correct" is whatever the reference SQL returns,
which makes the reference SQL both question and answer and tests nothing.

Generating inverts that: the effect is decided first ("Q3 profit falls while revenue
rises"), the data is built to contain it, and `planted_effects` records what a competent
analyst should be able to find. The generator being committed matters as much as the
data:

```
a committed CSV        opaque    "why is West's margin low?" — nobody knows
a committed generator  legible   the parameter that made it low is on line 79
```

Three datasets rather than one, because a single shape tests a single skill. Clean data
tests diagnosis; a 44-column table with duplicated metrics and mixed units tests reading
a schema carefully; hourly readings test reasoning over time.

**Tradeoffs.** Synthetic data cannot surprise you the way real data does — this measures
regression, not generality. Held-out real datasets are an M6 item, and the two are kept
distinct rather than conflated.

---

## D-020 — The scoreboard ships with three stub agents that calibrate its own scale

**Decision.** `eval/agents.py` provides `oracle`, `refusing` and `schema-only`, and they
are run before any real agent exists.

**Why.** A benchmark reporting 62% means nothing unless the instrument is known to read
100% for a perfect answer and 0% for a worthless one. Measured:

| stub | accuracy | values | what it proves |
|---|---|---|---|
| `oracle` | 100% | 100% | the grader's ceiling is reachable |
| `refusing` | 0% | 0% | saying nothing earns nothing |
| `schema-only` | 0% | 0% | **sounding right earns nothing** |

The third is the important one. It inspects the schema and writes a fluent, confident
answer containing no computed number — the most realistic failure mode of a weak agent.
It scores zero on values, and that is the property that makes the benchmark worth
running.

Building the oracle immediately found five grader bugs, all wording brittleness
(`shipping_cost` vs `null_shipping`, `variant b` vs `variant = B`), and one real regex
bug: the number extractor could not match `Q3`, because its lookbehind rejected any
digit preceded by a letter.

**Tradeoffs.** The `ambiguity` category has **no verified ceiling** — its check ("did the
answer say which column it used") needs narrative behaviour that no table-dumping stub
produces, so the oracle scores 0/2 there. Those questions are excluded from headline
accuracy, and this is stated rather than hidden.

---

## D-021 — Chart tools return a validated spec; the model sees a summary of it

**Decision.** `create_chart` returns a chart specification plus its data. The caller
receives every point; the model receives at most twelve, via `Tool.model_view`.

**Why.** Two audiences with opposite needs. The browser must have all the data to draw
the chart; the model needs to know a chart was made and what it shows. Measured on a
400-point scatter plot: **11,325 characters → 776**, roughly 3,000 tokens of context
recovered per chart, for numbers the model computed itself in order to request it.

Returning a spec rather than a rendered image is what makes this possible at all, and it
also removes `kaleido` and a headless Chromium from the dependency list. Validation
targets *plausible nonsense* rather than crashes: a line chart on an unordered category,
4,000 bars, a scatter plot of text — all render fine and mean nothing.

**Tradeoffs.** `model_view` is a second payload shape for any tool that overrides it.
Only `create_chart` does, and `ToolResult.model_data` stays `None` everywhere else.
