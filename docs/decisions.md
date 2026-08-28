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

---

## D-022 — The job queue is a Postgres table, not Celery and Redis

**Decision.** `analyses` is the queue. Workers claim rows with
`UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING`. No
broker, no result backend, no second stateful service.

**Why.** The reflex is to reach for Celery. Look at what it would actually add here:

```
Celery + Redis     a broker to run, monitor and back up; a result backend that is a
                   SECOND store the analysis lives in; a task id that is not the
                   thing the user asked about
Postgres queue     one table, one transaction, already backed up, and the queue row
                   IS the analysis -- question, status, result and audit trail
```

The hard part of a queue is not delivery, it is *exactly-once claiming*, and Postgres
has solved that since 9.5. `SKIP LOCKED` is the specific primitive: it takes a row lock
and walks past rows another transaction already holds, instead of blocking on them. Two
workers polling in the same millisecond get two different jobs and neither waits.

The claim is one statement on purpose. `SELECT` then `UPDATE` leaves a window between
them where another transaction can interleave; the single statement locates, locks and
mutates atomically.

This is not "Postgres scales forever". At a few thousand jobs a second the polling loop
becomes the bottleneck and a broker earns its keep. This system runs one local model on
one laptop, so that number is not in view, and the cost of being wrong is one afternoon
to swap the implementation behind `app/jobs/queue.py`.

**Tradeoffs.** Polling costs one indexed query per worker per second. A partial index
(`WHERE status = 'PENDING'`) keeps that query proportional to the backlog rather than to
history. No fan-out, no priorities, no scheduled tasks — none of which M5 needs.

---

## D-023 — Liveness is a heartbeat, and every terminal write is guarded by worker identity

**Decision.** A running worker refreshes `heartbeat_at` every 5 s from a background
thread. A sweep requeues any RUNNING row whose heartbeat is older than 30 s. Every write
that finishes a job carries `AND worker_id = :me AND status = 'RUNNING'`.

**Why.** Requeueing an orphaned job is the easy half, and doing only that half creates a
worse bug than the one it fixes:

```
12:00:00  worker A claims #7
12:00:31  A is paused -- long GC, suspended laptop -- and misses its heartbeats
12:00:35  the sweep decides A is dead and requeues #7
12:00:36  worker B claims #7 and starts over
12:00:40  A wakes up, finishes, and writes its result
```

Without the guard, A's write lands on a row B now owns. With it, A's UPDATE matches zero
rows and is silently a no-op — correct, because A was, as far as the system is
concerned, dead. The worker treats a rejected write as "my result is unwanted" and drops
it. That is the counter-intuitive part and the reason it is stated here.

The timeout must be several beats wide, and `app/config.py` refuses to start if it is
not (`>= 3 x` the interval). At one beat, a single slow write hands a live worker's job
to somebody else.

The beat runs on its own thread rather than between analysis steps because a single step
is exactly when liveness matters most — one M5 model call is 10-60 s of silence. The
same round trip also reads back `cancel_requested`, so one query answers "am I alive",
"do I still own this" and "does anyone still want it".

Attempts are counted at *claim* time, not completion. A job that kills its worker has
still consumed an attempt, and after `analysis_max_attempts` the sweep fails it instead
of feeding it another worker forever.

**Tradeoffs.** One UPDATE per worker per 5 s, and a job whose worker dies is invisible
for up to 30 s. Shortening the timeout trades recovery latency for the risk of stealing
live work; 30 s was chosen because it is longer than any GC pause and shorter than a
user's patience.

---

## D-024 — Cancellation is a request to the worker, not a status change

**Decision.** `POST /analyses/{id}/cancel` cancels a PENDING job outright. For a RUNNING
job it sets `cancel_requested` and returns RUNNING; the worker notices at its next
heartbeat and moves the row to CANCELLED itself.

**Why.** The API cannot stop the work. The worker owns the DuckDB connection, the open
Parquet file and, in M5, an in-flight model call. Writing CANCELLED from the API would
produce a row that says cancelled while a process is still computing a result for it —
and that result would then arrive, guard or no guard, as a confusing event trail.

Returning RUNNING is deliberate honesty. The alternative is to report CANCELLED
immediately and have the client believe something that is not yet true. One heartbeat
interval later it is true, and the status endpoint says so.

**Tradeoffs.** Up to one beat interval of latency, and a UI that must tolerate
"cancelling" as a state between the request and the fact.

---

## D-025 — POST /analyses is idempotent by key, and answers 200 or 201 to say which

**Decision.** The request may carry an `idempotency_key`. The insert is
`ON CONFLICT (idempotency_key) DO NOTHING RETURNING id`; a key that already exists
returns the existing analysis with **200** instead of **201**.

**Why.** A dropped connection after the server committed is indistinguishable, from the
client's side, from a request that never arrived. Without a key the only safe options
are "retry and risk a duplicate job" or "do not retry and risk losing the request".

Check-then-insert does not solve it: two retries can both find nothing and both insert,
and one of them then fails on the unique index — turning a successful retry into a 500.
`ON CONFLICT` lets the database decide atomically, and the loser reads what the winner
wrote.

The status code is the useful half. 201 means "this is new"; 200 means "your first
attempt already landed, here it is". A client can tell those apart without guessing.

**Tradeoffs.** A key is optional, so a client that omits it gets no protection. The
column is nullable and unique — Postgres permits many NULLs in a unique index, which is
exactly the behaviour wanted.

---

## D-026 — The M4 analysis is hardcoded, and produces the exact result shape M5 must fill

**Decision.** The worker runs a fixed sequence — `inspect_schema`, then
`compare_groups`, then `create_chart` — with no model anywhere. Its output is
`{engine, question, dataset, answer, steps[], table, chart}`.

**Why.** Two arguments.

*Isolating the failure.* If a model went in now, every broken thing would have two
candidate causes, and the boring one (the plumbing) would be blamed on the interesting
one (the model), or worse, the reverse. With a deterministic analysis, anything that
fails is the plumbing, and there is no argument about it.

*Fixing the contract while it is cheap.* That result shape is stored in a JSONB column,
serialised by a pydantic model, and rendered by a React component. Getting it wrong now
means changing three layers later. Getting it right now means M5 replaces one function.
`engine` is stored with the result so a row from M4 stays interpretable next to an
`agent-v1` row forever.

It goes through `ToolRegistry.call` rather than running SQL directly, which is how M4
proves the M3 action space works end to end while there is still no model around to
confuse the picture.

**Tradeoffs.** The analysis is not an answer to the question, and both the answer text
and the UI say so plainly rather than implying otherwise.

---

## D-027 — The UI is unstyled React, and the API is polled rather than pushed

**Decision.** Five unstyled components in `frontend/`, Vite dev server proxying to the
API. The result view polls `GET /analyses/{id}/events?after=<cursor>` once a second.

**Why unstyled.** M4's job is to surface API design flaws while they are still cheap to
fix. It already worked: the events endpoint returns `status` alongside the events
because building the polling component made it obvious that a UI needing both would
otherwise issue two requests per tick and have to reconcile them disagreeing.

**Why polling.** A websocket needs connection state, reconnection logic, and a way to
replay what was missed while disconnected. The event cursor gives replay for free — a
client asks for everything after the highest id it has seen — and each poll costs an
indexed lookup. At M6, if a live cursor is genuinely wanted, the cursor semantics are
already the right shape for SSE.

`setTimeout` chained after each response, never `setInterval`: an interval fires on
schedule regardless of whether the last request returned, so one slow response stacks
requests and the event list flickers backwards.

**Tradeoffs.** Up to one second of latency on a status change, and a poll per second per
open tab. Both are irrelevant at this scale and neither survives contact with a real
deployment, which is a M6 problem.

---

## D-028 — The schema is handed to the model, not discovered by it

**Decision.** Before the agent loop starts, `inspect_schema` is called deterministically
and its result — plus three real sample rows — is rendered into the system prompt.
`inspect_schema` and `profile_column` remain in the action space anyway.

**Why.** The textbook agent starts with an empty prompt and lets the model call
`inspect_schema` as its first move. That costs a full turn — 10 to 40 seconds on a 4B
model with reasoning on — to produce a result that is completely deterministic and
takes 40 ms to fetch.

It is also the turn most likely to go wrong. A model that has not seen the column names
has nothing to ground its first call in, so it guesses, gets an error, and spends a
second turn recovering. Handing the schema over deletes an entire failure class and
roughly a third of the wall clock.

Three sample rows cost about 150 tokens and answer what a type list cannot: is
`InvoiceDate` an ISO timestamp or `1/10/11 10:04`? Is `Country` 'UK' or 'United
Kingdom'? A model writing `WHERE Country = 'UK'` against a column that says 'United
Kingdom' gets an empty result and confidently reports zero — which is the exact failure
this system exists to prevent, arrived at through correct SQL.

**Tradeoffs.** A larger prompt on every turn, and a dataset with 200 columns would need
the rendering truncated. Both tools stay available for the model that wants more.

---

## D-029 — Two budgets, and hitting either asks for an answer rather than failing

**Decision.** `agent_max_steps` (6) bounds model turns; `agent_time_budget_s` (300)
bounds wall clock. Hitting either stops the loop, removes the tools from the request,
and asks for a final answer. The result carries a warning saying which budget ran out.

**Why two.** They are different failure modes. Six fast turns that go nowhere should
stop at six. One turn that takes four minutes because the machine is swapping should
stop on time, whatever the step count says. A single budget always gets one of these
wrong.

**Why not a failure.** A partial analysis that reports what it established is more
useful than an error, and the alternative — discarding real computed results because
the agent was slow to converge — is the worst trade available.

**Why the final turn drops the tools entirely.** Asking a model to "stop calling tools
now" while still passing it tools is asking it to resist the strongest signal in its
prompt. Removing them makes a tool call impossible rather than discouraged.

**Tradeoffs.** Six steps is not enough for a genuinely multi-part question, and the
answer then says so rather than pretending otherwise.

---

## D-030 — The evidence table and chart are built from tool results, never by the model

**Decision.** After the loop, the last successful tabular tool result becomes the
evidence table, and a bar chart is derived from that table. `create_chart`'s output wins
if the agent chose to call it.

**Why the table is independent.** The answer is a claim; the table is what makes it
checkable. If the model produced both, they could agree with each other and be wrong
together — and a UI showing agreeing numbers is more convincing than one showing none,
which makes it worse.

**Why the LAST result.** An agent's early calls are orientation: a schema read, a
distinct-value check, a failed guess at a column name. The call that produced the answer
is the last one that returned rows. Picking the first shows a person the agent's
throat-clearing and labels it evidence.

**Why the chart is not `create_chart`.** That tool charts *stored columns*: it resolves
`x` and `y` against the schema and aggregates in SQL. That is right for "sales by
region" and cannot express the most common real question at all — "which country
generated the most revenue" has no `revenue` column, revenue is `Quantity * UnitPrice`,
and the result exists only as rows in a tool payload. Asking `create_chart` for it fails
with "no such column", and a model that just computed the answer correctly then burns
two turns arguing with a tool.

A chart is deliberately **not** produced for a single-row result. A bar chart with one
bar is decoration presented as analysis.

**Tradeoffs.** A question whose answer is spread across two queries gets a table from
one of them. The full trace is in `steps` either way.

---

## D-031 — Every figure in the answer is traced back to a computed number

**Decision.** `app/agent/verify.py` extracts every figure from the answer, extracts
every number from every successful tool result, and attaches a warning naming the
figures that match nothing. It does not rewrite the answer and does not fail the
analysis.

**Why.** The first end-to-end run on real data produced this:

> "WORLD WAR 2 GLIDERS ASSTD DESIGNS with 53,847 units, significantly exceeding the
> next highest product JUMBO BAG RED RETROSPOT by 16,484 units"

53,847 and 47,363 both came from DuckDB. 16,484 came from the model's head, and the
correct difference is 6,484. Every guard was working — real schema, real SQL, real table
underneath the sentence — and the model still asserted a number nobody computed, in the
one clause where it reads as completely natural.

No prompt wording removes this. A language model does arithmetic by autocomplete, and
autocomplete is right most of the time. A prompt rule was added ("do not do arithmetic
yourself; put the difference in the query") and it did fix this instance — but a rule
that works most of the time is exactly what this check is for.

**Why it warns rather than corrects.** Dropping the sentence or quietly fixing the
number would be a different way of asserting more than is known. A figure the system
cannot trace is a fact about the system's confidence and belongs in front of the user.

**Why the matching is loose.** An answer writes 8,187,806.36 for a stored
8187806.363998184, and 35% for a stored share of 0.3528. A figure matches if it equals a
computed number, is that number rounded to any sensible number of places, or is within
0.5% of it; fractions are also matched against their percentage form. Figures below 100
are skipped entirely — "the top 10 countries" and "3 groups" are numbers no tool needs
to have produced.

The bias is deliberately towards NOT warning. A warning on a correct answer teaches
people to ignore warnings, and then the one that matters is ignored too.

**Tradeoffs.** A correct figure the model rounded unusually could be flagged, and a
wrong figure that coincidentally matches an unrelated number in a payload is missed.
Neither is silent: the trace and the table are both shown.

---

## D-032 — The deterministic engine survived M5, and is selected by configuration

**Decision.** `ANALYSIS_ENGINE` chooses `agent` (the model) or `fixed` (M4's hardcoded
analysis). Both fill the same result contract. The test suite forces `fixed` by an
autouse fixture.

**Why.** The obvious move was to delete the fixed engine once the agent worked. Keeping
it answers one question in one command: *is this broken, or is the model just bad at
it?* With `ANALYSIS_ENGINE=fixed` the whole stack runs with no model in it, and if it
still fails the model was never the problem.

It is also what makes the suite runnable on a machine with no Ollama. A test suite whose
result depends on which weights happen to be pulled is not a test suite; the agent loop
is tested against a scripted model instead, and the genuinely non-deterministic question
— can qwen3:4b actually answer this? — is what `eval/` is for.

**Tradeoffs.** Two engines to keep filling one contract. The contract is pinned by a
test that fails if either drifts.

---

## D-033 — The test suite runs against its own database, created by the suite

**Decision.** `tests/conftest.py` sets `POSTGRES_DB=adi_test` before `app.config` is
imported, creates that database if it does not exist, and migrates it to head.

**Why.** The `db` fixture empties tables. Pointed at the development database that the
running app also uses, that is a data-loss bug wearing a test fixture: it deleted a real
542,000-row upload twice in one afternoon, and the only symptom was the API answering
`no dataset <uuid>` some minutes later, during an unrelated task.

The redirect is one environment variable because the engine and Alembic both read the
same `get_settings()`, and because pydantic-settings reads the environment before the
`.env` file. Creating and migrating the database in code rather than in a README step is
the point: a setup instruction that can be skipped eventually is, and the failure mode
of skipping this one is running the suite against real data.

**Tradeoffs.** A second database on the dev machine, and the first run pays for a
migration.

---

## D-034 — The tools are taken away before the answer is written

**Decision.** The agent runs at most `agent_max_tool_rounds` (2) turns with the tool
definitions attached, leaves that phase the moment a tool has succeeded, and writes the
answer in a separate call with `tools=None` **and a different system prompt**.

**Why.** A user reported five minutes per question. The measurement, on an identical
conversation asked for the same prose:

```
tools attached      41.8 s   2,098 output tokens   7,972 characters of thinking
tools omitted        9.0 s     431 output tokens   1,547 characters of thinking
```

A model holding tools spends its reasoning re-arguing whether to call one. It does this
on every turn, including the turn where it has all the results and only has to write
three sentences. Removing them from the request makes a tool call impossible rather
than discouraged, and makes the turn four times faster as a side effect.

The system prompt is swapped too, for the same reason. The planning prompt is a page of
rules about batching calls, avoiding identifier columns and repairing failed queries —
none of which applies once the results are in. A small model does not ignore
instructions it cannot use; it reasons about them.

The old loop handed the model the tools again just so it could say "I am done". That
turn cost a full model call AND was the slowest one in the run.

**Tradeoffs.** The model cannot decide to run one more query after seeing its results,
unless the first round failed outright. That is what batching (D-036) is for: it asks
for everything at once instead. A genuinely iterative question is worse served, and
`AGENT_MAX_TOOL_ROUNDS` is the dial for anyone who wants to pay for that.

---

## D-035 — The context window is 8192, because 16384 silently ran on the CPU

**Decision.** `llm_num_ctx = 8192`.

**Why.** Measured on the target hardware, a 6 GB RTX 4050, with qwen3:4b:

```
ctx=4096    61.3 tok/s   3.5 GB   100% GPU
ctx=8192    61.4 tok/s   4.1 GB   100% GPU
ctx=16384   29.6 tok/s   5.4 GB   17% CPU / 83% GPU     <- 2.07x slower
```

At 16k the KV cache pushes the model past available VRAM and Ollama spills layers to
the CPU. **Nothing reports this.** No error, no warning, no log line — the same request
simply takes twice as long, and `ollama ps` is the only place the split is visible. It
had been in place since the agent was written, doubling every question.

8192 rather than 4096, which was what was asked for: Ollama's *default* of 4096
truncates silently, and a real conversation here is ~2,400 prompt tokens plus up to
2,000 generated. Over 4096 the head of the prompt goes — which is the schema and the
grounding rules — with no error. 8192 is measurably identical in speed and cannot
truncate, so the risk buys nothing.

**Tradeoffs.** A dataset with a very large schema, or a tool result near the 50-row cap
with wide text columns, has less headroom. The row and character caps in
`app/agent/analyst.py` already bound that; if it ever binds, the fix is a smaller
result, not a bigger window that halves the speed of every request.

**Also here:** `keep_alive: 30m`. Ollama unloads after five minutes by default, so a
user who thinks for six between questions pays a 20-30 second cold load on the next one
and reasonably concludes the app is slow.

---

## D-036 — Several tool calls per turn, rather than several turns

**Decision.** Up to four tool calls from one model turn are executed, and the system
prompt tells the model to ask for everything at once.

**Why.** The economics, measured:

```
one tool call            30-70 milliseconds
one turn of the model    45-90 seconds
```

A model turn is roughly a thousand times more expensive than the work it authorises. An
earlier version of the prompt said "call ONE tool per turn", on the reasoning that a
small model handles one result at a time better. That rule was costing a minute to save
the model forty milliseconds of reading.

Told it may batch, qwen3:4b asked for three queries in a single 51-second turn: the top
products, the busiest countries, and the overall total. That is a better answer than any
of them alone, for the price of the cheapest one.

This is also the answer to "the trace is always two steps and there is no creativity in
it". More analysis per question comes from a **wider** turn, not more turns. More turns
is the same analysis, slower.

**Tradeoffs.** The model has to anticipate what it needs before seeing any of it, and
sometimes asks for something the first result makes redundant. A wasted 40 ms is not a
cost worth optimising.

---

## D-037 — The chart type is inferred from the result, not chosen by the model

**Decision.** `app/agent/charts.py` picks between bar, line, pie, scatter, histogram and
**no chart** by inspecting the result's columns, types and row count. The model is told
in its prompt which shapes produce which charts, so it can aim for one, but it never
names a chart type.

**Why.** Asking the model costs a turn, and a turn is 45-90 seconds. The type is also
not really a judgement: it is a function of the shape, and the shape is fully known by
the time the choice is made. Code decides it in microseconds and cannot pick `pie` for
four hundred rows.

```
10 labels, 1 value       -> bar
a date or month column   -> line
<= 8 positive shares     -> pie
two numeric columns      -> scatter
one numeric column       -> histogram
a single row             -> NO CHART
```

The last line is the one that prompted this. "Is there a relationship between quantity
and unit price" produced a bar chart of one bar under a sentence that already said the
number.

The division is the same one the whole system runs on: the model chooses **what to
compute**, and the computation decides how it looks. That is better than letting it name
a type, because a model that has been told to draw a pie will draw a pie of forty
slices.

**Tradeoffs.** A user who wants a specific chart of a result cannot ask for one yet. The
ordering of the rules is load-bearing and was wrong first time — `Month` and `Revenue`
are both numeric, so a "two numeric columns means scatter" test that ran before the time
test turned a monthly revenue series into a scatter plot. Time is now checked first.

---

## D-038 — A result that is one number is shown as a number

**Decision.** A 1x1 evidence table renders as a labelled figure, not as a table.

**Why.** It arrived as a full bordered table with a sticky header, a column title and a
"1 row" caption, to hold `-0.0012` — directly beneath a sentence that had already said
-0.001. All of the furniture of a table and none of its purpose: nothing to scan,
nothing to compare, nothing to sort.

The evidence still has to be shown, because the premise of the whole system is that an
answer is worth what its evidence is worth. So it is shown as what it actually is: one
computed figure, labelled with the expression that produced it.

**Tradeoffs.** One more branch in the renderer. The alternative — hiding it entirely
because the prose repeats it — would mean the one case where the answer and the evidence
cannot be compared is the case with no evidence displayed.

---

## D-039 — The model is chosen per question, from a catalogue with its measurements attached

**Decision.** `POST /analyses` accepts `model` and `thinking`; both are stored on the
analysis row and travel with the claim to the worker. `GET /models` returns a catalogue
with per-model accuracy, speed and weaknesses. The UI is a picker beside the composer.

**Why per question rather than per deployment.** The trade is real and has no right
answer, measured on this project's own evaluation set:

```
                     qwen3:4b     qwen2.5:3b-instruct
  overall              60%              29%
  lookup              100%              75%
  aggregation         100%              58%
  trend                50%               0%
  data quality         50%              12%
  comparison            0%               0%
  diagnosis             0%               0%
  per question        ~150 s            ~3 s
```

Fifty times faster for roughly half the accuracy. Which side of that a person wants
depends on the question they are about to ask — "which country earns most" is answered
correctly by both, and one of them does it in ten seconds. That is a decision for the
asker, and `LLM_MODEL` in a `.env` file cannot express it.

**Why the menu carries numbers rather than adjectives.** Every model picker says "fast"
and "smart", and neither helps. The real question is "will this answer MY question, and
how long will I wait". Both halves are measured, so both are printed — including what
each model is BAD at. A chooser that lists only strengths is an advert, and the user is
picking between two things that are each bad at something.

**Why it is an allowlist.** `model` arrives as a string in a JSON body from a browser.
Passing it to Ollama would let any request pull and load an arbitrary model on the host
— resource exhaustion, not a feature. An operator can still run anything through
`LLM_MODEL`; a request may only name what has been measured.

**Why it is stored on the row.** For the same reason `dataset_version` is. Two answers
to the same question can legitimately differ because one was asked of a 3B model and one
of a 4B, and a stored answer that cannot say which is one nobody can act on. NULL means
"whatever the worker was configured with", which is also what every row written before
these columns existed means.

**The reasoning toggle is offered, and labelled honestly.** It appears only for a model
that reasons at all, and its tooltip says what was measured: turning reasoning off does
**not** make qwen3 faster (42.1 s versus 43.6 s), because the model emits the same
tokens either way and the flag only decides whether they land in `thinking` or in
`content` — where, on the answer turn, they become the answer. It is exposed because it
was asked for and because it is the model's own switch; it is labelled so nobody reaches
for it expecting speed.

**Tradeoffs.** Two models to keep measured, and a catalogue that goes stale if the
evaluation set changes without a re-run. The numbers name their source in the menu
footer (`python -m eval.runner --agent local-model`) so a sceptical reader can check
them rather than trust them.

---

## D-040 — A conversation is a database entity, not a React array

**Decision.** `conversations` is a table; `analyses.conversation_id` and `turn_index`
place a question in a thread. The worker loads the last three successful exchanges into
the system prompt. A thread is pinned to one dataset AND one version.

**Why in the database.** The thing that needs the history is the *worker*, a different
process from the browser. Keeping the thread in React state would mean shipping the whole
conversation with every request — unbounded growth, and the client deciding what the
model is told.

**What is carried is the interesting part.** Not the conversation. A turn is compressed
to three fields:

```
the question   so "last year" has something to attach to
the answer     so "why is that?" has a referent
THE SQL        so "same thing but for France" is one edit away
```

The SQL is the one people leave out and the most valuable of the three. A model that can
see `SELECT Country, SUM(Quantity*UnitPrice) ... GROUP BY Country` writes the follow-up by
changing one clause; a model given only the prose has to rediscover the whole query from
the schema. Tool payloads, event trails and reasoning traces are excluded: they are large
and nothing ever refers back to them.

**Why three turns.** The literature converges on 3-5 for a rolling window. The binding
constraint here is tighter than the literature's: at ~120 tokens per compressed turn
against an 8,192 window where the schema and samples already cost ~1,200, three fits
comfortably and five starts competing with the results themselves.

**Why the version is pinned.** The same reason it is pinned on an analysis. A follow-up
must be about the data the first answer described, and half a thread about `retail` beside
half about `sensors` is exactly how a fact from one ends up in an answer about the other.

**Tradeoffs.** A conversation cannot span datasets — a real limitation, deliberately
taken. Threads are never garbage-collected yet.

---

## D-041 — Postgres caches answers, because a result is a pure function of four things

**Decision.** `answer_cache`, keyed on `(dataset_id, dataset_version, question_hash,
llm_model)`. A hit returns the stored result in about five milliseconds, marked
`cached: true`, and the UI says so in words.

**Why.** An analysis costs 90-190 seconds of local GPU, and every input to it is known
before any work begins. The second ask has nothing to compute. This is also the honest
answer to "you are only using Postgres for metadata": it is the queue, the event log, and
now the memoisation table.

**Every part of the key prevents a specific wrong answer.** The version, so a new upload
cannot serve an answer about the old file. The model, because the two available models
score 60% and 29% and one's answer must never reach somebody who asked for the other. The
hash is of a *normalised* question, so capitalisation and a trailing question mark do not
create a second entry.

**What is NOT cached matters more than what is.** Failures, and any answer carrying a
warning — out of budget, answered without querying, or containing a figure that could not
be traced to a computation. **Speed reads as confidence**: an answer that appears
instantly looks retrieved and certain, and this system's premise is that a claim is worth
what its evidence is worth.

**The next version, designed for and not built.** Semantic caching: embed the question,
match above ~0.8 cosine, so "which country earns most" also hits "top country by revenue"
— published implementations report 60-70% hit rates against roughly 10% for exact match.
It needs pgvector (this runs `postgres:16-alpine`, which does not ship it) and an
embedding model. D-042 recovers part of the same benefit for free.

---

## D-042 — The sub-agent is a smaller model, not another copy of the same one

**Decision.** A query rewriter runs on qwen2.5:3b-instruct (~1 s) and turns "what about
France?" into a standalone question before the analyst sees it. A cheap keyword gate skips
it entirely when the question already stands alone.

**Why not fan-out.** The instinct with sub-agents is parallelism: a planner, a SQL writer
and a critic, all at once. **On one GPU that is strictly a loss.** The model is already
100% resident and saturating the card; two concurrent calls do not run in parallel, they
split the same tokens per second between them and add KV-cache pressure. Fan-out buys
thoroughness and pays for it in latency — the thing this project spent a day removing.

**What pays is a model of a different size.** At ~1 s against the analyst's ~45 s, the
rewriter costs about 2% of a turn and earns three things: the follow-up is resolved, the
analyst's prompt gets shorter, and — the one that is easy to miss — **the cache starts
working on follow-ups.** "What about France?" means something different in every thread
and can never hash to anything useful; rewritten, it hashes identically to somebody asking
it directly.

The rule worth keeping: *a sub-agent earns its place when it is cheaper than the model it
serves, not when it is another copy of it.*

**Tradeoffs.** A rewriter that "improves" a standalone question changes what was asked, so
it is constrained hard and every failure path returns the original untouched. An
optimisation that can change the answer is a bug.

---

## D-043 — LISTEN/NOTIFY wakes the worker; SKIP LOCKED remains the queue

**Decision.** `enqueue` issues `NOTIFY analyses_new`. The worker holds a dedicated
listening connection and blocks on it instead of sleeping between polls. The poll interval
survives as that wait's timeout.

**Why it is only a wake-up.** Notifications are **not persisted**: anything sent while no
worker is listening is gone forever. A queue built on them loses every job enqueued during
a restart. So the table remains the queue and this removes only the waiting — up to a
second of it, on every question.

**Why the payload is empty.** Sending the analysis id and skipping the claim would be a
second, unreliable queue: the notification can be lost, two workers can both receive it,
and payloads cap at 8 KB. The signal says "something changed, go and look"; the database
stays the only source of truth about what to work on.

**Why a dedicated connection.** `LISTEN` occupies a connection for as long as it listens.
Borrowing one from the pool would remove it from circulation and hand it back still
subscribed.

**Tradeoffs.** One extra connection per worker. Every failure path — no permission, an
older server, a proxy that drops notifications — degrades to plain polling, so correctness
never depends on a notification arriving.

---

## D-044 — The toolkit is an MCP server; the agent is not

**Decision.** `app/mcp_server.py` publishes `list_datasets`, `inspect_schema`,
`profile_column`, `execute_sql`, `compare_groups` and `correlation` as MCP **tools**, and a
dataset's schema and sample rows as MCP **resources**. The agent loop and the job queue are
not published.

**Why the tools and not the agent.** Publishing "ask a question" would bury a 90-second
local model call behind one opaque call, and it duplicates reasoning — the client already
has a model, almost certainly better than qwen3:4b. What this project owns that no client
has is the deterministic half: an exact profiler, a sandboxed executor that resolves column
names against a live schema, a grouping tool that refuses meaningless groupings. Those are
worth publishing. The reasoning is not.

The practical result: Claude Desktop can analyse a 542,000-row Parquet file through the
same guard rails, with the same "the model never computes anything" property, and none of
the local model's latency.

**Tools versus resources maps onto a distinction this codebase already made.** Tools are
model-controlled verbs; resources are application-controlled nouns with addresses
(`dataset://{id}/schema`). A client can attach the schema up front at no tool call — the
same argument `app/agent/prompt.py` makes when it hands the schema over rather than making
the agent go and fetch it.

**The security boundary moves, and is re-established.** `dataset_id` was previously
unreachable by any model, living only in `ToolContext`. Over MCP the caller supplies it, so
it is parsed as a UUID and checked against stored datasets before a context exists.
Everything after that point is the same `registry.call` the local agent uses — one
implementation, two front doors, so row caps and the SQL allowlist cannot drift apart.
Verified: `DROP TABLE dataset` through MCP is refused by the same sandbox, with the same
message.

**Tradeoffs.** The wrappers are written out by hand rather than generated, so a new
registry tool must be added here too. Generated wrappers needed a private attribute to
carry the schema, and the descriptions a model actually reads deserve to be visible in the
source rather than assembled at import time.

---

## D-045 — Groq is a second client, and the provider decides the loop's shape

**Decision.** `app/agent/groq.py` speaks Groq's OpenAI-compatible API; `app/agent/llm.py`
still speaks Ollama's. `app/agent/factory.py` picks one from the model name. Three hosted
models join the picker: `openai/gpt-oss-20b` (production), and `qwen/qwen3.6-27b` and
`qwen/qwen3.8-27b` (both preview).

**Why two clients and not a branch.** The two APIs disagree about nearly every field that
matters — endpoint, where `tool_calls` live, whether reasoning is a boolean or an effort
level, whether `num_ctx` exists, how usage is reported. One client with seven `if`
statements is one client that is wrong about one of them. Two classes returning the same
`ModelTurn` means the agent loop never learns which it is holding.

**Two translation details are silent failures rather than errors.** A tool result must
carry `tool_call_id` matching the assistant turn that requested it — Ollama matches by
name and ignores the field, OpenAI drops the message. And `arguments` must be a JSON
*string*: sending an object is accepted and then ignored, which looks exactly like a model
that forgot what it just asked for.

**The provider decides how many tool rounds are affordable.** This was one constant and
cannot stay one:

```
local   a round is 45-90 s      two is already the limit of patience
Groq    a round is under 1 s    four is cheaper than one local round
```

Two rounds on Groq would leave the entire benefit unspent; four locally would be a
six-minute wait. So `tool_rounds` is read from the catalogue, not from configuration.

**And it decides whether sub-agents may fan out.** `ModelProfile.fans_out` is computed,
not configured, because *the same architecture is right in one deployment and wrong in the
other*: on one local GPU two concurrent calls split one card, while against a hosted API
they are genuinely concurrent. The rewriter follows the same rule and now runs on the
cheapest model of the **same provider** — a 1-second local call in front of a 2-second
hosted answer is 50% overhead, where the same call in front of a 150-second local answer
is 2%.

**Measured, on the 541,909-row retail dataset:** 2.6 s against ~150 s locally, for the
same question and the same answer.

**Two things the free tier taught us immediately.**

*The limit is 8,000 tokens per minute, not the 250,000 the plan comparison implies*, and
one analysis of a wide schema is comfortably 2,500. The first version turned a 429 into a
FAILED analysis that discarded two successful queries. A 429 is the one HTTP error that is
neither a bug nor permanent — it is an instruction to wait, and Groq says exactly how long,
so the client now honours it. Retries only on 429 and 5xx: a 401 will not be different in
two seconds.

*And a model failure mid-planning is no longer fatal.* If queries have already succeeded,
the loop breaks out and answers from what it has, with a warning saying why. Losing real
computed work to a transient rate limit was the worst trade in the system.

**Tradeoffs.** The hosted path sends the schema, three sample rows and the tool results
(capped at 50 rows) off-machine — never the file. The picker says so, and the local models
remain fully capable, so "local-first" stays true as a default rather than as a slogan.

---

## D-046 — The cache key includes a prompt version

**Decision.** `question_hash` hashes `PROMPT_VERSION` together with the normalised
question.

**Why.** A rule was added telling the model to write "84%" rather than
"0.8399690286861813". The next run served the previous answer from cache in 552 ms, and
the fix looked like it had simply not worked.

A cache keyed only on the question pins yesterday's behaviour to today's build. Every
input to an answer belongs in its key, and the prompt is an input — arguably the most
important one, since it is the only one a developer changes.

**Tradeoffs.** Bumping the version discards every entry, which is correct and cheap: the
entries were computed under rules that no longer apply. It has to be bumped by hand, which
is a step that can be forgotten — the alternative, hashing the prompt text itself, would
invalidate the cache on a typo fix in a comment.

---

## D-047 — `create_chart` left the agent's action space

**Decision.** `AGENT_TOOLS` no longer offers `create_chart`. It stays in the registry and
stays published over MCP.

**Why.** Since D-037 the chart type is inferred from the shape of the result, so the tool
was never called — it sat in every prompt as a distractor, costing tokens and selection
accuracy to advertise a capability the model had no reason to use. Removing an unused tool
is not tidying: fewer tools measurably improves selection on small models.

**And the chart now appears in "how it got there".** It was missing entirely, which was a
real gap rather than a cosmetic one: that section is the page's honesty guarantee, and a
chart appearing with no entry explaining where it came from is exactly the unexplained
artefact the section exists to prevent. It has no step because nothing called a tool, so
the trace says that — "bar chart · inferred from the shape of the result, no tool call".

---

## D-048 — An MCP server logs every call, because "is it being used?" is unanswerable otherwise

**Decision.** Every tool call through `app/mcp_server.py` writes one line to stderr and,
when `MCP_LOG_FILE` is set, appends it to a file. `--selftest` exercises the server
in-process.

**Why.** An MCP server runs as a subprocess of its client with **stdout owned by the
protocol**, so there is no console to watch and a stray `print` would corrupt the JSON-RPC
stream. Without a log there is no way to answer "did the client really call my tools, or
did it answer from memory?" other than trusting its UI — and a model answering from memory
looks identical to one that called a tool.

stderr is free (the protocol does not use it) so it becomes the audit trail. The file
exists because the question is usually asked *after* the fact, and reproducing it means
asking the same question again.

`--selftest` separates two failures that look the same: a green selftest with a silent log
means the server works and the client is misconfigured.

**Tradeoffs.** The log grows without rotation and records question-derived SQL, so it is
gitignored. Off by default; stderr alone is enough for the interactive case.
