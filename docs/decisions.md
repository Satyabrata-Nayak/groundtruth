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
