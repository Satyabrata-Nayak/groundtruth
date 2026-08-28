# Architecture

Current state: **data layer and tool/evaluation layer complete, no AI yet**. This
document describes the target and marks what exists today.

---

## The one-sentence version

A user uploads a dataset and asks a question in English. An agent inspects the data,
chooses from a fixed set of deterministic tools, executes SQL / statistics / ML through
them, verifies the numbers in its own answer against real computations, and returns an
evidence-backed result showing every step it took.

---

## Component map

```
                     React + TypeScript + Vite          [M4 skeleton, M6 real]
                              │
                              │  HTTP/JSON  (poll for job status)
                              ▼
                        FastAPI                          [M4]
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          PostgreSQL                  Analysis Worker    [M4]
       [M2: datasets + profiles]      python -m app.worker
        · dataset metadata                  │
        · analysis jobs (queue)             ├── Agent loop          [M5]
        · execution events                  │      │
        · verified claims                   │      └── Ollama ──> qwen3  [M1 ✓]
                                            │           (native, GPU)
                                            ├── Tool registry    [M3 ✓]
                                            │      · inspect_schema
                                            │      · execute_sql
                                            │      · profile_column
                                            │      · compare_groups
                                            │      · correlation
                                            │      · create_chart
                                            │      · detect_anomalies   [M6]
                                            │
                                            ├── DuckDB (sandboxed)  [M2 ✓]
                                            │      └── reads Parquet, read-only
                                            ├── SciPy / scikit-learn [M6]
                                            └── Verification layer   [M6]

       data/datasets/<dataset-id>/<version>/data.parquet   — immutable   [M2]

       eval/  50 golden questions, 3 generated datasets, graded runner   [M3 ✓]
              scored against the SAME tool registry the agent will use
```

**Legend:** ✓ done · [Mn] arrives in that milestone.

---

## The three ideas the design rests on

### 1. The LLM decides *what*; deterministic code decides *how*

The model never computes a number. It selects a tool and supplies arguments; the tool
computes. This is what makes results reproducible and auditable, and it is why
"arbitrary LLM-generated Python" is on the permanent exclusion list — it would collapse
the two roles into one.

```
LLM  ──"call execute_sql with this query"──►  our code
                                                 │
                                       validate ─┤ ← the security boundary
                                                 │
                                         execute ─┴──►  deterministic result
```

### 2. Every number in the answer must trace to a computation

The final answer is not trusted because a model produced it. Numbers appearing in it
are matched against the values that actually came out of tool calls, and classified
`verified` / `inconsistent` / `unsupported` (M6). Hallucinated figures become visible
rather than plausible.

### 3. Nothing that runs long runs inside an HTTP request

`POST /analyses` creates a row and returns immediately. A separate worker process
claims the job from Postgres and does the work. The client polls. This is why local
inference taking 30–90 seconds does not mean a 90-second HTTP request, and it is what
lets a crashed worker's job be reclaimed rather than lost.

---

## Trust boundaries

Two untrusted inputs. Both are treated as data, never as instruction:

| Source | Threat | Control |
|---|---|---|
| Model-generated SQL | reads/writes outside the dataset | sqlglot AST allowlist + DuckDB with `enable_external_access=false` + only this dataset registered [M2] |
| Dataset *contents* | prompt injection in a cell (`"ignore previous instructions..."`) | fenced and labelled as data in the prompt; tested with a hostile fixture [M5] |

The model never sees or supplies a filesystem path. It supplies a `dataset_id`; the
backend maps that to a trusted path.

---

## What exists after M4

```
app/config.py              typed settings from .env, single source of truth
app/db/base.py             DeclarativeBase
app/db/models.py           Dataset, DatasetVersion, ColumnProfile,
                           Analysis, AnalysisEvent  <- the M4 job queue
app/db/session.py          engine, pooled connections, transactional session_scope
app/db/migrations/         Alembic, URL injected from app.config
app/data/storage.py        dataset_id -> path. THE TRUST BOUNDARY.
app/data/ingest.py         validate, CSV->Parquet, immutable versioning
app/data/profile.py        row/column stats, exact null and distinct counts, flags
app/data/sandbox.py        four-layer read-only SQL executor
app/data/service.py        create/list/get/delete — the operations the API will call

app/tools/base.py          Tool, ToolContext, ToolResult, ToolRegistry. THE CONTRACT.
app/tools/_common.py       resolve_column — a model's column name never reaches SQL
app/tools/inspect.py       inspect_schema, profile_column
app/tools/query.py         execute_sql — the general tool
app/tools/stats.py         compare_groups, correlation (Pearson + Spearman)
app/tools/chart.py         create_chart — a validated spec, never a rendered image

app/jobs/queue.py          claim / heartbeat / reclaim / finish. FOR UPDATE SKIP LOCKED.
app/worker/loop.py         the worker process: claim, run, record, repeat
app/worker/heartbeat.py    a background beat that also reads back "still wanted?"
app/worker/analysis.py     M4's hardcoded analysis — the shape M5 must fill
app/worker/__main__.py     `python -m app.worker`
app/api/main.py            FastAPI app, CORS, domain-error handlers, /healthz
app/api/deps.py            per-request transaction; why the endpoints are sync `def`
app/api/schemas.py         the published contract, decided rather than inherited
app/api/routes/            datasets.py, analyses.py

frontend/src/App.jsx       five unstyled components, one piece of state each way
frontend/src/api.js        every call to the backend, and the only place errors parse
frontend/src/Result.jsx    cursor polling with setTimeout, never setInterval

eval/datasets/             seeded generators: ecommerce, marketing, sensors
eval/questions/*.yaml      50 golden questions with hand-written reference SQL
eval/answers/*.json        ground truth, COMPUTED by running that SQL
eval/suite.py              loads and validates the question set
eval/expected.py           computes ground truth through the sandbox
eval/grader.py             number extraction, tolerance, ranking, must_mention
eval/agents.py             oracle / refusing / schema-only — scale calibration
eval/runner.py             scores an agent, reports by category
eval/build.py              regenerate everything; --check fails on drift

docker-compose.yml         Postgres 16 on host port 5433, health-checked
scripts/bench_model.py     the M1 deliverable: measures the model choice
tests/                     335 tests, incl. a 30-query SQL attack corpus and a
                           crash drill that hard-kills a real worker process
docs/                      decisions D-001..D-027, learning notes, benchmarking
```

**Model selected: `qwen3:4b`, reasoning enabled** (D-006, D-009). Measured 53.8 tok/s
fully GPU-resident, 100% on JSON planning, tool selection, tool arguments and SQL
correctness. `qwen3:8b` matched every capability metric but ran 5.7x slower because it
cannot fit alongside a KV cache and spills 38% of its layers to system RAM.

Deliberately absent: **any LLM client in `app/`.** Three milestones in, nothing in the
application talks to a model. M2 made deterministic analysis correct, M3 defined what a
model will be allowed to do and built the scoreboard it will be developed against, and
M4 built the entire request path around a hardcoded analysis — so that when the model
arrives in M5, anything that breaks is the model.

**The scoreboard is calibrated.** Three stub agents bracket its scale before any real
agent exists — `oracle` (executes the reference SQL) scores 100%, `refusing` scores 0%,
and `schema-only` (fluent, confident, no computed numbers) also scores 0%. A future
score of 62% therefore sits on a scale whose endpoints were measured, not assumed.

---

## The M2 data path, as built

```
  file (CSV / Parquet)
        |
        v  ingest.validate_source      size cap, extension, non-empty
        v  ingest._convert_csv_to_parquet   sample_size=-1: full-file type inference
        v  storage.allocate_version_dir     mkdir(exist_ok=False) -> atomic claim
        |
   data/datasets/<uuid>/v<n>/data.parquet   IMMUTABLE once written
        |
        v  profile.profile_parquet     SUMMARIZE + exact nulls/distincts + duplicates
        v  service.create_dataset      commit metadata; delete the file if that fails
        |
   PostgreSQL: datasets / dataset_versions / column_profiles
        |
        v  sandbox.execute_sql(dataset_id, version, sql)
             L1 sqlglot allowlist  -> L2 confined DuckDB -> L3 one view -> L4 limits
        |
   QueryResult(columns, rows, truncated, execution_ms)
```

Measured end to end on a 5,000-row dataset: ingest + profile + persist, then a
grouped profit-margin query returning in ~11 ms.

---

## The M3 tool path, as built

```
  model proposes            {"name": "compare_groups",
                             "arguments": {"group_column": "Reveune", ...}}
        |
        v  ToolRegistry.call()
        |
        +-- unknown tool?     -> ToolResult(ok=False, "Available tools: ...")
        |
        v  schema validation  types, enums, bounds, no undeclared keys
        |
        +-- bad arguments?    -> ToolResult(ok=False, "must be integer, got boolean")
        |
        v  Tool.execute(ToolContext(dataset_id, version), **cleaned)
        |     |
        |     v  resolve_column("Reveune")  -> looked up in the LIVE schema
        |     |
        |     +-- no match?   -> ToolResult(ok=False, "Did you mean: revenue?")
        |     |
        |     v  SQL built from the CANONICAL name, never the model's string
        |     v  app.data.sandbox.execute_sql   (the same four layers as always)
        |
        v  Tool.model_view(data)     trims what re-enters the context window
        |
   ToolResult(ok, data, model_data, summary, duration_ms)
        |
        +--> caller / browser   gets `data`        (every chart point)
        +--> the model          gets `model_data`  (a 12-point summary)
```

`ToolContext` carries `dataset_id` and `version`. Neither appears in any tool's JSON
schema, so the model cannot name a dataset — it is told which one it is looking at.

---

## The M3 evaluation path, as built

```
  eval/datasets/*.py          seeded generators, committed
        |
        v  python -m eval.build
        |
        v  generate CSV -> service.create_dataset -> registered in Postgres
        |
        v  execute each question's reference_sql THROUGH THE SANDBOX
        |     (so a question the agent could not answer fails here, not later)
        |
   eval/answers/*.json         ground truth, committed and diffable
        |
        v  python -m eval.runner --agent <name>
        |
        v  for each question: agent.answer(question, context, registry)
        |
        v  grader: extract numbers from prose, normalise, compare within tolerance
        |          check ranking order, check must_mention concepts
        |
   accuracy / values / mentions / tool calls, broken down by category
```

`python -m eval.build --check` recomputes ground truth and exits non-zero if it moved,
naming every question affected. That is what turns a generator edit from a silent
change into a reviewed one.

---

## The M4 request path, as built

Two processes, one database, and nothing slow inside an HTTP request.

```
  browser
    │
    │ POST /analyses  {dataset_id, question}
    ▼
  FastAPI ──────────────────────────────────────────────┐
    │  resolve the dataset version and PIN it            │  ~4 ms
    │  INSERT one row, status = PENDING                  │  no analysis here
    │  201 {id, status: "PENDING"}                       │
    └───────────────────────────────────────────────────┘
    │
    │                        ┌──────────── Postgres ────────────┐
    │                        │  analyses          (the queue)   │
    │                        │  analysis_events   (the trail)   │
    │                        └──────────────────────────────────┘
    │                                     ▲
    │                                     │  UPDATE ... WHERE id = (
    │                                     │      SELECT id ... FOR UPDATE
    │                                     │      SKIP LOCKED LIMIT 1)
    │                                     │  RETURNING ...
    │                                     │
    │                        ┌──────── python -m app.worker ────┐
    │                        │  claim  (PENDING -> RUNNING)     │
    │                        │    │                             │
    │                        │    ├─ heartbeat thread, 5 s ─────┼─► "alive?"
    │                        │    │                             │   "still mine?"
    │                        │    │                             │   "still wanted?"
    │                        │    ▼                             │
    │                        │  run_analysis                    │
    │                        │    inspect_schema                │
    │                        │    compare_groups   ── M3 tools ─┼─► DuckDB sandbox
    │                        │    create_chart                  │
    │                        │    │                             │
    │                        │    ▼                             │
    │                        │  succeed | fail | cancelled      │
    │                        │  ...or LOST: write nothing       │
    │                        └──────────────────────────────────┘
    │
    │ GET /analyses/{id}/events?after=<cursor>     every 1 s
    ▼
  {events: [...], next_after: 384, status: "SUCCEEDED"}
```

**Every terminal write carries `AND worker_id = :me AND status = 'RUNNING'.'** That one
clause is what makes crash recovery safe: a worker that was reclaimed while it was slow
finds its UPDATE matches zero rows, and drops a perfectly good result rather than
writing a second, conflicting story about the same row.

**The status travels with the events** so a polling client makes one request per tick
rather than two — and never has to reconcile a trail and a status read a few
milliseconds apart.

---

## The M4 state machine

```
                 POST /analyses
                       │
                       ▼
                   ┌────────┐  cancel  ┌───────────┐
                   │PENDING ├─────────►│ CANCELLED │
                   └───┬────┘          └───────────┘
        claim_next()   │                     ▲
   FOR UPDATE SKIP     │                     │ worker sees the flag
        LOCKED         ▼                     │ at its next heartbeat
                   ┌────────┐                │
        ┌──────────┤RUNNING ├────────────────┘
        │          └───┬────┘
        │              │
  heartbeat goes       ├──────────────► SUCCEEDED   result written
  stale for 30 s       │
        │              └──────────────► FAILED      reason written
        │
        ├── attempts <  max  ──► back to PENDING   (worker_id cleared)
        └── attempts >= max  ──► FAILED            "abandoned after N attempts"
```

Nothing moves a row *out* of a terminal state. That is what makes "the result is final"
true rather than hoped for.

Attempts are counted at **claim** time, not completion, so a job that kills its worker
has still consumed an attempt — otherwise a poison job is retried until it has killed
every worker you own.

---

## What exists after M5

The question is now answered by a language model that is not allowed to compute
anything. `app/agent/` is the whole of the reasoning layer:

```
app/agent/
  contract.py   AnalysisFailed, Emit, Checkpoint, the shared result shape
  llm.py        the ONLY code that talks to Ollama
  prompt.py     what the model is told before it is asked anything
  analyst.py    the loop: model proposes, tools compute, evidence is collected
  evidence.py   the table and chart shown underneath the answer
  verify.py     every figure in the answer, traced back to a computation
```

## The M5 request path, as built

```
  browser ── POST /analyses ──► FastAPI ── INSERT one PENDING row ──► 201 in ~4 ms
                                              (unchanged from M4)
                                                    │
  ┌──────────────────── python -m app.worker claims it ─────────────────────────┐
  │                                                                              │
  │  STEP 0  deterministic, before a single token is generated                   │
  │    inspect_schema  ──────────────────────────► 40 ms, cannot be wrong        │
  │    SELECT * LIMIT 3  ─────────────────────────► three real sample rows       │
  │              │                                                               │
  │              ▼                                                               │
  │    system prompt = rules + schema + samples                                  │
  │                                                                              │
  │  STEP 1..6  the loop, bounded by 6 turns OR 300 seconds                      │
  │                                                                              │
  │      ┌───► model turn ──┬── tool call ──► ToolRegistry.call ──► DuckDB       │
  │      │   (30-60 s each) │                       │                            │
  │      │                  │      result JSON ◄────┘                            │
  │      └──────────────────┤      (a failure is a REPAIR MESSAGE, not a crash)  │
  │                         │                                                    │
  │                         └── prose, no tool call ──► the answer               │
  │                                                                              │
  │  STEP 7  presentation, built from the RESULTS and not from the model         │
  │    evidence table  ◄── the last successful tabular result                    │
  │    chart           ◄── derived from that table (or create_chart, if used)    │
  │    verification    ◄── every figure in the answer vs every computed number   │
  │                                                                              │
  └──────────────────────────────────────────────────────────────────────────────┘
                                       │
  browser ◄── GET /analyses/{id}/events?after=<cursor> ── every 1 s (unchanged)
```

**The one line that everything else follows from:** the model chooses *what* to
compute and never computes anything. Every number a user sees came out of DuckDB.

## Where the guards are

| failure | where it is caught |
|---|---|
| invented column names | the schema is in the prompt (`prompt.py`) |
| `WHERE Country = 'UK'` on 'United Kingdom' | three real sample rows in the prompt |
| a bad tool call | `ToolResult(ok=False)` fed back as a repair message |
| the same call forever | `attempted` set; the repeat is refused with an explanation |
| parallel duplicate calls | at most 2 honoured per turn |
| a loop that goes nowhere | step budget and time budget, independently |
| an answer with nothing behind it | pushed back once, then flagged in `warnings` |
| a table that agrees with a wrong answer | the table is built from tool results |
| arithmetic done in the model's head | `verify.py` traces every figure |
| "is it the model or the plumbing?" | `ANALYSIS_ENGINE=fixed` removes the model |

## The two engines

```
                     run_analysis()
                           │
        ANALYSIS_ENGINE ───┤
                           │
      "agent" ─────────────┴───────────── "fixed"
         │                                   │
  app/agent/analyst.py              app/worker/analysis.py
  a model choosing tools            a fixed sequence, no model
         │                                   │
         └──────────► the same result shape ◄┘

  {engine, question, dataset, answer, steps[], table, chart, warnings[]}
```

`engine` is stored with every result, so a row written by either stays interpretable
next to the other forever. The contract is pinned by a test that fails if either drifts.
