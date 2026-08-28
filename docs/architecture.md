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

## What exists after M3

```
app/config.py              typed settings from .env, single source of truth
app/db/base.py             DeclarativeBase
app/db/models.py           Dataset, DatasetVersion, ColumnProfile
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
tests/                     262 tests, incl. a 30-query SQL attack corpus
docs/                      decisions D-001..D-021, learning notes, benchmarking
```

**Model selected: `qwen3:4b`, reasoning enabled** (D-006, D-009). Measured 53.8 tok/s
fully GPU-resident, 100% on JSON planning, tool selection, tool arguments and SQL
correctness. `qwen3:8b` matched every capability metric but ran 5.7x slower because it
cannot fit alongside a KV cache and spills 38% of its layers to system RAM.

Deliberately absent: FastAPI, any HTTP route, any LLM client in `app/`. M2's job was
to make deterministic analysis correct before any AI exists; M3's was to define what
the AI will be allowed to do, and to build the scoreboard it will be developed against.

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
