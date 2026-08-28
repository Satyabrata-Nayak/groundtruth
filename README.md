# groundtruth

**Local-first AI data analysis where every number is traceable to a real computation.**

Upload a CSV or Parquet file, ask a
question in English, and get an evidence-backed answer — with every SQL query,
statistic and chart the system used shown alongside it, and every number in the answer
checked against a real computation.

**Runs entirely on your own machine. No API keys, no cloud services, no cost.**

> **Status: in active development.** The data layer, the deterministic tool registry
> and the 50-question evaluation set are complete and passing. The agent itself is
> next — see the roadmap below.

---

## Why this is not another "chat with your CSV"

Three design commitments, in order of importance:

1. **The model never does arithmetic.** It chooses which deterministic tool to call and
   with what arguments. The tool computes. Results are reproducible and auditable.
2. **Every number in the answer is traced back to a computation** and labelled
   `verified`, `inconsistent` or `unsupported`. Hallucinated figures become visible
   instead of plausible.
3. **Model-generated SQL is treated as hostile input** — parsed to an AST, allowlisted
   to `SELECT`/`WITH`, and executed on a connection with filesystem and network access
   switched off.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async, typed, automatic validation |
| Analytics | DuckDB over Parquet | SQL is a surface the model can emit and we can *validate* — see [D-001](docs/decisions.md) |
| Metadata + job queue | PostgreSQL | `FOR UPDATE SKIP LOCKED` gives a durable queue without Redis |
| LLM | Qwen3 via Ollama | local, GPU-accelerated, free |
| Frontend | React + Vite | deliberately unstyled — it exists to surface API design flaws, not to look finished |
| Env | uv | fast, reproducible lockfile |

Not used, deliberately: LangChain, Redis, a vector database, Kubernetes, cloud storage,
paid APIs, arbitrary LLM-generated Python.

---

## Setup

### Prerequisites

```
Python 3.12+     Node.js LTS     Docker Desktop     Ollama     NVIDIA driver
```

### 1. Python environment

```bash
# install uv if needed:  irm https://astral.sh/uv/install.ps1 | iex
uv sync --extra dev
```

### 2. Database

```bash
cp .env.example .env
docker compose up -d           # Postgres 16 on host port 5433
.venv/Scripts/alembic.exe upgrade head
```

### 3. Model

```bash
ollama serve                   # if not already running
ollama pull qwen3:4b
```

### 4. Frontend

```bash
cd frontend && npm install
```

### 5. Verify the model

```bash
uv run python scripts/bench_model.py qwen3:4b
```

This measures latency, structured-output validity, tool-calling reliability and SQL
correctness on your hardware. Results for the reference machine are in
[`docs/benchmarking.md`](docs/benchmarking.md).

---

## Running it

Three processes. The split is the point: the API answers in milliseconds because it
never does the work.

```bash
uv run uvicorn app.api.main:app --reload    # http://127.0.0.1:8000  (docs at /docs)
uv run python -m app.worker                 # claims jobs from Postgres and runs them
cd frontend && npm run dev                  # http://localhost:5173
```

Then: upload a CSV, select it, ask a question, and watch the status go
`PENDING -> RUNNING -> SUCCEEDED` with the tool calls appearing as they happen.

The worker is safe to kill at any point. Its job is reclaimed and finished by the next
one — there is a test that hard-kills a real worker process to prove it
(`tests/test_worker_recovery.py`).

It also needs **Ollama** running with the model pulled:

```bash
ollama serve
ollama pull qwen3:4b
```

Expect **90 seconds to three minutes** per question on a laptop. That is a local 4B
model reasoning, not the plumbing: the queue answers `POST /analyses` in about 4 ms and
every step appears in the UI as it happens.

> **Why it takes that long, and how to make it faster.** qwen3 is a *reasoning* model:
> it emits ~2,500 tokens of thinking per turn whatever you ask of it, and at ~61 tok/s
> on a 6 GB laptop GPU that is ~40 seconds a turn. Setting `think: false` does not help
> — measured at 42.1 s versus 43.6 s, because the model produces the same tokens and
> merely moves them from `thinking` into `content`. The lever is the model:
> `ollama pull qwen2.5:3b-instruct` and `LLM_MODEL=qwen2.5:3b-instruct` answers in
> about three seconds instead of ninety. See `docs/benchmarking.md` for what that costs
> in accuracy.

### What actually happens when you ask

```
  your question
       │
       ▼
  the schema and three real sample rows are fetched deterministically   ~40 ms
       │                       (not asked of the model: it cannot be wrong,
       │                        and it is the turn most likely to go wrong)
       ▼
  the model chooses ONE tool per turn ──► DuckDB, in a sandbox ──► result JSON back
       │  up to 6 turns, or 300 seconds, whichever comes first
       ▼
  prose answer + the table it came from + a chart of that table
       │
       ▼
  every figure in the answer is checked against the numbers actually computed
```

The model **chooses what to compute and never computes anything**. Every number a user
sees came out of DuckDB.

> **The last step is the one worth explaining.** The first real run answered "53,847
> units, exceeding the next product by 16,484" — and 53,847 − 47,363 is 6,484. Both
> totals came from the database; the gap came from the model's head, in a sentence
> where everything else was right. `app/agent/verify.py` now traces every figure in an
> answer back to a computed number and names the ones it cannot. It does not rewrite
> the sentence: a figure the system cannot vouch for is a fact about its confidence,
> and quietly fixing it would be another way of claiming more than is known.

### Running it without a model

```bash
ANALYSIS_ENGINE=fixed uv run python -m app.worker
```

The whole stack — API, queue, worker, tools, sandbox, UI — then runs with no model in
it, using M4's deterministic analysis. If something still fails, the model was never
the problem.

---

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | components, trust boundaries, data flow |
| [`docs/decisions.md`](docs/decisions.md) | every non-obvious choice, with alternatives and why they lost |
| [`docs/learning-notes.md`](docs/learning-notes.md) | the concepts, explained — quantization, KV cache, tool calling, constrained decoding |
| [`docs/benchmarking.md`](docs/benchmarking.md) | measured numbers, never estimated ones |
| [`eval/`](eval/) | the golden question set, its generated datasets and the graded runner |

---

## Roadmap

| | |
|---|---|
| **Foundation** | environment, PostgreSQL + migrations, local model evaluation — **done** |
| **Data layer** | CSV/Parquet ingestion, profiling, sandboxed DuckDB SQL execution — **done** |
| **Tools + evaluation** | deterministic tool registry, golden question set with hand-written reference SQL — **done** |
| **Application** | FastAPI, durable PostgreSQL job queue, web UI — **done** |
| **Agent** | tool-calling loop, execution trace, answer verification — **done** |
| **Verification** | numeric claim verification, charts, statistics, anomaly detection, performance benchmarks |

Deliberate ordering: the data layer and the evaluation set are built **before** the
agent, so the AI is added to a system already proven to work without it — and there is
a scoreboard to develop it against from the first commit.

### The scoreboard is calibrated before it is used

A benchmark reporting 62% means nothing unless you know it reads 100% for a perfect
answer and 0% for a worthless one. Three stub agents establish that, and they run today:

| stub agent | accuracy | what it proves |
|---|---|---|
| `oracle` — executes each question's reference SQL | **100%** | the ceiling is reachable |
| `refusing` — answers "I don't know" | **0%** | saying nothing earns nothing |
| `schema-only` — fluent answer, no computed numbers | **0%** | **sounding right earns nothing** |

```bash
python -m eval.build                    # generate data, compute ground truth
python -m eval.runner --agent oracle    # 50 questions, scored by category
python -m eval.build --check            # CI: fail if any expected answer moved
```

### The queue survives a killed worker

The job queue is one Postgres table. No Celery, no Redis — `FOR UPDATE SKIP LOCKED`
already solves the hard part, and the queue row *is* the analysis: same row, same
transaction, one thing to back up.

```sql
UPDATE analyses SET status = 'RUNNING', worker_id = :me, attempts = attempts + 1
WHERE id = (SELECT id FROM analyses WHERE status = 'PENDING'
            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING ...;
```

One statement, so there is no window between locating a job and claiming it. Two
workers polling in the same millisecond get two different jobs and neither blocks.

A running worker heartbeats every 5 s; anything quiet for 30 s is requeued. The clause
that makes that *safe* is on every terminal write:

```sql
... WHERE id = :id AND worker_id = :me AND status = 'RUNNING'
```

A worker that was merely slow, got reclaimed, and then finished finds its UPDATE matches
zero rows — so it drops a perfectly good result instead of overwriting the worker that
took over. `tests/test_worker_recovery.py` spawns a real worker, kills the process
outright, and asserts the job is reclaimed and completed.

---

## Reference hardware

All benchmark figures come from:

```
GPU   RTX 4050 Laptop, 6 GB VRAM
RAM   16 GB
CPU   Ryzen 7 7435HS, 8C/16T
OS    Windows 11
```
