# groundtruth

**Local-first AI data analysis where every number is traceable to a real computation.**

Upload a CSV or Parquet file, ask a
question in English, and get an evidence-backed answer — with every SQL query,
statistic and chart the system used shown alongside it, and every number in the answer
checked against a real computation.

**Runs entirely on your own machine. No API keys, no cloud services, no cost.**

> **Status: in active development.** The foundation, local model evaluation and
> benchmark harness are complete; see the roadmap below.

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
| Frontend | React + TypeScript + Vite | |
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

### 4. Verify

```bash
.venv/Scripts/python.exe scripts/bench_model.py qwen3:4b
```

This measures latency, structured-output validity, tool-calling reliability and SQL
correctness on your hardware. Results for the reference machine are in
[`docs/benchmarking.md`](docs/benchmarking.md).

---

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | components, trust boundaries, data flow |
| [`docs/decisions.md`](docs/decisions.md) | every non-obvious choice, with alternatives and why they lost |
| [`docs/learning-notes.md`](docs/learning-notes.md) | the concepts, explained — quantization, KV cache, tool calling, constrained decoding |
| [`docs/benchmarking.md`](docs/benchmarking.md) | measured numbers, never estimated ones |

---

## Roadmap

| | |
|---|---|
| **Foundation** | environment, PostgreSQL + migrations, local model evaluation — **done** |
| **Data layer** | CSV/Parquet ingestion, profiling, sandboxed DuckDB SQL execution |
| **Tools + evaluation** | deterministic tool registry, golden question set with hand-written reference SQL |
| **Application** | FastAPI, durable PostgreSQL job queue, web UI |
| **Agent** | tool-calling loop, execution trace, graded continuously against the evaluation set |
| **Verification** | numeric claim verification, charts, statistics, anomaly detection, performance benchmarks |

Deliberate ordering: the data layer and the evaluation set are built **before** the
agent, so the AI is added to a system already proven to work without it — and there is
a scoreboard to develop it against from the first commit.

## Reference hardware

All benchmark figures come from:

```
GPU   RTX 4050 Laptop, 6 GB VRAM
RAM   16 GB
CPU   Ryzen 7 7435HS, 8C/16T
OS    Windows 11
```
