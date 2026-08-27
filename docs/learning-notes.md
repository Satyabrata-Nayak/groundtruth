# Learning notes

Concepts written down as they became necessary, not in advance. Each entry answers
"what is this, why did we need it here, and what would break without it."

---

# M1

## 1. Quantization, and what `Q4_K_M` means

A model's weights are numbers. Trained, they are 16-bit floats (FP16). An 8-billion
parameter model at FP16 is therefore `8e9 × 2 bytes ≈ 16 GB` — more than twice this
laptop's entire VRAM.

**Quantization** stores those weights at lower precision. `Q4_K_M` means roughly
4 bits per weight, so:

```
8B params × ~0.65 bytes/param  ≈  5.2 GB     (qwen3:8b  Q4_K_M)
4B params × ~0.65 bytes/param  ≈  2.5 GB     (qwen3:4b  Q4_K_M)
```

The `_K_M` part is the *scheme*: K-quants don't quantize every weight identically —
they use a block structure with per-block scaling factors, and the `M` (medium) variant
keeps certain sensitive tensors (attention layers, embeddings) at higher precision
because degrading those hurts quality disproportionately. This is why Q4_K_M loses far
less quality than the 4× compression suggests.

**Why it matters here:** quantization is what makes local inference possible at all on
a 6 GB card. The cost is a small, real accuracy loss — which is precisely why we
benchmark rather than assume.

---

## 2. KV cache — the thing that actually eats your VRAM

This is the concept most people miss, and it is why "the model is 5.2 GB and I have
6 GB, so it fits" is wrong.

When a transformer generates token #500, it needs to attend to all 499 previous tokens.
Recomputing their internal representations every step would be quadratic and hopeless.
So the engine **caches** the key and value vectors for every token processed so far.
That cache is the **KV cache**, it lives in VRAM alongside the weights, and it grows
linearly with context length.

```
VRAM  =  weights  +  KV cache  +  activations  +  whatever Windows is already using

qwen3:8b:   5.2 GB  +  KV cache  +  ...   vs ~5.4 GB free  →  ~200 MB left
qwen3:4b:   2.5 GB  +  KV cache  +  ...   vs ~5.4 GB free  →  ~2.9 GB left
```

With 200 MB of headroom, an 8B model can hold only a very short context before the
engine offloads layers to CPU RAM. Once that happens, every generated token crosses the
PCIe bus and throughput collapses — typically 4–8× slower.

**Why this bites *this* project specifically.** An agent turn is not one model call. It
is: system prompt + tool definitions + dataset schema + the question + every prior tool
call and its result, re-sent on every iteration. Context grows monotonically through
the loop. An agent doing 6 tool calls might reach 8–15k tokens by the final turn. That
is exactly the regime where KV cache size decides whether you stay on the GPU.

So context length is not free, and "how much context can I afford?" is really "how much
VRAM is left after weights?"

---

## 3. What "tool calling" actually is

Demystifying this early matters, because the name suggests something magical.

The model **cannot execute anything**. It has no filesystem, no network, no database.
What actually happens:

```
1.  We send: the conversation + a JSON description of available tools
        [{name: "execute_sql", parameters: {query: {type: "string"}}}, ...]

2.  The model emits TEXT in a special format its training taught it, which the
    serving layer parses into:
        {"name": "execute_sql", "arguments": {"query": "SELECT SUM(revenue)..."}}

3.  OUR CODE reads that, validates it, decides whether to run it, and runs it.

4.  We append the result to the conversation as a new message and call the model again.

5.  Repeat until the model emits a normal answer instead of a tool call.
```

Step 3 is the entire security boundary of this project. The model *proposes*; our code
*disposes*. That is why the roadmap forbids arbitrary Python execution — it would
collapse steps 2 and 3 into one and delete the boundary.

The one-sentence version, worth being able to say cold:

> Our backend sends a list of available tools to the model. The model chooses a tool
> and returns structured arguments. Our application validates and executes the tool;
> the model never touches the filesystem or the database directly.

---

## 4. Constrained decoding — the highest-leverage trick for small models

**This is the hardest and most important idea in M1. Read this section twice.**

### The problem

A model generates one token at a time. At each step it produces a probability
distribution over its whole vocabulary (~150k tokens) and one is sampled. Nothing in
that mechanism knows what JSON is. When we ask a 4B model for
`{"intent": "aggregate", ...}`, we are *hoping* the sampled path happens to be
well-formed. Small models fail this constantly — a trailing comma, a missing brace,
a helpful `Here's the JSON:` preamble, a hallucinated extra field.

### The trick

Before sampling, **mask the distribution**: set the probability of every token that
could not legally continue a valid document to zero, then sample from what remains.

Concretely, having emitted `{"intent": "` — the schema says `intent` is an enum of
`aggregate | compare | trend | distribution | unknown`. So the *only* tokens with
nonzero probability are those that begin one of those five strings. The model
physically cannot emit `"banana"`. Having emitted `{"intent": "aggregate"`, the only
legal next characters are `,` or `}`.

This is implemented as a grammar (Ollama compiles your JSON Schema into one) applied
as a logit mask at every decoding step.

### Why it matters so much

Malformed output stops being *unlikely* and becomes **impossible**. You are not
persuading the model with a better prompt — you are removing its ability to be wrong
about *structure*.

Crucially, note what it does and does not fix:

| | Constrained decoding |
|---|---|
| Invalid JSON syntax | **Eliminated** |
| Field missing from schema | **Eliminated** |
| Value outside an enum | **Eliminated** |
| Wrong *choice* of tool | Not helped |
| Semantically wrong SQL inside a valid string | Not helped |

It buys you **structural** reliability, which is exactly the failure mode small models
suffer from most. The remaining failures are *reasoning* failures — and those are what
the M3 eval set and the M6 verification layer exist to catch.

### Why we measured it

`section_structured()` in the benchmark runs the same 10 tasks twice — once free-form,
once constrained — precisely so the gap is a number in `docs/benchmarking.md` rather
than a claim. It is also why the benchmark contains `extract_json()`, a salvage routine
that digs JSON out of markdown fences and prose: quantifying how much of that hack we
get to delete is part of the point.

### What the measurement actually showed — and how it corrected me

Prediction going in (written into the M1 plan): constrained decoding would be *the*
lever making a 4B model viable, taking validity from something like 70% to near 100%.

Measured, `qwen3:4b --no-think`, 10 planner tasks:

```
free-form   (schema described in the prompt)   10/10   100%
constrained (schema enforced by the sampler)   10/10   100%
                                               ─────────────
delta                                                    0
```

**The prediction was wrong.** Told clearly what object to produce, the model produced it
correctly every time without any constraint at all. The earlier 0/10 free-form score
that seemed to support the prediction was a broken test — the free-form arm had not
been given the schema (see `benchmarking.md`, methodology errors).

Revised position, which is the defensible one:

> Constrained decoding is a **floor guarantee, not a capability upgrade.** It makes
> malformed output structurally impossible, which is worth having in production because
> one unparseable response fails an entire analysis run. But it is not what makes a
> small model usable, and *asking clearly* matters far more than *constraining tightly*.

Caveats kept honestly: this schema is easy (four fields, one flat enum, no nesting) and
ten tasks is a small sample. The nested/enum `create_chart` schema in section C is a
harder test. The result may not survive a more complex object — but on the evidence we
have, the mechanism does not earn the billing I gave it.

**The transferable lesson:** before reaching for machinery to force a model to behave,
check whether you actually asked it clearly. That mistake cost an hour here; in M5 it
would have meant building a fallback ladder to solve a problem that did not exist.

---

## 4b. Why the GPU sits at 28% of its clock speed — bandwidth, not compute

Observed mid-benchmark, and confusing at first:

```
utilization.gpu     94 %          ← looks maxed out
clocks.sm          885 MHz        ← of a 3105 MHz maximum
power.draw       34.55 W          ← of a 140 W limit
temperature         68 °C         ← not hot
throttle reasons    none active   ← not power-capped, not thermally limited
```

The GPU is "94% utilised" while running at 28% of its clock and a quarter of its power
budget, and nothing is throttling it. That combination looks impossible until you know
what single-stream token generation actually does.

**To generate one token, the engine reads every weight in the model out of VRAM.**
Every token. There is no way around it: each weight participates in the forward pass.
At batch size 1 there is almost no arithmetic per byte read — a multiply-accumulate and
move on — so the SM cores spend nearly all their time *waiting for memory*. The driver
sees compute units idling and drops the clock, because raising it would not produce a
single extra token.

This gives a hard ceiling you can compute in advance:

```
max tokens/sec  ≈  memory bandwidth ÷ bytes that must be read per token
                ≈  memory bandwidth ÷ model size resident in VRAM
```

For this machine — RTX 4050 Laptop, 96-bit bus, GDDR6 at 16 Gbps ≈ **192 GB/s**:

```
qwen3:4b   3.5 GB resident   →  192 / 3.5  ≈  55 tok/s ceiling
                                measured:    38–61 tok/s     ← at the hardware limit
```

So the model is not "slow"; it is running about as fast as this GPU physically can.
No prompt change, quantization tweak or driver update moves that much.

### The prediction this makes about 8B

The same arithmetic, applied before running the experiment:

```
qwen3:8b   ~5.2 GB weights   →  192 / 5.2  ≈  37 tok/s ceiling  — IF it fits
```

But it will not fit. Only ~5.4 GB of VRAM is free after the Windows desktop, so 5.2 GB
of weights leaves essentially nothing for the KV cache (§2), and layers spill to CPU
RAM. Once that happens the relevant bandwidth is no longer 192 GB/s VRAM but ~50 GB/s
system RAM across PCIe — and throughput should fall much further than the 1.5× the
size ratio alone suggests.

**Prediction: `qwen3:8b` lands well below 37 tok/s, not around it.** That is a falsifiable
claim, made before the measurement, and the benchmark will confirm or refute it. Making
the prediction first is the difference between measuring and merely observing.

### The practical consequence

Bigger model → proportionally slower, always, on this hardware. Doubling parameters
roughly halves throughput even in the best case where it still fits. That is the real
cost curve behind the model decision, and it is why "just use the biggest model that
fits" is bad advice: the largest model that *fits* is usually the one that fits with
nothing left over, which is exactly where performance falls off a cliff.

---

## 4c. Reasoning models: `think: false` does not do what it sounds like

**The most consequential thing learned in M1, and it inverted a decision.**

Qwen3 is a reasoning model: it deliberates before answering. Ollama exposes a `think`
parameter, and the obvious reading is "set it false to skip deliberation and go faster."

That reading is wrong. Measured on identical prompts at temperature 0:

```
think = default (on)    content: 241 chars   thinking: 1683 chars   eval: 428 tokens
think = false           content: 1758 chars  thinking:    0 chars   eval: 393 tokens
```

Total output is nearly the same. The model deliberates either way. What actually
changes is **which field the deliberation lands in**:

```
think ON                                think OFF
─────────                               ─────────
message.thinking  ← deliberation        message.content  ← deliberation
message.content   ← clean answer                         ← AND the answer,
                                                            mixed together
```

With reasoning on, Ollama parses the `<think>` block out for us and `content` arrives
clean. With it off, the deliberation floods into `content` and there is nothing
machine-readable left. An actual observed response to "What is the total revenue?":

```
"Okay, let's see. The user wants the total revenue from the sales table. The schema
 shows that the sales table has a column called revenue of type DOUBLE... The query
 should be SELECT SUM(revenue) FROM sales;  Wait, let me check if there are any
 conditions or joins needed..."
```

The SQL is correct and it is in there. Extracting it requires exactly the kind of
brittle prose-scraping this project exists to avoid.

**Consequence: for a machine consumer, `think: false` is strictly worse.** It does not
save the deliberation cost, and it destroys the structure that makes the answer usable.
The agent in M5 keeps reasoning ON and reads `message.content`, letting the server do
the separation.

This also explains an anomaly that looked like a bug: `eval_count` was identical across
both modes on the same prompts. Not a counter error — the model really was producing
about the same number of tokens either way.

**The generalisable point:** a flag named after an *outcome* ("stop thinking") may only
control a *representation* (where the thinking is stored). Check what it changes in the
response, not what its name promises.

---

## 5. Time-to-first-token vs tokens/sec

Two different numbers, and they fail differently.

- **TTFT** = prompt processing + scheduling, before any output appears. Dominated by
  how *long the prompt is*.
- **tokens/sec** = generation throughput after that. Dominated by memory bandwidth and
  whether you are fully on the GPU.

For a chat UI, TTFT dominates perceived speed. **For this project, neither alone tells
the story**: the agent makes 5–8 sequential calls, and the user waits for
`Σ(TTFT + generation)` across all of them. A model that is fast per-token but has a
long TTFT is punished badly here, because TTFT is paid on every loop iteration and the
prompt gets longer each time.

That is why the benchmark reports both, and why total agent latency gets broken down
by stage in M6.

---

## 6. Why the benchmark unloads the model before measuring cold load

`keep_alive: 0` evicts the model from VRAM first. Without it, the second run of the
benchmark would report a "cold load" of ~0 s because the model was still resident from
the previous run — a number that looks great and means nothing.

General principle worth internalising: **a benchmark that does not control its initial
state measures the state, not the system.**

---

## 7. Database migrations, and why an empty one

`alembic upgrade head` applies migrations in order and records which have run in a
table called `alembic_version`. That table is how the database knows what it is.

Our first migration creates *nothing*. That is deliberate: it establishes the chain
from a known-empty database. Every future migration descends from it, so there is never
a moment where the schema exists but is unversioned — the state where someone runs
autogenerate against an ad-hoc database and Alembic writes a migration that drops
tables it never knew about.

**The autogenerate footgun to remember:** Alembic diffs `Base.metadata` (what your
Python models say) against the live database (what actually exists). A model class that
is never *imported* is not in `Base.metadata`, so autogenerate concludes its table is
unwanted and emits a `drop_table`. This is why `app/db/models.py` exists as the single
import point and why `env.py` imports it.

---

## 8. Why the health check in docker-compose is not optional

Without `healthcheck`, `docker compose up -d` returns as soon as the *container* has
started — not when Postgres inside it is accepting connections. Postgres takes a few
seconds to initialise. The result is a flaky first connection, and worse, a class of
bug that only appears on cold-start machines and never on yours.

`pg_isready` is the correct probe here because it tests the thing we actually depend
on (the server accepting connections), not a proxy for it like "is the process alive."

---

# M2

## 9. OLTP vs OLAP — why this project runs two databases

The obvious objection to the architecture is "you already have DuckDB, why also run
Postgres?" They are built for opposite workloads, and using either for the other's job
is genuinely bad.

```
                    PostgreSQL  (OLTP)          DuckDB  (OLAP)
                    ──────────────────          ───────────────
storage             row-oriented                column-oriented
optimised for       many small reads/writes     few enormous scans
typical query       "give me order #4821"       "average revenue over 5M rows"
runs as             a server, always on         a library, inside your process
concurrent writers  hundreds, safely            effectively one
transactions        full ACID, row locks        minimal
```

**Row-oriented** means one record's fields are stored together, so fetching a single
order touches one place on disk. **Column-oriented** means all values of one column are
stored together, so summing `revenue` over five million rows reads one contiguous run
of numbers and never touches the other columns at all. Each layout is close to optimal
for its workload and poor for the other.

In this project:

```
Postgres   what datasets exist, versions, profiles, later the job queue and claims
Parquet    the actual rows — read only by DuckDB, never by Postgres
```

A 5,000-row dataset produces **8 rows** of Postgres metadata. The ratio only improves
with size.

### The requirement that settles it

M4 needs several worker processes each claiming the next analysis job, with no two ever
claiming the same one:

```sql
SELECT * FROM analyses WHERE status = 'PENDING'
FOR UPDATE SKIP LOCKED LIMIT 1;
```

Row-level locks, real transactions, many concurrent writers. **DuckDB has no
equivalent** — that is not a flaw, it is simply not a coordination database. And the
reverse substitution is just as bad: a `GROUP BY` over millions of rows in Postgres
would be far slower than DuckDB reading Parquet.

---

## 10. Two stores that cannot commit together

Writing a Parquet file and inserting a database row cannot be one atomic operation:
the filesystem has no transactions. So a crash between them leaves an inconsistent
state, and the design question is *which* inconsistency to accept.

```
file without a row      invisible to every listing, occupies disk forever,
                        findable only by walking directories
row without a file      appears in listings, looks healthy, fails only when
                        someone finally queries it
```

The second is worse — a broken record that advertises itself as working. So:

```
create:   write Parquet ──► commit metadata ──► on failure, delete the Parquet
delete:   delete rows   ──► then delete files
```

Both orders follow the same rule: **the database is the source of truth about what
exists, so never let it point at something that is not there.** An orphaned file is
recoverable by hand; a dangling reference corrupts every listing that touches it.

This is the same reasoning behind ordinary write-ahead logging, at a much smaller scale.

---

## 11. HyperLogLog, and estimates wearing the name of counts

DuckDB's `SUMMARIZE` reports `approx_unique`, which sounds like a distinct count and is
not one. It is **HyperLogLog**: a probabilistic sketch that estimates cardinality in
fixed memory — a few kilobytes regardless of whether it is counting a thousand values
or a billion. It works by hashing each value and tracking the longest run of leading
zeros seen; long runs are unlikely, so seeing one implies many distinct values.

Brilliant, and wrong for this use. Measured here:

```
30 rows, 30 genuinely distinct values
approx_unique      →  27      (10% error)
count(DISTINCT x)  →  30
```

At small N the error is proportionally large, and it was enough to flip a
high-cardinality threshold during development.

**The rule this project follows:** an estimate may be *displayed*, but never *stored in
a field named like a count* and never used in arithmetic. The same argument had already
forced exact null counts (SUMMARIZE's `null_percentage` is pre-rounded — 0.4% of 2.4M
rows is off by thousands), so accepting an estimate here would have been inconsistent.

Use HLL when you have billions of values and need an answer in constant memory. Do not
use it when the exact answer costs one extra scan at ingest time.

---

## 12. Prepared statements do not work everywhere

Parameterised queries (`?` placeholders) are the standard defence against SQL
injection, and the reflex is to use them for every value. But they are only supported
where the engine can plan around a value it does not yet know — and that is not
everywhere. Two failures hit this project:

```sql
COPY (SELECT * FROM read_csv(?)) TO ?          -- misbinds SILENTLY; DuckDB tried to
                                               -- READ the destination path
CREATE VIEW dataset AS SELECT * FROM read_parquet(?)
                                               -- "This type of statement can't be prepared"
```

The first is the dangerous one: no error, just wrong behaviour, and it took reading a
confusing "no files found" message to notice.

The reason is that a prepared parameter is a *value* in a query plan. A `COPY` target
and a `CREATE VIEW` body are part of the statement's **structure**, decided at plan time,
so there is nothing to substitute into.

**What to do instead.** Where parameters are unavailable, inline only values you
generated yourself — here, paths built from a validated UUID — and still escape them.
`storage.sql_path_literal()` doubles embedded quotes even though its inputs are
server-generated, because a function cannot verify that its caller kept that promise.

Never conclude "parameters do not work here, so user input can be interpolated." The
correct conclusion is "this position cannot accept user input at all."

---

## 13. Allowlists beat blocklists — and the mistake I made one level down

The sandbox rejects SQL by parsing it and requiring the root node to be a SELECT. This
is an allowlist, and it exists because keyword blocklists lose to comments, nesting and
string literals:

```sql
SELECT 1; /*x*/ DROP TABLE t
WITH a AS (SELECT 1) SELECT * FROM read_csv('~/.ssh/id_rsa')
```

Having got that right at the statement level, I then wrote a **blocklist of function
names** for table functions (`read_csv`, `read_parquet`, `glob`, ...) — and it silently
failed, because sqlglot parses `read_csv` and `read_parquet` into *dedicated AST
classes* (`ReadCSV`, `ReadParquet`) rather than the generic function node the check
looked for. The two most obvious attacks slipped past L1 entirely. L2 stopped them, but
only because the layering existed.

The fix was not to add two names. It was to notice the check had the wrong *shape*:

```
blocklist:  reject FROM sources whose function name is in a list I wrote
allowlist:  require every FROM source to be a plain identifier
```

The second rejects every table-valued function, including ones that do not exist yet.

**The transferable lesson:** knowing the principle is not the same as applying it
everywhere it applies. I had written the allowlist argument into a design document and
then implemented a blocklist two functions later, in the same file.
