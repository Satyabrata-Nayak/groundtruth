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

---

# M3 — tools and evaluation

## 14. Does a fixed tool set cripple the model?

This is the first question anyone asks about a tool-calling agent, and it deserves a
real answer rather than reassurance.

The worry is reasonable: if the model can only call six functions, surely it can only
do six things? The resolution is that the six are not the same *kind* of thing.

```
execute_sql          GENERAL     any read-only SELECT
                                 window functions, CTEs, self-joins, percentiles,
                                 cohort analysis, CASE, date arithmetic...

the other five       SPECIFIC    guard rails and shortcuts
```

So the action space is "everything expressible in SQL, plus five conveniences", not
"six canned reports". Almost every analytical question an analyst asks of a single
table is a SQL query, which is why `execute_sql` alone covers so much ground.

### Then why have the other five at all?

Because they refuse things *before* execution, in words a model can act on. Compare:

```
model calls  corr(region, revenue)          region is text

raw SQL   -> DuckDB: "Binder Error: No function matches corr(VARCHAR, DOUBLE)"
             a dead end. The model does not know what would have worked.

our tool  -> "column 'region' is categorical (VARCHAR), but this tool needs a
              numeric column. Suitable columns: order_id, revenue, cost, units"
             a repair instruction. The next call is likely to be right.
```

They also enforce things a model writing its own SQL reliably forgets. `compare_groups`
always returns the row count and share of total alongside each group, because *"North
has the highest average order value"* is misleading when North has four orders — and a
model asked for an average asks for an average.

### Where the ceiling actually is

Not at "six tools". At **what SQL cannot express**:

```
regression          fitting a line and reporting its confidence
clustering          k-means, segmentation
decomposition       separating trend from seasonality from noise
significance        is this A/B difference real, or noise?
```

Those are M6, added as more tools backed by SciPy and scikit-learn. The architecture
does not change; the list gets longer. What stays permanently excluded is *arbitrary
model-written Python*, because that collapses "the model decides what" and "our code
decides how" into one role, and every guarantee in this project rests on keeping them
apart.

### The cost of adding tools is not zero

Each tool is another entry in the prompt and another chance to pick wrong. Selection
accuracy on small models degrades as the list grows. That is why `detect_anomalies` is
*not* in M3 despite being easy to write — it waits until there is an agent to measure
it against.

---

## 15. Why the model's column name never reaches the SQL

`execute_sql` is safe because the whole statement goes through the four-layer sandbox.
But `compare_groups` **builds** SQL from arguments:

```python
f"SELECT {group_column} ... GROUP BY {group_column}"  # <- the classic disaster
```

The instinctive fix is to quote the identifier. That is the wrong fix, or rather an
insufficient one. The actual fix is that **the model's string is never used**:

```
model sends   group_column = "Reveune"
                     |
                     v   resolve_column()  reads the LIVE schema
                     |
              matched to the real column  "revenue"
                     |
                     v
SQL contains  "revenue"       <- our string, from DuckDB, not the model's
```

If the name matches nothing, nothing is built — the call returns an error listing the
columns that exist, plus a `difflib` suggestion. So the injection surface is not
"quoted incorrectly", it is *empty*: no path exists from the model's text to the query.

This is the same shape as `storage.parse_dataset_id`, which turns an untrusted id into
a UUID before it can become a path. Both are the pattern **validate by lookup, not by
escaping** — and the difference matters, because escaping is a thing you can get subtly
wrong, while substitution is a thing you either did or did not do.

### The case-insensitivity is free

Small models get casing wrong constantly. Since the canonical name is what proceeds,
accepting `"REGION"` for `region` costs no safety whatsoever and removes a whole
category of failure that teaches nothing.

---

## 16. Errors as values: why `call()` never raises

Normal Python advice is to let exceptions propagate. This module does the opposite on
purpose, and the reason is entirely about what happens in M5.

The agent loop looks like:

```
model: "call compare_groups with metric_column='catgory'"
  |
  v  registry.call(...)
  |
  +-- raises  -> the run dies. One typo ends the analysis.
  |
  +-- returns ToolResult(ok=False, error="column 'catgory' does not exist.
  |                                       Did you mean: category?")
  |
  v  feed that back into the conversation
  |
model: "call compare_groups with metric_column='category'"   <- recovered
```

The error message is not a log line. It is **the interface**, read by the model, and it
is why every message in the tool layer names the valid alternatives instead of only
stating what was wrong.

### Three kinds of failure, kept apart

```
unknown tool name   the model is not working from the action space it was given.
                    Not a slip — a sign the prompt or the tool list is wrong.

bad arguments       a repairable mistake. Retry is likely to succeed.

ToolError           a repairable semantic problem (wrong type, bad grouping).

anything else       a bug in OUR code. Reported as "internal error", because
                    inviting the model to retry it would loop forever.
```

Collapsing these into one generic error is exactly how an agent ends up retrying a call
that cannot ever work.

---

## 17. Two audiences for one result: `model_view`

A chart tool has a problem no other tool has. Its output has two consumers who need
opposite things:

```
the browser   needs EVERY data point, or it cannot draw the chart
the model     needs to know a chart was made, and roughly what it shows
```

Sending all the points to the model is not just wasteful, it is actively bad: a
400-point scatter plot is about 3,000 tokens of context spent on numbers the model
already computed in order to *request* the chart.

Measured, on `create_chart(scatter, revenue, cost)` over 400 rows:

```
payload to the caller   11,325 characters      unchanged
payload to the model         776 characters    14x smaller
```

The mechanism is a `Tool.model_view(data)` hook, identity for every tool except this
one. `ToolResult` carries `data` (the caller's) and `model_data` (the model's), and
`model_data` stays `None` when they are the same — so no other tool pays for the
distinction.

### The general principle

"What we computed" and "what the model should see" are different questions. This is the
first place they diverge; in M6 the verification layer will need the same split, since
it must check claims against the *full* results, not the summarised ones.

---

## 18. Why the benchmark builds its own data

This was the part I most expected to be pushed back on, so here is the argument in
full.

### The problem with a downloaded CSV

Suppose we benchmark against a Kaggle sales dataset and ask *"which category has the
highest profit margin?"*. What is the correct answer?

Whatever the reference SQL returns. There is no independent fact to check it against.
So the benchmark tests that DuckDB is deterministic — which it is, and which we knew.

### Generating inverts the direction

```
downloaded:   data -> run SQL -> that IS the answer          (circular)
generated:    decide the effect -> build data containing it -> the answer
              exists before any query is written             (independent)
```

`ecommerce` is built so that **Q3 revenue rises while Q3 profit falls**, with two
distinct causes: a low-margin category doubling its share, and the mean discount rising
from 5% to 14%. That is a fact about the data decided in a config dict, and a question
asking "why?" has a real answer that an analyst could reach and be graded against.

### The generator is checked in; the CSV is not

```
a committed CSV        3.5 MB, opaque.  "why is West's margin low?" — nobody knows
a committed generator  400 lines.        REGION_MARGIN_DELTA = {"West": -0.07}
```

The generator is the reviewable artefact. It also documents, in `planted_effects`,
exactly what a competent analyst should be able to find — which is the specification
the questions are written against.

### The honest limitation

**This measures regression, not generality.** It answers "did my change make it better
or worse", which is the only question that matters while building. It does *not* answer
"can this handle any dataset a stranger uploads" — that needs held-out real data, and
it is an M6 item kept deliberately separate rather than quietly conflated.

Three datasets rather than one, because a single shape tests a single skill:

```
ecommerce   14 clean columns    multi-step diagnosis
marketing   44 messy columns    reading a schema carefully
sensors     hourly readings     reasoning over time
```

---

## 19. Verifying the instrument before trusting its readings

The single most important thing in M3 is not the questions. It is that the scoreboard
was calibrated against known inputs *before* any real agent existed.

M1's lesson was blunt: five of six apparent "model failures" were bugs in my measuring
harness. So M3 ships three stub agents whose scores are known in advance:

```
oracle        executes the question's own reference SQL       expect ~100%
refusing      answers "I don't know", calls nothing           expect 0%
schema-only   inspects the schema, writes a fluent answer
              containing no computed number                   expect 0%
```

Measured:

| stub | accuracy | values correct |
|---|---|---|
| oracle | 100% (48/48) | 100% |
| refusing | 0% | 0% |
| schema-only | 0% | 0% |

The third one is the one that matters. It does real work and produces a confident,
plausible, entirely uncomputed answer — the most realistic failure mode of a weak
agent. **It scores zero.** That is the property that makes the number worth reporting.

### It found six bugs immediately

Running the oracle exposed five wording failures (`shipping_cost` required, but the
answer said `null_shipping`; `variant b` required, but the answer said `variant = B`)
and one real bug in the number extractor: its lookbehind rejected any digit preceded by
a letter, so **`Q3` could never match quarter 3** — and quarters are exactly how a real
answer refers to a quarter.

Every one of those would have shown up later as "the agent is bad at data-quality
questions".

### The gap I could not close

The `ambiguity` category grades "did the answer say which column it used". No stub that
renders a result table produces that sentence, so **the oracle scores 0/2 there and the
ceiling is unverified**. The honest response is to exclude those questions from headline
accuracy and say so — not to reword the oracle until it passes its own check, which
would make the calibration circular.

---

## 20. Grading free text against a number

Ground truth is `0.1531`. A correct answer might say:

```
"15.31%"        "about 15%"       "0.1531"      "15.3 percent"
"$1,234.56"     "2.5 million"     "15k"
```

A grader matching literal digits marks most correct answers wrong. And an
under-reporting benchmark is *worse than none*: it sends you optimising a model that
was already right.

So numbers are extracted from prose and normalised — commas, currency symbols, `k`/`M`/
`B` suffixes, percent signs — then compared within a relative tolerance the question
declares.

### The subtle part is percent

`15.31%` and `0.1531` are the same quantity, so rescaling by 100 has to be allowed. But
rescaling freely would mean an answer of `1` satisfies an expected value of `100`, which
is a silent free pass on every large number. The rule:

```
token carried a % sign        -> try both 15.31 and 0.1531
ground truth is a rate (<1)   -> try the answer / 100
otherwise                     -> face value only
```

That last line is the one that keeps the scoreboard honest, and it has its own test.

### Values and mentions are scored separately

A diagnosis question fails in two different ways:

```
wrong numbers                        -> the query was wrong
right numbers, no explanation        -> the analysis was not communicated
```

Collapsing both into one boolean loses the distinction exactly where it matters most,
so `Grade` carries `values_correct` and `mentions_present` apart, and the report shows
both. They also get fixed in different places — one needs a better query, the other
needs a better prompt.

`must_mention` accepts synonyms (`"differ|disagree|not the same"`) because requiring one
chosen verb measures vocabulary rather than understanding. And it is counted only over
questions that actually carry a requirement — counting the rest as passes made a stub
that mentions nothing score 58%, which is a statistic about the question set, not about
the agent.

---

## 21. Four bugs the data found that the code did not show

Every one of these was invisible in the generator and obvious the moment the data was
queried. This is why the generators have tests asserting their *effects*.

**1. Three event dates, all off by one day.** I wrote "day 45 is 2024-05-15" by hand.
April has 30 days, so day 45 is 16 May. All three sensor events were wrong. Every one
of those dates is quoted in a golden question's reference SQL, so the queries would have
run, returned the wrong window, and produced confidently incorrect "ground truth". The
dates are now *derived* (`START + timedelta(days=SPIKE_DAY)`), never typed.

**2. A column that was a perfect alias of another.** `AUDIENCES[index % 5]` and
`CHANNELS[index % 5]` — both lists have five entries, so campaign *i* always paired
audience *i* with channel *i*. `audience_segment` carried no independent information,
and "which audience performs best?" was "which channel performs best?" in disguise. It
surfaced because two unrelated questions returned the identical number, 6.0538.

**3. A trap that was not a trap.** `ecom-005` asked for the *lowest*-margin category to
catch a model reusing its previous answer — but the highest-revenue category (Electronics)
*is* the lowest-margin one, so repeating yourself scored correct. Asking for the
*highest* margin (Books) is what actually separates them.

**4. A planted effect that did not fire.** `sens-010` was built on "the all-period
average hides the fault". It did not: the drifting sensor edged out the genuinely hot
one even over 90 days. Fixed in the data (raising the hot sensor's baseline and moving
the dead-sensor fault onto a different unit so the two effects stay independent), not by
softening the question.

There is a fifth in the same family, found by the tool smoke test rather than the data:
`compare_groups` accepted grouping by `order_id` — 400 groups of one row each — because
the cardinality limit was an absolute 1,000 rather than *relative to row count*.
Uniqueness is relative: 400 distinct values is fine in a million rows and meaningless in
four hundred.

**The pattern in all five:** the code was self-consistent and the intent was wrong.
Nothing but querying the output would have shown it.

---

## 22. `connect_timeout`, and a skip-guard that could not skip

The test suite has a fixture that skips integration tests when Postgres is unreachable:

```python
try:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
except Exception:
    pytest.skip("Postgres not available")
```

Correct logic, useless in practice. With Docker Desktop stopped, nothing is listening on
port 5433, and a TCP connect to a dead port on Windows blocks for around 21 seconds
before the OS gives up — per attempt. The suite did not skip; it hung, and looked like
an infinite loop.

The fix is one line of engine configuration:

```python
connect_args = {"connect_timeout": settings.db_connect_timeout_s}  # 5 seconds
```

**The lesson:** a guard that depends on an operation *failing* has to specify how long
it is willing to wait for that failure. "It will error out" is not a plan if the error
takes 21 seconds and the default is to wait forever.

---

## 23. `SKIP LOCKED`, and the race you cannot see in a single-threaded test

Two workers poll the queue in the same millisecond. Both run:

```sql
SELECT id FROM analyses WHERE status = 'PENDING' ORDER BY created_at LIMIT 1;
UPDATE analyses SET status = 'RUNNING' WHERE id = :id;
```

Both read row #7. Both mark it RUNNING. The question is answered twice, and in M5 the
model is invoked twice for one answer on a laptop that can barely afford one.

What made this click was that the obvious fixes are each wrong in an instructive way:

```
a global lock            correct, and the queue is now single-file: workers stop
                         being workers and become a queue of one
SELECT ... FOR UPDATE    correct, but worker B BLOCKS on A's lock waiting for a row
                         it does not want. Under load everyone queues behind the
                         first row in the table.
optimistic retry         correct, and every worker does throwaway work in proportion
                         to how many workers there are
```

`FOR UPDATE SKIP LOCKED` is different in kind: it takes the lock, and where a row is
already locked by another transaction it *walks past it* instead of waiting. B does not
block on A. B gets row #8.

The second half is that the claim must be ONE statement:

```sql
UPDATE analyses SET status = 'RUNNING', worker_id = :me, attempts = attempts + 1
WHERE id = (
    SELECT id FROM analyses WHERE status = 'PENDING'
    ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
)
RETURNING id, dataset_id, dataset_version, question, attempts;
```

Select-then-update leaves a window between the two statements. One statement has no
window, and `RETURNING` hands back exactly what was written.

**The part I had to build to believe.** A test that claims twice in sequence passes
against a completely broken queue. The only test that observes the bug opens two real
sessions and claims in both *before either commits* — which is what
`test_two_workers_claiming_at_once_get_different_jobs` does. It is also why this cannot
be tested against SQLite: `SKIP LOCKED`, row-level locks and partial indexes are the
things under test, and a fake would only test the fake.

---

## 24. The heartbeat is the easy half; the ownership guard is the half that matters

A worker dies. Its job sits in RUNNING forever. Obvious fix: have workers stamp a
`heartbeat_at`, and sweep anything that has gone quiet back to PENDING.

That is half a solution, and shipping only that half creates a worse bug than the one it
fixes:

```
12:00:00  worker A claims #7
12:00:31  A pauses -- long GC, laptop suspended, disk stall -- and misses beats
12:00:35  the sweep decides A is dead, requeues #7
12:00:36  worker B claims #7 and starts over
12:00:40  A wakes up, finishes, and writes its result
```

A is not dead. It was slow. Now two workers write results for one analysis, and which
one survives depends on scheduling.

The fix is that a worker's identity is part of every terminal write:

```sql
UPDATE analyses SET status = 'SUCCEEDED', result = :result
WHERE id = :id AND worker_id = :me AND status = 'RUNNING';
```

A's UPDATE matches zero rows. `succeed()` returns False, and the worker logs that its
result was rejected and *drops it*. Discarding a correct result computed from good data
is the right behaviour, and it took me a while to be comfortable with that: another
worker owns the story of that row now, and a second story is worse than none.

Three details that follow from the same reasoning:

- **The timeout must be several beats wide.** At one beat, one slow write hands a live
  worker's job away. `app/config.py` refuses to start if
  `worker_heartbeat_timeout_s < 3 x worker_heartbeat_interval_s` — a config error caught
  at import beats a race caught in production.
- **Time comes from the database.** Every `now()` is evaluated by Postgres. If workers
  stamped their own clocks, a few seconds of skew between two machines would either
  resurrect live jobs or never reclaim dead ones.
- **Attempts are counted at claim, not at completion.** A job that reliably kills its
  worker has still consumed an attempt. Otherwise it is retried forever, taking a worker
  with it each time.

---

## 25. Why the beat needs its own thread and its own session

The tempting design is to heartbeat between analysis steps. It is simpler and it fails
in exactly the case that matters: one step can be the slow one. An M5 model call is
10-60 seconds of silence, and that is precisely when the worker most needs to be saying
"still alive". Beating between steps makes the gap between beats equal to the duration
of the longest step.

So the beat runs on a background thread at a fixed interval, independent of what the
work is doing.

Two things I had to get right:

**Its own session.** A SQLAlchemy `Session` is not thread-safe. Two threads sharing one
interleave statements on a single DBAPI connection and corrupt transaction state in ways
that surface as unrelated errors much later. The thread opens its own short transaction
per beat, from the same process-wide pool.

**It listens as well as speaks.** The same UPDATE that stamps the timestamp returns
`cancel_requested`, and fails to match if the row was reclaimed. One round trip answers
three questions:

```
did the UPDATE match?          -> do I still own this job
what did it return?            -> does anyone still want the answer
did it happen?                 -> yes, and the timestamp is refreshed
```

The thread sets flags; the analysis polls them at `checkpoint()` between steps. It does
not raise across the thread boundary, because an exception raised in the beat thread
could not interrupt the work anyway — pretending otherwise would hide the fact that the
work only stops at checkpoints.

---

## 26. Four outcomes, not two

I started writing `process()` with try/except/else — succeeded or failed — and it was
wrong. A worker has four things that can happen to it:

```
succeeded          write the result
AnalysisFailed     write the reason; the user asked something that cannot be answered
cancelled          write CANCELLED; someone asked it to stop
lost ownership     write NOTHING
```

The fourth is always forgotten, and its correct behaviour is the one that looks like a
bug: say nothing at all. Another worker owns this analysis now. It is also why
`StopRequested` carries a `reason` rather than being two exception classes — the worker
has to *distinguish* "stop and report" from "stop and be quiet", and a single flag at
the raise site is where that decision belongs.

There is a fifth case that is not the worker's: an unexpected exception. That is a bug
in us, and it gets `internal error: RuntimeError: ...` with the type name, so a
maintainer can tell it apart from a bad question at a glance.

---

## 27. Why long work must not run inside an HTTP request

Stated plainly because it is the thing M4 exists to demonstrate.

```
POST /analyses  --> run the analysis --> return the answer     (60 seconds)
POST /analyses  --> INSERT one row   --> return an id          (4 milliseconds)
```

The first version fails in five ways that have nothing to do with each other:

1. The browser holds a connection open for a minute, and so does a worker process.
2. Every proxy in the path applies its own idea of a timeout. A cloud load balancer's
   default is 60 seconds, and it will cut the connection mid-answer.
3. A page refresh abandons work that is still running and cannot be found again.
4. Restarting the API loses every request in flight.
5. Nothing can report progress, because nothing has committed yet — an uncommitted row
   is invisible to every other session.

Point 5 is the one I had not thought through, and it shaped the worker: **each event is
written in its own short transaction**. Doing the whole analysis in one transaction
would leave the trail invisible until the very end, so the UI would show nothing for a
minute and then everything at once. Short transactions are what make the trail live
rather than retrospective.

The same argument says the analysis must run *outside* the claiming transaction.
Otherwise the claim's row lock and its pooled connection are held for the entire
analysis. The heartbeat is what replaces the lock as the liveness signal.

---

## 28. Idempotency: making a retry safe to send

A client POSTs, the server commits, the connection drops before the response arrives.
From the client's side that is indistinguishable from a request that never arrived.
Retry and you may get two analyses; do not retry and you may get none.

An idempotency key fixes it, but only if the insert is atomic:

```python
# WRONG: two retries can both find nothing and both insert. One then dies on the
# unique index -- a successful retry turned into a 500.
if not session.scalar(select(Analysis).where(Analysis.idempotency_key == key)):
    session.add(Analysis(...))

# RIGHT: the database decides, once.
pg_insert(Analysis).values(...).on_conflict_do_nothing(
    index_elements=[Analysis.idempotency_key]
).returning(Analysis.id)
```

If nothing comes back, somebody else won; read their row and return it.

The part worth copying is the **status code**. 201 means "this is new"; 200 means "your
first attempt already landed, here it is". A client can tell those apart without
guessing, which is the whole point of the pattern.

Also: the column is nullable AND unique. Postgres permits many NULLs in a unique index,
so callers that do not care are not forced to invent a key.

---

## 29. Four bugs the output showed and the code did not — again

M3's lesson repeated itself exactly. The first end-to-end run succeeded, and its answer
was nonsense:

> The largest total **order_id** is in region = **None** at 4410927, **0.3528%** of the
> total

Three bugs in one sentence, and a fourth hiding behind the first fix.

**1. It summed the primary key.** I had assumed `inspect_schema` would flag an
identifier. It does not: `is_high_cardinality` is only computed for *categorical*
columns, because a numeric column with many distinct values is usually a measurement.
So `order_id` arrived unflagged and was the first numeric column in position order.

**2. The fix for #1 rejected every float.** "Distinct values ≈ row count means
identifier" is right for `order_id` and wrong for `revenue` — a continuous measurement
over 5,000 rows also has ~5,000 distinct values. Ratios measured on the real data:
`revenue` 0.97, `cost` 0.96, `unit_price` 0.90, all above my 0.9 threshold. The
analysis silently downgraded to whichever small-range integer survived, which is the
harder kind of wrong to notice because the output stops being absurd.

The discriminator is the *type*: near-uniqueness implies an identifier only for
integers. Nobody stores a primary key as a DOUBLE.

**3. `region = None`.** `compare_groups` keys the group label as `"group"`, not as the
column's name, so `top[group_column]` was always `None` — printed directly above a table
whose first row said `West`. An error would have been better; this looked like an
answer.

**4. `0.3528%`.** `share_of_total` is `value / total` — a fraction. Appending `%`
understated it a hundredfold, and 0.35% is a plausible-looking number, which is exactly
why it survived.

The pattern, for the third milestone running: **the code was self-consistent and my
assumption about someone else's payload was wrong.** Reading the tool's actual output
found all four in about a minute. Reading the code found none of them. Each now has a
named regression test in `tests/test_worker.py`, which opens with the list.

---

## 30. Small things that were not obvious

**`except X as e` deletes `e` at the end of the block.** Ruff (F821) caught this:

```python
except AnalysisFailed as failure:
    self._write(claimed, lambda s: queue.fail(s, ..., str(failure)))  # NameError risk
```

Python unbinds the exception name when the `except` block exits, so any closure
capturing it is a latent `NameError` — waiting for the one path that defers the call.
Bind to a plain string first.

**`Depends()` in a default argument is a live object built at import time.** FastAPI
accepts it; ruff flags it (B008); and it breaks the moment the function is called
outside FastAPI, as a test would. `Annotated[Session, Depends(get_session)]` puts the
dependency in the *type*, where it describes rather than defaults.

**A partial index keeps a queue's cost proportional to its backlog.**

```sql
CREATE INDEX ix_analyses_pending ON analyses (created_at) WHERE status = 'PENDING';
```

Only pending rows are in it. A million finished analyses do not slow the claim down by a
single page read.

**A CHECK constraint beats a native Postgres ENUM for a status column.**
`ALTER TYPE ... ADD VALUE` cannot run in the same transaction that then uses the new
value, so adding a state and backfilling rows with it needs two migrations. A CHECK
constraint is one `ALTER TABLE` and gives the same guarantee.

**`autoincrement` beats `max(seq) + 1`.** Event ordering could have been a per-analysis
counter. That is a read-then-write race: two writers read 4, both write 5. It happens to
be safe today because one worker owns a running analysis — but "safe because of an
invariant enforced somewhere else" is how races ship. A BIGINT identity column is
allocated by a sequence, needs no coordination, and doubles as the polling cursor.

**`setTimeout` chained after each response, not `setInterval`.** An interval fires on
schedule regardless of whether the previous request returned, so one slow response
stacks requests, they arrive out of order, and the event list flickers backwards.
Chaining means never more than one request in flight and the gap measured from the
*reply* — which is what "poll every second" always meant.

**React's `StrictMode` double-invokes effects on purpose.** It is not a nuisance to
switch off: it is what proves the polling effect cleans up after itself. An effect that
leaks a timer works fine without StrictMode and leaks one poll per mount with it.

**A worker left running in another terminal will eat your tests.** Two concurrency tests
passed alone and failed in the full suite. The cause was not a race in the code: I had
`python -m app.worker` still running from a manual demo, and it was polling the same
database the tests use, claiming their PENDING rows out from under them within a second.

The queue was behaving *exactly* as designed — a worker claims available work, and it
has no way to know some of that work belongs to a test. The lesson is about the test
environment, not the queue: a shared database plus a background consumer means test
isolation is no longer something the test file can guarantee on its own. Worth
remembering before debugging a "flaky" concurrency test for an hour. The real fix, if
this recurs, is a dedicated test database rather than a cleverer fixture.

---

## 31. The agent loop is small; everything around it is the work

The loop itself is about forty lines: ask the model, run what it asks for, feed the
result back, stop when it writes prose instead. Writing that took an afternoon. What
took the rest of the time was everything defending it, and each defence exists because
of something that actually happened rather than something imagined:

```
the schema is in the prompt        invented column names
one tool call per turn honoured    qwen3 emits the same call three times in parallel
tool errors fed back as text       a bad call is repaired instead of retried forever
repeat calls refused               the loop cannot spin on one idea
step budget AND time budget        a confused agent stops and says what it has
answer with no tool call           pushed back once, flagged if it persists
evidence table from the results    the table cannot agree with a wrong answer
figures traced to computations     the model does arithmetic in its head
```

The lesson worth keeping: **the interesting part of an agent is not the loop, it is the
list of ways the loop is wrong.** A tutorial agent is the forty lines. A working one is
the forty lines plus a reason for every guard.

---

## 32. `think: false` is slower AND worse, which is not what I expected

M1 measured that disabling qwen3's reasoning block produced worse machine-readable
output, because the model reasons anyway and, told not to, does it inside `content`
where it then has to be stripped with a regex. I re-measured on the real thing, expecting
at least a speed win to trade against:

```
                    tool-call turn   answer turn   total
reasoning on             50 s            80 s      130 s
reasoning off            67 s           153 s      220 s   + non-Latin characters in
                                                             the output
```

Slower on both turns. The reason is the same as M1's: the reasoning happens either way.
With `think: true` it goes into `message.thinking`, which is a separate field we never
show and never store. With `think: false` it goes into `content`, which is longer to
generate, has to be stripped, and — on this run — came back with characters that
crashed a `print()` on a cp1252 Windows console.

So "turn off thinking to go faster" is exactly backwards for this model. What DID make
it faster was telling it not to recite the result table in prose (135 s → 80 s on the
answer turn): the table is already shown to the reader, so listing ten rows in a
sentence is pure output tokens for negative value.

**The general shape:** on a small local model, latency is dominated by output tokens,
so the cheapest speed-ups are the ones that ask for less text — not the ones that ask
for less thinking.

---

## 33. Four bugs the output showed and the code did not — for the fourth milestone running

The pattern has now held for M3, M4 and M5. Reading the code found none of these.
Reading one real answer found all of them in about a minute.

**1. `LIMIT 1`, so there was no chart.** Asked "which country generated the most
revenue", the model wrote `ORDER BY total DESC LIMIT 1`. Correct SQL, correct answer —
and a one-row table, which cannot be charted, and which shows nobody whether the UK won
by a mile or a rounding error. A prompt rule ("when the question asks which is highest,
return the top 10") fixed it and made the answer better as well as prettier.

**2. It invented a currency.** "The United Kingdom generated the most revenue at
$8,187,806.36". The dataset is a UK retailer, the column is `UnitPrice`, and nothing
anywhere says dollars. A fabricated unit is a fabricated fact and it is not obviously
one, which is what makes it dangerous.

**3. Arithmetic in its head — the important one.** See note 34.

**4. A false positive in my own grounding check.** Asked for customer ages on a dataset
that has none, the agent inspected the schema, correctly answered "this data does not
record that", and was flagged with *"the answer was written without running a query
against the data"*. My evidence rule excluded `inspect_schema` by name, on the reasoning
that a schema read is not a computation. But **establishing that the data cannot answer
a question is done precisely by looking at the data.** The rule is now "any successful
tool call the MODEL chose" — the automatic pre-fetch is excluded because nobody chose
it, and everything else counts.

That fourth one is the one I would have defended in review. It was a rule that sounded
principled and had a category of correct behaviour on the wrong side of it.

---

## 34. The number that came from nowhere

This is the single most instructive output of the milestone.

> "WORLD WAR 2 GLIDERS ASSTD DESIGNS with 53,847 units, significantly exceeding the
> next highest product JUMBO BAG RED RETROSPOT by **16,484** units"

53,847 came from DuckDB. 47,363 came from DuckDB. 53,847 − 47,363 = **6,484**.

Every guard in the system was working. The schema was real, the SQL was real, the table
printed underneath the sentence was real and correct, and the two figures either side of
the wrong one were exact. The model subtracted two numbers, got it wrong, and wrote it
into the only clause of the sentence where nobody would think to check.

Two things follow, and the second is the one I would not have predicted.

**A prompt rule helps.** Adding "do not do arithmetic yourself — differences, ratios and
percentages must be computed by SQL" fixed this instance. The rerun quoted both totals
and let the reader compare them, which is also a better sentence.

**A prompt rule is not enough, and its success is the problem.** It worked, so the next
hundred answers will look fine, and the failure will return on the one where the model
is a little more confident. A rule that works most of the time is indistinguishable from
a rule that works, right up until it matters.

So the check is mechanical: pull every figure out of the answer, pull every number out
of every successful tool result, report what does not match. It is thirty lines
(`app/agent/verify.py`) and it catches the class rather than the instance.

**It warns; it does not correct.** This took the longest to settle. Rewriting the
sentence to remove the bad figure, or silently substituting the right one, would be a
different way of asserting more than the system knows — and it would hide the fact that
the model is capable of this. A figure that cannot be traced is a fact about confidence,
and it belongs in front of the reader, above the evidence rather than below it.

**Matching has to be loose, and the bias has to be towards silence.** An answer writes
`8,187,806.36` for a stored `8187806.363998184` and `35%` for a stored `0.3528`. So a
figure matches if it equals a computed number, is it rounded to any sensible number of
places, or is within 0.5% of it — and fractions are matched against their percentage
form too. Figures under 100 are skipped entirely, because "the top 10 countries" and
"3 groups" are numbers no tool produced and flagging them would bury the one that
matters. **A warning on a correct answer teaches people to ignore warnings.**

---

## 35. `list.extend(generator over the same list)` hangs forever

The verifier needed percentage forms of any fraction, so:

```python
numbers.extend(value * 100 for value in numbers if 0.0 <= value <= 1.0)
```

This reads correctly and hung the entire test suite until pytest was killed at five
minutes. `extend` consumes the generator lazily while appending to the list the
generator is iterating. Most values are harmless — `0.35 * 100` is 35, outside the
range, not re-emitted. But a single `0.0` anywhere in the data produces `0.0`, which IS
in range, which produces `0.0`, forever.

Real datasets are full of zeros. The fix is to materialise first:

```python
percentages = [value * 100 for value in numbers if 0.0 <= value <= 1.0]
return numbers + percentages
```

**The general rule:** never let a lazy iterator's source be the thing you are mutating.
The version with a list comprehension inside `extend(...)` would also have been safe,
which makes the failure depend on a single pair of brackets.

---

## 36. A test fixture deleted 542,000 rows of real data, twice

`tests/conftest.py`'s `db` fixture opens with `DELETE FROM datasets`. Entirely
reasonable — a test needs a known starting state. It was also pointed at the
development database, which is the same database the running app uses.

So `uv run pytest` destroyed a real upload. Twice, because the first time the symptom
appeared much later and somewhere else: the API answered `no dataset <uuid>` during an
unrelated end-to-end check, and the obvious hypothesis was a bug in the new code.

Two things worth keeping:

**The damage was invisible at the time it happened.** The tests passed. Nothing failed.
The consequence surfaced minutes later in a different process, which is the hardest kind
of causation to see.

**The fix belongs in code, not in a README.** One line, before `app.config` is imported:

```python
os.environ["POSTGRES_DB"] = os.environ.get("TEST_POSTGRES_DB", "adi_test")
```

An environment variable beats the `.env` file in pydantic-settings, and both the engine
and Alembic read the same `get_settings()`, so one line redirects everything
consistently. A session fixture creates the database if missing and migrates it. A setup
step that a person has to remember is a setup step that eventually gets skipped, and the
failure mode of skipping this one is deleting real data.

---

## 37. Why the model never computes anything, stated once

The division this whole milestone rests on:

```
the model decides WHAT to compute        which column, which grouping, which filter
DuckDB decides WHAT THE ANSWER IS        every number a user ever sees
```

Nothing else would work. A 4B model reasons well enough to pick `Quantity * UnitPrice`
grouped by `Country` — that is a language problem, and language is what it is for. It
cannot be trusted to sum 541,909 rows, and asking it to would be asking the wrong
question of the wrong tool.

Everything else in `app/agent/` follows from holding that line: tools are the only path
to computation, tool results are the only source of numbers, the evidence table comes
from the results rather than the prose, and the figures in the prose get checked back
against the results. The model is the part of the system with judgement and no
authority.
