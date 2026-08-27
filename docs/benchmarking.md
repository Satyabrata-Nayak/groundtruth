# Benchmarking

**Rule for this file: every number here was measured on this machine. Nothing is
estimated, rounded up, or carried over from a blog post. If a number is missing, the
measurement has not been run yet.**

---

## Reference hardware

```
GPU     NVIDIA RTX 4050 Laptop — 6141 MiB VRAM, driver 571.96
RAM     15.8 GB
CPU     AMD Ryzen 7 7435HS, 8 cores / 16 threads
OS      Windows 11
Ollama  0.18.2   (native, not containerized — direct GPU access)
```

---

## M1 — model selection

### What is measured, and why these four things

| Section | Measures | Why it decides the model |
|---|---|---|
| A. Latency | cold load, time-to-first-token, tokens/sec | The agent loop is 5–8 sequential calls. The user waits for the sum. TTFT is paid on *every* iteration and the prompt grows each time. |
| B. Structured output | valid-JSON rate, free-form vs schema-constrained | Small models fail on *structure*. The gap between the two arms says how much of that is fixable for free. |
| C. Tool calling | emitted a call / chose the right tool / args valid | If the model cannot reliably emit a valid tool call, no amount of architecture saves the project. |
| D. SQL | correct value after **executing** the query | The one that actually matters. A query that reads beautifully and returns the wrong number is a failure. |

### Method

- `temperature = 0` everywhere. We are measuring capability; sampling noise would make
  runs non-reproducible, and the real agent wants the most probable tool call anyway.
- Timings come from Ollama's native `/api/chat`, which reports `load_duration`,
  `prompt_eval_duration` and `eval_duration` in nanoseconds straight from the inference
  engine — not wall-clock estimates polluted by HTTP and Python overhead.
- The model is explicitly unloaded (`keep_alive: 0`) before the cold-load measurement.
  A benchmark that does not control its initial state measures the state, not the system.
- SQL is graded by **execution**: the model's query runs against a DuckDB fixture table
  and its scalar result is compared to a reference value computed from hand-written SQL,
  within 0.1% relative tolerance. No regex, no partial credit.
- Tool definitions in the benchmark deliberately mirror the shape of the real M3
  registry (including a nested/enum schema in `create_chart`). A model that scores well
  on toy one-argument tools tells us nothing.
- **Nothing else heavy runs during a benchmark.** Learned the hard way: an `ollama pull`
  running concurrently with section A dropped measured throughput from ~60 tok/s to
  36.3 tok/s. That run was discarded and re-run. Correctness sections (B/C/D) are
  immune to this, but latency numbers are worthless if the machine is busy.

### Methodology errors found and corrected

Kept deliberately, because the corrections are more instructive than the final numbers.

**1. TTFT measured the wrong event.** The first version waited for the first streamed
chunk containing `message.content`. But Qwen3 streams its reasoning block first, as
`message.thinking`, leaving `content` empty for the whole deliberation. The script
therefore reported "TTFT" values of 2.6 s, 14.4 s and 61.0 s on prompts of similar
length — which is not time-to-first-token at all, it is time-until-it-stopped-thinking.
Prompt processing actually takes ~0.2 s. The script now reports `ttft_any` (first token
of any kind — genuine engine latency) and `ttft_answer` (first token of the answer —
what a human actually waits through) as separate numbers.

**2. The structured-output comparison was rigged in favour of constrained decoding.**
The free-form arm was never told the target schema; only the constrained arm received it,
via the `format` parameter. The model duly returned well-formed JSON with invented field
names — `{"step": 1, "action": "SELECT SUM(revenue) FROM sales"}` — and scored 0/10.
Published as-is, that would have shown constrained decoding taking validity from 0% to
~90%: a spectacular result, and entirely an artifact of an unfair test. Both arms are now
given the schema in the system prompt. One is *constrained* to it, the other is merely
*asked* — and the remaining gap is the real effect.

**3. The SQL grader could not find the SQL.** Section D scored 0/10 with a uniform
`ParserException`. The model's queries were correct; with `think: false` it puts its
deliberation in `content` as prose and the grader passed whole paragraphs to DuckDB.
Fixed by extracting the statement and — more importantly — by reporting *correctness*
and *output cleanliness* as two separate numbers, since they are different failures
with different fixes (better model vs structured output).

**4. The first fix decapitated CTEs.** Taking the last `SELECT|WITH` keyword returns the
inner SELECT of `WITH t AS (SELECT ...) SELECT MAX(r) FROM t`, producing a query that
references a CTE that no longer exists — scoring a correct answer as a failure. Caught
by a unit test, not by a benchmark run. Now covered by `tests/test_bench_extraction.py`.

### Open caveat: what `eval_count` counts

Unresolved, and deliberately not chased further because it does not affect the decision.

Both reasoning modes report identical `eval_count` on the same prompts (254 / 1868 /
892), which is inconsistent with thinking tokens being included — yet the wall-clock
arithmetic for prompt 3 (892 tokens ÷ 60.2 tok/s ≈ 14.8 s ≈ the observed 15.1 s wall,
with thinking at 88% of output characters) only reconciles if they *are* included.
Additionally `ttft_answer` (61.8 s, measured on a streaming call) exceeded the total
wall time of the equivalent non-streaming call (15.1 s), so the two calls are not
behaving identically at temperature 0.

**Consequence, and why it is parked:** tokens/sec is not trustworthy *across* reasoning
modes. Wall-clock per call is, and it answers the same question. All cross-mode claims
in this document rest on wall-clock and on directly observed output structure, never on
tokens/sec. Within a single mode, tokens/sec remains useful for comparing models.

The tell for the first two bugs was **uniformity**: all ten structured-output tasks failed with
the identical message. Genuine model failures are messy and varied; identical failures
across a whole suite point at the harness, not the model. Worth remembering when the
M3 eval set starts producing suspiciously clean numbers.

Reproduce with:

```bash
.venv/Scripts/python.exe -u scripts/bench_model.py qwen3:4b
.venv/Scripts/python.exe -u scripts/bench_model.py qwen3:4b --no-think
.venv/Scripts/python.exe -u scripts/bench_model.py qwen3:8b
```

Raw JSON lands in `scripts/_bench_raw/` (gitignored — summaries here are the record).

---

## Observations before the numbers

These were confirmed live, not predicted:

**`qwen3:4b` runs 100% on GPU.** `ollama ps` reports 3.5 GB resident, `PROCESSOR:
100% GPU`, with `nvidia-smi` showing 5090 / 6141 MiB used at 93% utilisation during
generation. This is the KV-cache argument from `learning-notes.md` §2 confirmed in
practice: 3.5 GB of model plus context leaves the card nearly full at only a 4096-token
default context. An 8B Q4_K_M at ~5.2 GB of weights cannot fit alongside any meaningful
KV cache on this GPU.

**Qwen3 is a reasoning model.** It emits a `<think>` block before answering. Ollama
returns this separately as `message.thinking`, so it never contaminates `content` or a
constrained-JSON document — but it costs tokens, and therefore latency, on *every*
agent turn. Spot measurement on a short prompt:

```
think = model default   wall 7.9 s   428 eval tokens   1683 thinking chars   241 content chars
think = disabled        wall 6.9 s   393 eval tokens      0 thinking chars  1758 content chars
```

Two things worth noting. The latency difference on a short prompt is small (~1 s), and
disabling reasoning made the model *more verbose in its answer*, not less — it moved
the deliberation into the visible response. Whether reasoning helps enough on tool
choice and SQL to justify its cost is a real M5 decision, which is why `--no-think` is
a benchmark arm rather than a hardcoded setting.

**Throughput is roughly 54 tok/s** on this spot check (428 tokens / 7.9 s wall,
including HTTP overhead). Section A measures this properly.

---

## Results

Generated from `scripts/_bench_raw/*.json` by `scripts/bench_report.py` — never typed
by hand, so the table cannot drift from the runs that produced it.

| Metric                    | qwen3:4b | qwen3:4b --no-think | qwen3:8b |
|---------------------------|----------|---------------------|----------|
| tokens/sec (median)       | 53.8     | 54.9                | 9.4      |
| ttft - any token (s)      | 0.12     | 0.13                | 0.68     |
| ttft - answer token (s)   | 16.52    | 0.13                | 20.36    |
| wall per call (s, median) | 15.1     | 15.1                | 30.9     |
| thinking share of output  | 57%      | 0%                  | 61%      |
| cold load (s)             | 2.25     | 2.35                | 11.07    |
| JSON valid - free-form    | 100%     | 100%                | 100%     |
| JSON valid - constrained  | 100%     | 100%                | 100%     |
| tool call emitted         | 15/15    | 15/15               | 15/15    |
| correct tool chosen       | 15/15    | 15/15               | 15/15    |
| tool + valid args         | 15/15    | 15/15               | 15/15    |
| SQL correct on execution  | 10/10    | 7/10                | 10/10    |
| SQL emitted cleanly       | 10/10    | 0/10                | 10/10    |
| benchmark wall time (s)   | 1243     | 891                 | 3097     |

Only `--no-think` failures occurred; all three were `ParserException` — the extractor
could not recover a statement from prose:
`order_count`, `filter_count`, `group_max`.

### How to read this table

**1. Every capability metric is a tie.** JSON validity, tool selection, tool arguments
and SQL correctness are 100% for both a 4B and an 8B model. There is no quality gap for
the larger model to close.

**2. The only real difference is throughput, and it is large.** 53.8 vs 9.4 tokens/sec —
5.7×. Per agent call: 15.1 s vs 30.9 s. Over a 6-call analysis: ~1.5 min vs ~3–4 min.

**3. The cause is placement, not size.** `ollama ps` reported `38%/62% CPU/GPU` for 8B
against `100% GPU` for 4B. 5.2 GB of weights into ~5.4 GB of free VRAM leaves nothing
for the KV cache, so a third of the model is read from system RAM over PCIe. Predicted
in `learning-notes.md` §4b before the run, including the mechanism.

**4. `ttft - answer token` is not a defect.** 16.5 s for 4B looks alarming next to 0.13 s
for `--no-think`, but that is the reasoning block being generated. The `ttft - any token`
row (0.12 s) is the real engine latency. The reasoning is *why* the answer arrives clean.

**5. Constrained decoding contributed nothing measurable** — 100% both arms, all three
configurations. See `learning-notes.md` §4, where the original prediction is kept next to
the result that refuted it.

**6. `--no-think` is worse, not faster.** Same wall-clock (15.1 s), same tokens/sec, but
`SQL emitted cleanly 0/10` versus `10/10`. Disabling reasoning does not remove it; it
relocates it into `content` where it destroys parseability (§4c).

### Cost of the benchmark itself

The 8B run took **3097 s (~52 min)** versus 1243 s (~21 min) for 4B — the throughput
penalty compounding across 45 sequential calls. Worth noting for M3: an eval suite that
takes an hour per run will not be run often enough to be useful as a development
instrument, and keeping it fast is a design constraint, not a nicety.
