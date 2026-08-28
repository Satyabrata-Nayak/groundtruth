"""Benchmark a local Ollama model on the four things this project actually needs.

Run:
    .venv/Scripts/python.exe scripts/bench_model.py qwen3:4b
    .venv/Scripts/python.exe scripts/bench_model.py qwen3:8b

WHY THIS SCRIPT EXISTS
----------------------
The roadmap said "target Qwen3 8B". That is a guess from VRAM arithmetic. This script
replaces the guess with measurements, because the model decision drives everything
downstream: if the model cannot reliably emit a valid tool call, no amount of good
architecture saves the project.

It deliberately has NO dependency on the app package. It talks to Ollama over HTTP and
nothing else, so it stays runnable even while the rest of the codebase is in flux.

WHY THE NATIVE /api/chat ENDPOINT, NOT THE OpenAI-COMPATIBLE ONE
----------------------------------------------------------------
M5 will use Ollama's OpenAI-compatible endpoint (so swapping providers is a base-URL
change). But for *measurement* the native endpoint is strictly better: it returns
`load_duration`, `prompt_eval_count/duration` and `eval_count/duration` in nanoseconds,
straight from the inference engine. That gives exact tokens/sec instead of
wall-clock estimates polluted by HTTP and Python overhead. Measure with the precise
instrument; ship with the portable one.

THE FOUR MEASUREMENTS
---------------------
A. Latency      — cold load time, time-to-first-token, generation tokens/sec.
B. Structured   — can it emit JSON conforming to a schema? Run twice: once free-form,
                  once with schema-constrained decoding, so we can measure how much
                  the constraint actually buys.
C. Tool calling — given tool definitions, does it pick the right tool with valid args?
D. SQL          — given a schema, does its SQL EXECUTE and return the right number?
                  Graded by running it in DuckDB against reference values computed
                  from hand-written SQL. No regex, no partial credit for looking right.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import httpx

OLLAMA = "http://127.0.0.1:11434"
TIMEOUT = httpx.Timeout(300.0, connect=10.0)
NS = 1_000_000_000  # Ollama reports durations in nanoseconds

# Qwen3 is a reasoning model: by default it emits a <think> block before answering.
# Ollama returns that separately as message.thinking, so it never pollutes the content
# or the JSON — but it does cost tokens, and therefore latency, on EVERY agent turn.
# Whether to pay that is a real M5 decision, so we make it a measurable flag.
THINK: bool | None = None  # None = model default (on for qwen3); False = disabled


def log(msg: str = "") -> None:
    """Print immediately.

    Python line-buffers stdout only when attached to a TTY; when piped to a file it
    uses a 8 KB block buffer, so a long-running script appears frozen. A benchmark
    that looks hung is a benchmark people kill before it finishes.
    """
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture dataset — small, deterministic, and shaped like the real thing.
# Reference answers are computed by EXECUTING hand-written SQL below, never typed
# in by hand. This is the same discipline the M3 eval set will use.
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE sales (
    order_id    INTEGER,
    order_date  DATE,
    region      VARCHAR,
    category    VARCHAR,
    revenue     DOUBLE,
    cost        DOUBLE
);
INSERT INTO sales VALUES
    (1,  '2024-01-15', 'North', 'Electronics', 1200.0,  800.0),
    (2,  '2024-01-20', 'South', 'Furniture',    450.0,  300.0),
    (3,  '2024-02-10', 'North', 'Electronics',  980.0,  700.0),
    (4,  '2024-02-14', 'East',  'Furniture',    620.0,  410.0),
    (5,  '2024-03-05', 'South', 'Electronics', 1500.0, 1100.0),
    (6,  '2024-03-22', 'North', 'Clothing',     230.0,  140.0),
    (7,  '2024-04-01', 'East',  'Electronics',  875.0,  600.0),
    (8,  '2024-04-18', 'South', 'Clothing',     310.0,  190.0),
    (9,  '2024-05-09', 'North', 'Furniture',    720.0,  500.0),
    (10, '2024-05-30', 'East',  'Clothing',     190.0,  120.0);
"""

SCHEMA_PROMPT = """Table: sales
Columns:
  order_id   INTEGER
  order_date DATE
  region     VARCHAR  (values: North, South, East)
  category   VARCHAR  (values: Electronics, Furniture, Clothing)
  revenue    DOUBLE
  cost       DOUBLE"""

# (label, question, reference SQL used to compute the true answer)
SQL_TASKS: list[tuple[str, str, str]] = [
    ("total_revenue", "What is the total revenue?", "SELECT SUM(revenue) FROM sales"),
    ("avg_revenue", "What is the average revenue per order?", "SELECT AVG(revenue) FROM sales"),
    ("order_count", "How many orders are there?", "SELECT COUNT(*) FROM sales"),
    (
        "filter_region",
        "What is the total revenue from the North region?",
        "SELECT SUM(revenue) FROM sales WHERE region = 'North'",
    ),
    (
        "filter_count",
        "How many orders were in the Electronics category?",
        "SELECT COUNT(*) FROM sales WHERE category = 'Electronics'",
    ),
    (
        "group_max",
        "Which category has the highest total revenue? Return only its total revenue.",
        "SELECT SUM(revenue) AS t FROM sales GROUP BY category ORDER BY t DESC LIMIT 1",
    ),
    (
        "derived_profit",
        "What is the total profit, where profit is revenue minus cost?",
        "SELECT SUM(revenue - cost) FROM sales",
    ),
    (
        "group_filter",
        "What is the total profit for the North region?",
        "SELECT SUM(revenue - cost) FROM sales WHERE region = 'North'",
    ),
    (
        "time_group",
        "What was the total revenue in March 2024?",
        "SELECT SUM(revenue) FROM sales "
        "WHERE order_date >= '2024-03-01' AND order_date < '2024-04-01'",
    ),
    (
        "ratio",
        "What is the overall profit margin as a fraction (total profit / total revenue)?",
        "SELECT SUM(revenue - cost) / SUM(revenue) FROM sales",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool-calling tasks
#
# These tool definitions are intentionally close to the real M3 registry, so the
# measurement predicts real behaviour. A model that scores well on toy one-arg tools
# tells us nothing about a tool with a nested schema.
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Run a read-only SQL SELECT query against the dataset and return rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A DuckDB SELECT query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Return column names, types and row count for the dataset. "
            "Use this first when the dataset structure is unknown.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_column",
            "description": "Return detailed statistics for one column: nulls, distinct "
            "count, min, max, mean, stddev.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Build a chart specification from dataset columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "bar", "scatter", "histogram"],
                    },
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["chart_type", "x_column", "y_column"],
            },
        },
    },
]

# (label, prompt, acceptable tool names, required arg keys)
TOOL_TASKS: list[tuple[str, str, set[str], set[str]]] = [
    ("schema_unknown", "What columns does this dataset have?", {"inspect_schema"}, set()),
    ("schema_rows", "How many rows are in the dataset?", {"inspect_schema", "execute_sql"}, set()),
    ("sql_sum", "What is the total revenue?", {"execute_sql"}, {"query"}),
    ("sql_filter", "How much revenue came from the North region?", {"execute_sql"}, {"query"}),
    ("sql_group", "Which category is most profitable?", {"execute_sql"}, {"query"}),
    (
        "profile_1",
        "Give me detailed statistics about the revenue column.",
        {"profile_column"},
        {"column"},
    ),
    (
        "profile_2",
        "Are there any missing values in the cost column?",
        {"profile_column", "execute_sql"},
        set(),
    ),
    ("chart_bar", "Draw a bar chart of revenue by region.", {"create_chart", "execute_sql"}, set()),
    (
        "chart_line",
        "Show me a line chart of revenue over time.",
        {"create_chart", "execute_sql"},
        set(),
    ),
    ("sql_count", "How many orders are in the Electronics category?", {"execute_sql"}, {"query"}),
    ("sql_avg", "What is the average order value?", {"execute_sql"}, {"query"}),
    ("sql_time", "How did revenue change month by month?", {"execute_sql"}, {"query"}),
    (
        "profile_3",
        "What is the distribution of the category column?",
        {"profile_column", "execute_sql"},
        set(),
    ),
    ("sql_margin", "What is the profit margin per category?", {"execute_sql"}, {"query"}),
    ("chart_hist", "Plot a histogram of revenue.", {"create_chart", "execute_sql"}, set()),
]


# ─────────────────────────────────────────────────────────────────────────────
# Structured-output tasks
#
# The schema mirrors the "analysis plan" object M5 will ask the model to produce.
# ─────────────────────────────────────────────────────────────────────────────

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["aggregate", "compare", "trend", "distribution", "unknown"],
        },
        "columns_needed": {"type": "array", "items": {"type": "string"}},
        "requires_sql": {"type": "boolean"},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "columns_needed", "requires_sql", "steps"],
}

JSON_TASKS: list[tuple[str, str]] = [
    ("j_total", "What is the total revenue?"),
    ("j_compare", "Compare profit between North and South regions."),
    ("j_trend", "How did revenue change over the year?"),
    ("j_dist", "What does the distribution of order values look like?"),
    ("j_multi", "Why did profit fall even though revenue rose?"),
    ("j_vague", "Tell me something interesting."),
    ("j_margin", "Which category has the best profit margin?"),
    ("j_count", "How many orders came from the East region?"),
    ("j_outlier", "Are there any unusually large orders?"),
    ("j_corr", "Is revenue correlated with cost?"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChatResult:
    """One model response plus the engine-reported timing counters."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    load_s: float = 0.0
    prompt_tokens: int = 0
    prompt_s: float = 0.0
    eval_tokens: int = 0
    eval_s: float = 0.0
    wall_s: float = 0.0
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float:
        return self.eval_tokens / self.eval_s if self.eval_s > 0 else 0.0


def chat(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    fmt: dict | None = None,
    temperature: float = 0.0,
) -> ChatResult:
    """One non-streaming /api/chat call.

    temperature=0 throughout: we are measuring capability, and sampling noise would
    make runs non-reproducible. Low temperature is also what the real agent will use —
    we want the most probable tool call, not a creative one.

    `fmt` is Ollama's `format` field. Passing a JSON Schema here switches on
    CONSTRAINED DECODING: the engine masks out every token that could not continue a
    valid document. Malformed JSON stops being unlikely and becomes impossible.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    if THINK is not None:
        payload["think"] = THINK

    t0 = time.perf_counter()
    try:
        r = client.post(f"{OLLAMA}/api/chat", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # noqa: BLE001 — a failed call is a data point, not a crash
        return ChatResult(content="", wall_s=time.perf_counter() - t0, error=repr(e)[:200])

    msg = d.get("message", {})
    return ChatResult(
        content=msg.get("content", "") or "",
        tool_calls=msg.get("tool_calls", []) or [],
        thinking=msg.get("thinking", "") or "",
        load_s=d.get("load_duration", 0) / NS,
        prompt_tokens=d.get("prompt_eval_count", 0),
        prompt_s=d.get("prompt_eval_duration", 0) / NS,
        eval_tokens=d.get("eval_count", 0),
        eval_s=d.get("eval_duration", 0) / NS,
        wall_s=time.perf_counter() - t0,
    )


def measure_ttft(client: httpx.Client, model: str, prompt: str) -> tuple[float, float, float]:
    """Streaming latency. Returns (ttft_any, ttft_content, total) in seconds.

    WHY TWO DIFFERENT "FIRST TOKEN" NUMBERS
    ---------------------------------------
    On a reasoning model these are wildly different numbers, and conflating them is a
    measurement bug I made on the first version of this script.

    Qwen3 streams its <think> block first, as `message.thinking`. During that entire
    phase `message.content` is empty. So:

      ttft_any     — first token of ANY kind. This is true time-to-first-token:
                     prompt processing + scheduling. Roughly constant per prompt size.
      ttft_content — first token of the actual ANSWER. On a reasoning model this is
                     ttft_any PLUS the whole thinking phase, which was observed to run
                     from 2 s to 60 s on prompts of similar length.

    Reporting only the second and calling it "TTFT" makes prompt processing look
    catastrophically slow when what you actually measured was deliberation time.
    Both are reported because they answer different questions: ttft_any tells you how
    fast the engine is, ttft_content tells you how long a human stares at a blank screen.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.0},
    }
    if THINK is not None:
        payload["think"] = THINK

    t0 = time.perf_counter()
    ttft_any = ttft_content = -1.0
    with client.stream("POST", f"{OLLAMA}/api/chat", json=payload, timeout=TIMEOUT) as r:
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if ttft_any < 0 and (msg.get("content") or msg.get("thinking")):
                ttft_any = time.perf_counter() - t0
            if ttft_content < 0 and msg.get("content"):
                ttft_content = time.perf_counter() - t0
            if chunk.get("done"):
                break
    return ttft_any, ttft_content, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Grading helpers
# ─────────────────────────────────────────────────────────────────────────────


def validate_plan(obj: Any) -> tuple[bool, str]:
    """Minimal hand-rolled check against PLAN_SCHEMA.

    Hand-rolled rather than jsonschema-the-library because we only need these four
    rules and this keeps the script dependency-light. M5 uses pydantic for real.
    """
    if not isinstance(obj, dict):
        return False, "not an object"
    for key in PLAN_SCHEMA["required"]:
        if key not in obj:
            return False, f"missing '{key}'"
    if obj["intent"] not in PLAN_SCHEMA["properties"]["intent"]["enum"]:
        return False, f"bad intent '{obj['intent']}'"
    if not isinstance(obj["columns_needed"], list):
        return False, "columns_needed not a list"
    if not isinstance(obj["requires_sql"], bool):
        return False, "requires_sql not a bool"
    if not isinstance(obj["steps"], list):
        return False, "steps not a list"
    return True, "ok"


def extract_json(text: str) -> Any:
    """Best-effort JSON recovery from free-form text.

    Used ONLY for the unconstrained arm of the structured-output test, to be fair to
    the model: if it wrapped valid JSON in a ```json fence or in prose, we still count
    it as a success. This is exactly the salvage logic that constrained decoding makes
    unnecessary — quantifying how much of it we can delete is part of the point.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


def strip_sql(text: str) -> tuple[str, bool]:
    """Extract a SQL statement from a model response. Returns (sql, needed_salvage).

    WHY THIS IS MORE THAN A .strip()
    --------------------------------
    The first version only unwrapped markdown fences, and section D scored 0/10 with a
    uniform ParserException. The cause was not bad SQL — it was that with `think: false`
    the model puts its deliberation in `content` as prose:

        "Okay, let's see. The user wants the total revenue... The query should be
         SELECT SUM(revenue) FROM sales;  Wait, let me check if there are any joins..."

    The SQL is in there and it is correct. Disabling reasoning does not stop the model
    reasoning; it stops the model *separating* its reasoning from its answer. Ollama
    returns a real <think> block as `message.thinking`, leaving `content` clean — so
    `think: false` is strictly WORSE for a machine consumer than leaving it on.

    Separating the two questions matters for the model decision:
        - can it write CORRECT SQL?            <- what grade_sql measures
        - does it emit CLEAN, parseable output? <- what needed_salvage measures

    Conflating them (as the first version did) reports a formatting problem as a
    correctness failure, and would have wrongly disqualified the model.
    """
    t = text.strip()

    # 1. Fenced block, if present — the common well-behaved case.
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            body = parts[1]
            for prefix in ("sql\n", "sql\r\n", "sql "):
                if body.lower().startswith(prefix):
                    body = body[len(prefix) :]
                    break
            cleaned = body.strip().rstrip(";").strip()
            if cleaned:
                return cleaned, True

    # 2. Already a bare statement — the ideal case, no salvage needed.
    if re.match(r"^\s*(SELECT|WITH)\b", t, re.IGNORECASE):
        stmt = t.split(";")[0]
        return stmt.strip(), False

    # 3. Prose containing a statement somewhere inside it.
    #
    # Naive "take the last SELECT|WITH" is WRONG, and a unit test caught it: for a CTE
    # like `WITH t AS (SELECT ...) SELECT MAX(r) FROM t`, the last keyword is the inner
    # SELECT, so it would return `SELECT MAX(r) FROM t` — a query referencing a CTE that
    # no longer exists. A correct answer scored as a failure.
    #
    # Rules that survive the cases actually seen:
    #   a. Statements are separated by ';'. Use the LAST segment that contains SQL —
    #      models deliberate first and commit last.
    #   b. Inside that segment, a WITH opens the statement, so take the FIRST WITH.
    #   c. Otherwise take the last SELECT that still has a FROM after it — this skips
    #      prose uses of the word ("I will select the revenue column...").
    #
    # Known limitation: splitting on ';' ignores semicolons inside string literals.
    # Acceptable here; M2's sqlglot parser replaces this heuristic entirely.
    segments = t.split(";")
    sql_segments = [s for s in segments if re.search(r"\b(SELECT|WITH)\b", s, re.IGNORECASE)]
    if not sql_segments:
        return "", True

    seg = sql_segments[-1]

    with_match = re.search(r"\bWITH\b", seg, re.IGNORECASE)
    if with_match:
        stmt = seg[with_match.start() :]
    else:
        candidates = [
            m
            for m in re.finditer(r"\bSELECT\b", seg, re.IGNORECASE)
            if re.search(r"\bFROM\b", seg[m.start() :], re.IGNORECASE)
        ]
        if not candidates:
            return "", True
        stmt = seg[candidates[-1].start() :]

    # Prose often resumes after the query on a new paragraph.
    stmt = stmt.split("\n\n")[0]
    return stmt.strip().rstrip(";").strip(), True


def grade_sql(
    con: duckdb.DuckDBPyConnection, generated: str, expected: float
) -> tuple[bool, bool, str]:
    """Execute the model's SQL and compare its scalar result to the reference value.

    This is the whole philosophy of the project in one function: we do not ask whether
    the SQL *looks* right, we run it and check the number. A query that reads
    beautifully and returns the wrong figure is a failure.

    Returns (correct, needed_salvage, reason). `needed_salvage` is reported separately
    because "wrote the right query" and "emitted it cleanly" are different failures with
    different fixes — the first needs a better model, the second needs structured output.
    """
    sql, salvaged = strip_sql(generated)
    if not sql:
        return False, salvaged, "no SQL found in response"
    try:
        rows = con.execute(sql).fetchall()
    except Exception as e:  # noqa: BLE001
        return False, salvaged, f"exec error: {type(e).__name__}"
    if not rows or not rows[0]:
        return False, salvaged, "no rows"
    val = rows[0][0]
    if val is None:
        return False, salvaged, "null result"
    try:
        got = float(val)
    except (TypeError, ValueError):
        return False, salvaged, f"non-numeric: {val!r}"
    tol = max(abs(expected) * 0.001, 1e-6)  # 0.1% relative tolerance for float math
    if abs(got - expected) <= tol:
        return True, salvaged, "ok"
    return False, salvaged, f"got {got:.4f} want {expected:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark sections
# ─────────────────────────────────────────────────────────────────────────────


def section_latency(client: httpx.Client, model: str) -> dict[str, Any]:
    log("\n[A] Latency")
    # Cold load: unload the model first so load_duration reflects a real cold start.
    client.post(
        f"{OLLAMA}/api/chat",
        json={"model": model, "messages": [], "keep_alive": 0},
        timeout=TIMEOUT,
    )
    time.sleep(2)

    cold = chat(client, model, [{"role": "user", "content": "Say OK."}])
    log(f"    cold load          : {cold.load_s:6.2f} s")

    anys, contents, rates, walls, think_ratio = [], [], [], [], []
    log(
        "      ttft_any = true first token | ttft_ans = first ANSWER token "
        "(after the <think> block)"
    )
    for i, prompt in enumerate(
        (
            "Explain what a database index is, in about 80 words.",
            "List five common data quality problems in CSV files.",
            "Describe the difference between a LEFT JOIN and an INNER JOIN in 80 words.",
        ),
        1,
    ):
        log(f"      {i}/3 streaming...")
        t_any, t_content, _ = measure_ttft(client, model, prompt)
        res = chat(client, model, [{"role": "user", "content": prompt}])

        # Thinking tokens are not reported separately by Ollama, so approximate the
        # share of generation spent deliberating from the character split. Rough, but
        # enough to see whether reasoning dominates the turn.
        total_chars = len(res.thinking) + len(res.content)
        ratio = len(res.thinking) / total_chars if total_chars else 0.0

        anys.append(t_any)
        contents.append(t_content)
        rates.append(res.tokens_per_sec)
        walls.append(res.wall_s)
        think_ratio.append(ratio)
        log(
            f"      ttft_any {t_any:5.2f}s | ttft_ans {t_content:6.2f}s | "
            f"{res.tokens_per_sec:5.1f} tok/s | {res.eval_tokens:4d} tok | "
            f"wall {res.wall_s:5.1f}s | thinking {100 * ratio:3.0f}% of output"
        )

    return {
        "cold_load_s": round(cold.load_s, 2),
        "ttft_any_median_s": round(statistics.median(anys), 2),
        "ttft_answer_median_s": round(statistics.median(contents), 2),
        "wall_median_s": round(statistics.median(walls), 2),
        "tokens_per_sec_median": round(statistics.median(rates), 1),
        "thinking_share_median": round(statistics.median(think_ratio), 3),
    }


def section_structured(client: httpx.Client, model: str) -> dict[str, Any]:
    """Run the same 10 tasks twice — free-form vs schema-constrained.

    The delta between the two arms is the single most important number in this script:
    it tells us how much of the small model's unreliability is fixable for free.
    """
    log("\n[B] Structured output  (free-form vs schema-constrained)")
    # BOTH arms are told the target schema, in the prompt. This matters:
    # an earlier version of this script described the schema ONLY via the `format`
    # parameter, so the free-form arm was being asked to guess our field names. It
    # scored 0/10 by returning perfectly valid JSON like {"step": 1, "action": "..."}.
    # That is not a structured-output failure, it is an unfair test, and it would have
    # made constrained decoding look miraculous for entirely the wrong reason.
    #
    # The honest comparison: both arms know exactly what is wanted. One is *constrained*
    # to produce it; the other is merely *asked*. The delta is the real value of
    # constrained decoding.
    system = (
        "You are an analysis planner. Given a question about a sales dataset, respond "
        "with a JSON object describing how to answer it.\n\n"
        "Respond with JSON only - no prose, no markdown fences. The object must match "
        "this schema exactly:\n" + json.dumps(PLAN_SCHEMA, indent=2)
    )
    out: dict[str, Any] = {}

    for arm, fmt in (("freeform", None), ("constrained", PLAN_SCHEMA)):
        ok = 0
        failures: list[str] = []
        for i, (label, question) in enumerate(JSON_TASKS, 1):
            log(f"      {arm:11s} {i:2d}/{len(JSON_TASKS)}  {label}")
            res = chat(
                client,
                model,
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Schema:\n{SCHEMA_PROMPT}\n\nQuestion: {question}",
                    },
                ],
                fmt=fmt,
            )
            if res.error:
                failures.append(f"{label}: {res.error[:40]}")
                continue
            parsed = json.loads(res.content) if fmt else extract_json(res.content)
            valid, why = (False, "unparseable") if parsed is None else validate_plan(parsed)
            ok += valid
            if not valid:
                failures.append(f"{label}: {why}")
        pct = 100 * ok / len(JSON_TASKS)
        log(f"    {arm:12s}: {ok}/{len(JSON_TASKS)} valid ({pct:.0f}%)")
        for f in failures[:3]:
            log(f"        - {f}")
        out[arm] = {
            "valid": ok,
            "total": len(JSON_TASKS),
            "pct": round(pct, 1),
            "failures": failures,
        }
    return out


def section_tools(client: httpx.Client, model: str) -> dict[str, Any]:
    log("\n[C] Tool calling")
    system = (
        "You are a data analysis agent. Use the provided tools to answer questions "
        "about a sales dataset. Always call a tool rather than answering from memory.\n\n"
        f"Dataset schema:\n{SCHEMA_PROMPT}"
    )
    emitted = correct = args_ok = 0
    failures: list[str] = []

    for i, (label, prompt, allowed, required_args) in enumerate(TOOL_TASKS, 1):
        log(f"      {i:2d}/{len(TOOL_TASKS)}  {label}")
        res = chat(
            client,
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tools=TOOLS,
        )
        if res.error or not res.tool_calls:
            failures.append(f"{label}: no tool call")
            continue
        emitted += 1
        fn = res.tool_calls[0].get("function", {})
        name = fn.get("name", "")
        if name not in allowed:
            failures.append(f"{label}: chose '{name}', wanted {allowed}")
            continue
        correct += 1
        raw = fn.get("arguments", {})
        args = json.loads(raw) if isinstance(raw, str) else raw
        missing = required_args - set(args or {})
        if missing:
            failures.append(f"{label}: missing args {missing}")
            continue
        args_ok += 1

    n = len(TOOL_TASKS)
    log(f"    emitted a tool call: {emitted}/{n} ({100 * emitted / n:.0f}%)")
    log(f"    chose right tool   : {correct}/{n} ({100 * correct / n:.0f}%)")
    log(f"    args valid too     : {args_ok}/{n} ({100 * args_ok / n:.0f}%)")
    for f in failures[:5]:
        log(f"        - {f}")
    return {
        "emitted": emitted,
        "correct_tool": correct,
        "valid_args": args_ok,
        "total": n,
        "pct": round(100 * args_ok / n, 1),
        "failures": failures,
    }


def section_sql(client: httpx.Client, model: str) -> dict[str, Any]:
    log("\n[D] SQL generation  (graded by execution)")
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA_DDL)

    system = (
        "You write DuckDB SQL. Given a schema and a question, respond with ONE SQL "
        "SELECT statement that returns a single numeric value. Output only the SQL, "
        "with no explanation and no markdown fences."
    )
    ok = 0
    salvaged = 0  # responses where the SQL had to be dug out of prose or a fence
    failures: list[str] = []
    for i, (label, question, ref_sql) in enumerate(SQL_TASKS, 1):
        log(f"      {i:2d}/{len(SQL_TASKS)}  {label}")
        expected = float(con.execute(ref_sql).fetchone()[0])
        res = chat(
            client,
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Schema:\n{SCHEMA_PROMPT}\n\nQuestion: {question}"},
            ],
        )
        if res.error:
            failures.append(f"{label}: {res.error[:40]}")
            continue
        passed, needed_salvage, why = grade_sql(con, res.content, expected)
        ok += passed
        salvaged += needed_salvage
        if not passed:
            failures.append(f"{label}: {why}")

    n = len(SQL_TASKS)
    log(f"    correct value      : {ok}/{n} ({100 * ok / n:.0f}%)")
    log(
        f"    clean output       : {n - salvaged}/{n} "
        f"(SQL had to be extracted from prose/fences in {salvaged})"
    )
    for f in failures[:5]:
        log(f"        - {f}")
    con.close()
    return {
        "correct": ok,
        "total": n,
        "pct": round(100 * ok / n, 1),
        "clean_output": n - salvaged,
        "salvaged": salvaged,
        "failures": failures,
    }


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    global THINK
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    model = args[0] if args else "qwen3:4b"

    # --no-think disables Qwen3's reasoning block. Worth measuring as its own arm:
    # reasoning may improve tool choice and SQL, but it is paid on EVERY agent turn,
    # and an agent doing 6 turns pays it 6 times.
    if "--no-think" in flags:
        THINK = False

    log("=" * 70)
    mode = "model default" if THINK is None else str(THINK)
    log(f"  BENCHMARK: {model}   (reasoning: {mode})")
    log("=" * 70)

    with httpx.Client() as client:
        try:
            tags = client.get(f"{OLLAMA}/api/tags", timeout=10).json()
        except Exception as e:  # noqa: BLE001
            log(f"ERROR: cannot reach Ollama at {OLLAMA}: {e}")
            return 1
        available = [m["name"] for m in tags.get("models", [])]
        if model not in available:
            log(f"ERROR: model '{model}' not installed. Have: {available}")
            return 1

        t0 = time.perf_counter()
        results = {
            "model": model,
            "reasoning": "default" if THINK is None else str(THINK),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "latency": section_latency(client, model),
            "structured": section_structured(client, model),
            "tools": section_tools(client, model),
            "sql": section_sql(client, model),
        }
        results["total_bench_s"] = round(time.perf_counter() - t0, 1)

    out_dir = Path(__file__).parent / "_bench_raw"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if THINK is None else "_nothink"
    out_file = out_dir / f"{model.replace(':', '_')}{suffix}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log("\n" + "=" * 70)
    log(f"  SUMMARY: {model}")
    log("=" * 70)
    lat, st, tl, sq = (results[k] for k in ("latency", "structured", "tools", "sql"))
    log(f"  tokens/sec (median)     : {lat['tokens_per_sec_median']}")
    log(f"  ttft, any token         : {lat['ttft_any_median_s']} s")
    log(
        f"  ttft, answer token      : {lat['ttft_answer_median_s']} s  "
        f"(thinking = {100 * lat['thinking_share_median']:.0f}% of output)"
    )
    log(f"  wall per call (median)  : {lat['wall_median_s']} s")
    log(f"  cold load               : {lat['cold_load_s']} s")
    log(f"  JSON valid, free-form   : {st['freeform']['pct']}%")
    log(f"  JSON valid, constrained : {st['constrained']['pct']}%   <-- the lever")
    log(f"  tool call + valid args  : {tl['pct']}%")
    log(f"  SQL correct on execution: {sq['pct']}%")
    log(f"\n  raw -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
