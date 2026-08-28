"""The analysis toolkit, exposed over the Model Context Protocol.

    Claude Desktop / any MCP client  ──stdio──►  this server  ──►  ToolRegistry
                                                                        │
                                                                        ▼
                                                              DuckDB, sandboxed

WHY EXPOSE THE TOOLS AND NOT THE AGENT
--------------------------------------
The obvious move is to publish "ask a question" as one tool and let the client's model
drive our agent. That is the wrong seam. It buries a 90-second local model call behind a
single opaque call, and it duplicates reasoning: the client already has a model, and it
is almost certainly a better one than qwen3:4b.

What this project owns that no client has is the **deterministic half** — an exact
profiler, a sandboxed SQL executor whose column names are resolved against a live
schema, a grouping tool that refuses meaningless groupings, and a chart builder. Those
are worth publishing. So MCP gets the verbs, and whoever connects brings their own
reasoning.

The practical consequence is that Claude Desktop can analyse a 542,000-row Parquet file
through the same guard rails the local agent uses, with the same "the model never
computes anything" property, and none of the local model's latency.

TOOLS AND RESOURCES ARE NOT THE SAME THING
------------------------------------------
MCP draws a line that maps exactly onto a distinction this codebase already made:

    tools      model-controlled    verbs, chosen mid-conversation
    resources  application-controlled  nouns, addressable documents

So `execute_sql` and `compare_groups` are tools — the model decides when to run one.
A dataset's schema and its stored profile are *resources*: they are documents with
addresses (`dataset://{id}/schema`), the client can attach them up front, and doing so
costs no tool call and no round trip through the model's decision-making. The same
argument as `app/agent/prompt.py`, which hands the schema over rather than making the
agent fetch it.

THE SECURITY BOUNDARY IS UNCHANGED, AND THAT IS THE POINT
----------------------------------------------------------
`ToolContext` carries `dataset_id` and `version`; a tool's JSON schema has no field for
either. Over MCP the caller supplies the dataset id, so it moves from "never in the
model's reach" to "an argument like any other" — which is why it is validated as a UUID
against stored datasets before a `ToolContext` is built. Everything after that point is
the identical code path the local agent uses: same registry, same sandbox, same
allowlisted SQL. One implementation, two front doors.

RUN IT
------
    uv run python -m app.mcp_server

In Claude Desktop's config:

    {"mcpServers": {"ground-truth": {
        "command": "uv",
        "args": ["run", "--directory", "<repo>", "python", "-m", "app.mcp_server"],
        "env": {"MCP_LOG_FILE": "<repo>/mcp-calls.log"}}}}

HOW TO PROVE IT IS ACTUALLY BEING USED
--------------------------------------
This is a real question and it has a real answer, because a client that answers from
memory looks identical to one that called a tool.

    1. `uv run python -m app.mcp_server --selftest` runs a call through the whole
       server in-process and prints what came back. It proves the server works.
    2. Set MCP_LOG_FILE and `tail -f` it. Every call appends one line with the tool, the
       dataset, the arguments and the duration. If the file does not grow while the
       client is answering, the client did not use the tools.
    3. The numbers themselves. Ask for something no model could know — a sum over
       541,909 rows — and check it against `SELECT sum(...)` yourself.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from mcp.server.mcpserver import MCPServer

from app.data import service
from app.db.session import session_scope
from app.tools import get_registry
from app.tools.base import ToolContext

log = logging.getLogger(__name__)

# Published: the deterministic verbs. NOT published: the agent loop or the job queue —
# a client that brings its own model does not want ours, and burying a 90-second local
# model call behind one opaque tool would be the wrong seam entirely.
#
# `create_chart` is also left out, deliberately: its value is a spec a renderer draws,
# and an MCP client has no renderer for it. A chart returned as JSON to a chat window is
# a table with extra steps.

mcp = MCPServer(
    "ground-truth",
    instructions=(
        "Analyse a local tabular dataset. Every number you report must come from a tool "
        "result here — the tools compute, you decide what to compute. Call "
        "list_datasets first to get a dataset_id, then read its schema before writing "
        "SQL. The dataset is exposed to SQL as a single table named 'dataset'."
    ),
)


# --------------------------------------------------------------------------------
# Resources: nouns the client can attach without spending a tool call
# --------------------------------------------------------------------------------


@mcp.resource("dataset://{dataset_id}/schema")
def dataset_schema(dataset_id: str) -> str:
    """The columns, types and quality flags of a dataset's latest version."""
    context = _context(dataset_id)
    result = get_registry().call("inspect_schema", context, {"include_statistics": True})
    if not result.ok:
        raise ValueError(result.error)
    return json.dumps(result.data, indent=2, default=str)


@mcp.resource("dataset://{dataset_id}/sample")
def dataset_sample(dataset_id: str) -> str:
    """Three real rows.

    Worth its own resource for the reason `app/agent/prompt.py` puts samples in the
    prompt: a type list cannot tell you whether Country reads 'UK' or 'United Kingdom',
    and a query written against the wrong one returns nothing and reads as a real zero.
    """
    context = _context(dataset_id)
    result = get_registry().call(
        "execute_sql", context, {"sql": 'SELECT * FROM "dataset"', "max_rows": 3}
    )
    if not result.ok:
        raise ValueError(result.error)
    return json.dumps(result.data, indent=2, default=str)


# --------------------------------------------------------------------------------
# Tools: verbs the model chooses
# --------------------------------------------------------------------------------


@mcp.tool()
def list_datasets() -> str:
    """List the datasets available to analyse, with their ids, sizes and versions."""
    with session_scope() as session:
        datasets = service.list_datasets(session)
        return json.dumps(
            [
                {
                    "dataset_id": str(dataset.id),
                    "name": dataset.name,
                    "latest_version": max((v.version for v in dataset.versions), default=None),
                    "rows": max((v.row_count for v in dataset.versions), default=0),
                    "columns": max((v.column_count for v in dataset.versions), default=0),
                }
                for dataset in datasets
            ],
            indent=2,
        )


def _run(name: str, dataset_id: str, arguments: dict[str, Any]) -> str:
    """Execute one registry tool for an MCP caller.

    Every published tool funnels through here and therefore through `registry.call` —
    the same entry point the local agent uses. A second execution path would mean a
    second set of guarantees about row caps, column resolution and error wording, and
    the two would drift.

    EVERY CALL IS LOGGED, and that is a feature rather than debris. An MCP server runs
    as a subprocess of its client with stdout owned by the protocol, so there is no
    console to watch: without a log there is no way to answer "is it actually being
    used?" other than trusting the client's UI. stderr is free — the protocol does not
    use it — so it becomes the audit trail. See MCP_LOG_FILE.
    """
    started = time.perf_counter()
    try:
        context = _context(dataset_id)
    except ValueError as exc:
        _audit(name, dataset_id, arguments, ok=False, detail=str(exc), ms=0.0)
        return json.dumps({"ok": False, "error": str(exc)})

    result = get_registry().call(name, context, arguments)
    _audit(
        name,
        dataset_id,
        arguments,
        ok=result.ok,
        detail=result.summary if result.ok else (result.error or ""),
        ms=(time.perf_counter() - started) * 1000,
    )
    return json.dumps(result.to_model_payload(), indent=2, default=str)


def _audit(
    tool: str, dataset_id: str, arguments: dict[str, Any], *, ok: bool, detail: str, ms: float
) -> None:
    """One line per call, to stderr and optionally to a file.

    A file, because the interesting question is usually asked after the fact: an MCP
    client may not surface its subprocess's stderr at all, and "did Claude Desktop
    really run my SQL, or did it answer from memory?" is not a question you want to
    have to reproduce.
    """
    line = (
        f"{datetime.now(UTC).isoformat(timespec='seconds')} "
        f"{'ok ' if ok else 'ERR'} {tool} dataset={dataset_id[:8]} {ms:.0f}ms "
        f"args={json.dumps(arguments, default=str)[:200]} :: {detail[:160]}"
    )
    log.info(line)
    path = os.environ.get("MCP_LOG_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            # An unwritable audit path must never break a working tool call.
            pass


# Written out rather than generated. A generated wrapper would need the schema poked in
# through a private attribute, and the whole value of an MCP tool is the description and
# argument names the model reads — those deserve to be visible in the source.


@mcp.tool()
def inspect_schema(dataset_id: str, include_statistics: bool = True) -> str:
    """List the dataset's columns with types, null counts, distinct counts and quality
    flags. Read this before writing any SQL so column names are known, not guessed."""
    return _run("inspect_schema", dataset_id, {"include_statistics": include_statistics})


@mcp.tool()
def profile_column(dataset_id: str, column: str, top_values: int = 10) -> str:
    """Profile one column in depth, including its most frequent values."""
    return _run("profile_column", dataset_id, {"column": column, "top_values": top_values})


@mcp.tool()
def execute_sql(dataset_id: str, sql: str, max_rows: int = 50) -> str:
    """Run a read-only SELECT against the dataset, which is exposed as a single table
    named 'dataset'. Full DuckDB SELECT dialect: CTEs, window functions, aggregates.
    Only SELECT and WITH are permitted. Always aggregate rather than listing raw rows."""
    return _run("execute_sql", dataset_id, {"sql": sql, "max_rows": max_rows})


@mcp.tool()
def compare_groups(
    dataset_id: str,
    group_column: str,
    metric_column: str,
    aggregation: str = "sum",
    order: str = "desc",
    limit: int = 20,
) -> str:
    """Aggregate a numeric metric across the categories of another column and rank the
    result. Refuses a grouping that would return roughly one row per record."""
    return _run(
        "compare_groups",
        dataset_id,
        {
            "group_column": group_column,
            "metric_column": metric_column,
            "aggregation": aggregation,
            "order": order,
            "limit": limit,
        },
    )


@mcp.tool()
def correlation(dataset_id: str, column_a: str, column_b: str) -> str:
    """Pearson and Spearman correlation between two numeric columns."""
    return _run("correlation", dataset_id, {"column_a": column_a, "column_b": column_b})


# --------------------------------------------------------------------------------


def _context(dataset_id: str) -> ToolContext:
    """Turn a client-supplied id into a context, or refuse.

    Over MCP the dataset id arrives from outside for the first time, so it is parsed as
    a UUID and checked against stored datasets before anything touches the filesystem.
    `app/data/storage.py` remains the only place an id becomes a path.
    """
    try:
        parsed = uuid.UUID(str(dataset_id))
    except (ValueError, AttributeError):
        raise ValueError(f"{dataset_id!r} is not a valid dataset id") from None

    with session_scope() as session:
        dataset = service.get_dataset(session, parsed)
        if dataset is None:
            raise ValueError(f"no dataset {parsed}")
        version = max((v.version for v in dataset.versions), default=None)
        if version is None:
            raise ValueError(f"dataset {parsed} has no versions to analyse")

    return ToolContext(dataset_id=parsed, version=version)


def selftest() -> int:
    """Exercise the server in-process and print what came back.

    Answers "is the server itself working?" without a client, which is the first thing
    to establish when a client appears to be ignoring it: a green selftest and a silent
    log means the problem is the client's configuration, not this code.
    """
    import asyncio

    async def run() -> int:
        datasets = json.loads((await mcp.call_tool("list_datasets", {})).content[0].text)
        print(f"list_datasets -> {len(datasets)} dataset(s)")
        if not datasets:
            print("no datasets to test against; upload one first")
            return 1

        target = datasets[0]
        print(f"  using {target['name']} ({target['rows']:,} rows)")

        schema = json.loads(
            (await mcp.call_tool("inspect_schema", {"dataset_id": target["dataset_id"]}))
            .content[0]
            .text
        )
        print(f"inspect_schema -> {schema.get('summary', 'ok')}")

        blocked = json.loads(
            (
                await mcp.call_tool(
                    "execute_sql",
                    {"dataset_id": target["dataset_id"], "sql": "DROP TABLE dataset"},
                )
            )
            .content[0]
            .text
        )
        print(f"execute_sql(DROP) -> refused: {blocked.get('error', '')[:70]}")
        print("\nserver is working. If a client still is not using it, the problem is")
        print("the client's configuration — check its MCP server list and its logs.")
        return 0

    return asyncio.run(run())


def main() -> None:
    # stdio, not HTTP: the client launches this as a subprocess, so there is no port to
    # secure and no way to reach it from another machine. For a server holding a
    # sandboxed SQL executor over local files, that is the right default.
    # stderr, because stdout IS the protocol on a stdio transport — a stray print
    # would corrupt the JSON-RPC stream and look like a client bug.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s mcp %(message)s",
    )
    log.info("ground-truth MCP server starting (stdio)")
    if os.environ.get("MCP_LOG_FILE"):
        log.info("auditing tool calls to %s", os.environ["MCP_LOG_FILE"])
    mcp.run(transport="stdio")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    main()
