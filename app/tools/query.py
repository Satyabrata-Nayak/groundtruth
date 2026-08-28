"""The general tool. Everything else in the registry is a guard rail around this one.

WHY ONE OPEN-ENDED TOOL DOES NOT DEFEAT THE POINT OF A FIXED TOOL SET
--------------------------------------------------------------------
It is fair to ask what a fixed action space buys if one of the actions is "run any
SQL". The answer is that SQL is the largest surface we can offer that is still
*checkable before it runs*:

    arbitrary Python   unbounded    cannot be validated without running it
    arbitrary SQL      large        parses to an AST we can allowlist, on a
                                    connection with the filesystem switched off
    fixed analyses     tiny         safe, and useless for real questions

So the model keeps the expressive power it needs -- joins, CTEs, window functions,
percentiles, cohorts -- while every statement still passes the four layers in
`app.data.sandbox`. The tool's job here is not safety, which the sandbox already owns.
It is *shaping the result for a context window* and *turning failures into repair
instructions*.

THE ROW CAP IS THE INTERESTING PART
-----------------------------------
The sandbox will return up to 10,000 rows. A model must not see 10,000 rows: it would
consume the entire context and invite the model to eyeball what it should have
aggregated. So this caps at 50 and says so in the payload. A model that sees
`"truncated": true` can rewrite the query with GROUP BY; a model handed a silently
clipped result will confidently describe a sample as the whole.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.data.sandbox import TABLE_NAME
from app.tools._common import rows_as_lists, run_sql
from app.tools.base import Tool, ToolContext


class ExecuteSqlTool(Tool):
    name = "execute_sql"
    description = (
        f"Run a read-only SQL SELECT query against the dataset. The dataset is exposed "
        f"as a single table named '{TABLE_NAME}'. Supports the full DuckDB SELECT "
        f"dialect including CTEs, window functions, aggregates and CASE expressions. "
        f"Only SELECT and WITH are permitted; the query cannot modify data or read "
        f"other files. Always aggregate rather than returning many raw rows."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    f"A single SQL SELECT statement querying the '{TABLE_NAME}' table. "
                    f"Do not include a trailing semicolon or multiple statements."
                ),
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum rows to return. Keep small; aggregate instead.",
                "default": 50,
                "minimum": 1,
                "maximum": 500,
            },
        },
        "required": ["sql"],
    }

    def execute(
        self, context: ToolContext, *, sql: str, max_rows: int = 50
    ) -> tuple[dict[str, Any], str]:
        cap = min(int(max_rows), get_settings().max_tool_result_rows)
        result = run_sql(context, sql, max_rows=cap)

        data: dict[str, Any] = {
            "columns": result.columns,
            "rows": rows_as_lists(result),
            "row_count": result.row_count,
            "truncated": result.truncated,
            "execution_ms": round(result.execution_ms, 2),
        }
        if result.truncated:
            data["note"] = (
                f"only the first {cap} rows are shown. Rewrite the query with GROUP BY "
                f"or an aggregate if you need to describe all of the data."
            )

        shape = f"{result.row_count} row{'s' if result.row_count != 1 else ''}"
        return data, f"{shape} x {len(result.columns)} column(s) in {result.execution_ms:.0f} ms"
