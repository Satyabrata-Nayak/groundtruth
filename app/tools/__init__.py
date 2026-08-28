"""The tool registry: everything the agent is allowed to do.

    inspect_schema    what columns exist, their types and data quality
    profile_column    one column in depth, including its frequent values
    execute_sql       any read-only SELECT -- the general capability
    compare_groups    a metric aggregated and ranked across categories
    correlation       Pearson and Spearman between two numeric columns
    create_chart      a validated chart specification plus its data

WHY SIX AND NOT SIXTY
---------------------
Every tool added is another entry in the model's prompt and another chance to pick the
wrong one. Selection accuracy on a 4B model degrades as the list grows, so the list
earns its length: `execute_sql` covers everything expressible in SQL, and the other
five exist only where they enforce something SQL cannot -- type checking before
execution, a guard against meaningless groupings, a chart that is actually readable.

`detect_anomalies` and the regression / clustering tools are deliberately absent until
M6. They need SciPy and scikit-learn, and adding a dependency before there is an agent
to use it is how a dependency list stops meaning anything.

A NOTE ON THE DEFAULT REGISTRY BEING A SINGLETON
------------------------------------------------
`get_registry()` is cached because tool instances are stateless -- all per-call state
lives in `ToolContext` and the arguments. Tests that need an isolated action space
build their own `ToolRegistry` rather than mutating this one.
"""

from __future__ import annotations

from functools import lru_cache

from app.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from app.tools.chart import CreateChartTool
from app.tools.inspect import InspectSchemaTool, ProfileColumnTool
from app.tools.query import ExecuteSqlTool
from app.tools.stats import CompareGroupsTool, CorrelationTool

TOOL_CLASSES: tuple[type[Tool], ...] = (
    InspectSchemaTool,
    ProfileColumnTool,
    ExecuteSqlTool,
    CompareGroupsTool,
    CorrelationTool,
    CreateChartTool,
)


def build_registry() -> ToolRegistry:
    """A fresh registry holding one instance of every built-in tool."""
    registry = ToolRegistry()
    for tool_class in TOOL_CLASSES:
        registry.register(tool_class())
    return registry


@lru_cache
def get_registry() -> ToolRegistry:
    """The process-wide registry."""
    return build_registry()


__all__ = [
    "TOOL_CLASSES",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_registry",
    "get_registry",
]
