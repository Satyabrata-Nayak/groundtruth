"""The contract between a language model and this system's deterministic code.

    model  --"call execute_sql with {sql: ...}"-->  Registry.call()
                                                        |
                                          validate args |  <- schema, then semantics
                                                        |
                                              execute   +-->  ToolResult(ok=True, ...)
                                                        +-->  ToolResult(ok=False, error=...)

THREE RULES THIS MODULE ENFORCES
--------------------------------

1. THE MODEL NEVER SUPPLIES A DATASET OR A PATH.
   `dataset_id` and `version` live in `ToolContext`, which the caller -- not the model
   -- constructs. A tool's JSON schema, the only thing the model sees, has no field for
   either. This continues the boundary `app/data/storage.py` established: an id becomes
   a path in exactly one place, and nothing the model emits reaches it.

2. `call()` NEVER RAISES.
   A failed tool call is a *value*, not an exception. This looks like sloppy error
   handling and is the opposite: in M5 the agent loop must be able to hand the model
   back "column 'reveune' does not exist; valid columns are: ..." and let it retry.
   An exception would abort the run; a ToolResult with `ok=False` is a repair prompt.
   Every error message here is written to be read by a model, which is why they name
   the valid alternatives rather than only stating what was wrong.

3. RESULTS ARE CAPPED IN ROWS, NOT BYTES.
   A tool result is going into a context window. `app.config.max_result_rows` (10,000)
   bounds what the *database* may return; `max_tool_result_rows` (50) bounds what the
   *model* may see. Truncation is always reported in the payload, so the model knows
   it is looking at a sample and can aggregate instead of eyeballing.

WHY A HAND-WRITTEN SCHEMA VALIDATOR
-----------------------------------
`_validate_arguments` implements the subset of JSON Schema the tools actually use. A
full validator (`jsonschema`) would be a dependency whose error strings are written
for developers -- "None is not of type 'string'" -- when the consumer here is a 4B
model that has to act on the message. Owning the validator means owning the wording.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# JSON Schema type name -> the Python types that satisfy it.
# `bool` is excluded from integer/number deliberately: in Python `True == 1`, so a
# model that sends `true` for a `limit` would otherwise silently mean 1.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolError(Exception):
    """A tool call failed for a reason the model may be able to fix.

    The message is fed back into the conversation verbatim, so it must be actionable:
    say what was wrong AND what would have been right.
    """


@dataclass(frozen=True)
class ToolContext:
    """What the caller knows and the model does not.

    Constructed by the worker from a request, never from model output. Adding a field
    here is how a tool gains access to something without exposing it to the model.
    """

    dataset_id: uuid.UUID
    version: int


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call -- success or failure, uniformly shaped."""

    tool: str
    ok: bool
    arguments: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    # Set from `Tool.model_view` when the model should see less than the caller does.
    # None means "the same thing", which is the case for every tool but create_chart.
    model_data: dict[str, Any] | None = None

    def to_model_payload(self) -> dict[str, Any]:
        """The JSON the model sees after its call.

        Failures carry `ok: false` and an error string and nothing else -- no partial
        data to misread as an answer.
        """
        if not self.ok:
            return {"ok": False, "tool": self.tool, "error": self.error}
        visible = self.data if self.model_data is None else self.model_data
        return {"ok": True, "tool": self.tool, "summary": self.summary, **visible}


class Tool(ABC):
    """One deterministic capability the model may invoke.

    Subclasses declare a JSON Schema for their arguments and implement `execute`.
    Schema validation, timing and error capture are handled by the registry, so a tool
    body contains only its own logic.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def execute(self, context: ToolContext, **kwargs: Any) -> tuple[dict[str, Any], str]:
        """Do the work. Returns (data, one-line summary).

        Raise `ToolError` for anything the model could plausibly correct. Anything else
        raising is a bug in us, and the registry reports it as such rather than
        inviting the model to retry a call that will never work.
        """

    def model_view(self, data: dict[str, Any]) -> dict[str, Any]:
        """What the MODEL sees, when that differs from what the caller gets.

        Identity for almost every tool. It exists for `create_chart`, whose result has
        two audiences with opposite needs: the browser needs every data point in order
        to draw the chart, and the model needs to know a chart was produced and what it
        shows. Sending 400 scatter points to a 4B model costs roughly 3,000 tokens of
        context and tells it nothing its own query did not already establish.

        Overriding this is how a tool stays useful to both without the caller having to
        strip its own payload.
        """
        return data

    def spec(self) -> dict[str, Any]:
        """This tool as Ollama / OpenAI tool-calling JSON."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """The agent's entire action space.

    Everything the model is permitted to do is registered here, and `call` is the only
    way to do it. There is no second path from model output to execution.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"unknown tool '{name}'. Available tools: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        """Tool definitions for the model's prompt.

        `only` narrows the action space for a single run. Offering four relevant tools
        instead of seven measurably improves selection accuracy on small models, and it
        is also how the evaluation set pins a question to the tools it should need.
        """
        chosen = self.names() if only is None else [n for n in self.names() if n in set(only)]
        return [self._tools[n].spec() for n in chosen]

    def call(
        self, name: str, context: ToolContext, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Validate and execute one model-proposed tool call. Never raises.

        The order matters: an unknown tool, a bad argument shape and a runtime failure
        are three different messages, and collapsing them into one generic error is how
        an agent ends up retrying the same broken call forever.
        """
        arguments = dict(arguments or {})
        started = time.perf_counter()

        def failed(message: str) -> ToolResult:
            return ToolResult(
                tool=name,
                ok=False,
                arguments=arguments,
                error=message,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            tool = self.get(name)
        except ToolError as exc:
            return failed(str(exc))

        try:
            cleaned = _validate_arguments(tool, arguments)
        except ToolError as exc:
            return failed(str(exc))

        try:
            data, summary = tool.execute(context, **cleaned)
        except ToolError as exc:
            return failed(str(exc))
        except Exception as exc:  # noqa: BLE001 - a bug in us, not a repairable call
            return failed(f"internal error in tool '{name}': {type(exc).__name__}: {exc}")

        try:
            model_data = tool.model_view(data)
        except Exception as exc:  # noqa: BLE001 - never lose a good result to a bad view
            return failed(f"internal error summarising '{name}': {type(exc).__name__}: {exc}")

        return ToolResult(
            tool=name,
            ok=True,
            arguments=cleaned,
            data=data,
            summary=summary,
            model_data=None if model_data is data else model_data,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


# --------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------


def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check model-supplied arguments against the tool's schema.

    Returns a NEW dict containing only declared properties, with defaults filled in.
    Returning a copy rather than mutating is what guarantees an undeclared key cannot
    reach `execute` as a surprise keyword argument.
    """
    schema = tool.parameters
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        accepted = ", ".join(sorted(properties)) or "none"
        raise ToolError(
            f"unexpected argument{plural} {', '.join(repr(u) for u in unknown)} "
            f"for tool '{tool.name}'. Accepted arguments: {accepted}"
        )

    missing = [key for key in required if key not in arguments or arguments[key] is None]
    if missing:
        plural = "s" if len(missing) > 1 else ""
        raise ToolError(
            f"missing required argument{plural} "
            f"{', '.join(repr(m) for m in missing)} for tool '{tool.name}'"
        )

    cleaned: dict[str, Any] = {}
    for key, spec in properties.items():
        if key not in arguments or arguments[key] is None:
            if "default" in spec:
                cleaned[key] = spec["default"]
            continue
        cleaned[key] = _validate_value(tool.name, key, arguments[key], spec)
    return cleaned


def _validate_value(tool_name: str, key: str, value: Any, spec: dict[str, Any]) -> Any:
    expected = spec.get("type")

    if expected in _JSON_TYPES:
        allowed = _JSON_TYPES[expected]
        # A bool is an int in Python; refuse it where a number was asked for.
        if isinstance(value, bool) and expected in ("integer", "number"):
            raise ToolError(
                f"argument '{key}' of tool '{tool_name}' must be {expected}, got boolean"
            )
        if not isinstance(value, allowed):
            raise ToolError(
                f"argument '{key}' of tool '{tool_name}' must be {expected}, "
                f"got {type(value).__name__}"
            )
        if expected == "number":
            value = float(value)

    if "enum" in spec and value not in spec["enum"]:
        options = ", ".join(repr(o) for o in spec["enum"])
        raise ToolError(
            f"argument '{key}' of tool '{tool_name}' must be one of {options}, got {value!r}"
        )

    if expected == "array":
        item_spec = spec.get("items", {})
        value = [
            _validate_value(tool_name, f"{key}[{i}]", item, item_spec)
            for i, item in enumerate(value)
        ]
        if "minItems" in spec and len(value) < spec["minItems"]:
            raise ToolError(
                f"argument '{key}' of tool '{tool_name}' needs at least "
                f"{spec['minItems']} item(s), got {len(value)}"
            )

    if "minimum" in spec and value < spec["minimum"]:
        raise ToolError(
            f"argument '{key}' of tool '{tool_name}' must be >= {spec['minimum']}, got {value}"
        )
    if "maximum" in spec and value > spec["maximum"]:
        raise ToolError(
            f"argument '{key}' of tool '{tool_name}' must be <= {spec['maximum']}, got {value}"
        )

    return value
