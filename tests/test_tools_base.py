"""The tool contract: argument validation, error shape, and the model/caller split.

These tests use throwaway tools rather than the real six, because what is under test
is the REGISTRY -- schema validation, error wording, the guarantee that `call` never
raises. Using real tools here would make every failure ambiguous between "the contract
is broken" and "that tool is broken".
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
)


class EchoTool(Tool):
    name = "echo"
    description = "echo arguments back"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "count": {"type": "integer", "default": 1, "minimum": 1, "maximum": 10},
            "mode": {"type": "string", "enum": ["loud", "quiet"], "default": "quiet"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "ratio": {"type": "number"},
        },
        "required": ["text"],
    }

    def execute(self, context: ToolContext, **kwargs: Any) -> tuple[dict[str, Any], str]:
        return dict(kwargs), f"echoed {kwargs.get('text')!r}"


class ExplodingTool(Tool):
    name = "explode"
    description = "always fails"
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def execute(self, context: ToolContext, **kwargs: Any) -> tuple[dict[str, Any], str]:
        raise ZeroDivisionError("boom")


class RepairableTool(Tool):
    name = "repairable"
    description = "fails in a way the model could fix"
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def execute(self, context: ToolContext, **kwargs: Any) -> tuple[dict[str, Any], str]:
        raise ToolError("column 'x' does not exist. Available columns: a, b")


class BigResultTool(Tool):
    name = "big"
    description = "returns more than the model should see"
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def execute(self, context: ToolContext, **kwargs: Any) -> tuple[dict[str, Any], str]:
        return {"points": list(range(1000)), "keep": "yes"}, "1000 points"

    def model_view(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"points": data["points"][:3], "truncated": True, "keep": data["keep"]}


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (EchoTool(), ExplodingTool(), RepairableTool(), BigResultTool()):
        reg.register(tool)
    return reg


@pytest.fixture
def context() -> ToolContext:
    return ToolContext(dataset_id=uuid.uuid4(), version=1)


# --------------------------------------------------------------------------- happy path


def test_valid_call_returns_data_and_summary(registry, context):
    result = registry.call("echo", context, {"text": "hi"})
    assert result.ok
    assert result.data["text"] == "hi"
    assert result.summary == "echoed 'hi'"
    assert result.duration_ms >= 0


def test_defaults_are_filled_in(registry, context):
    result = registry.call("echo", context, {"text": "hi"})
    assert result.data["count"] == 1
    assert result.data["mode"] == "quiet"


def test_absent_optional_without_default_is_omitted(registry, context):
    """`ratio` has no default, so it must not appear rather than arriving as None."""
    result = registry.call("echo", context, {"text": "hi"})
    assert "ratio" not in result.arguments


# ------------------------------------------------------------------------- validation


def test_unknown_tool_names_the_alternatives(registry, context):
    result = registry.call("nope", context, {})
    assert not result.ok
    assert "unknown tool 'nope'" in result.error
    # The message has to be actionable for a model, so it lists what does exist.
    assert "echo" in result.error


def test_missing_required_argument(registry, context):
    result = registry.call("echo", context, {})
    assert not result.ok
    assert "missing required argument 'text'" in result.error


def test_unexpected_argument_lists_accepted_ones(registry, context):
    result = registry.call("echo", context, {"text": "hi", "txt": "typo"})
    assert not result.ok
    assert "'txt'" in result.error
    assert "count" in result.error


def test_wrong_type_is_rejected(registry, context):
    result = registry.call("echo", context, {"text": 42})
    assert not result.ok
    assert "must be string, got int" in result.error


def test_bool_is_not_accepted_as_integer(registry, context):
    """In Python `True == 1`, so a bool would silently mean count=1 without this."""
    result = registry.call("echo", context, {"text": "hi", "count": True})
    assert not result.ok
    assert "got boolean" in result.error


def test_enum_violation_lists_options(registry, context):
    result = registry.call("echo", context, {"text": "hi", "mode": "shouting"})
    assert not result.ok
    assert "'loud'" in result.error and "'quiet'" in result.error


def test_bounds_are_enforced(registry, context):
    assert not registry.call("echo", context, {"text": "h", "count": 0}).ok
    assert not registry.call("echo", context, {"text": "h", "count": 99}).ok
    assert registry.call("echo", context, {"text": "h", "count": 5}).ok


def test_array_items_are_validated(registry, context):
    result = registry.call("echo", context, {"text": "hi", "tags": ["a", 3]})
    assert not result.ok
    assert "tags[1]" in result.error


def test_array_min_items(registry, context):
    result = registry.call("echo", context, {"text": "hi", "tags": []})
    assert not result.ok
    assert "at least 1" in result.error


def test_number_accepts_int_and_coerces(registry, context):
    result = registry.call("echo", context, {"text": "hi", "ratio": 2})
    assert result.ok
    assert isinstance(result.data["ratio"], float)


# ------------------------------------------------------------------- error containment


def test_call_never_raises_on_internal_error(registry, context):
    """An unexpected exception becomes a failed result, not a crashed agent loop."""
    result = registry.call("explode", context, {})
    assert not result.ok
    assert "internal error" in result.error
    assert "ZeroDivisionError" in result.error


def test_tool_error_is_reported_without_the_internal_prefix(registry, context):
    """A ToolError is repairable, so it must not be dressed up as an internal fault."""
    result = registry.call("repairable", context, {})
    assert not result.ok
    assert "internal error" not in result.error
    assert "Available columns: a, b" in result.error


def test_failed_payload_carries_no_partial_data(registry, context):
    payload = registry.call("explode", context, {}).to_model_payload()
    assert payload["ok"] is False
    assert set(payload) == {"ok", "tool", "error"}


# ------------------------------------------------------------------- the model/caller split


def test_model_view_trims_what_the_model_sees(registry, context):
    result = registry.call("big", context, {})
    assert len(result.data["points"]) == 1000  # the caller gets everything
    payload = result.to_model_payload()
    assert len(payload["points"]) == 3  # the model gets a sample
    assert payload["truncated"] is True
    assert payload["keep"] == "yes"


def test_model_data_is_none_when_views_are_identical(registry, context):
    """Only tools that actually differ pay for a second payload."""
    result = registry.call("echo", context, {"text": "hi"})
    assert result.model_data is None


# --------------------------------------------------------------------------- registry


def test_duplicate_registration_is_refused():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(EchoTool())


def test_specs_are_ollama_shaped(registry):
    spec = next(s for s in registry.specs() if s["function"]["name"] == "echo")
    assert spec["type"] == "function"
    assert spec["function"]["parameters"]["required"] == ["text"]


def test_specs_can_be_narrowed(registry):
    names = [s["function"]["name"] for s in registry.specs(only=["echo"])]
    assert names == ["echo"]


def test_context_is_not_in_any_tool_schema():
    """The model must never be offered a dataset_id or a path. This is the boundary."""
    from app.tools import get_registry

    for spec in get_registry().specs():
        properties = spec["function"]["parameters"].get("properties", {})
        assert "dataset_id" not in properties
        assert "version" not in properties
        assert not any("path" in key.lower() for key in properties)
