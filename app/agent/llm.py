"""The only place in this system that talks to a language model.

    agent loop  ──►  LlmClient.chat(messages, tools)  ──►  POST /api/chat  ──►  Ollama
                            │
                            └──► ModelTurn(content, tool_calls, thinking, tokens)

WHY A THIN CLIENT AND NOT THE `ollama` PACKAGE
----------------------------------------------
The wire format is one POST with a JSON body. A client library would add a dependency
whose only job is to build that body, and would hide the two fields this project
actually has opinions about (`think` and `options.temperature`) behind defaults that
change between releases. `httpx` is already a dependency because M1's benchmark used
it, and the benchmark's findings are only transferable if the request looks the same.

WHY /api/chat AND NOT THE OpenAI-COMPATIBLE /v1/chat/completions
----------------------------------------------------------------
Ollama's native endpoint returns Qwen3's reasoning block as a SEPARATE `message.
thinking` field. The OpenAI-compatible shim concatenates it into `content`, so every
consumer then has to strip `<think>...</think>` with a regex — and M1 measured that
going wrong: `think: false` produced WORSE parseable output than leaving reasoning on,
because the model reasoned anyway and did it inside the content field.

Keeping reasoning on, and reading it from its own field, is the configuration this
project benchmarked. `thinking` is returned here so it can be logged, and is never
persisted: see the note on `EventKind` about not storing unverifiable narration.

FAILURES ARE TYPED, BECAUSE THE WORKER TREATS THEM DIFFERENTLY
--------------------------------------------------------------
`ModelUnavailable` means the operator has something to fix — Ollama is not running,
the model is not pulled, the machine ran out of memory. It carries an instruction, not
a stack trace, because it is going to be shown in a browser to whoever started the
process. Everything else is a `ModelError` the agent loop may be able to work around.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings

# Ollama reports its timing counters in nanoseconds.
_NS = 1_000_000_000


class ModelError(Exception):
    """The model call failed in a way the agent loop may be able to continue past."""


class ModelUnavailable(ModelError):
    """The model cannot be reached or does not exist. The operator must act.

    Separate from `ModelError` because the message is an instruction for a human, and
    because retrying it inside the agent loop is pointless: nothing the agent does will
    start a process that is not running.
    """


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation proposed by the model.

    `arguments` is normalised to a dict here. Ollama returns it as a JSON object for
    most models and as a JSON *string* for some — a difference that shows up as
    `'str' object has no attribute 'get'` three layers away if it is not fixed at the
    boundary.
    """

    name: str
    arguments: dict[str, Any]
    raw: Any = None


@dataclass
class ModelTurn:
    """One response from the model."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmClient:
    """A synchronous chat client for one Ollama server.

    Synchronous on purpose: it is called from a worker process whose whole job is to
    run one analysis at a time, and whose other blocking dependency (DuckDB) is
    synchronous too. Introducing async here would mean an event loop that exists to
    wait on one request.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        think: bool | None = None,
        num_ctx: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout_s = timeout_s if timeout_s is not None else settings.llm_timeout_s
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.think = think if think is not None else settings.llm_think
        self.num_ctx = num_ctx if num_ctx is not None else settings.llm_num_ctx
        # connect=5s, read=timeout_s. A refused connection must fail immediately —
        # waiting out a 300 second read timeout to learn Ollama is not running is the
        # kind of delay that gets diagnosed as "the agent hung".
        self._timeout = httpx.Timeout(self.timeout_s, connect=5.0)
        self._client = httpx.Client(timeout=self._timeout)

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the one call ------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelTurn:
        """One non-streaming turn. Raises `ModelError` rather than returning a failure.

        Non-streaming because nothing downstream can use a partial answer: a tool call
        is only actionable once its arguments are complete, and the UI's progress comes
        from the event trail, which is a better signal than tokens appearing.

        Passing `tools=None` is how the loop asks for prose. Ollama will happily emit a
        tool call whenever tools are present, so the final-answer turn removes them
        entirely rather than asking the model nicely to stop.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think

        started = time.perf_counter()
        try:
            response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise ModelUnavailable(
                f"cannot reach the language model at {self.base_url}. "
                f"Start it with 'ollama serve' and confirm the model is pulled "
                f"('ollama pull {self.model}')."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ModelError(
                f"the language model did not respond within {self.timeout_s:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"the language model call failed: {exc}") from exc

        if response.status_code == 404:
            raise ModelUnavailable(
                f"the model '{self.model}' is not available on {self.base_url}. "
                f"Pull it with 'ollama pull {self.model}'."
            )
        if response.status_code >= 400:
            raise ModelError(
                f"the language model returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ModelError("the language model returned a response that was not JSON") from exc

        message = body.get("message") or {}
        return ModelTurn(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
            thinking=(message.get("thinking") or "").strip(),
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
            wall_s=time.perf_counter() - started,
        )

    # -- readiness ---------------------------------------------------------------

    def check_available(self) -> None:
        """Raise `ModelUnavailable` unless the server is up and the model is pulled.

        Called once before the agent starts rather than discovering it on the first
        turn. The difference matters in the UI: a job that fails in 200 ms with "start
        ollama" is a fixable message, and one that fails after the queue has been
        polled and the schema loaded looks like the analysis itself broke.
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            available = {m.get("name", "") for m in response.json().get("models", [])}
        except httpx.HTTPError as exc:
            raise ModelUnavailable(
                f"cannot reach the language model at {self.base_url}. Start it with 'ollama serve'."
            ) from exc

        # Ollama reports "qwen3:4b"; a configured "qwen3" should match it, because the
        # server itself resolves the implicit ':latest' tag the same way.
        if self.model in available:
            return
        if any(name.split(":")[0] == self.model for name in available):
            return
        listed = ", ".join(sorted(available)) or "none"
        raise ModelUnavailable(
            f"the model '{self.model}' is not pulled on {self.base_url} "
            f"(available: {listed}). Run 'ollama pull {self.model}'."
        )


def _parse_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    """Normalise Ollama's tool-call payloads into `ToolCall`s.

    Anything unparseable becomes a call with an empty name, which the agent loop turns
    into a repair message for the model instead of a crash. A malformed tool call is
    the model's mistake, and the model is the only thing that can fix it.
    """
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = (raw or {}).get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        calls.append(ToolCall(name=name, arguments=arguments, raw=raw))
    return calls
