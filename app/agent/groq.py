"""Talking to Groq, which is OpenAI-shaped rather than Ollama-shaped.

WHY A SECOND CLIENT AND NOT A BRANCH INSIDE THE FIRST
------------------------------------------------------
The two APIs disagree about almost every field name that matters:

    Ollama                          Groq (OpenAI-compatible)
    /api/chat                       /openai/v1/chat/completions
    message.tool_calls[].function   choices[0].message.tool_calls[].function
    message.thinking (own field)    reasoning, or nothing at all
    options.num_ctx                 no equivalent — the window is the model's
    "think": true|false             "reasoning_effort": low|medium|high
    keep_alive                      meaningless; there is nothing to keep alive
    eval_count / prompt_eval_count  usage.completion_tokens / usage.prompt_tokens

A single client with `if provider == ...` at seven points would be a client that is
wrong about one of them. Two classes with one shared result type (`ModelTurn`) means the
agent loop never learns which one it is holding.

WHAT CHANGES FOR THE AGENT WHEN IT RUNS ON GROQ
-----------------------------------------------
Everything the local design fought for stops binding:

    context     8,192 tokens locally (a hard VRAM cliff)  ->  131,072
    speed       ~61 tokens/second                          ->  ~1,000
    concurrency one GPU, so parallel calls SPLIT it        ->  server-side, genuinely
                                                               parallel

The last line is the important one. `app/agent/rewrite.py` argues that fan-out is a loss
on one local GPU because two calls share one card. That argument does not survive the
move to a hosted API — which is why the sub-agent policy is a function of the provider
rather than a constant (see `app/agent/models.py`).

WHAT DOES NOT CHANGE
--------------------
The model still computes nothing. Every number still comes out of DuckDB on this
machine; only the choosing moves off it. And the data does not move: the tool results
are the only thing sent, capped at 50 rows, which is what makes a hosted model
acceptable for a local-first tool at all.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.agent.llm import ModelError, ModelTurn, ModelUnavailable, ToolCall
from app.config import get_settings

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

log = logging.getLogger(__name__)

# Three attempts. The free tier's window is a minute, so a fourth wait would exceed any
# reasonable patience; better to fail with a message naming the limit.
_MAX_RETRIES = 3
# Groq puts the wait in the message text as well as the header, and the text is the more
# precise of the two ("try again in 8.745s" against a whole-second header).
_RETRY_SECONDS = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """How long to wait, preferring what the server said over a guess."""
    match = _RETRY_SECONDS.search(response.text or "")
    if match:
        # A small margin: waiting exactly the stated time races the window's edge.
        return min(float(match.group(1)) + 0.5, 30.0)
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, 30.0)
        except ValueError:
            pass
    return min(2.0 * (2**attempt), 30.0)


class GroqClient:
    """A chat client for Groq, exposing the same surface as `LlmClient`.

    Same attributes (`model`, `think`), same `chat`/`check_available`/`close`, same
    `ModelTurn` out. The agent loop is written against that shape and never asks which
    provider it is talking to.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        # Far shorter than the local default of 300 s. A hosted model that has not
        # answered in a minute is not thinking, it is failing, and waiting out five
        # minutes to discover that is the difference between an error and a hang.
        self.timeout_s = timeout_s if timeout_s is not None else settings.groq_timeout_s
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.think = think
        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=10.0))

    # -- lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GroqClient:
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
        """One turn. Same contract as the local client: raises rather than returning bad."""
        if not self.api_key:
            raise ModelUnavailable(
                "GROQ_API_KEY is not set. Add it to .env, or choose a local model in the picker."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai(messages),
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            # "auto" rather than "required": the loop's whole design is that a turn with
            # NO tool call is the signal that the model is ready to answer. Forcing a
            # call would remove the only way it has to say it is finished.
            payload["tool_choice"] = "auto"
        if self.think is False:
            # gpt-oss exposes reasoning as an effort dial rather than a boolean. "low"
            # is the closest honest translation of "the user asked for less thinking";
            # there is no way to switch it off entirely, and pretending otherwise in the
            # UI would be worse than mapping it.
            payload["reasoning_effort"] = "low"

        started = time.perf_counter()
        response = self._post_with_backoff(payload)
        _raise_for_status(response, self.model)

        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}

        return ModelTurn(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
            # gpt-oss returns its reasoning separately, like Ollama does. Read so it can
            # be counted; never stored, for the same reason as everywhere else.
            thinking=(message.get("reasoning") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            wall_s=time.perf_counter() - started,
        )

    def _post_with_backoff(self, payload: dict[str, Any]) -> httpx.Response:
        """POST, waiting out rate limits rather than failing on them.

        THE FREE TIER IS 8,000 TOKENS PER MINUTE, not the 250,000 the plan comparison
        implies, and one analysis of a wide schema is comfortably 2,500. So a burst of
        three questions hits the limit, and the first version of this client turned that
        into a FAILED analysis that threw away two successful queries.

        A 429 is the one HTTP error that is neither a bug nor permanent — it is an
        instruction to wait, and Groq says exactly how long, both in `retry-after` and in
        the message text. Honouring it costs nine seconds; not honouring it costs the
        user their analysis.

        Retries only on 429 and 5xx. A 401 or a 404 is not going to be different in two
        seconds, and retrying it would turn "check your API key" into three copies of
        the same wait.
        """
        last: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                last = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.TimeoutException as exc:
                raise ModelError(f"Groq did not respond within {self.timeout_s:.0f}s") from exc
            except httpx.HTTPError as exc:
                raise ModelUnavailable(f"cannot reach Groq at {self.base_url}: {exc}") from exc

            if last.status_code not in (429, 500, 502, 503) or attempt == _MAX_RETRIES - 1:
                return last

            delay = _retry_after(last, attempt)
            log.info("Groq %s; waiting %.1fs before retry %d", last.status_code, delay, attempt + 1)
            time.sleep(delay)

        return last  # unreachable; the loop returns on its last attempt

    # -- readiness ---------------------------------------------------------------

    def check_available(self) -> None:
        """Fail early and with an instruction, rather than mid-analysis with a 401."""
        if not self.api_key:
            raise ModelUnavailable(
                "GROQ_API_KEY is not set. Put it in .env and restart the worker, or "
                "choose a local model in the picker."
            )
        try:
            response = self._client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"cannot reach Groq at {self.base_url}: {exc}") from exc

        if response.status_code == 401:
            raise ModelUnavailable("Groq rejected the API key. Check GROQ_API_KEY in .env.")
        if response.status_code >= 400:
            raise ModelUnavailable(f"Groq returned HTTP {response.status_code} listing models")

        available = {entry.get("id") for entry in response.json().get("data", [])}
        if available and self.model not in available:
            raise ModelUnavailable(
                f"Groq has no model {self.model!r}. It may have been retired — preview "
                f"models are not guaranteed to stay."
            )


def _raise_for_status(response: httpx.Response, model: str) -> None:
    """Turn Groq's failures into the two kinds the agent loop distinguishes.

    `ModelUnavailable` is for things an operator fixes — a bad key, a retired model, a
    quota. `ModelError` is everything else. Getting this split right is what decides
    whether a user sees "add your key to .env" or a stack trace.
    """
    if response.status_code < 400:
        return

    detail = ""
    try:
        detail = str((response.json().get("error") or {}).get("message") or "")[:300]
    except ValueError:
        detail = response.text[:200]

    if response.status_code == 401:
        raise ModelUnavailable(f"Groq rejected the API key. Check GROQ_API_KEY in .env. {detail}")
    if response.status_code == 404:
        raise ModelUnavailable(f"Groq has no model {model!r}. {detail}")
    if response.status_code == 429:
        # A rate limit is not a bug and not a permanent failure, and saying which it is
        # matters: the user's next move is to wait or to switch models, not to debug.
        raise ModelError(
            f"Groq rate limit reached for {model}. Wait a moment, or switch model. {detail}"
        )
    if response.status_code >= 500:
        raise ModelError(f"Groq is having trouble (HTTP {response.status_code}). {detail}")
    raise ModelError(f"Groq returned HTTP {response.status_code}: {detail}")


def _to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the loop's Ollama-shaped conversation into OpenAI's shape.

    Two differences bite, and both are silent failures rather than errors:

    1. A tool result must carry `tool_call_id` matching the assistant turn that asked
       for it. Ollama matches by name and ignores the field; OpenAI drops the message.
       So ids are synthesised in order and paired up here.
    2. `arguments` must be a JSON STRING, not an object. Sending an object is accepted
       and then ignored, which looks like a model that forgot what it just asked for.
    """
    out: list[dict[str, Any]] = []
    pending: list[str] = []

    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            calls = []
            pending = []
            for index, raw in enumerate(message["tool_calls"]):
                function = (raw or {}).get("function") or {}
                call_id = f"call_{len(out)}_{index}"
                pending.append(call_id)
                arguments = function.get("arguments")
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": (
                                arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments or {})
                            ),
                        },
                    }
                )
            out.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
            )
            continue

        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": pending.pop(0) if pending else "call_0",
                    "content": message.get("content", ""),
                }
            )
            continue

        out.append({"role": role, "content": message.get("content", "")})

    return out


def _parse_tool_calls(raw_calls: list[Any]) -> list[ToolCall]:
    """Normalise OpenAI-shaped tool calls, whose arguments arrive as a JSON string.

    `raw` keeps the ORIGINAL payload, because the loop replays the assistant turn back
    to the model verbatim and a rewritten call makes the transcript disagree with what
    the model believes it said.
    """
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = (raw or {}).get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(name=str(function.get("name") or ""), arguments=arguments, raw=raw))
    return calls
