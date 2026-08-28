"""Turning "what about France?" into a question that stands on its own.

WHY THIS IS THE RIGHT SHAPE FOR A "SUB-AGENT" ON ONE GPU
--------------------------------------------------------
The instinct with sub-agents is to fan out: a planner, a SQL writer, a critic, all
working at once. On a machine with one GPU that is strictly a **loss**. The model is
already 100% resident and saturating the card; two concurrent calls do not run in
parallel, they split the same tokens per second between them and add KV-cache pressure.
Fan-out buys thoroughness and pays for it in latency, and latency is the thing this
project has spent a day fixing.

What *does* pay is a second model that is a different size. This rewriter runs on
qwen2.5:3b-instruct — measured at about **one second** against qwen3:4b's forty-five —
so it costs ~2% of a turn and earns three things:

    it resolves the follow-up      "what about France?" becomes a question the planner
                                   can answer without re-reading the thread
    it shortens the big prompt     the analyst no longer needs the full history block,
                                   only the standalone question
    it makes the cache work        a cache keyed on the question text can never hit a
                                   follow-up, because "what about France?" means
                                   something different in every thread. Rewritten, it
                                   hashes to the same key as somebody asking it directly

That is the general rule worth stating: **a sub-agent earns its place when it is
cheaper than the model it serves, not when it is another copy of it.**

WHY IT MUST NEVER INVENT
------------------------
A rewriter that "improves" a standalone question is a rewriter that changes what was
asked. So it is constrained hard: if the question already stands alone it is returned
untouched, and any failure — an unreachable model, an empty reply, an answer longer than
the input plus the history could justify — falls back to the original. A rewriter is an
optimisation, and an optimisation that can change the answer is a bug.
"""

from __future__ import annotations

from app.agent.llm import LlmClient, ModelError
from app.agent.memory import Turn

# The rewriter runs on the CHEAPEST model of the SAME provider as the analyst.
#
# Same provider, because mixing them makes the cheap step the slow one: a 1-second local
# call in front of a 2-second Groq answer is 50% overhead, where the same call in front
# of a 150-second local answer is 2%.
#
# Cheapest, because the whole argument for a sub-agent is that it costs less than the
# model it serves. A sub-agent that costs what the main agent costs is just two main
# agents, and the user waits for both.
FALLBACK_REWRITER = "qwen2.5:3b-instruct"

# A rewrite is a resolution of pronouns and ellipsis, not an essay. Anything longer than
# this means the model started explaining rather than rewriting.
_MAX_REWRITE_CHARS = 300

SYSTEM = """\
You rewrite a follow-up question so that it can be understood on its own, without the \
conversation around it.

Replace pronouns and references — "it", "that", "those", "the same", "what about X" — \
with the thing they refer to, taken from the previous questions.

RULES
- If the question already stands on its own, return it EXACTLY as given. Do not improve
  it, expand it, or make it more specific.
- Never add a filter, a metric, a time period or a limit that nobody asked for.
- Keep it to one sentence.
- Reply with the rewritten question and nothing else. No preamble, no quotes, no
  explanation."""


def needs_rewriting(question: str, turns: list[Turn]) -> bool:
    """A cheap gate, so the common case does not pay even one second.

    Most questions are standalone, and a call that returns the input unchanged is a
    second spent to learn nothing. The signals are crude on purpose: a reference word,
    or a fragment too short to be a full question.
    """
    if not turns:
        return False

    lowered = f" {question.lower().strip()} "
    references = (
        " it ",
        " its ",
        " that ",
        " those ",
        " these ",
        " they ",
        " them ",
        " their ",
        " same ",
        " also ",
        " instead ",
        " previous ",
        " above ",
        " earlier ",
        " what about",
        " how about",
        " and for ",
        " but for ",
        " why ",
    )
    if any(token in lowered for token in references):
        return True
    # "France?" or "by month" — too short to carry its own subject. Three, not four:
    # "which country earns most" is four words and stands perfectly well on its own,
    # and a gate that fires on it would spend a second per question learning nothing.
    return len(question.split()) <= 3


def rewrite(
    question: str,
    turns: list[Turn],
    *,
    main_model: str | None = None,
    client: LlmClient | None = None,
) -> str:
    """The standalone form of `question`, or `question` itself.

    Never raises. Every failure path returns the original, because an unreachable
    rewriter must slow nothing down and change nothing.
    """
    if not needs_rewriting(question, turns):
        return question

    history = "\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in turns)
    owns_client = client is None
    if client is None:
        from app.agent.factory import build_client, cheapest_peer

        client = build_client(cheapest_peer(main_model) or FALLBACK_REWRITER, think=False)

    try:
        turn = client.chat(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{history}\n\nRewrite this follow-up so it stands alone:\n"
                        f"{question.strip()}"
                    ),
                },
            ],
            tools=None,
        )
    except ModelError:
        return question
    finally:
        if owns_client:
            client.close()

    return _accept(turn.content, question)


def _accept(rewritten: str, original: str) -> str:
    """Take the rewrite only if it looks like a rewrite.

    A small model asked for one sentence sometimes returns a paragraph explaining what
    it changed. That is not a question, and sending it to the analyst would replace the
    user's words with the rewriter's commentary.
    """
    candidate = (rewritten or "").strip().strip('"').strip()
    # Some models still open with "Sure, here is the rewritten question:".
    if ":" in candidate[:60] and "\n" in candidate:
        candidate = candidate.split("\n")[-1].strip()

    if not candidate or len(candidate) > _MAX_REWRITE_CHARS or "\n" in candidate:
        return original
    return candidate
