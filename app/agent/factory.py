"""Building the right client for a model name, and sizing the loop to what it costs.

ONE PLACE DECIDES WHICH PROVIDER A NAME BELONGS TO
---------------------------------------------------
`qwen3:4b` runs on this machine and `openai/gpt-oss-20b` runs on Groq's. Everything
downstream — the agent loop, the rewriter, the worker — should never need to know which,
so the decision lives here and nowhere else. Add a provider by adding a branch in this
file and a catalogue entry; nothing else changes.

THE LOOP IS SIZED BY WHAT A ROUND COSTS, NOT BY A CONSTANT
------------------------------------------------------------
`agent_max_tool_rounds` was a single number when there was a single provider. It cannot
stay one:

    local   a round is 45-90 seconds   two rounds is already the limit of patience
    Groq    a round is under a second  four rounds is cheaper than one local round

Two rounds on Groq would be leaving the entire benefit unspent. Four on a local model
would be a six-minute wait. The right number is a property of the provider, so it is
read from the catalogue rather than from configuration.
"""

from __future__ import annotations

from app.agent.llm import LlmClient
from app.agent.models import profile_for
from app.config import get_settings


def build_client(model: str | None, *, think: bool | None = None):
    """The client for `model`, or for whatever is configured when `model` is None."""
    name = model or get_settings().llm_model
    profile = profile_for(name)

    if profile is not None and profile.provider == "groq":
        # Imported lazily so a purely local deployment never pays for it and never
        # needs the key to exist.
        from app.agent.groq import GroqClient

        return GroqClient(model=name, think=think)

    return LlmClient(model=name, think=think)


def rounds_for(model: str | None) -> int:
    """How many tool rounds this model can afford. See the module note."""
    profile = profile_for(model or get_settings().llm_model)
    if profile is None:
        # An unmeasured model an operator pointed LLM_MODEL at. Use the configured
        # value rather than inventing a budget for something we know nothing about.
        return get_settings().agent_max_tool_rounds
    return profile.tool_rounds


def cheapest_peer(model: str | None) -> str | None:
    """The cheapest model of the SAME provider, for a sub-agent to run on.

    Same provider, because a hosted main model with a local sub-agent would make the
    cheap step the slow one: a 1-second local call in front of a 2-second hosted answer
    is 50% overhead, where the same call in front of a 150-second local answer is 2%.

    The sub-agent must be LIGHTER than the model it serves. That is the whole argument
    for having one — a sub-agent that costs what the main agent costs is just two main
    agents, and the user waits for both.
    """
    from app.agent.models import CATALOGUE

    profile = profile_for(model or get_settings().llm_model)
    if profile is None:
        return None
    peers = [
        candidate
        for candidate in CATALOGUE
        if candidate.provider == profile.provider and not candidate.preview
    ]
    if not peers:
        return None
    # Cheapest by cost for hosted models; by size for local ones, where cost is zero and
    # the smaller model is the faster one.
    cheapest = min(peers, key=lambda c: (c.cost_in, c.size_gb))
    return cheapest.name if cheapest.name != profile.name else cheapest.name
