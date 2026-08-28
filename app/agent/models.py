"""The models a user may choose between, and what is actually known about each.

WHY A CATALOGUE AND NOT JUST A FREE-TEXT MODEL NAME
---------------------------------------------------
`LLM_MODEL` already lets an operator point at any model Ollama has. That is the right
control for an operator and the wrong one for a user: "qwen2.5:3b-instruct" tells you
nothing about whether your question will be answered well, and the honest answer to
"which should I pick?" is a table of measurements, not a text box.

So the choice is presented as two named options with their real numbers attached. Every
figure below was measured on this project's own evaluation set — the same golden
questions, the same grader, the same machine — and not taken from a model card.

WHAT THE MEASUREMENTS SAY
-------------------------
On the 20-question ecommerce slice, graded by executing reference SQL:

    qwen3:4b              60% correct     ~150 s per question
    qwen2.5:3b-instruct   ~33% correct    ~3 s per question

Fifty times faster for roughly half the accuracy. That is a real trade with no right
answer, which is exactly the kind of decision that belongs to the person asking.

The per-category numbers matter more than the total, because they say WHICH questions
each model can be trusted with:

                     qwen3:4b     qwen2.5:3b-instruct
    lookup            100%              75%
    aggregation       100%              58%
    trend              50%               0%
    data quality       50%              12%
    comparison          0%               0%
    diagnosis           0%               0%

A direct "which X has the most Y" is answered well by both, and the cheap one answers
it in three seconds. Anything needing several facts held together is where the larger
model earns its two minutes — and neither model can currently do "why" questions at
all, which is stated plainly rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """One selectable model, with what is known about it rather than what is claimed."""

    name: str
    label: str
    tagline: str
    # "ollama" runs on this machine; "groq" is a hosted API. The provider decides which
    # client is built, and — because it decides whether concurrent calls are genuinely
    # concurrent — it also decides the sub-agent policy. See `fans_out`.
    provider: str = "ollama"
    # Why you would pick this one, and what it will do badly. Both, always: a chooser
    # that only lists strengths is an advert, and the user is choosing between two
    # things that are each bad at something.
    good_at: list[str] = field(default_factory=list)
    weak_at: list[str] = field(default_factory=list)
    typical_seconds: tuple[int, int] = (0, 0)
    accuracy_pct: int | None = None
    # Whether the model emits a reasoning trace at all. Decides whether the thinking
    # toggle means anything for it.
    reasons: bool = False
    size_gb: float = 0.0
    # Dollars per million tokens, in and out. Zero for a local model, which is the whole
    # argument for keeping one: a question costs electricity rather than money.
    cost_in: float = 0.0
    cost_out: float = 0.0
    # Preview models carry no production guarantee and can be retired without notice.
    preview: bool = False
    # How many tool rounds are affordable. Locally a round is 45-90 s, so two is the
    # limit before a user gives up; on Groq a round is under a second, so the agent can
    # afford to look, think, and look again — which is where analytical depth comes from.
    tool_rounds: int = 2

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    @property
    def fans_out(self) -> bool:
        """May this model afford a parallel sub-agent?

        THE ANSWER IS A PROPERTY OF THE PROVIDER, NOT OF THE MODEL. On one local GPU two
        concurrent calls do not run in parallel — they split one card's tokens per second
        and add KV-cache pressure, so fan-out is strictly a latency loss. Against a hosted
        API the calls really are concurrent and the same design becomes free.

        This is why the sub-agent policy is computed rather than configured: the same
        architecture is right in one deployment and wrong in the other.
        """
        return not self.is_local

    @property
    def cost_label(self) -> str:
        if self.is_local:
            return "free"
        return f"${self.cost_in:g}/${self.cost_out:g} per 1M"

    @property
    def speed_label(self) -> str:
        low, high = self.typical_seconds
        if high < 10:
            return f"~{high}s"
        return f"{low // 60}-{high // 60} min" if low >= 60 else f"{low}-{high}s"


CATALOGUE: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="openai/gpt-oss-20b",
        label="GPT-OSS 20B",
        tagline="Hosted on Groq. Answers in seconds, and the only production-grade option here.",
        provider="groq",
        good_at=[
            "Speed — about 1,000 tokens/second, so a full answer in a few seconds",
            "A 131,072-token context: the whole schema and long results fit easily",
            "Multi-step questions, because it can afford several rounds of querying",
        ],
        weak_at=[
            "Needs GROQ_API_KEY and sends your tool RESULTS (not your file) off-machine",
            "Costs money, though very little: about $0.075 per million input tokens",
        ],
        typical_seconds=(2, 8),
        reasons=True,
        cost_in=0.075,
        cost_out=0.30,
        # Cheap and fast enough that the agent can look, think and look again — which is
        # where analytical depth comes from, and what two local rounds cannot buy.
        tool_rounds=4,
    ),
    ModelProfile(
        name="qwen/qwen3.6-27b",
        label="Qwen3.6 27B",
        tagline="Hosted on Groq. A larger model for questions the others get wrong.",
        provider="groq",
        good_at=[
            "More capable on reasoning and code than the 20B",
            "Still fast — around 500 tokens/second",
        ],
        weak_at=[
            "PREVIEW on Groq: no production guarantee, and it can be retired",
            "Eight times the input cost of GPT-OSS 20B ($0.60 per 1M)",
        ],
        typical_seconds=(3, 12),
        reasons=True,
        cost_in=0.60,
        cost_out=3.00,
        preview=True,
        tool_rounds=4,
    ),
    ModelProfile(
        name="qwen/qwen3.8-27b",
        label="Qwen3.8 27B",
        tagline="Hosted on Groq. The newest Qwen, aimed at agentic work.",
        provider="groq",
        good_at=[
            "The newest of the three, and built for tool-driven work",
            "131,072-token context",
        ],
        weak_at=[
            "PREVIEW on Groq: no production guarantee, and it can be retired",
            "The most expensive here — $0.80 in, $4.00 out per 1M — and the slowest of "
            "the hosted three",
        ],
        typical_seconds=(3, 14),
        reasons=True,
        cost_in=0.80,
        cost_out=4.00,
        preview=True,
        tool_rounds=4,
    ),
    ModelProfile(
        name="qwen3:4b",
        label="Qwen3 4B",
        tagline="Thinks before answering. Slower, and better at questions with more than one part.",
        good_at=[
            "Rankings and totals — 100% on the evaluation set",
            "Direct lookups — 100%",
            "Holding several facts together in one answer",
        ],
        weak_at=[
            "Takes 1-3 minutes per question on a laptop GPU",
            "“Why” questions — 0% on the diagnosis category",
        ],
        typical_seconds=(90, 190),
        accuracy_pct=60,
        reasons=True,
        size_gb=2.5,
    ),
    ModelProfile(
        name="qwen2.5:3b-instruct",
        label="Qwen2.5 3B",
        tagline="Answers in seconds. No reasoning step, so it is best on questions with one part.",
        good_at=[
            "Speed — about 3 seconds instead of two minutes",
            "Direct lookups — 75% on the evaluation set",
            "Trying several phrasings of a question quickly",
        ],
        weak_at=[
            "Roughly half the overall accuracy of Qwen3 (29% vs 60%)",
            "Multi-part questions, trends and data-quality checks",
        ],
        typical_seconds=(2, 5),
        accuracy_pct=29,
        reasons=False,
        size_gb=1.9,
    ),
)

BY_NAME: dict[str, ModelProfile] = {profile.name: profile for profile in CATALOGUE}


def profile_for(name: str) -> ModelProfile | None:
    """What is known about a model name, or None for one we have not measured.

    Returning None rather than inventing a profile is deliberate: an operator who points
    `LLM_MODEL` at something else gets a working system and no fabricated claims about
    how accurate it is.
    """
    return BY_NAME.get(name)


def is_selectable(name: str) -> bool:
    """May a REQUEST name this model?

    The catalogue is an allowlist. `model` arrives in a JSON body from a browser, and
    the alternative — passing it through to Ollama — would let a request pull and run
    any model on the host, which is a resource-exhaustion hole rather than a feature.
    An operator can still run anything via `LLM_MODEL`; a request cannot.
    """
    return name in BY_NAME
