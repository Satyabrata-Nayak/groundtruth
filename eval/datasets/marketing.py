"""Marketing campaign data: the WIDE and MESSY dataset. 44 columns, on purpose.

WHY A DELIBERATELY CONFUSING SCHEMA IS THE POINT
------------------------------------------------
The e-commerce dataset has seven obvious columns, so choosing the right one is free.
Real analytics tables do not look like that: they accumulate a decade of renamed
metrics, duplicated fields, mixed units and dead columns nobody dares delete. An agent
that succeeds on a tidy seven-column table and fails here has learned nothing
transferable, and M1 already showed the model can write correct SQL when the schema is
unambiguous -- so the untested skill is *reading a schema carefully*, and that is what
this dataset tests.

THE FOUR TRAPS, AND WHAT EACH ONE CATCHES
-----------------------------------------
    conversions vs conv        `conv` is a legacy column from an older attribution
                               model and is systematically ~15% lower. Picking it
                               gives a plausible, confidently wrong number.
    conversion_rate vs cvr     the SAME metric in different units -- a fraction
                               (0.032) and a percentage (3.2). A model that averages
                               one and reports the other is out by 100x.
    cpc vs cost_per_click      byte-identical duplicates. Harmless, but they inflate
                               the schema and punish skim-reading.
    dead columns               `account_currency` and `data_source` are constant;
                               `notes` is 95% null. Grouping by a constant returns one
                               row, and a model that reports it as a finding is
                               describing an artefact.

FAIRNESS RULE FOR THE QUESTION SET
----------------------------------
A trap the question does not let you resolve is not a test, it is a coin flip. So
questions that touch an ambiguous pair NAME the column they mean ("using the
`conversions` column"), which makes the question about attention rather than
telepathy. The genuinely ambiguous phrasings live in their own question category
where the correct behaviour is to state the assumption, and they are graded on that
rather than on a number.

THE REAL SIGNAL UNDERNEATH THE MESS
-----------------------------------
Variant B converts about 18% better than variant A, and it is a real difference over
enough rows to be found. Mobile takes the most clicks and converts the worst. Search
returns the most per unit of spend, Display the least.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from eval.datasets.base import DatasetSpec

CAMPAIGNS = 40
DAYS = 150
START = date(2024, 2, 1)

# (channel, base click-through rate, base conversion rate, revenue per conversion)
CHANNELS = (
    ("Search", 0.045, 0.038, 92.0),
    ("Social", 0.021, 0.022, 61.0),
    ("Display", 0.008, 0.009, 44.0),
    ("Email", 0.062, 0.051, 78.0),
    ("Affiliate", 0.034, 0.041, 70.0),
)

# Cost per click by channel. Together with each channel's conversion rate and revenue
# per conversion, these set the return on ad spend, and the ordering they produce is
# what the benchmark questions ask about:
#
#     Email ~6.0x   Affiliate ~2.5x   Search ~2.1x   Social ~1.7x   Display ~0.8x
#
# Email's CPC was originally 0.11, which made its ROAS 32x -- true to the arithmetic
# and useless as data, because no channel returns 32x and a question whose answer is
# absurd tests nothing. Display keeps the cheapest clicks and still returns less than
# it costs, which is the genuinely interesting shape here: cheap traffic is not the
# same as good traffic.
CHANNEL_CPC = {
    "Search": 1.45,
    "Social": 0.72,
    "Display": 0.38,
    "Email": 0.58,
    "Affiliate": 0.95,
}

# THE PLANTED A/B RESULT: B converts ~18% better than A.
VARIANT_CONVERSION_MULTIPLIER = {"A": 1.0, "B": 1.18}

# Mobile attracts clicks and converts badly -- the classic mobile funnel gap.
DEVICE_WEIGHT = {"Mobile": 1.6, "Desktop": 1.0, "Tablet": 0.3}
DEVICE_CTR_MULTIPLIER = {"Mobile": 1.35, "Desktop": 1.0, "Tablet": 0.85}
DEVICE_CONVERSION_MULTIPLIER = {"Mobile": 0.62, "Desktop": 1.0, "Tablet": 0.88}

AUDIENCES = ("Lookalike", "Retargeting", "Broad", "Interest", "Custom")
COUNTRIES = ("US", "GB", "DE", "IN", "CA", "AU")
BID_STRATEGIES = ("Manual CPC", "Target CPA", "Maximize Clicks", "Target ROAS")
PLACEMENTS = ("Feed", "Sidebar", "In-Stream", "Search Results", "Native")
CREATIVE_FORMATS = ("Image", "Video", "Carousel", "Text")

# `conv` came from an attribution model that missed roughly this share of conversions.
LEGACY_ATTRIBUTION_FACTOR = 0.85

NOTES_FILL_RATE = 0.05
LEGACY_ID_FILL_RATE = 0.40

COLUMNS = (
    "row_id", "date", "week_number", "campaign_id", "campaign_name", "channel",
    "ad_group", "variant", "audience_segment", "device", "country", "placement",
    "creative_id", "creative_format", "bid_strategy",
    "impressions", "clicks", "ctr",
    "conversions", "conv", "conversion_rate", "cvr",
    "spend", "cpc", "cost_per_click", "budget_daily",
    "revenue", "roas",
    "sessions", "new_users", "returning_users", "bounce_rate",
    "avg_session_sec", "pages_per_session",
    "video_views", "video_completions",
    "engagement_score", "quality_score",
    "is_active", "attribution_window",
    "account_currency", "data_source",
    "legacy_campaign_id", "notes",
)

NOTE_TEXTS = (
    "budget increased mid-flight",
    "creative refresh",
    "paused for review",
    "tracking pixel reinstalled",
    "seasonal push",
)


def build(destination: Path, seed: int) -> int:
    rng = random.Random(seed)

    campaigns = []
    for index in range(CAMPAIGNS):
        channel, base_ctr, base_cvr, revenue_per_conversion = CHANNELS[index % len(CHANNELS)]

        # Audience must vary INDEPENDENTLY of channel. Both lists have five entries, so
        # the original `AUDIENCES[index % 5]` paired audience i with channel i for
        # every campaign -- making audience_segment a perfect alias of channel, and
        # "which audience performs best?" a differently-worded copy of "which channel
        # performs best?". Dividing before the modulo changes the stride, so each
        # channel meets every audience.
        audience = AUDIENCES[(index // len(CHANNELS)) % len(AUDIENCES)]

        campaigns.append(
            {
                "id": f"CMP-{1000 + index}",
                "name": f"{channel} {audience} Q{index % 4 + 1}",
                "channel": channel,
                "base_ctr": base_ctr,
                "base_cvr": base_cvr,
                "revenue_per_conversion": revenue_per_conversion,
                # A per-campaign quality factor, so campaigns within a channel differ
                # and "which campaign performed best" is not answered by channel alone.
                "quality": rng.uniform(0.75, 1.3),
                "audience": audience,
            }
        )

    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)

        for day_offset in range(DAYS):
            current = START + timedelta(days=day_offset)
            for campaign in campaigns:
                rows += 1
                device = _weighted_choice(rng, list(DEVICE_WEIGHT.items()))
                variant = "A" if rng.random() < 0.5 else "B"

                impressions = int(rng.uniform(2_000, 60_000) * campaign["quality"])
                ctr = max(
                    0.0005,
                    campaign["base_ctr"]
                    * DEVICE_CTR_MULTIPLIER[device]
                    * rng.uniform(0.7, 1.3),
                )
                clicks = max(1, int(impressions * ctr))

                conversion_rate_true = max(
                    0.0002,
                    campaign["base_cvr"]
                    * VARIANT_CONVERSION_MULTIPLIER[variant]
                    * DEVICE_CONVERSION_MULTIPLIER[device]
                    * campaign["quality"]
                    * rng.uniform(0.75, 1.25),
                )
                conversions = _binomial(rng, clicks, min(conversion_rate_true, 0.95))

                # The legacy column: the same conversions seen through an older
                # attribution model that under-counted.
                legacy_conversions = int(round(conversions * LEGACY_ATTRIBUTION_FACTOR))

                # Reported rate is derived from the ACTUAL counts, not from the
                # probability used to draw them -- otherwise the rate and the counts
                # would disagree and every question about either would be unanswerable.
                observed_rate = conversions / clicks if clicks else 0.0

                cpc = round(CHANNEL_CPC[campaign["channel"]] * rng.uniform(0.8, 1.25), 4)
                spend = round(clicks * cpc, 2)
                revenue = round(
                    conversions * campaign["revenue_per_conversion"] * rng.uniform(0.85, 1.2), 2
                )
                roas = round(revenue / spend, 4) if spend else 0.0

                sessions = max(clicks, int(clicks * rng.uniform(1.0, 1.4)))
                new_users = int(sessions * rng.uniform(0.35, 0.75))
                video_views = int(clicks * rng.uniform(0.0, 2.5))

                writer.writerow(
                    [
                        rows,
                        current.isoformat(),
                        current.isocalendar().week,
                        campaign["id"],
                        campaign["name"],
                        campaign["channel"],
                        f"{campaign['id']}-AG{rng.randrange(1, 5)}",
                        variant,
                        campaign["audience"],
                        device,
                        rng.choice(COUNTRIES),
                        rng.choice(PLACEMENTS),
                        f"CR-{rng.randrange(100, 400)}",
                        rng.choice(CREATIVE_FORMATS),
                        rng.choice(BID_STRATEGIES),
                        impressions,
                        clicks,
                        round(clicks / impressions, 6) if impressions else 0.0,
                        conversions,
                        legacy_conversions,
                        round(observed_rate, 6),           # fraction, 0-1
                        round(observed_rate * 100, 4),     # SAME metric as a percentage
                        f"{spend:.2f}",
                        f"{cpc:.4f}",
                        f"{cpc:.4f}",                      # exact duplicate of cpc
                        round(rng.uniform(50, 900), 2),
                        f"{revenue:.2f}",
                        roas,
                        sessions,
                        new_users,
                        sessions - new_users,
                        round(rng.uniform(0.18, 0.78), 4),
                        round(rng.uniform(25, 420), 1),
                        round(rng.uniform(1.1, 6.5), 2),
                        video_views,
                        int(video_views * rng.uniform(0.1, 0.6)),
                        round(rng.uniform(0, 100), 2),
                        rng.randrange(1, 11),
                        "true" if rng.random() < 0.92 else "false",
                        "28d",
                        "USD",          # constant
                        "ads_api_v2",   # constant
                        f"LEG{rng.randrange(1000, 9999)}"
                        if rng.random() < LEGACY_ID_FILL_RATE
                        else "",
                        rng.choice(NOTE_TEXTS) if rng.random() < NOTES_FILL_RATE else "",
                    ]
                )

    return rows


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in options)
    threshold = rng.random() * total
    running = 0.0
    for value, weight in options:
        running += weight
        if threshold <= running:
            return value
    return options[-1][0]


def _binomial(rng: random.Random, trials: int, probability: float) -> int:
    """Number of successes in `trials` independent draws.

    Drawn properly rather than as `round(trials * probability)` because the A/B test
    has to be a real statistical question. A deterministic product would make every
    conversion rate exactly its parameter, and "is B better than A" would stop being
    something to measure and become something to read off. Capped at 4,000 trials so
    a very large click count cannot make generation slow.
    """
    if trials > 4000:
        scale = trials / 4000
        return int(round(sum(1 for _ in range(4000) if rng.random() < probability) * scale))
    return sum(1 for _ in range(trials) if rng.random() < probability)


SPEC = DatasetSpec(
    name="marketing",
    description=(
        "6,000 daily marketing campaign rows across 44 columns, including duplicated "
        "metrics, mixed units, constant columns and mostly-empty legacy fields."
    ),
    seed=20240517,
    planted_effects=(
        "Variant B has a conversion rate about 20% higher than variant A "
        "(3.79% against 3.15%), over roughly 3,000 rows each.",
        "`conv` is a legacy attribution column about 15% lower than `conversions`; "
        "the two disagree for almost every row.",
        "`conversion_rate` is a fraction (0-1) and `cvr` is the same metric as a "
        "percentage (0-100).",
        "`cost_per_click` is an exact duplicate of `cpc`.",
        "Mobile receives the most clicks and has the lowest conversion rate.",
        "Email has the highest return on ad spend and Display the lowest -- Display "
        "returns less than it costs, despite having the cheapest clicks.",
        "`account_currency` and `data_source` are constant across every row.",
        "`notes` is about 95% empty and `legacy_campaign_id` about 60% empty.",
        "Email has the highest click-through rate; Display has the lowest.",
    ),
    build=build,
)
