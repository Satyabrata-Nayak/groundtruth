"""E-commerce orders: the clean dataset, carrying a multi-step diagnostic story.

THE CENTRAL PLANTED EFFECT
--------------------------
Q3 revenue is HIGHER than Q2 while Q3 profit is LOWER. That is the shape of a real
analytical question, and it cannot be answered in one query. Finding it requires:

    quarterly revenue      -> up
    quarterly profit       -> down                (so the premise is real)
    margin by quarter      -> falling             (so it is not a volume story)
    mix / discount by qtr  -> Electronics grows, discounts deepen   (the cause)

Two independent causes are planted, because a single-cause story lets a model stop at
the first thing it finds and still look right:

    1. MIX SHIFT. Electronics carries a ~12% gross margin against ~45% for Apparel and
       Books. Its share of orders roughly doubles in Q3.
    2. DISCOUNTING. Mean discount rises from ~5% in Q1/Q2 to ~14% in Q3.

THE DELIBERATE TRAP
-------------------
West has the HIGHEST total revenue and the LOWEST profit margin. "Which region
performs best?" therefore has two defensible answers and one lazy one. A model that
reports revenue alone gets a different region than one that checks profitability, and
the question set asks both forms separately so the difference is measurable rather
than a matter of opinion.

OTHER EFFECTS, EACH SUPPORTING AT LEAST ONE QUESTION
----------------------------------------------------
    returns          concentrated in Apparel (~15%) against <6% elsewhere
    customer_segment VIP orders are ~2.6x the value of New
    shipping_cost    ~2% null, so a data-quality question has a real answer
    channel          Partner is small but high-margin, so "biggest" != "best"
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from eval.datasets.base import DatasetSpec, quarter_of

ROWS = 5000
START = date(2024, 1, 1)
DAYS = 366  # 2024 is a leap year

REGIONS = ("North", "South", "East", "West")
CHANNELS = ("Online", "Retail", "Partner")
SEGMENTS = ("New", "Returning", "VIP")

# (category, gross margin, base unit price, popularity weight)
# The margin column is the engine of the whole story: shifting the popularity of a
# low-margin category between quarters is what makes profit fall while revenue rises.
CATEGORIES = (
    ("Electronics", 0.12, 320.0, 1.0),
    ("Home & Garden", 0.32, 85.0, 1.0),
    ("Apparel", 0.45, 55.0, 1.0),
    ("Sports", 0.28, 120.0, 1.0),
    ("Books", 0.46, 18.0, 1.0),
)

# Multiplier on each category's popularity, per quarter. Q3 doubles Electronics and
# suppresses the high-margin categories -- the mix shift, stated as data.
QUARTER_MIX: dict[int, dict[str, float]] = {
    1: {"Electronics": 1.0, "Home & Garden": 1.0, "Apparel": 1.2, "Sports": 1.0, "Books": 1.0},
    2: {"Electronics": 1.1, "Home & Garden": 1.2, "Apparel": 1.1, "Sports": 1.3, "Books": 0.9},
    3: {"Electronics": 2.2, "Home & Garden": 0.9, "Apparel": 0.6, "Sports": 0.8, "Books": 0.6},
    4: {"Electronics": 1.4, "Home & Garden": 1.0, "Apparel": 1.4, "Sports": 0.9, "Books": 1.3},
}

# Mean discount by quarter. The second cause of the Q3 margin fall.
QUARTER_DISCOUNT: dict[int, float] = {1: 0.05, 2: 0.05, 3: 0.14, 4: 0.09}

# Regional revenue weight, and a margin adjustment. West sells the most and keeps the
# least -- the trap.
REGION_WEIGHT = {"North": 1.0, "South": 0.85, "East": 0.75, "West": 1.45}
REGION_MARGIN_DELTA = {"North": 0.02, "South": 0.01, "East": 0.03, "West": -0.07}

CHANNEL_WEIGHT = {"Online": 1.0, "Retail": 0.55, "Partner": 0.18}
CHANNEL_MARGIN_DELTA = {"Online": 0.0, "Retail": -0.02, "Partner": 0.10}

SEGMENT_WEIGHT = {"New": 1.0, "Returning": 0.9, "VIP": 0.25}
SEGMENT_VALUE_MULTIPLIER = {"New": 1.0, "Returning": 1.35, "VIP": 3.0}

RETURN_RATE = {
    "Apparel": 0.18,
    "Electronics": 0.05,
    "Home & Garden": 0.03,
    "Sports": 0.04,
    "Books": 0.02,
}

SHIPPING_NULL_RATE = 0.02

COLUMNS = (
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "category",
    "channel",
    "customer_segment",
    "units",
    "unit_price",
    "discount_pct",
    "revenue",
    "cost",
    "shipping_cost",
    "returned",
)


def _weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    """Pick one option, probability proportional to its weight.

    Written out rather than using `random.choices` so the number of `rng` draws per
    row is fixed and obvious -- the whole file's determinism depends on the call
    sequence, and a helper that sometimes draws twice would silently break it.
    """
    total = sum(weight for _, weight in options)
    threshold = rng.random() * total
    running = 0.0
    for value, weight in options:
        running += weight
        if threshold <= running:
            return value
    return options[-1][0]


def build(destination: Path, seed: int) -> int:
    rng = random.Random(seed)
    margins = {name: margin for name, margin, _, _ in CATEGORIES}
    prices = {name: price for name, _, price, _ in CATEGORIES}

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)

        for order_id in range(1, ROWS + 1):
            order_date = START + timedelta(days=rng.randrange(DAYS))
            quarter = quarter_of(order_date.month)

            region = _weighted_choice(rng, list(REGION_WEIGHT.items()))
            channel = _weighted_choice(rng, list(CHANNEL_WEIGHT.items()))
            segment = _weighted_choice(rng, list(SEGMENT_WEIGHT.items()))

            mix = QUARTER_MIX[quarter]
            category = _weighted_choice(
                rng, [(name, weight * mix[name]) for name, _, _, weight in CATEGORIES]
            )

            units = rng.randint(1, 5)
            unit_price = round(
                prices[category]
                * SEGMENT_VALUE_MULTIPLIER[segment]
                * rng.uniform(0.8, 1.25),
                2,
            )

            # Discount is drawn around the quarter's mean, clamped so it stays a
            # discount: a negative one would silently become a price increase.
            discount = max(0.0, min(0.6, rng.gauss(QUARTER_DISCOUNT[quarter], 0.04)))
            discount = round(discount, 3)

            gross = units * unit_price
            revenue = round(gross * (1 - discount), 2)

            # Cost is derived from the margin the category is *supposed* to earn, then
            # nudged per region and channel. Cost does not fall with the discount,
            # which is exactly why discounting destroys margin.
            margin = (
                margins[category]
                + REGION_MARGIN_DELTA[region]
                + CHANNEL_MARGIN_DELTA[channel]
                + rng.gauss(0.0, 0.03)
            )
            cost = round(gross * (1 - max(0.02, margin)), 2)

            shipping = "" if rng.random() < SHIPPING_NULL_RATE else round(
                rng.uniform(3.5, 24.0), 2
            )
            returned = rng.random() < RETURN_RATE[category]

            writer.writerow(
                [
                    order_id,
                    order_date.isoformat(),
                    f"C{rng.randrange(1, 1201):05d}",
                    region,
                    category,
                    channel,
                    segment,
                    units,
                    f"{unit_price:.2f}",
                    f"{discount:.3f}",
                    f"{revenue:.2f}",
                    f"{cost:.2f}",
                    shipping,
                    "true" if returned else "false",
                ]
            )

    return ROWS


SPEC = DatasetSpec(
    name="ecommerce",
    description=(
        "5,000 e-commerce orders across 2024, with region, product category, sales "
        "channel, customer segment, revenue, cost, discount and return flag."
    ),
    seed=20240301,
    planted_effects=(
        "Q3 revenue is higher than Q2 (793k against 576k) but Q3 profit is far "
        "lower (20k against 97k): margin falls from 16.8% to 2.6%.",
        "The Q3 margin fall has two causes: Electronics (low margin) roughly doubles "
        "its share of orders, and the mean discount rises from ~5% to ~14%.",
        "West has the highest total revenue and the lowest profit margin.",
        "Electronics has the lowest gross margin of any category; Books the highest.",
        "Apparel is returned far more often than anything else: ~15.3% against "
        "~5.6% for Electronics and under 4% for everything else.",
        "VIP customers place orders worth about 2.6x those of New customers "
        "(974 against 377 on average).",
        "Partner is the smallest channel by revenue and the most profitable by margin.",
        "About 2% of orders have no shipping_cost recorded.",
    ),
    build=build,
)
