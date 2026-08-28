"""What an evaluation dataset is, and why they are generated rather than downloaded.

THE CASE FOR SYNTHETIC DATA HERE
--------------------------------
A benchmark question needs a known-correct answer. For a Kaggle CSV, "correct" means
whatever a query returns -- which makes the reference SQL both the question and the
answer, and tests nothing except that DuckDB is deterministic.

Generating the data inverts that. The effect is decided first ("Q3 profit falls while
revenue rises"), the data is built to contain it, and the question then has an answer
that exists independently of how it is queried. `planted_effects` is that list: it is
the specification the questions are written against, and every entry is something a
competent analyst should be able to find.

The generator being checked in matters as much as the data:

    a committed CSV        opaque    "why is West's margin low?" -- nobody knows
    a committed generator  legible   the parameter that made it low is on line 60

DETERMINISM
-----------
Each spec carries a fixed seed and generators use only `random.Random(seed)`, whose
core methods are stable across CPython versions. The same commit therefore produces
byte-identical data on any machine, which is what allows expected answers to be
computed once and checked in. A generator that used `numpy` or unseeded `random`
would make every expected value a machine-specific claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """One evaluation dataset: how to build it, and what is true about it."""

    name: str
    description: str
    seed: int
    # Written in analyst's language, not implementation language. These are the
    # findings the question set is allowed to ask for.
    planted_effects: tuple[str, ...]
    build: Callable[[Path, int], int]  # (destination_csv, seed) -> row count

    def generate(self, directory: Path) -> Path:
        """Write this dataset's CSV into `directory` and return its path."""
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{self.name}.csv"
        self.build(destination, self.seed)
        return destination


def quarter_of(month: int) -> int:
    return (month - 1) // 3 + 1
