"""The evaluation datasets, and the registry that names them.

    ecommerce   5,000 orders, 14 columns   -- clean; tests multi-step diagnosis
    marketing   6,000 rows, 44 columns     -- messy; tests reading a schema carefully
    sensors     17,280 readings, 11 cols   -- hourly; tests reasoning over time

Adding a fourth dataset is a new module exposing a `SPEC`, plus one line here. That is
deliberate: the runner, the question loader and the expected-answer builder are all
dataset-agnostic, so the cost of covering a new data shape is writing the data and the
questions, never changing the harness.
"""

from __future__ import annotations

from pathlib import Path

from eval.datasets import ecommerce, marketing, sensors
from eval.datasets.base import DatasetSpec

SPECS: tuple[DatasetSpec, ...] = (ecommerce.SPEC, marketing.SPEC, sensors.SPEC)
BY_NAME: dict[str, DatasetSpec] = {spec.name: spec for spec in SPECS}


def get_spec(name: str) -> DatasetSpec:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown evaluation dataset '{name}'. Known: {', '.join(sorted(BY_NAME))}"
        ) from None


def generate_all(directory: Path) -> dict[str, Path]:
    """Write every dataset's CSV into `directory`, returning name -> path."""
    return {spec.name: spec.generate(directory) for spec in SPECS}


__all__ = ["BY_NAME", "SPECS", "DatasetSpec", "generate_all", "get_spec"]
