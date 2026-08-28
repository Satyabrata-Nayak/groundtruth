"""Build the evaluation environment: generate data, register it, compute ground truth.

    python -m eval.build              rebuild everything
    python -m eval.build --check      rebuild and FAIL if the checked-in answers moved

WHAT `--check` IS FOR
---------------------
The expected answers are committed. `--check` recomputes them and exits non-zero if
anything differs from what is on disk. That turns a silent change into a visible one:
edit a generator parameter and the check tells you which of the forty answers you just
invalidated, before a benchmark run reports a score against stale truth.

IDEMPOTENCE
-----------
Re-running deletes and recreates each evaluation dataset rather than adding a version.
Evaluation data is derived, not owned -- there is no history worth keeping, and
accumulating v1..v12 of an identical file would make the runner's choice of version
ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.data.service import create_dataset, delete_dataset
from app.db.models import Dataset
from app.db.session import session_scope
from eval import expected as expected_module
from eval.datasets import SPECS
from eval.suite import load_suite

DATA_DIR = Path(__file__).parent / "data"


def log(message: str) -> None:
    print(message, flush=True)


def _drop_existing(name: str) -> None:
    """Remove any previously registered evaluation dataset with this name."""
    with session_scope() as session:
        existing = list(session.scalars(select(Dataset.id).where(Dataset.name == name)))
    for dataset_id in existing:
        delete_dataset(dataset_id)


def build(*, check: bool) -> int:
    suite = load_suite()
    log(f"question set: {len(suite.questions)} questions across {len(SPECS)} datasets")
    for category, count in suite.categories().items():
        log(f"  {category:<14} {count}")

    registry: dict[str, dict[str, object]] = {}
    for spec in SPECS:
        log(f"\n[{spec.name}] generating...")
        csv_path = spec.generate(DATA_DIR)

        _drop_existing(spec.name)
        created = create_dataset(csv_path, name=spec.name, description=spec.description)
        registry[spec.name] = {
            "dataset_id": str(created.dataset_id),
            "version": created.version,
            "row_count": created.row_count,
            "column_count": created.column_count,
        }
        log(
            f"[{spec.name}] registered {created.dataset_id} v{created.version}: "
            f"{created.row_count} rows x {created.column_count} columns"
        )

    expected_module.save_registry(registry)

    log("\ncomputing expected answers by executing reference SQL...")
    answers = expected_module.compute_all(suite, registry)
    total = sum(len(entries) for entries in answers.values())
    log(f"computed {total} expected answers")

    if check:
        drift = _find_drift(answers)
        if drift:
            log(f"\nEXPECTED ANSWERS CHANGED for {len(drift)} question(s):")
            for question_id in drift:
                log(f"  {question_id}")
            log("\nRe-run without --check to accept, then review the JSON diff.")
            return 1
        log("\nexpected answers match what is checked in")
        return 0

    written = expected_module.save(answers)
    for path in written:
        log(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return 0


def _find_drift(answers: dict[str, list[expected_module.ExpectedAnswer]]) -> list[str]:
    """Question ids whose computed answer differs from the checked-in one."""
    try:
        stored = expected_module.load()
    except FileNotFoundError:
        return []

    changed = []
    for entries in answers.values():
        for entry in entries:
            previous = stored.get(entry.question_id)
            # Compared through JSON so that float formatting is identical on both
            # sides; comparing dataclasses directly would flag 1.0 against 1 as drift.
            if previous is None or json.dumps(
                previous.to_json(), sort_keys=True, default=str
            ) != json.dumps(entry.to_json(), sort_keys=True, default=str):
                changed.append(entry.question_id)
    return sorted(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the recomputed answers differ from the checked-in ones",
    )
    args = parser.parse_args(argv)
    try:
        return build(check=args.check)
    except Exception as exc:  # noqa: BLE001 - a CLI; a traceback helps nobody here
        log(f"\nbuild failed: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
