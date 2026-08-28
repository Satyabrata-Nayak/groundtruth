"""`python -m app.worker` — the second process.

Same package as the API, separate process. Sharing the package means the worker and
the API cannot drift onto different models, different config or different tool
definitions; separate processes mean a slow analysis cannot block an HTTP request and
a crashed worker cannot take the API down with it.
"""

from __future__ import annotations

import argparse
import logging

from app.worker.loop import Worker


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.worker",
        description="Claim analyses from the Postgres queue and run them.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one job and exit (used by tests and by the crash drill)",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="override the generated worker id; useful when reproducing a specific run",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    worker = Worker(worker_id=args.worker_id)
    if args.once:
        # No signal handlers: a one-shot run should die instantly when killed, which
        # is exactly what the crash drill in tests/test_worker_recovery.py relies on.
        return 0 if worker.tick() else 1

    worker.install_signal_handlers()
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
