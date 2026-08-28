"""Render every benchmark run in scripts/_bench_raw/ as one Markdown comparison table.

    .venv/Scripts/python.exe scripts/bench_report.py

Why a script rather than typing the table into docs/benchmarking.md by hand: a
hand-copied number is a number that can silently drift from the run that produced it.
The docs table is generated, so it cannot disagree with the raw JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RAW = Path(__file__).parent / "_bench_raw"

# (row label, dotted path into the results dict, format spec)
ROWS: list[tuple[str, str, str]] = [
    ("tokens/sec (median)", "latency.tokens_per_sec_median", "{:.1f}"),
    ("ttft - any token (s)", "latency.ttft_any_median_s", "{:.2f}"),
    ("ttft - answer token (s)", "latency.ttft_answer_median_s", "{:.2f}"),
    ("wall per call (s, median)", "latency.wall_median_s", "{:.1f}"),
    ("thinking share of output", "latency.thinking_share_median", "{:.0%}"),
    ("cold load (s)", "latency.cold_load_s", "{:.2f}"),
    ("JSON valid - free-form", "structured.freeform.pct", "{:.0f}%"),
    ("JSON valid - constrained", "structured.constrained.pct", "{:.0f}%"),
    ("tool call emitted", "tools.emitted", "{:.0f}/15"),
    ("correct tool chosen", "tools.correct_tool", "{:.0f}/15"),
    ("tool + valid args", "tools.valid_args", "{:.0f}/15"),
    ("SQL correct on execution", "sql.correct", "{:.0f}/10"),
    ("SQL emitted cleanly", "sql.clean_output", "{:.0f}/10"),
    ("benchmark wall time (s)", "total_bench_s", "{:.0f}"),
]


def dig(d: dict[str, Any], path: str) -> Any:
    """Follow a dotted path, returning None if any segment is missing.

    Missing rather than raising, so a partially-completed or older run still renders.
    """
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    files = sorted(RAW.glob("*.json"))
    if not files:
        print(f"No runs found in {RAW}. Run scripts/bench_model.py first.")
        return 1

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    headers = [
        f"{r['model']}{'' if r.get('reasoning') == 'default' else ' --no-think'}" for r in runs
    ]

    width = max(len(label) for label, _, _ in ROWS)
    print(f"| {'Metric'.ljust(width)} | " + " | ".join(headers) + " |")
    print(f"|{'-' * (width + 2)}|" + "|".join("-" * (len(h) + 2) for h in headers) + "|")

    for label, path, fmt in ROWS:
        cells = []
        for r in runs:
            v = dig(r, path)
            cells.append("—" if v is None else fmt.format(v))
        print(
            f"| {label.ljust(width)} | "
            + " | ".join(c.ljust(len(h)) for c, h in zip(cells, headers, strict=True))
            + " |"
        )

    print("\nFailure samples (first 3 per section):")
    for r, h in zip(runs, headers, strict=True):
        print(f"\n  {h}")
        for section in ("structured.freeform", "structured.constrained", "tools", "sql"):
            fails = dig(r, f"{section}.failures") or []
            if fails:
                print(f"    {section}: " + "; ".join(fails[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
