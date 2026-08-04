#!/usr/bin/env python3
"""Check every claim this project makes, in one command.

Run it:

    python verify.py

The point is that the README's claims are not asked to be believed. Each of the
checks below corresponds to something stated there, and the exit code is the
answer. A claim with no check next to it should be treated as decoration.

Use ``--fast`` to skip the calibration sweep when iterating; the calibration is
the slow part and also the part that matters, so it runs by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent

EFFICIENCY_TOLERANCE = 1e-9
"""Shapley efficiency is exact in algebra; this allows only for float rounding."""


@dataclass(frozen=True, slots=True)
class Check:
    """One verifiable claim.

    Attributes:
        claim: What the project asserts, in the words a reader would use.
        command: The command that settles it.
        slow: Whether it is skipped under ``--fast``.
    """

    claim: str
    command: list[str]
    slow: bool = False


CHECKS: tuple[Check, ...] = (
    Check(
        "the source passes its linter with no exemptions beyond those in pyproject",
        [sys.executable, "-m", "ruff", "check", "src", "tests", "verify.py"],
    ),
    Check(
        "the source is formatted",
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "verify.py"],
    ),
    Check(
        "the source and the tests both type-check under mypy --strict",
        [sys.executable, "-m", "mypy"],
    ),
    Check(
        "the test suite passes",
        [sys.executable, "-m", "pytest", "-q", "-m", "not slow"],
    ),
    Check(
        "the instrument recovers planted artifacts and stays silent on clean models",
        [sys.executable, "-m", "pytest", "-q", "-m", "slow", "-k", "calibration"],
        slow=True,
    ),
    Check(
        "the Shapley shares sum to the observed drop, and the axioms hold",
        [sys.executable, "-m", "pytest", "-q", "tests/test_stats_shapley.py"],
    ),
    Check(
        "the package installs on numpy and scipy alone; the rest stay optional",
        [sys.executable, "-m", "pytest", "-q", "tests/test_optional_dependencies.py"],
    ),
    Check(
        "no scorer reads the answer key",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_score.py::test_a_scorer_never_reads_the_answer_key",
        ],
    ),
)


def _run(check: Check) -> tuple[bool, float, str]:
    """Run one check.

    Args:
        check: The check.

    Returns:
        Whether it passed, how long it took, and its combined output.
    """
    started = time.monotonic()
    completed = subprocess.run(
        check.command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, elapsed, output


def _check_demo_is_reproducible() -> tuple[bool, str]:
    """Run the demo twice and require byte-identical machine-readable output.

    Returns:
        Whether the outputs matched, and a description of the result.
    """
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"
        for destination in (first, second):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evalassay.cli",
                    "demo",
                    "--bootstrap",
                    "2000",
                    "--json",
                    str(destination),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return False, (completed.stdout + completed.stderr).strip()

        left = first.read_text(encoding="utf-8")
        right = second.read_text(encoding="utf-8")
        if left != right:
            return False, "two runs of the same audit produced different output"

        parsed = json.loads(left)
        shares = sum(c["estimate"]["point"] for c in parsed["components"])
        if abs(shares - parsed["total_drop"]) > EFFICIENCY_TOLERANCE:
            return False, f"shares sum to {shares}, total drop is {parsed['total_drop']}"
        return True, "identical output, and the shares sum to the observed drop"


def main() -> int:
    """Run every check and report.

    Returns:
        Zero if every check passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the calibration sweep")
    args = parser.parse_args()

    selected = [check for check in CHECKS if not (args.fast and check.slow)]
    width = 74
    print("Verifying EvalAssay")
    print("=" * width)

    failures: list[tuple[str, str]] = []
    for index, check in enumerate(selected, start=1):
        print(f"[{index}/{len(selected) + 1}] {check.claim}")
        passed, elapsed, output = _run(check)
        if passed:
            print(f"      PASS ({elapsed:.1f}s)")
        else:
            print(f"      FAIL ({elapsed:.1f}s)")
            failures.append((check.claim, output))

    print(f"[{len(selected) + 1}/{len(selected) + 1}] two runs of the same audit agree exactly")
    reproducible, detail = _check_demo_is_reproducible()
    if reproducible:
        print(f"      PASS  {detail}")
    else:
        print("      FAIL")
        failures.append(("two runs of the same audit agree exactly", detail))

    print("=" * width)
    if not failures:
        print(f"All {len(selected) + 1} checks passed.")
        if args.fast:
            print("Calibration was skipped; run without --fast before believing anything.")
        return 0

    print(f"{len(failures)} of {len(selected) + 1} checks FAILED.\n")
    for claim, output in failures:
        print(f"--- {claim} ---")
        print(output[-3000:])
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
