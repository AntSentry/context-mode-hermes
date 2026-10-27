"""Fail CI when a JUnit XML report shows skips, failures, errors or zero tests.

The real-engine integration suite must never turn a missing engine
prerequisite into a silent green skip, so a passing run requires that every
collected test actually ran and passed. Stdlib only; run with any Python.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

USAGE = "usage: assert_junit_all_passed.py <junit.xml>"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        root = ET.parse(argv[1]).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"FAIL: cannot read JUnit XML {argv[1]}: {exc}", file=sys.stderr)
        return 2

    totals = {"tests": 0, "skipped": 0, "failures": 0, "errors": 0}
    for suite in root.iter("testsuite"):
        for key in totals:
            totals[key] += int(suite.get(key, 0))

    problems = []
    if totals["tests"] <= 0:
        problems.append("no tests were collected")
    for key in ("skipped", "failures", "errors"):
        if totals[key]:
            problems.append(f"{totals[key]} {key}")
    if problems:
        print(f"FAIL: {'; '.join(problems)} (totals: {totals})", file=sys.stderr)
        return 1
    print(f"OK: {totals['tests']} tests ran and passed with no skips")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
