"""Run the N1-N7 neuronal-validity tests and record the outcome as a gate.

The post-revision audit's finding: the release manifest required only D1-D5 and
REPLAY, so a release could be built while the neuronal-behaviour tests were
failing or had never been run.  Statements that N1-N7 are "automatically
enforced fail-closed" were therefore premature.

This tool closes that hole by producing the machine-readable evidence the
manifest needs.  It runs the tests, maps each one to its gate, and writes a
``by_gate`` record that ``tools/build_release_manifest.py`` ingests like any
other gate file.

Mapping is by test NAME (``test_n3_...`` -> N3) where the convention exists, and
by FILE for the one gate whose tests predate the convention (N5).  A gate with no
evidence is reported with ``n=0``, which the manifest treats as missing -- and a
missing gate is a failure, not a pass.

**N2 is deliberately absent.**  Strict reproduction of published traces is not
implemented as a test: ``compass.validation.bounds`` says so explicitly -- its
intervals are "unit-error and dead-channel detectors, not goodness-of-fit
tests", and true source reproduction "needs the matching published stimulus,
temperature and recording site".  Recording N2 as absent is the honest state and
keeps the release blocked until someone implements it, rather than letting a
descriptive model audit stand in for a gate.

    python tools/record_numerical_gates.py            # runs the fast tests
    python tools/record_numerical_gates.py --slow     # includes slow panel tests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "validation" / "numerical_gates.json"

# Gates whose tests follow the `test_n<N>_...` naming convention.
NAME_RE = re.compile(r"::test_(n[1-7])_", re.I)

# Gates whose tests predate the convention, mapped by file.
FILE_GATES = {"test_rheobase_failure_states.py": "N5"}

# Every gate the project claims to enforce.  A gate absent from the run is
# emitted with n=0 so the manifest blocks on it.
ALL_GATES = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]


class _Collector:
    """Records one row per test outcome."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001
        if report.when != "call" and not (report.when == "setup" and report.skipped):
            return
        gate = None
        m = NAME_RE.search(report.nodeid)
        if m:
            gate = m.group(1).upper()
        else:
            fname = report.nodeid.split("::")[0].split("/")[-1]
            gate = FILE_GATES.get(fname)
        if gate is None:
            return
        self.rows.append({"gate": gate, "subject": report.nodeid,
                          "outcome": report.outcome})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", action="store_true",
                    help="include tests marked slow (the full-panel N1/N4 "
                         "simulation tests). Without this the record states "
                         "that the slow tests were not exercised.")
    args = ap.parse_args()

    coll = _Collector()
    argv = ["-q", "--no-header", "-p", "no:cacheprovider",
            str(ROOT / "tests")]
    if not args.slow:
        argv += ["-m", "not slow"]
    pytest.main(argv, plugins=[coll])

    by_gate: dict[str, dict] = {}
    for g in ALL_GATES:
        rows = [r for r in coll.rows if r["gate"] == g]
        passed = [r for r in rows if r["outcome"] == "passed"]
        failed = [r for r in rows if r["outcome"] == "failed"]
        skipped = [r for r in rows if r["outcome"] == "skipped"]
        # A skip produces no evidence, so it counts neither as a pass nor as a
        # failure -- `n` is the number of tests that actually decided something.
        # Skips stay visible, and if EVERY test for a gate skips then n == 0 and
        # the manifest sees the gate as missing, which is the correct outcome.
        by_gate[g] = {
            "n": len(passed) + len(failed),
            "passed": len(passed),
            "skipped": len(skipped),
            "skipped_subjects": [r["subject"] for r in skipped],
            "failed_subjects": [r["subject"] for r in failed],
        }

    record = {
        "tool": "record_numerical_gates.py",
        "written_utc": datetime.now(UTC).isoformat(),
        "slow_tests_included": bool(args.slow),
        "note": ("N2 has no automated test: strict reproduction of published "
                 "traces is not implemented, and the canonical-bounds checks are "
                 "explicitly unit-error detectors rather than goodness-of-fit "
                 "tests. It is reported with n=0 so the release manifest treats "
                 "it as missing evidence."),
        "by_gate": by_gate,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1))

    worst = 0
    for g in ALL_GATES:
        v = by_gate[g]
        state = "MISSING" if v["n"] == 0 else (
            "pass" if v["passed"] == v["n"] else "FAIL")
        if state != "pass":
            worst = 1
        skip = f"  ({v['skipped']} skipped)" if v["skipped"] else ""
        print(f"  {g}: {v['passed']}/{v['n']} {state}{skip}")
    print(f"-> {OUT}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
