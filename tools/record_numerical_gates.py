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

**N2 is implemented for the sourceable subset.**  It reported ``0/0`` through
three audits; ``tests/integration/test_n2_source_reproduction.py`` now compares
each model's canonical operating point against a published scalar declared in
``data/provenance/n2_source_targets.json``.  It is NOT trace-level reproduction
-- the published stimulus, temperature and recording site are not distributed
with these models -- and the models with no sourceable scalar are declared not
evaluable with a reason, appearing as ``exempt`` in the validity matrix rather
than as passes.  Every claim requiring source-level reproduction is withdrawn
for those models.

    python tools/record_numerical_gates.py            # runs the fast tests
    python tools/record_numerical_gates.py --slow     # includes slow panel tests
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.validation.gates import environment_record, git_tree  # noqa: E402

OUT = ROOT / "results" / "validation" / "numerical_gates.json"

# pytest's documented exit statuses.  0 and 1 are ordinary outcomes; everything
# else means the run itself is untrustworthy.
_RC_MEANING = {
    0: "all collected tests passed",
    1: "tests were collected and run, some failed",
    2: "run interrupted by the user",
    3: "internal error",
    4: "pytest usage error",
    5: "no tests were collected",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=ROOT, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""

# Gates whose tests follow the `test_n<N>_...` naming convention.
NAME_RE = re.compile(r"::test_(n[1-7])_", re.I)

# Gates whose tests predate the convention, mapped by file.
FILE_GATES = {"test_rheobase_failure_states.py": "N5"}

# Every gate the project claims to enforce.  A gate absent from the run is
# emitted with n=0 so the manifest blocks on it.
ALL_GATES = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]

# Tests that police the EXCLUSION BOOKKEEPING rather than measure a model's
# neuronal validity.
#
# These matter -- they are what stops a known failure from being quietly
# delisted -- but they are a different kind of evidence, and pooling them with
# the validity tests makes a roll-up say something it does not mean.  "N3 46/46"
# read as "every model passes N3" while four models were exempted from the
# validity test altogether and two tests were passing *because* those models
# still fail (post-response audit, section 3.5).  They are now counted and
# reported separately, and `by_gate` -- the roll-up the release manifest
# consumes -- contains validity tests only.
POLICY_TESTS = re.compile(r"::test_n[1-7]_known_failures_", re.I)

# `nodeid[param]` -> param.  Panel tests are parametrised by model id, which is
# what makes a per-model matrix possible at all.
PARAM_RE = re.compile(r"\[([^\]]+)\]$")


class _Collector:
    """Records one row per test outcome."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.collect_errors: list[str] = []

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
        pm = PARAM_RE.search(report.nodeid)
        self.rows.append({
            "gate": gate, "subject": report.nodeid, "outcome": report.outcome,
            "kind": "policy" if POLICY_TESTS.search(report.nodeid) else "validity",
            "param": pm.group(1) if pm else None,
        })

    def pytest_collectreport(self, report) -> None:  # noqa: ANN001
        # A collection error means whole files never ran.  Without this the
        # roll-up could be assembled from a partial run and look complete.
        if report.outcome == "failed":
            self.collect_errors.append(str(report.nodeid))


def _subject_records(matrix: dict[str, dict[str, str]]) -> list[dict]:
    """Flatten the validity matrix into one gate record per (gate, model).

    ``untested`` and ``exempt`` are recorded as FAIL, not as absent evidence:
    a release must be able to say, per model, that a gate was satisfied, and
    "we never ran it here" is the answer the coverage check exists to catch.
    """
    out = []
    for gate, row in sorted(matrix.items()):
        if gate.startswith("_"):
            continue
        for model, state in sorted(row.items()):
            out.append({
                "gate": gate,
                "subject": model,
                "status": "PASS" if state == "pass" else "FAIL",
                "detail": state,
                "failures": [] if state == "pass" else [f"{gate} is {state} for {model}"],
                "measurements": {"validity_state": state},
            })
    return out


def _validity_matrix(rows: list[dict]) -> dict[str, dict[str, str]]:
    """gate -> model_id -> pass | fail | exempt | untested.

    The release evidence the post-response audit asked for: a per-model
    scientific-validity statement that cannot be read as a pass merely because
    the model was never tested.  Built from the parametrised node ids, so it
    reflects what actually ran rather than what the gate intends to cover.

    ``exempt`` means the model is on a documented known-failure list and was
    therefore not parametrised into that gate's validity tests.  It is a
    recorded failure with a remediation note, never a pass -- the models on
    those lists are also required to be in ``PRIMARY_EXCLUSIONS``, which the
    policy tests check separately.
    """
    from compass.adapters.models import MODEL_REGISTRY

    sys.path.insert(0, str(ROOT / "tests" / "integration"))
    from test_convergence_contract import N3_KNOWN_FAILURES
    from test_real_trace_features import N6_KNOWN_FAILURES

    exempt: dict[str, dict[str, list[str]]] = {
        "N3": {m: ["all N3 acceptance checks"] for m in N3_KNOWN_FAILURES},
        "N6": {},
        # N2 is evaluable only where a published scalar exists to compare
        # against; the rest are DECLARED not evaluable, with a reason, in
        # data/provenance/n2_source_targets.json. Declared-and-explained is not
        # the same as passing, so they are exempt, not pass.
        "N2": {m: [v.get("reason", "declared not evaluable")]
               for m, v in json.loads(
                   (ROOT / "data" / "provenance" / "n2_source_targets.json"
                    ).read_text())["models"].items()
               if not v.get("evaluable")},
    }
    for m, check in N6_KNOWN_FAILURES:
        exempt["N6"].setdefault(m, []).append(check)

    models = sorted(MODEL_REGISTRY)
    out: dict[str, dict[str, str]] = {}
    for g in ALL_GATES:
        sel = [r for r in rows
               if r["gate"] == g and r["kind"] == "validity" and r["param"]]
        row: dict[str, str] = {}
        for m in models:
            mine = [r for r in sel if r["param"] == m]
            decided = [r for r in mine if r["outcome"] in ("passed", "failed")]
            # Exemption DOMINATES a pass, and this ordering is the whole point.
            # A gate is several tests: for N3 the decisive one
            # (`production_setting_is_accepted`) is parametrised over the
            # non-failing models only, while two ancillary ones cover all
            # fourteen.  Checking `decided` first therefore reported all four
            # documented N3 failures as "pass" -- reproducing, inside the matrix
            # that was meant to expose it, exactly the false all-clear this
            # matrix exists to prevent.
            if any(r["outcome"] == "failed" for r in decided):
                row[m] = "fail"
            elif m in exempt.get(g, {}):
                row[m] = "exempt"
            elif decided:
                row[m] = "pass"
            else:
                row[m] = "untested"
        # A gate whose tests are not parametrised by model (N5, N7 in part) has
        # nothing per-model to say; omit it rather than report every model as
        # untested, which would be misleading in the other direction.
        if any(v != "untested" for v in row.values()):
            out[g] = row
    out["_exemptions"] = {g: {m: sorted(c) for m, c in v.items()}
                          for g, v in exempt.items() if v}
    out["_legend"] = {
        "pass": "every validity test that ran for this model passed",
        "fail": "at least one validity test failed",
        "exempt": ("on this gate's documented known-failure list, so the "
                   "decisive validity test was not parametrised for it. A "
                   "recorded failure with a remediation note -- never a pass. "
                   "Each such model is separately required to appear in "
                   "compass.analysis.cohorts.PRIMARY_EXCLUSIONS, which the "
                   "policy tests enforce."),
        "untested": "no validity test for this gate ran with this model",
    }
    return out


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
    t0 = time.time()
    # The global exit status was previously discarded.  It is the only signal
    # that reports an internal error, a usage error or an interrupted run --
    # states in which the per-test rows below can look perfectly healthy because
    # the tests that would have failed never ran (post-response audit, 3.5).
    pytest_rc = int(pytest.main(argv, plugins=[coll]))
    duration_s = round(time.time() - t0, 1)

    by_gate: dict[str, dict] = {}
    policy: dict[str, dict] = {}
    for g in ALL_GATES:
        rows = [r for r in coll.rows if r["gate"] == g]
        for kind, sink in (("validity", by_gate), ("policy", policy)):
            sel = [r for r in rows if r["kind"] == kind]
            passed = [r for r in sel if r["outcome"] == "passed"]
            failed = [r for r in sel if r["outcome"] == "failed"]
            skipped = [r for r in sel if r["outcome"] == "skipped"]
            # A skip produces no evidence, so it counts neither as a pass nor as
            # a failure -- `n` is the number of tests that actually decided
            # something.  Skips stay visible, and if EVERY test for a gate skips
            # then n == 0 and the manifest sees the gate as missing, which is
            # the correct outcome.
            sink[g] = {
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
        # --- provenance ---------------------------------------------------- #
        # Without these the record was a bare tally that could have come from any
        # revision, any environment and any subset of the suite.
        "provenance": {
            "git_revision": _git("rev-parse", "HEAD"),
            "git_tree": git_tree(),
            "command": [sys.executable, "-m", "pytest", *argv],
            "pytest_version": pytest.__version__,
            "pytest_exit_code": pytest_rc,
            "pytest_exit_meaning": _RC_MEANING.get(
                pytest_rc, f"unrecognised pytest exit code {pytest_rc}"),
            "duration_s": duration_s,
            "collection_errors": coll.collect_errors,
            "environment": environment_record(),
        },
        "note": ("N2 compares each model's canonical operating point against a "
                 "SOURCED published scalar declared in "
                 "data/provenance/n2_source_targets.json. It is NOT trace-level "
                 "reproduction: the published stimulus, temperature and "
                 "recording site are not distributed with these models. Models "
                 "with no sourceable scalar are declared not evaluable with a "
                 "reason and appear as `exempt` in validity_matrix, never as a "
                 "pass; every claim requiring source-level reproduction is "
                 "withdrawn for them."),
        "by_gate": by_gate,
        "by_gate_semantics": (
            "by_gate counts NEURONAL-VALIDITY tests only. Tests that police the "
            "known-failure bookkeeping -- 'this documented failure still fails' "
            "and 'this documented failure is excluded from primary inference' -- "
            "are counted in policy_tests instead, because a pass there means the "
            "exclusion is intact, not that a model is valid. Models exempted "
            "from a gate's validity test are listed per model in "
            "validity_matrix; `passed == n` for a gate therefore means 'every "
            "model that was TESTED passed', which is not the same as 'every "
            "model passes'."),
        "policy_tests": policy,
        "validity_matrix": _validity_matrix(coll.rows),
        # Per-MODEL gate records in the same shape every other gate file uses.
        # The roll-up alone is keyed by gate name, so the release manifest could
        # only ever ask "did N4 run?", never "did N4 run for tomko_ca1?"
        # (2026-08-09 audit, 3.1: "require named gate subjects").
        "records": _subject_records(_validity_matrix(coll.rows)),
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
        pol = policy[g]
        polstr = f"  [+{pol['passed']}/{pol['n']} policy]" if pol["n"] else ""
        print(f"  {g}: {v['passed']}/{v['n']} {state}{skip}{polstr}")
    mat = record["validity_matrix"]
    for g, row in mat.items():
        if g.startswith("_"):
            continue
        ms = sorted(m for m, s in row.items() if s == "exempt")
        bad = sorted(m for m, s in row.items() if s == "fail")
        if ms:
            print(f"  {g}: exempted from the decisive validity test: {ms} "
                  f"(documented failures, not passes)")
        if bad:
            print(f"  {g}: FAILS for {bad}")
    # A non-zero pytest status with no recorded failure means something went
    # wrong outside the mapped tests; the roll-up must not be trusted.
    if pytest_rc not in (0, 1):
        print(f"  !! pytest exited {pytest_rc} "
              f"({record['provenance']['pytest_exit_meaning']}); the roll-up "
              f"above may be incomplete")
        worst = 1
    if coll.collect_errors:
        print(f"  !! collection errors: {coll.collect_errors}")
        worst = 1
    print(f"-> {OUT}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
