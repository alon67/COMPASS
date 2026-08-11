"""Machine-readable gate records (2026-08-06 audit, section 6.4).

Every gate must "emit a machine-readable record containing the Git tree or
commit, configuration hashes, model-asset and mechanism hashes, dependency and
runtime versions, random seeds, input shard hashes, tested row IDs, tolerances,
status and failure reasons. A skipped, missing or indeterminate mandatory gate
is a failure, not a pass."

:class:`GateResult` is that record.  It has exactly three states and ``passed``
is true only for ``PASS`` -- ``SKIPPED`` and ``INDETERMINATE`` are failures by
construction, so a gate cannot be waived by not running it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["GateResult", "environment_record", "gate_record", "git_tree"]

ROOT = Path(__file__).resolve().parents[3]


# THE TREE IDENTITY IS THE SOURCE REVISION, NOT THE WHOLE WORKING DIRECTORY.
#
# `git_tree()` used to run `git rev-parse HEAD` plus `git status --porcelain`
# over everything, and generated output is tracked in this repository -- 176 of
# 204 modified files at the 2026-08-11 release were under `results/`. Two
# consequences, and together they made the release's own acceptance item
# "one clean identified revision" unreachable:
#
#   * every gate record carried `<rev>-dirty` forever, because running a gate
#     writes a tracked artifact and therefore dirties the tree it is stamping;
#   * committing changed HEAD, which instantly made all 4,780 campaign records,
#     84 N-gate records and every waveform record stale -- and re-running them
#     to fix that dirtied the tree again. A fixed point with no exit.
#
# The claim a gate record actually makes is "this evidence was produced by this
# SOURCE revision". So the identity is now the last commit that touched a
# declared INPUT path, and the dirty flag is computed over those paths alone.
# Regenerating results no longer changes it; changing a line of `src/` does.
#
# The path set is read from configs/release_inventory.json -- the `source` and
# `configuration` classes, the same two the clean-build check calls inputs -- so
# it cannot drift from the inventory's own notion of what an input is.
_INPUT_CLASSES = ("source", "configuration")


def _source_paths() -> list[str]:
    """Top-level paths of the declared input classes, from the inventory."""
    inv = ROOT / "configs" / "release_inventory.json"
    try:
        classes = json.loads(inv.read_text())["classes"]
    except Exception:  # noqa: BLE001
        return ["src", "tools", "tests", "configs", "pyproject.toml"]
    out: set[str] = set()
    for name in _INPUT_CLASSES:
        for glob in classes.get(name, {}).get("include", []):
            out.add(glob.split("/")[0].split("*")[0] or glob)
    return sorted(p for p in out if p)


def git_tree() -> str:
    """The source revision, with a dirty marker over source paths only.

    ``unversioned`` if this is not a git repository.
    """
    try:
        paths = _source_paths()
        rev = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", *paths],
            capture_output=True, text=True, timeout=20)
        if rev.returncode != 0:
            return "unversioned"
        commit = rev.stdout.strip()
        if not commit:                      # no commit has touched source yet
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=20)
            commit = head.stdout.strip() if head.returncode == 0 else ""
            if not commit:
                return "unversioned"
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", *paths],
            capture_output=True, text=True, timeout=30)
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unversioned"


def environment_record() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "machine": platform.machine()}
    for mod in ("neuron", "efel", "numpy", "scipy", "pandas", "sklearn",
                "statsmodels", "SALib"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            env[mod] = "absent"
    return env


@dataclass
class GateResult:
    """One gate's verdict for one subject (a model, a shard, or the campaign)."""

    gate: str                     # "N1", "D3", ...
    subject: str                  # model_id / shard name / "campaign"
    status: str                   # "PASS" | "FAIL" | "SKIPPED" | "INDETERMINATE"
    detail: str = ""
    failures: list[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    tolerances: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)   # hashes, seeds, row ids
    git_tree: str = field(default_factory=git_tree)
    environment: dict = field(default_factory=environment_record)
    recorded_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "SKIPPED", "INDETERMINATE"}:
            raise ValueError(f"unknown gate status {self.status!r}")

    @property
    def passed(self) -> bool:
        """Only an explicit PASS passes.  Absence of evidence is a failure."""
        return self.status == "PASS"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def gate_record(gate: str, subject: str, failures: list[str],
                measurements: dict | None = None,
                tolerances: dict | None = None,
                inputs: dict | None = None,
                detail: str = "") -> GateResult:
    """Build a result whose status follows from the failure list."""
    return GateResult(
        gate=gate, subject=subject,
        status="PASS" if not failures else "FAIL",
        detail=detail, failures=list(failures),
        measurements=measurements or {}, tolerances=tolerances or {},
        inputs=inputs or {},
    )


def n2_coverage() -> dict:
    """The single generated statement of what gate N2 does and does not cover.

    N2's roll-up reads ``7/7``, which is six model tests plus one
    declaration-coverage test -- and a reader who sees ``7/7`` beside a gate
    named "quantitative agreement with published values" will read it as seven
    source-reproduced models out of seven (2026-08-09 post-response-3 audit,
    P1-2).  The published summary is generated from the target file instead, so
    every document quotes the same sentence and the arithmetic cannot drift.
    """
    spec = json.loads(
        (ROOT / "data" / "provenance" / "n2_source_targets.json").read_text())
    models = spec["models"]
    ev = sorted(m for m, v in models.items() if v.get("evaluable"))
    ne = sorted(m for m, v in models.items() if not v.get("evaluable"))
    n_targets = sum(len(models[m].get("targets", [])) for m in ev)
    n_unmatched = sum(1 for m in ev for t in models[m].get("targets", [])
                      if not t.get("protocol_matched"))
    return {
        "statement": (f"source-referenced canonical scalar agreement: "
                      f"{len(ev)}/{len(models)} evaluable implementations"),
        "n_evaluable": len(ev),
        "n_registered": len(models),
        "evaluable": ev,
        "not_evaluable": ne,
        "n_scalar_targets": n_targets,
        "n_targets_with_unmatched_protocol": n_unmatched,
        "is_not": ("trace-level reproduction of a published figure. The "
                   "published stimulus, temperature and recording site are not "
                   "distributed with these models; every target is a published "
                   "SCALAR compared against the COMPASS canonical measurement, "
                   "and targets whose protocol does not match are marked as "
                   "such with a widened tolerance."),
        "roll_up_note": (f"the N2 gate roll-up counts {len(ev)} model tests plus "
                         f"one declaration-coverage test; it is not "
                         f"{len(ev) + 1} reproduced models"),
    }


def write_gate_records(records: list[GateResult], path: Path) -> Path:
    """Persist a gate set plus a summary that fails closed on any non-PASS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_utc": datetime.now(UTC).isoformat(),
        "git_tree": git_tree(),
        "environment": environment_record(),
        "n_records": len(records),
        "n_passed": sum(r.passed for r in records),
        "all_passed": bool(records) and all(r.passed for r in records),
        "by_gate": {},
        "records": [r.as_dict() for r in records],
    }
    for r in records:
        g = payload["by_gate"].setdefault(
            r.gate, {"n": 0, "passed": 0, "failed_subjects": []})
        g["n"] += 1
        g["passed"] += int(r.passed)
        if not r.passed:
            g["failed_subjects"].append(f"{r.subject}[{r.status}]")
    text = json.dumps(payload, indent=1, default=str)
    path.write_text(text)
    payload["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return path
