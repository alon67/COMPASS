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


def git_tree() -> str:
    """Current commit, with a dirty marker.  ``unversioned`` if not a repo."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return "unversioned"
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
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
