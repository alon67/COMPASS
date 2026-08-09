#!/usr/bin/env python3
"""Verify the stored convergence evidence without re-running the audits.

Why this is a stage of its own
------------------------------
``tools/run_all.sh`` used to have a ``convergence`` stage whose entire body was
``echo "convergence audits are run per model: ..."``.  The post-response audit
called that out: a stage that prints instructions reproduces nothing, but it
occupies a slot in an ordered pipeline and reports "ok", so a reader of the log
cannot tell the difference between evidence that was checked and evidence that
was merely mentioned.

Re-running the audits genuinely is not an option inside a pipeline: a single
model costs hours and the panel costs days, and the audits *supersede* rather
than overwrite, so a casual re-run damages the record.  What can be done cheaply
is to verify that the stored evidence exists, is internally consistent, and
still describes the solver settings the campaigns actually used -- and to say,
in a machine-readable record, that it was verified rather than regenerated.

What is checked, per registered model
-------------------------------------
1. a convergence audit file exists;
2. it names a ``chosen`` setting, or the model is a documented N3 failure;
3. the ``chosen`` setting equals the campaign solver contract in
   ``CAMPAIGN_SOLVER`` -- the check that catches a model whose production dt was
   edited after its audit was run;
4. the reference setting is strictly finer than the chosen one; and
5. the audit is not older than the adapter or mechanism sources it describes.

Writes ``results/validation/convergence_evidence.json`` with an explicit
``regenerated: false`` and the reason.

    python tools/verify_convergence_evidence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROV = ROOT / "data" / "provenance"
OUT = ROOT / "results" / "validation" / "convergence_evidence.json"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=ROOT, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    from compass.adapters.models import MODEL_REGISTRY
    from compass.workflows.simulate import CAMPAIGN_SOLVER

    sys.path.insert(0, str(ROOT / "tests" / "integration"))
    from test_convergence_contract import N3_KNOWN_FAILURES

    per_model: dict[str, dict] = {}
    problems: list[str] = []

    for mid in sorted(MODEL_REGISTRY):
        path = PROV / f"convergence_audit_{mid}.json"
        rec: dict = {"file": path.name if path.exists() else None}
        if not path.exists():
            rec["status"] = "MISSING"
            problems.append(f"{mid}: no convergence audit file")
            per_model[mid] = rec
            continue

        d = json.loads(path.read_text())
        chosen = d.get("chosen")
        contract = CAMPAIGN_SOLVER.get(mid)
        rec.update({
            "chosen": chosen,
            "campaign_solver": contract,
            "reference_dt_ms": (d.get("reference") or {}).get("dt_ms"),
            "n_points": d.get("n_points"),
            "written_utc": d.get("written_utc"),
            "known_n3_failure": mid in N3_KNOWN_FAILURES,
        })

        if chosen is None:
            # A null `chosen` is the audit's way of saying "no tested setting
            # converged".  That is a legitimate recorded outcome, but only for a
            # model the project already treats as an N3 failure.
            if mid in N3_KNOWN_FAILURES:
                rec["status"] = "no-converged-setting (documented N3 failure)"
            else:
                rec["status"] = "FAIL"
                problems.append(f"{mid}: audit records chosen=null but the model "
                                f"is not a documented N3 failure")
            per_model[mid] = rec
            continue

        ok = True
        # (3) the audited setting must be the one the campaigns actually used.
        c_dt = chosen.get("dt_ms") if isinstance(chosen, dict) else None
        k_dt = (contract or {}).get("dt_ms")
        if c_dt is not None and k_dt is not None and abs(c_dt - k_dt) > 1e-12:
            ok = False
            problems.append(f"{mid}: audit chose dt={c_dt} ms but CAMPAIGN_SOLVER "
                            f"uses dt={k_dt} ms; the evidence describes a setting "
                            f"the campaign did not run")
        # (4) a reference that is not finer than the chosen setting proves nothing.
        r_dt = (d.get("reference") or {}).get("dt_ms")
        if c_dt is not None and r_dt is not None and r_dt >= c_dt:
            ok = False
            problems.append(f"{mid}: reference dt={r_dt} ms is not finer than the "
                            f"chosen dt={c_dt} ms")
        # (5) evidence older than the code it describes is stale.
        adapters = [ROOT / "src" / "compass" / "adapters" / "models.py",
                    ROOT / "src" / "compass" / "adapters" / "replication.py"]
        newest_src = max((p.stat().st_mtime for p in adapters if p.exists()),
                         default=0.0)
        if path.stat().st_mtime < newest_src:
            rec["stale_vs_adapter"] = True
            # Recorded, not fatal: an unrelated edit to a shared adapter module
            # touches every model's timestamp comparison at once, so treating it
            # as a failure would make the check unusable.  It is surfaced so a
            # reviewer can decide.
        rec["status"] = "verified" if ok else "FAIL"
        per_model[mid] = rec

    record = {
        "tool": "verify_convergence_evidence.py",
        "written_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git("rev-parse", "HEAD"),
        "regenerated": False,
        "regeneration_note": (
            "the convergence audits themselves were NOT re-run. Each costs hours "
            "on one model and the panel costs days, and tools/convergence_audit.py "
            "supersedes rather than overwrites its evidence. This stage verifies "
            "the stored record against the solver contract; it does not reproduce "
            "it. To regenerate: python tools/convergence_audit.py <model_id>."),
        "n_models": len(per_model),
        "n_verified": sum(1 for v in per_model.values() if v["status"] == "verified"),
        "n_documented_n3_failures": sum(
            1 for v in per_model.values() if v.get("known_n3_failure")),
        "problems": problems,
        "per_model": per_model,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, default=str))

    for mid, v in per_model.items():
        print(f"  {mid:22s} {v['status']}")
    print(f"\n{record['n_verified']}/{record['n_models']} verified, "
          f"{record['n_documented_n3_failures']} documented N3 failures, "
          f"{len(problems)} problem(s) -> {OUT}")
    for p in problems:
        print(f"  ! {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
