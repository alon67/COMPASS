#!/usr/bin/env python3
"""Immutable release manifest and provenance graph (audit 6.4.2 D6, 6.4.5).

Collects, in one signed-by-hash document:

* the Git tree and whether it is dirty;
* the full dependency and runtime environment;
* every downloaded model asset with its recorded hash and license;
* every compiled mechanism library with its binary hash;
* the solver contract and convergence verdict for each model;
* every result artifact with its size, row count and content hash;
* every gate record found under ``results/validation``;
* the effective inferential sample (lineages, not model ids); and
* a fail-closed verdict.

The verdict is **RELEASE-BLOCKED** unless every mandatory gate has an explicit
PASS.  A missing gate record is a block, not a pass -- the audit's release rule
7 in executable form.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.validation.gates import environment_record, git_tree  # noqa: E402

OUT = ROOT / "results" / "validation" / "release_manifest.json"

MANDATORY_GATES = ["D1", "D2", "D3", "D4", "D5", "REPLAY"]


def file_record(p: Path) -> dict:
    rec = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:32]}
    if p.suffix == ".parquet":
        try:
            rec["rows"] = int(len(pd.read_parquet(p)))
        except Exception as exc:  # noqa: BLE001
            rec["rows"] = f"unreadable: {type(exc).__name__}"
    return rec


def main() -> int:
    from compass.adapters.models import MODEL_REGISTRY
    from compass.analysis.cohorts import PRIMARY_EXCLUSIONS
    from compass.workflows.simulate import CAMPAIGN_SOLVER

    prov = ROOT / "data" / "provenance"
    manifest: dict = {
        "written_utc": datetime.now(UTC).isoformat(),
        "git_tree": git_tree(),
        "environment": environment_record(),
    }

    # --- models -------------------------------------------------------- #
    models = {}
    for mid, cls in sorted(MODEL_REGISTRY.items()):
        m = cls.manifest
        conv_path = prov / f"convergence_audit_{mid}.json"
        conv = json.loads(conv_path.read_text()) if conv_path.exists() else {}
        models[mid] = {
            "cohort": m.cohort,
            "source_lineage_id": m.source_lineage_id,
            "implementation_id": m.implementation_id,
            "neuron_class_id": m.neuron_class_id,
            "source_accession": m.source_accession,
            "license": m.license,
            "protocol_coverage": m.protocol_coverage,
            "solver": CAMPAIGN_SOLVER.get(mid),
            "convergence_chosen": conv.get("chosen"),
            "convergence_file": conv_path.name if conv_path.exists() else None,
            "excluded_from_primary": PRIMARY_EXCLUSIONS.get(mid),
        }
    manifest["models"] = models
    lineages = sorted({v["source_lineage_id"] for v in models.values()})
    manifest["inferential_sample"] = {
        "n_implementations": len(models),
        "n_source_lineages": len(lineages),
        "lineages": {
            lin: sorted(m for m, v in models.items() if v["source_lineage_id"] == lin)
            for lin in lineages},
        "n_primary_implementations": sum(
            1 for v in models.values() if not v["excluded_from_primary"]),
        "n_primary_lineages": len({
            v["source_lineage_id"] for v in models.values()
            if not v["excluded_from_primary"]}),
    }

    # --- assets and mechanisms ------------------------------------------ #
    for name in ("downloads.json", "mechanism_builds.json",
                 "amplitude_calibration.json", "identity_backfill.json"):
        p = prov / name
        if p.exists():
            manifest.setdefault("provenance_files", {})[name] = file_record(p)

    # --- artifacts ------------------------------------------------------- #
    artifacts = []
    for d in ("results/analyses", "results/manifests", "results/validation"):
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in {".parquet", ".json"} \
                    and "_withdrawn" not in p.parts and p.name != OUT.name:
                artifacts.append(file_record(p))
    manifest["artifacts"] = artifacts

    # --- gates ------------------------------------------------------------ #
    gates: dict = {}
    vdir = ROOT / "results" / "validation"
    if vdir.exists():
        for p in sorted(vdir.glob("*.json")):
            if p.name == OUT.name:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if "by_gate" in d:
                for g, v in d["by_gate"].items():
                    cur = gates.setdefault(g, {"n": 0, "passed": 0, "failed": [],
                                               "sources": []})
                    cur["n"] += v["n"]
                    cur["passed"] += v["passed"]
                    cur["failed"] += v.get("failed_subjects", [])
                    cur["sources"].append(p.name)
    manifest["gates"] = gates

    missing = [g for g in MANDATORY_GATES if g not in gates]
    failed = [g for g, v in gates.items() if v["passed"] < v["n"]]
    blockers = []
    if missing:
        blockers.append(f"mandatory gate records absent: {missing} "
                        "(a missing gate is a failure, not a pass)")
    if failed:
        blockers.append(f"gates with failures: {failed}")
    if manifest["git_tree"].endswith("-dirty") or manifest["git_tree"] == "unversioned":
        blockers.append(f"working tree is {manifest['git_tree']}: reproducibility "
                        "claims apply only to a committed, tagged state")
    if PRIMARY_EXCLUSIONS:
        blockers.append(
            "models excluded on gate failure: "
            + ", ".join(f"{m} ({v['gate']})" for m, v in sorted(PRIMARY_EXCLUSIONS.items()))
            + " -- valid for a screening result, blocking for a confirmatory release")

    manifest["release_verdict"] = "RELEASE-BLOCKED" if blockers else "RELEASE-READY"
    manifest["blockers"] = blockers

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=1, default=str)
    OUT.write_text(text)
    print(f"release verdict: {manifest['release_verdict']}")
    for b in blockers:
        print(f"  - {b}")
    print(f"{len(artifacts)} artifacts, {len(models)} models, "
          f"{manifest['inferential_sample']['n_source_lineages']} lineages "
          f"-> {OUT}")
    print(f"manifest sha256 {hashlib.sha256(text.encode()).hexdigest()[:32]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
