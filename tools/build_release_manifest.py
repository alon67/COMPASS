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

import argparse
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

# Data-integrity gates, the deterministic-replay gate, and the neuronal-validity
# gates.  N1-N7 were previously absent, so a release could be assembled while the
# neuronal-behaviour tests were failing or had simply never been run -- the
# post-revision audit's point that "N1-N7 are automatically enforced fail-closed"
# was premature.  Evidence for them is produced by
# tools/record_numerical_gates.py, which writes results/validation/
# numerical_gates.json in the same `by_gate` form as every other gate file.
#
# N2 is listed even though nothing implements it.  That is deliberate: strict
# reproduction of published traces is not a test today, and a gate with no
# evidence must block rather than be quietly dropped from the list.
MANDATORY_GATES = ["D1", "D2", "D3", "D4", "D5", "REPLAY",
                   "N1", "N2", "N3", "N4", "N5", "N6", "N7"]


def file_record(p: Path) -> dict:
    rec = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:32]}
    if p.suffix == ".parquet":
        try:
            rec["rows"] = int(len(pd.read_parquet(p)))
        except Exception as exc:  # noqa: BLE001
            rec["rows"] = f"unreadable: {type(exc).__name__}"
    return rec


def confirmatory_gates(manifest: dict, gates: dict, retained_n4: list) -> dict:
    """The thirteen conditions the 2026-08-09 independent audit requires.

    Its section 7 lists what must hold "before describing COMPASS as
    confirmatory or release-ready".  Kept here as an evaluated checklist rather
    than as prose in a reply, so the answer to "how far along is this?" is a
    generated artifact that cannot quietly go stale.

    Each entry is ``met`` (bool or None) plus the evidence used.  ``None`` means
    the condition is not machine-decidable from the tree -- a human has to look
    -- and is treated as NOT met for the purpose of the summary count, because
    an unevaluated condition is not a satisfied one.
    """
    an = ROOT / "results" / "analyses"
    val = ROOT / "results" / "validation"

    def has(p: Path) -> bool:
        return p.is_file()

    def load(p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}

    ng = load(val / "numerical_gates.json")
    pinf = load(an / "perturbation_inference.json")
    dec = load(an / "decision.json")
    wave = load(val / "retained_waveforms_screening.json")

    n2 = gates.get("N2", {})
    rep = pinf.get("numerical_repeatability", {})
    per_model = rep.get("per_model", {}) if rep.get("status") == "ok" else {}
    pre = dec.get("perturbation_precondition", {})

    checks: dict[str, dict] = {
        "1_n2_source_trace_reproduction": {
            "requirement": "N2 strict source-trace reproduction is implemented",
            "met": bool(n2.get("n", 0)) and n2.get("passed") == n2.get("n"),
            "evidence": f"N2 roll-up {n2.get('passed')}/{n2.get('n')}",
        },
        "2_n1_and_n4_pass_for_included_models": {
            "requirement": "N1 and N4 pass for every inferentially included "
                           "implementation",
            "met": all(gates.get(g, {}).get("n", 0)
                       and gates[g]["passed"] == gates[g]["n"]
                       for g in ("N1", "N4")),
            "evidence": {g: f"{gates.get(g, {}).get('passed')}/"
                            f"{gates.get(g, {}).get('n')}" for g in ("N1", "N4")},
        },
        "3_n4_failures_excluded_or_confined": {
            "requirement": "BBP and Tomko are remediated, excluded, or explicitly "
                           "confined to exploratory sensitivity arms",
            # Partially met: the arms exist and the labelling is corrected, but
            # the implementations still vote in the PRIMARY arm, which is what
            # the audit objected to.
            "met": not retained_n4,
            "evidence": {
                "retained_in_primary": retained_n4,
                "arms_available": ["n4_strict", "n4_projected"],
                "cohort_label": manifest.get("cohort_label"),
            },
        },
        "4_half_step_battery_covers_every_primary_model": {
            "requirement": "the half-step perturbation battery covers every "
                           "primary model, and decision scoring obeys its "
                           "repeatability outcome",
            "met": bool(pre) and not pre.get("models_untested")
                   and "perturbation_precondition" in dec,
            "evidence": {
                "models_tested": sorted(per_model),
                "models_untested": pre.get("models_untested"),
                "n_repeatable": sum(bool(v.get("numerically_repeatable"))
                                    for v in per_model.values()),
                "scoring_obeys_it": "perturbation_precondition" in dec,
            },
        },
        "5_run_all_completes_in_dependency_order": {
            "requirement": "run_all.sh completes from a clean result tree and "
                           "orders perturbation generation before cross-model "
                           "scoring",
            # Order is verified by tests/unit/test_audit_regressions.py; whether
            # it COMPLETES from a clean tree is a days-long execution that no
            # inspection of this repository can settle.
            "met": None,
            "evidence": "stage order enforced by "
                        "tests/unit/test_audit_regressions.py::"
                        "test_perturbations_are_generated_before_cross_model_scoring; "
                        "end-to-end execution from a clean tree NOT performed",
        },
        "6_gate_evidence_is_tied_to_commit_and_environment": {
            "requirement": "numerical-gate evidence is tied to the exact commit "
                           "and environment, with raw pytest status retained",
            "met": bool(ng.get("provenance", {}).get("git_revision"))
                   and "pytest_exit_code" in ng.get("provenance", {}),
            "evidence": {k: ng.get("provenance", {}).get(k)
                         for k in ("git_revision", "git_tree", "pytest_exit_code",
                                   "pytest_version", "duration_s")},
        },
        "7_rollups_separate_validity_from_exclusion_policy": {
            "requirement": "gate roll-ups distinguish neuronal validity from "
                           "'known failure correctly excluded'",
            "met": "policy_tests" in ng and "validity_matrix" in ng,
            "evidence": {"has_policy_tests": "policy_tests" in ng,
                         "has_validity_matrix": "validity_matrix" in ng},
        },
        "8_high_impact_fixes_have_regression_tests": {
            "requirement": "new high-impact fixes have regression tests",
            "met": has(ROOT / "tests" / "unit" / "test_audit_regressions.py"),
            "evidence": "tests/unit/test_audit_regressions.py",
        },
        "9_report_tables_generated_from_one_manifest": {
            "requirement": "all report tables are generated from one "
                           "authoritative result manifest, and stale values are "
                           "tested for absence",
            "met": has(an / "claim_manifest.json"),
            "evidence": "tools/verify_report_numbers.py: claim manifest, retired "
                        "values, and per-criterion single-value enforcement",
        },
        "10_manifest_hashes_match_every_artifact": {
            "requirement": "release-manifest hashes match every released artifact",
            "met": None,
            "evidence": "run `python3 tools/build_release_manifest.py --verify` "
                        "on the candidate tree; not self-evaluable while the "
                        "manifest is being written",
        },
        "11_stratified_waveforms_retained_and_inspected": {
            "requirement": "a stratified set of replay voltage traces is retained "
                           "and inspected, not only derived features",
            "met": bool(wave) and wave.get("n_with_problems") == 0
                   and wave.get("n_selected", 0) > 0,
            "evidence": {k: wave.get(k) for k in
                         ("n_selected", "n_clean", "n_with_problems",
                          "problems_by_kind", "traces_dir")} or "not run",
        },
        "12_lineage_level_inference_throughout": {
            "requirement": "lineage-level folds, bootstraps, random effects and "
                           "weighting are applied consistently",
            "met": None,
            "evidence": "leave-one-lineage-out folds, lineage-clustered "
                        "bootstraps with equal cluster weight, and "
                        "run_robustness clustering on source_lineage_id are all "
                        "in place and tested; 'consistently' across every "
                        "consumer is a review judgement, not a computable one",
        },
        "13_reproduced_from_a_committed_tagged_revision": {
            "requirement": "the complete workflow is reproduced from a committed, "
                           "preferably tagged, clean revision",
            "met": not (manifest["git_tree"].endswith("-dirty")
                        or manifest["git_tree"] == "unversioned"),
            "evidence": manifest["git_tree"],
        },
    }
    met = [k for k, v in checks.items() if v["met"] is True]
    unmet = [k for k, v in checks.items() if v["met"] is False]
    undecidable = [k for k, v in checks.items() if v["met"] is None]
    return {
        "source": "COMPASS_INDEPENDENT_POST_RESPONSE_TECHNICAL_SCIENTIFIC_AUDIT_"
                  "2026-08-09.md, section 7",
        "n_met": len(met), "n_total": len(checks),
        "met": met, "not_met": unmet,
        "not_machine_decidable": undecidable,
        "note": ("a condition that cannot be decided from the tree is counted as "
                 "NOT met. These are the conditions for describing the project "
                 "as CONFIRMATORY; they are stricter than the fail-closed gate "
                 "set that decides the release verdict."),
        "checks": checks,
    }


def verify() -> int:
    """Re-hash every artifact the stored manifest lists and report mismatches.

    Release rule 10 in executable form: "release-manifest hashes match every
    released artifact".  Building the manifest records the hashes; nothing
    checked them afterwards, and the reviewed revision shipped a manifest whose
    recorded hash for ``claim_manifest.json`` no longer matched the file on
    disk.  Run this on a candidate tree before publishing it.
    """
    if not OUT.exists():
        print(f"no manifest at {OUT}; build one first")
        return 1
    man = json.loads(OUT.read_text())
    missing, changed = [], []
    for rec in man.get("artifacts", []):
        p = ROOT / rec["path"]
        if not p.is_file():
            missing.append(rec["path"])
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()[:32]
        if got != rec["sha256"]:
            changed.append(f"{rec['path']}: recorded {rec['sha256']}, on disk {got}")
    tree_now = git_tree()
    print(f"manifest built from {man.get('git_tree')} at {man.get('written_utc')}")
    print(f"working tree is now  {tree_now}")
    for m in missing:
        print(f"  MISSING  {m}")
    for c in changed:
        print(f"  CHANGED  {c}")
    n = len(man.get("artifacts", []))
    print(f"{n - len(missing) - len(changed)}/{n} artifacts match their recorded hash")
    if man.get("git_tree") != tree_now:
        print("  ! the manifest was built from a different revision than the one "
              "checked out; rebuild it before releasing")
    return 0 if not (missing or changed) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="do not rebuild; re-hash the artifacts the stored "
                         "manifest lists and fail on any mismatch")
    args = ap.parse_args()
    if args.verify:
        return verify()

    from compass.adapters.models import MODEL_REGISTRY
    from compass.analysis.cohorts import (
        N4_FAILING_MODELS,
        PRIMARY_EXCLUSIONS,
        UNIDENTIFIABLE_PARAMETERS,
        cohort_label,
    )
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

    # A gate present in the roll-up but carrying NO evidence (n == 0) is missing,
    # not passing.  Without the `n == 0` arm, `failed` below tests `passed < n`,
    # which is `0 < 0` -> False, so a gate that ran nothing would slip through as
    # though it had been satisfied.  N2 is exactly that case today.
    missing = [g for g in MANDATORY_GATES
               if g not in gates or gates[g].get("n", 0) == 0]
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
    # N4-failing implementations RETAINED in the primary cohort.
    #
    # The gate roll-up already blocks on N4, but that says only "N4 failed"; it
    # does not say that the failing implementations are still casting votes in
    # the headline decision. The post-response audit found exactly that gap --
    # the cohort was being described as "gate-passing" while bbp_l5pc supplied
    # one of its five lineage votes -- so the manifest states it separately.
    retained_n4 = sorted(N4_FAILING_MODELS - set(PRIMARY_EXCLUSIONS))
    if retained_n4:
        blockers.append(
            "N4-failing implementations retained in the primary cohort: "
            + ", ".join(f"{m} ({', '.join(UNIDENTIFIABLE_PARAMETERS[m])})"
                        for m in retained_n4)
            + " -- the primary cohort is N3-passing, NOT gate-passing. See the "
              "n4_strict and n4_projected sensitivity arms")
    manifest["cohort_label"] = cohort_label("primary")
    manifest["n4_unidentifiable_parameters"] = {
        m: UNIDENTIFIABLE_PARAMETERS[m] for m in sorted(UNIDENTIFIABLE_PARAMETERS)}
    manifest["confirmatory_release_gates"] = confirmatory_gates(
        manifest, gates, retained_n4)

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
    # Exit non-zero when the release is blocked.  Returning 0 meant a pipeline
    # could run to completion, print RELEASE-BLOCKED, and still report success to
    # its caller -- so CI, a shell `&&` chain or run_all.sh would all treat a
    # blocked release as a good run (post-revision audit).  The manifest is still
    # written either way; only the status changes.
    return 0 if manifest["release_verdict"] == "RELEASE-READY" else 1


if __name__ == "__main__":
    sys.exit(main())
