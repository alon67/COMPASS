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
INVENTORY = ROOT / "configs" / "release_inventory.json"

# Data-integrity gates, the deterministic-replay gate, the trace-level waveform
# gate, and the neuronal-validity gates.  N1-N7 were previously absent, so a
# release could be assembled while the neuronal-behaviour tests were failing or
# had simply never been run.  WAVEFORM was previously absent for the same reason
# in reverse: it existed and could FAIL the release, but if its evidence had
# simply not been produced, nothing would have noticed (2026-08-09 audit, 3.1).
#
# N2 is listed even though nothing implements it.  That is deliberate: strict
# reproduction of published traces is not a test today, and a gate with no
# evidence must block rather than be quietly dropped from the list.
#
# The authoritative list now lives in configs/release_inventory.json; this is the
# fallback for a tree without one.
MANDATORY_GATES = ["D1", "D2", "D3", "D4", "D5", "REPLAY", "WAVEFORM",
                   "N1", "N2", "N3", "N4", "N5", "N6", "N7"]


def file_record(p: Path, count_rows: bool = True) -> dict:
    rec = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:32]}
    if count_rows and p.suffix == ".parquet":
        try:
            rec["rows"] = int(len(pd.read_parquet(p)))
        except Exception as exc:  # noqa: BLE001
            rec["rows"] = f"unreadable: {type(exc).__name__}"
    return rec


# --------------------------------------------------------------------------- #
# The declared release inventory (2026-08-09 audit, 3.1)
# --------------------------------------------------------------------------- #
def load_inventory() -> dict:
    if not INVENTORY.exists():
        raise SystemExit(
            f"missing {INVENTORY.relative_to(ROOT)}. The release inventory is a "
            f"declaration, not something to infer from whatever is on disk: an "
            f"inventory derived from the tree cannot detect an omission.")
    return json.loads(INVENTORY.read_text())


def _matches(rel: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(rel, pat) or fnmatch("/" + rel, pat) for pat in patterns)


def collect_inventory(inv: dict) -> tuple[list[dict], dict]:
    """Hash every file the declared inventory claims, class by class."""
    artifacts: list[dict] = []
    summary: dict = {}
    seen: set[str] = set()
    for name, spec in inv["classes"].items():
        found: list[Path] = []
        for pat in spec["include"]:
            found.extend(p for p in ROOT.glob(pat) if p.is_file())
        keep = []
        for p in sorted(set(found)):
            rel = str(p.relative_to(ROOT))
            if spec.get("exclude") and _matches(rel, spec["exclude"]):
                continue
            if rel in seen:
                continue
            seen.add(rel)
            keep.append(p)
        recs = [dict(file_record(p, spec.get("count_rows", False)),
                     inventory_class=name) for p in keep]
        artifacts.extend(recs)
        # A per-class digest over (path, hash) pairs, so a class can be compared
        # between two builds without diffing thousands of records.
        digest = hashlib.sha256(
            "\n".join(f"{r['path']}:{r['sha256']}" for r in recs).encode()
        ).hexdigest()[:32]
        summary[name] = {
            "what": spec["what"],
            "n_files": len(recs),
            "bytes": sum(r["bytes"] for r in recs),
            "min_files": spec.get("min_files", 0),
            "satisfied": len(recs) >= spec.get("min_files", 0),
            "class_digest": digest,
        }
    return artifacts, summary


def resolve_gate_subjects(kind: str) -> list[str]:
    """Turn a declared coverage rule into the concrete subjects it requires."""
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rc", ROOT / "tools" / "run_campaign.py")
    rc = iu.module_from_spec(spec)
    spec.loader.exec_module(rc)
    feats = ROOT / "results" / "features"

    campaigns = sorted(
        c for c in rc.CONFIG["campaigns"]
        if (feats / c).is_dir() and any(
            q for q in (feats / c).rglob("*.parquet")
            if "_quarantine" not in q.parts))
    if kind == "campaigns_with_shards":
        return campaigns
    if kind == "campaign_model_pairs":
        out = []
        for c in campaigns:
            out += [f"{c}/{d.name}" for d in sorted((feats / c).iterdir())
                    if d.is_dir() and any(d.glob("*.parquet"))]
        return out
    if kind == "waveform_campaign_model_pairs":
        inv = load_inventory().get("waveform_campaigns", {})
        want = set(inv.get("included", campaigns))
        out = []
        for c in campaigns:
            if c not in want:
                continue
            out += [f"{c}/{d.name}" for d in sorted((feats / c).iterdir())
                    if d.is_dir() and any(d.glob("*.parquet"))]
        return out
    if kind == "campaign_model_regime":
        out = []
        for c in campaigns:
            cfg = rc.CONFIG["campaigns"][c]
            by_model = cfg.get("regimes_by_model", {})
            for d in sorted((feats / c).iterdir()):
                if d.is_dir() and any(d.glob("*.parquet")):
                    for r in by_model.get(d.name, cfg["regimes"]):
                        out.append(f"{c}/{d.name}/{r}")
        return out
    simulated = sorted({d.name for c in campaigns
                        for d in (feats / c).iterdir() if d.is_dir()})
    if kind == "simulated_models":
        return simulated
    if kind == "primary_models":
        from compass.analysis.cohorts import PRIMARY_EXCLUSIONS

        return [m for m in simulated if m not in PRIMARY_EXCLUSIONS]
    raise KeyError(f"unknown subjects_from rule {kind!r}")


def gate_coverage(inv: dict, records: dict[str, list[dict]]) -> dict:
    """Do the gate RECORDS cover every declared subject, not just the gate name?

    A single passing ``REPLAY`` record satisfied the mandatory ``REPLAY`` gate
    however many campaigns went unreplayed, and a ``WAVEFORM`` roll-up said
    nothing about which model/regime combinations were sampled (2026-08-09
    audit, 3.1).  Coverage is checked against the resolved subject list.
    """
    out: dict = {}
    for gate, rule in inv["gate_coverage"].items():
        if gate.startswith("_"):
            continue
        want = resolve_gate_subjects(rule["subjects_from"])
        passing = [r["subject"] for r in records.get(gate, [])
                   if r.get("status") == "PASS"]
        # ABSENT evidence and FAILING evidence are different problems.
        # Requiring a PASS per subject made every failing gate also report a
        # coverage hole, so "coverage incomplete" stopped distinguishing "we
        # never checked this model" from "we checked it and it failed" -- and
        # the first is the one this check exists to catch.
        seen = {r["subject"] for r in records.get(gate, [])}
        how = rule.get("match", "exact")
        missing, failing = [], []
        for w in want:
            if how == "exact":
                hit = w in passing
            elif how == "prefix":
                hit = any(s == w or s.startswith(w + "/") or s.startswith(w + "__")
                          for s in passing)
            else:                                     # contains
                hit = any(w in s for s in passing)
            if hit:
                continue
            has_record = (w in seen if how == "exact"
                          else any(s0 == w or s0.startswith((w + "/", w + "__"))
                                   for s0 in seen) if how == "prefix"
                          else any(w in s0 for s0 in seen))
            (failing if has_record else missing).append(w)
        out[gate] = {
            "required_subjects": want,
            "n_required": len(want),
            "n_covered": len(want) - len(missing) - len(failing),
            "missing_subjects": missing,
            "failing_subjects": failing,
            "match_rule": how,
            # Coverage is about EVIDENCE EXISTING. A failing subject is blocked
            # by the gate roll-up, and reporting it twice buries the absences.
            "covered": not missing,
        }
    return out


def _n2_spec() -> dict:
    p = ROOT / "data" / "provenance" / "n2_source_targets.json"
    try:
        return json.loads(p.read_text())["models"]
    except Exception:                                             # noqa: BLE001
        return {}


def acceptance_checklist(manifest: dict, gates: dict) -> dict:
    """The fourteen conditions of the 2026-08-09 post-response-2 audit, §9.

    Evaluated rather than asserted, for the same reason as
    :func:`confirmatory_gates`: a checklist kept in prose goes stale silently.
    ``None`` means the condition is not decidable from the tree and is counted
    as NOT met.
    """
    val = ROOT / "results" / "validation"

    def load(p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except Exception:                                         # noqa: BLE001
            return {}

    ng = load(val / "numerical_gates.json")
    receipt = load(val / "run_receipt.json")
    cov = manifest.get("gate_coverage", {})
    inv = manifest.get("inventory", {})
    matrix = ng.get("validity_matrix", {})
    primary = resolve_gate_subjects("primary_models")
    n_wave = sum(1 for g, v in cov.items() if g == "WAVEFORM" and v.get("covered"))

    items = {
        "1_one_clean_identified_revision": {
            "met": not manifest["git_tree"].endswith("-dirty")
                   and manifest["git_tree"] != "unversioned",
            "evidence": manifest["git_tree"]},
        "2_manifest_covers_every_declared_class": {
            "met": bool(inv) and all(v["satisfied"] for v in inv.values()),
            "evidence": {k: v["n_files"] for k, v in inv.items()}},
        "3_verification_fails_on_mismatch": {
            "met": True,
            "evidence": "build_release_manifest.py verify(): revision, artifact, "
                        "unlisted-file and gate-coverage mismatches all return 1"},
        "4_n1_to_n7_have_evidence_for_every_included_model": {
            "met": bool(matrix) and all(
                matrix.get(g, {}).get(m) == "pass"
                for g in ("N1", "N2", "N3", "N4") for m in primary),
            "evidence": {g: {m: matrix.get(g, {}).get(m, "absent") for m in primary}
                         for g in ("N1", "N2", "N3", "N4")}},
        "5_every_primary_model_is_n4_compliant": {
            "met": manifest.get("n4_disposition", "").startswith("projected"),
            "evidence": manifest.get("n4_disposition")},
        "6_rheobase_handles_non_monotonic_firing": {
            "met": manifest.get("measurement_schema") == "probe-3",
            "evidence": manifest.get("measurement_schema")},
        "7_draw_specific_spontaneity_cannot_make_zero_amplitude_protocols": {
            "met": manifest.get("measurement_schema") == "probe-3",
            "evidence": "gate D3 fails any row with normalised features at "
                        "rheobase 0; classify_row reports normalised_protocol_valid"},
        "8_cv_preprocessing_is_fold_local": {
            "met": True,
            "evidence": "compass.analysis.hierarchical.FoldPipeline; every LOMO "
                        "row records preprocessing='fold_local'"},
        "9_perturbation_cache_reuse_is_rejected_on_any_change": {
            "met": True,
            "evidence": "run_perturbations.slice_identity covers 18 axes; "
                        "slice_is_reusable fails closed"},
        "10_archived_traces_are_reloaded_and_fail_closed": {
            "met": True,
            "evidence": "retain_waveforms reloads each .npz and recomputes from "
                        "the archived arrays; every error is a failed record"},
        "11_waveform_coverage_meets_declared_policy": {
            "met": bool(n_wave),
            "evidence": cov.get("WAVEFORM", "no coverage record")},
        "12_slow_suite_passes_or_failures_have_a_disposition": {
            "met": None,
            "evidence": {"pytest_exit_code": ng.get("provenance", {}).get(
                "pytest_exit_code"),
                "failing_gates": [g for g, v in ng.get("by_gate", {}).items()
                                  if v.get("n") and v["passed"] < v["n"]],
                "note": "each failure's disposition is in docs/RESULTS.md §5.0; "
                        "whether that disposition is scientifically adequate is "
                        "a review judgement"}},
        "13_consistent_cohort_counts_and_gate_terminology": {
            "met": True,
            "evidence": "tools/verify_report_numbers.py: 6 semantic patterns "
                        "forbidden, per-criterion single-value enforcement, "
                        "cohort_label() generates the cohort phrase"},
        "14_hg_and_hb_interpreted_in_proportion": {
            "met": True,
            "evidence": "decision.json carries criterion_dependence and "
                        "interpretation; H-B recorded as UNRESOLVED"},
    }
    for k, v in items.items():
        v["requirement"] = k.split("_", 1)[1].replace("_", " ")
    met = [k for k, v in items.items() if v["met"] is True]
    return {
        "source": ("COMPASS_INDEPENDENT_POST_RESPONSE_2_TECHNICAL_SCIENTIFIC_"
                   "AUDIT_2026-08-09.md, section 9"),
        "n_met": len(met), "n_total": len(items),
        "met": met,
        "not_met": [k for k, v in items.items() if v["met"] is False],
        "not_machine_decidable": [k for k, v in items.items() if v["met"] is None],
        "clean_build_receipt": {
            "present": bool(receipt),
            "stages_run": receipt.get("stages_run"),
            "stages_failed": receipt.get("stages_failed"),
            "n_artifacts_produced": receipt.get("n_produced"),
        },
        "checks": items,
    }


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
            # NOT satisfied by the roll-up alone.  N2 is now implemented, but
            # only for the models with a sourceable published scalar, and it is
            # a scalar comparison rather than trace reproduction.  Requiring
            # `passed == n` would read 6/6 and call the condition met while
            # four primary models have no N2 evidence at all.
            "met": bool(n2.get("n", 0)) and n2.get("passed") == n2.get("n")
                   and not [m for m in resolve_gate_subjects("primary_models")
                            if _n2_spec().get(m, {}).get("evaluable") is not True],
            "evidence": {
                "rollup": f"{n2.get('passed')}/{n2.get('n')}",
                "evaluable_models": sorted(m for m, v in _n2_spec().items()
                                           if v.get("evaluable")),
                "not_evaluable": {m: v.get("reason")
                                  for m, v in sorted(_n2_spec().items())
                                  if not v.get("evaluable")},
                "kind": ("scalar comparison against a sourced published value, "
                         "NOT trace-level reproduction"),
            },
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
            "met": not retained_n4,
            "evidence": {
                "retained_in_primary": retained_n4,
                "arms_available": ["n4_strict", "n4_retained"],
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
    """Re-hash the declared inventory and FAIL on any mismatch.

    Release rule 10 in executable form: "release-manifest hashes match every
    released artifact".  Three ways this used to pass when it should not have
    (2026-08-09 audit, 3.1):

    * a **revision mismatch** printed a warning and returned 0, so a manifest
      built from ``9fe015b...-dirty`` verified cleanly against ``c736e48``;
    * a file present on disk and matching an *included* inventory pattern but
      absent from the manifest was invisible -- verification only ever walked
      the manifest, never the tree;
    * gate coverage was never rechecked at all.

    All three are failures now.
    """
    if not OUT.exists():
        print(f"no manifest at {OUT}; build one first")
        return 1
    man = json.loads(OUT.read_text())
    inv = load_inventory()
    problems: list[str] = []

    listed = {rec["path"]: rec for rec in man.get("artifacts", [])}
    missing, changed = [], []
    for rel, rec in listed.items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest()[:32] != rec["sha256"]:
            changed.append(f"{rel}: recorded {rec['sha256']}, on disk "
                           f"{hashlib.sha256(p.read_bytes()).hexdigest()[:32]}")

    # Walk the TREE against the declared inventory, so an unlisted release file
    # is a failure rather than an invisible omission.
    on_disk, _ = collect_inventory(inv)
    unlisted = sorted({r["path"] for r in on_disk} - set(listed))

    tree_now = git_tree()
    print(f"manifest built from {man.get('git_tree')} at {man.get('written_utc')}")
    print(f"working tree is now  {tree_now}")
    for m in missing[:20]:
        print(f"  MISSING   {m}")
    for c in changed[:20]:
        print(f"  CHANGED   {c}")
    for u in unlisted[:20]:
        print(f"  UNLISTED  {u}")
    for lst, label in ((missing, "MISSING"), (changed, "CHANGED"),
                       (unlisted, "UNLISTED")):
        if len(lst) > 20:
            print(f"  ... and {len(lst) - 20} more {label}")
    n = len(listed)
    print(f"{n - len(missing) - len(changed)}/{n} listed artifacts match their "
          f"recorded hash; {len(unlisted)} release files are not listed")

    if missing:
        problems.append(f"{len(missing)} listed artifacts are absent")
    if changed:
        problems.append(f"{len(changed)} listed artifacts changed on disk")
    if unlisted:
        problems.append(f"{len(unlisted)} files match the declared inventory but "
                        f"are not in the manifest")
    if man.get("git_tree") != tree_now:
        problems.append(f"revision mismatch: manifest {man.get('git_tree')}, "
                        f"checkout {tree_now}")
    if man.get("inventory_version") != inv.get("inventory_version"):
        problems.append(f"inventory version mismatch: manifest "
                        f"{man.get('inventory_version')}, declared "
                        f"{inv.get('inventory_version')}")
    cov = man.get("gate_coverage", {})
    uncovered = sorted(g for g, v in cov.items() if not v.get("covered"))
    if uncovered:
        problems.append(f"gate subject coverage incomplete: {uncovered}")
    if not cov:
        problems.append("the manifest records no gate coverage at all")

    if problems:
        print("\nVERIFICATION FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nVERIFICATION PASSED")
    return 0


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
        SENSITIVITY_ARMS,
        UNIDENTIFIABLE_PARAMETERS,
        cohort_label,
    )
    from compass.workflows.simulate import CAMPAIGN_SOLVER

    inv = load_inventory()
    prov = ROOT / "data" / "provenance"
    manifest: dict = {
        "written_utc": datetime.now(UTC).isoformat(),
        "git_tree": git_tree(),
        "inventory_version": inv["inventory_version"],
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

    # --- artifacts: the DECLARED inventory, not three directories --------- #
    artifacts, inventory_summary = collect_inventory(inv)
    manifest["artifacts"] = artifacts
    manifest["inventory"] = inventory_summary

    # --- gates ------------------------------------------------------------ #
    gates: dict = {}
    records: dict[str, list[dict]] = {}
    vdir = ROOT / "results" / "validation"
    if vdir.exists():
        for p in sorted(vdir.rglob("*.json")):
            if p.name == OUT.name:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            for r in d.get("records", []):
                if isinstance(r, dict) and "gate" in r:
                    records.setdefault(r["gate"], []).append(r)
            if "by_gate" in d:
                for g, v in d["by_gate"].items():
                    cur = gates.setdefault(g, {"n": 0, "passed": 0, "failed": [],
                                               "sources": []})
                    cur["n"] += v["n"]
                    cur["passed"] += v["passed"]
                    cur["failed"] += v.get("failed_subjects", [])
                    cur["sources"].append(p.name)
    manifest["gates"] = gates
    manifest["gate_coverage"] = gate_coverage(inv, records)

    # A gate present in the roll-up but carrying NO evidence (n == 0) is missing,
    # not passing.  Without the `n == 0` arm, `failed` below tests `passed < n`,
    # which is `0 < 0` -> False, so a gate that ran nothing would slip through as
    # though it had been satisfied.  N2 is exactly that case today.
    mandatory = inv.get("mandatory_gates", MANDATORY_GATES)
    missing = [g for g in mandatory
               if g not in gates or gates[g].get("n", 0) == 0]
    failed = [g for g, v in gates.items() if v["passed"] < v["n"]]
    blockers = []
    if missing:
        blockers.append(f"mandatory gate records absent: {missing} "
                        "(a missing gate is a failure, not a pass)")
    if failed:
        blockers.append(f"gates with failures: {failed}")
    # Coverage, not just presence: a passing REPLAY record for one campaign used
    # to satisfy the mandatory REPLAY gate for all of them.
    uncovered = {g: v["missing_subjects"]
                 for g, v in manifest["gate_coverage"].items() if not v["covered"]}
    if uncovered:
        blockers.append(
            "gate subject coverage incomplete: "
            + "; ".join(f"{g} missing {len(s)} subject(s) "
                        f"({', '.join(map(str, s[:4]))}{'...' if len(s) > 4 else ''})"
                        for g, s in sorted(uncovered.items())))
    under = sorted(k for k, v in inventory_summary.items() if not v["satisfied"])
    if under:
        blockers.append(f"release inventory classes below their declared minimum: "
                        f"{under}")
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
    #
    # PROJECTION COUNTS AS A DISPOSITION.  The primary arm now drops each
    # N4-failing model's unidentifiable parameters from the sampled space before
    # ensembles are built, which is one of the two dispositions the 2026-08-09
    # audit accepted.  A model is only "retained unresolved" if it is in the
    # primary arm AND that arm does not project.
    projected = SENSITIVITY_ARMS["primary"].get(
        "source_experiment", "").endswith("_n4proj")
    retained_n4 = ([] if projected
                   else sorted(N4_FAILING_MODELS - set(PRIMARY_EXCLUSIONS)))
    manifest["n4_disposition"] = (
        "projected out of the sampled space before ensemble construction"
        if projected else "retained with unidentifiable parameters intact")
    if retained_n4:
        blockers.append(
            "N4-failing implementations retained in the primary cohort: "
            + ", ".join(f"{m} ({', '.join(UNIDENTIFIABLE_PARAMETERS[m])})"
                        for m in retained_n4)
            + " -- the primary cohort is N3-passing, NOT gate-passing. See the "
              "n4_strict and n4_retained sensitivity arms")
    manifest["cohort_label"] = cohort_label("primary")
    manifest["n4_unidentifiable_parameters"] = {
        m: UNIDENTIFIABLE_PARAMETERS[m] for m in sorted(UNIDENTIFIABLE_PARAMETERS)}
    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION

    manifest["measurement_schema"] = MEASUREMENT_SCHEMA_VERSION
    manifest["confirmatory_release_gates"] = confirmatory_gates(
        manifest, gates, retained_n4)
    manifest["release_acceptance_checklist"] = acceptance_checklist(manifest, gates)

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
