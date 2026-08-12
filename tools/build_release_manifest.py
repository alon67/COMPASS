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
import subprocess
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

# The stages of tools/run_all.sh that actually simulate. A receipt covers the
# simulation only when it contains ALL of them; see `covers_simulation`.
SIMULATION_STAGES = ("campaign", "sobol")


# WHICH CODE CAN CHANGE A GATE RESULT?
#
# The revision tie (2026-08-09 audit, P0-4A) exists to stop evidence produced by
# DIFFERENT CODE counting towards a gate. Applied through the whole-source
# `git_tree()` it treats any tracked source byte as different code -- so editing
# a reporting tool that no gate imports invalidates thousands of records the
# edit provably cannot have affected, and the only remedy is hours of
# re-simulation reproducing identical numbers. That is bookkeeping cost with no
# epistemic gain, and it is why this narrower question is asked instead.
#
# The set is DERIVED, not listed: `src/` and `configs/` in full, plus exactly
# those tools that can WRITE a gate record, found by looking for the calls. A
# tool that starts emitting gate records is covered the moment it does; a
# reporting tool never is. Naming them by hand would let the set be quietly
# gerrymandered to avoid a re-run, which is the failure this comment exists to
# prevent -- so if you are reading this while tempted to add an exclusion,
# don't: change what the code does, or accept the re-run.
#
# Two limits, stated because they are real. It assumes a gate tool reaches its
# helpers through `compass.*` rather than by importing another tool, which holds
# today. And it is FILE-granular: any change to a gate-code file counts, with no
# reasoning about which function was touched.
_GATE_CALLS = ("gate_" + "record(", "write_gate_" + "records(")


def gate_code_paths() -> list[str]:
    """Paths whose contents can change a gate result."""
    out = {"src", "configs"}
    for f in sorted((ROOT / "tools").glob("*.py")):
        if f.name == Path(__file__).name:
            continue          # this file derives the set; it writes no gate
        try:
            txt = f.read_text()
        except Exception:                                         # noqa: BLE001
            continue
        if any(c in txt for c in _GATE_CALLS):
            out.add(f"tools/{f.name}")
    return sorted(out)


def gate_code_changed_since(revision: str) -> list[str]:
    """Gate-affecting files differing between `revision` and the tree now."""
    rev = (revision or "").replace("-dirty", "")
    if not rev or rev == "unversioned":
        return ["<unversioned>"]
    paths = gate_code_paths()
    changed: set[str] = set()
    for args in (["diff", "--name-only", rev, "HEAD", "--", *paths],
                 ["status", "--porcelain", "--", *paths]):
        try:
            out = subprocess.run(["git", "-C", str(ROOT), *args],
                                 capture_output=True, text=True, timeout=30)
        except Exception:                                         # noqa: BLE001
            return ["<git unavailable>"]
        if out.returncode != 0:
            return [f"<git {args[0]} failed>"]
        for line in out.stdout.splitlines():
            changed.add(line[3:].strip() if args[0] == "status" else line.strip())
    return sorted(c for c in changed if c)


def _unattributed(artifacts: list[dict], produced: set[str]) -> list[str]:
    """Which manifest artifacts the run receipt does not account for.

    Imported from tools/run_receipt.py rather than reimplemented: the standalone
    `run_receipt.py check` and this build must not be able to disagree about
    what "produced by this run" means.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rr", ROOT / "tools" / "run_receipt.py")
    rr = iu.module_from_spec(spec)
    spec.loader.exec_module(rr)
    return rr.unattributed(artifacts, produced)


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


def declared_campaign_design() -> dict[str, dict]:
    """The required campaigns and, per campaign, its declared models/regimes.

    Read from ``required_campaigns`` in the inventory and from each campaign's
    own configuration -- NEVER from what is on disk.  The previous version built
    this list from the campaign-config entries that currently have shards, so a
    missing campaign silently stopped being required and ``sobol``, which is not
    a ``campaigns.json`` entry at all, was in no coverage rule despite being
    listed in this same inventory (2026-08-09 post-response-3 audit, P0-4B).
    """
    import importlib.util as iu

    inv = load_inventory()
    req = inv.get("required_campaigns")
    if not req:
        raise SystemExit(
            "configs/release_inventory.json has no `required_campaigns`. "
            "The required set must be declared, not inferred from the files "
            "being checked.")

    spec = iu.spec_from_file_location("_rc", ROOT / "tools" / "run_campaign.py")
    rc = iu.module_from_spec(spec)
    spec.loader.exec_module(rc)
    sspec = iu.spec_from_file_location("_rs", ROOT / "tools" / "run_sobol.py")
    rs = iu.module_from_spec(sspec)
    sspec.loader.exec_module(rs)

    regimes_for = req.get("regimes_for", {})
    out: dict[str, dict] = {}
    for c in req["declared"]:
        if c in rc.CONFIG["campaigns"]:
            cfg = rc.CONFIG["campaigns"][c]
            models = sorted(cfg["sample_sizes"])
            by_model = cfg.get("regimes_by_model", {})
            out[c] = {"models": models,
                      "regimes": {m: list(by_model.get(m, cfg["regimes"]))
                                  for m in models},
                      "source": "configs/experiments/campaigns.json"}
        elif c == "sobol":
            models = sorted(rs.N_BASE)
            out[c] = {"models": models,
                      "regimes": dict.fromkeys(models,
                                               list(regimes_for.get("sobol", ["SOBOL"]))),
                      "source": "tools/run_sobol.py:N_BASE"}
        else:
            raise SystemExit(
                f"required campaign {c!r} is declared in the inventory but has "
                f"no design anywhere: add it to campaigns.json or give it a "
                f"`models_from` rule")
    return out


def resolve_gate_subjects(kind: str) -> list[str]:
    """Turn a declared coverage rule into the concrete subjects it requires."""
    design = declared_campaign_design()
    campaigns = sorted(design)

    if kind == "campaigns_with_shards":
        return campaigns
    if kind == "campaign_model_pairs":
        return [f"{c}/{m}" for c in campaigns for m in design[c]["models"]]
    if kind == "waveform_campaign_model_pairs":
        want = set(load_inventory().get("waveform_campaigns", {})
                   .get("included", campaigns))
        return [f"{c}/{m}" for c in campaigns if c in want
                for m in design[c]["models"]]
    if kind == "campaign_model_regime":
        return [f"{c}/{m}/{r}" for c in campaigns for m in design[c]["models"]
                for r in design[c]["regimes"][m]]
    declared_models = sorted({m for c in campaigns for m in design[c]["models"]})
    if kind == "simulated_models":
        return declared_models
    if kind == "primary_models":
        from compass.analysis.cohorts import PRIMARY_EXCLUSIONS

        return [m for m in declared_models if m not in PRIMARY_EXCLUSIONS]
    raise KeyError(f"unknown subjects_from rule {kind!r}")


def find_orphans(inv: dict) -> dict:
    """Files that exist, hash consistently, and should not be in the release.

    The inventory's `analyses` class accepts any JSON/Parquet under
    ``results/analyses``, so seven obsolete ``*_n4_projected`` outputs and the
    ``perturbations_fast5*`` family were released -- one of them still asserting
    an overall H-G conclusion that the current decision contradicts
    (2026-08-09 post-response-3 audit, P0-4C).
    """
    pol = inv.get("orphan_policy") or {}
    pats = list(pol.get("forbidden_basename_patterns", []))
    withdrawn = pol.get("withdrawn_dir", "results/analyses/_withdrawn")
    hits: list[str] = []
    for base in ("results", "docs", "notebooks"):
        d = ROOT / base
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if rel.startswith(withdrawn) or "_history/" in rel:
                continue
            if _matches(p.name, pats):
                hits.append(rel)
    return {"patterns": pats, "withdrawn_dir": withdrawn, "files": hits,
            "rationale": pol.get("rationale", {}),
            "policy": ("an arm name no longer in the declared registry, or an "
                       "output family the current runner does not generate, is "
                       "an orphan and blocks the release")}


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


def _rheobase_behaviour_check() -> dict:
    """Run the search against firing windows placed between its grid points.

    Executed here, in the release build, because the previous check was
    ``measurement_schema == "probe-3"`` -- a string comparison that stayed true
    while the search returned ``None`` for a cell firing from 0.30 to 0.40 nA.
    No NEURON: the synthetic adapter answers the semantic question.
    """
    try:
        import importlib.util as iu

        spec = iu.spec_from_file_location(
            "_rf", ROOT / "tests" / "unit" / "test_rheobase_failure_states.py")
        m = iu.module_from_spec(spec)
        spec.loader.exec_module(m)

        n = 16
        delta = 1.0 / n
        missed = []
        for k in range(n - 1):                       # a window between every pair
            start = (k + 0.5) * delta
            a = m.probe(lambda amp, s=start: "spike" if s <= amp < s + delta
                        else "silent")
            got = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200,
                                  tol_nA=1e-4, scan_points=n)
            if got is None or abs(got - start) > 1e-3:
                missed.append(round(start, 4))
        # and a low window under a SPIKING ceiling, which probe-3 never scanned
        a = m.probe(lambda amp: "spike" if (0.10 <= amp < 0.15 or amp >= 0.60)
                    else "silent")
        low = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200,
                              tol_nA=1e-3, scan_points=64)
        low_ok = low is not None and abs(low - 0.10) <= 1e-3
        from compass.adapters.base import (
            RHEOBASE_SEARCH_CONTRACT,
            NeuronAdapter,
        )
        return {
            "met": not missed and low_ok,
            "evidence": {
                "windows_between_every_pair_of_grid_points": f"{n - 1} tested, "
                                                             f"{len(missed)} missed",
                "missed_window_starts_nA": missed,
                "low_window_under_a_spiking_ceiling_found": low_ok,
                "scan_points": NeuronAdapter.RHEOBASE_SCAN_POINTS,
                "contract": RHEOBASE_SEARCH_CONTRACT,
            }}
    except Exception as exc:                                      # noqa: BLE001
        return {"met": False, "evidence": f"check could not run: {exc!r}"}


def _perturbation_identity_check() -> dict:
    """Do the STORED perturbation slices carry the identity axes they must?"""
    cache = ROOT / "results" / "analyses" / "_perturbation_cache"
    need = ("measurement_schema", "feature_schema", "sim_code_hash",
            "selection_code_hash", "background_vectors_hash",
            "candidate_rows_hash", "effective_dt_ms", "solver")
    sides = sorted(cache.glob("*.identity.json")) if cache.is_dir() else []
    if not sides:
        return {"met": False,
                "evidence": "no per-slice identity sidecars found under "
                            "results/analyses/_perturbation_cache"}
    incomplete = {}
    for s in sides:
        try:
            rec = json.loads(s.read_text())      # written flat by write_slice_identity
        except Exception:                                         # noqa: BLE001
            incomplete[s.name] = "unreadable"
            continue
        gaps = [k for k in need if rec.get(k) in (None, "")]
        if gaps:
            incomplete[s.name] = gaps
    agg = sorted((ROOT / "results" / "analyses").glob("perturbations_raw*.identity.json"))
    return {"met": not incomplete and bool(agg),
            "evidence": {"n_slice_sidecars": len(sides),
                         "required_axes": list(need),
                         "slices_missing_axes": incomplete,
                         "aggregate_sidecars": [p.name for p in agg]}}


def _waveform_reload_check() -> dict:
    """Did the waveform battery actually recompute from reloaded archives?"""
    val = ROOT / "results" / "validation"
    reports = sorted(val.glob("retained_waveforms_*.json"))
    if not reports:
        return {"met": False, "evidence": "no retained-waveform report found"}
    total = mismatched = 0
    from_archive = []
    for p in reports:
        try:
            d = json.loads(p.read_text())
        except Exception:                                         # noqa: BLE001
            return {"met": False, "evidence": f"{p.name} is unreadable"}
        total += int(d.get("n_features_recomputed", 0))
        mismatched += int(d.get("n_features_mismatched", 0))
        from_archive.append("reloaded" in str(d.get("features_recomputed_from", "")))
    return {"met": total > 0 and mismatched == 0 and all(from_archive),
            "evidence": {"campaigns": [p.name for p in reports],
                         "n_features_recomputed": total,
                         "n_features_mismatched": mismatched,
                         "all_recomputed_from_the_reloaded_archive": all(from_archive)}}


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
    unattributed_here = _unattributed(
        manifest.get("artifacts", []),
        set(receipt.get("artifacts_produced_by_this_run") or []))
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
            # Every REPORTED arm, not only the discovery headline (P0-5).
            "met": all(
                (manifest.get("n4_disposition_by_arm", {}).get(a, {})
                 .get("disposition") in ("projected", "not_applicable"))
                for a in ("primary", "discovery_only", "replication_only")),
            "evidence": {a: manifest.get("n4_disposition_by_arm", {}).get(a)
                         for a in ("primary", "discovery_only", "replication_only")},
            "kind": "runtime"},
        "6_rheobase_handles_non_monotonic_firing": {
            # EXECUTED, not asserted from a version string.  Marking this met
            # because `measurement_schema == "probe-3"` tested a label while the
            # search was still stepping over firing windows between its rungs
            # (2026-08-09 post-response-3 audit, P0-1/P0-4D).
            **_rheobase_behaviour_check(),
            "kind": "runtime"},
        "7_draw_specific_spontaneity_cannot_make_zero_amplitude_protocols": {
            "met": bool(gates.get("D3", {}).get("n"))
                   and gates["D3"]["passed"] == gates["D3"]["n"],
            "evidence": {"D3": f"{gates.get('D3', {}).get('passed')}/"
                               f"{gates.get('D3', {}).get('n')}",
                         "rule": "D3 fails any row with normalised features at "
                                 "rheobase 0 or a ghost step_rheo_* QC flag"},
            "kind": "runtime"},
        "8_cv_preprocessing_is_fold_local": {
            "met": True,
            "evidence": "compass.analysis.hierarchical.FoldPipeline; every LOMO "
                        "row records preprocessing='fold_local'",
            "kind": "implementation"},
        "9_perturbation_cache_reuse_is_rejected_on_any_change": {
            # Reads the identities actually written beside the stored slices,
            # rather than counting axes in the source.
            **_perturbation_identity_check(),
            "kind": "runtime"},
        "10_archived_traces_are_reloaded_and_fail_closed": {
            **_waveform_reload_check(),
            "kind": "runtime"},
        "11_waveform_coverage_meets_declared_policy": {
            "met": bool(n_wave),
            "evidence": cov.get("WAVEFORM", "no coverage record"),
            "kind": "runtime"},
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
                        "interpretation; H-B recorded as UNRESOLVED",
            "kind": "implementation"},
        "15_h_b_input_evidence_is_current": {
            "met": bool(gates.get("H_B_INPUT", {}).get("n"))
                   and gates["H_B_INPUT"]["passed"] == gates["H_B_INPUT"]["n"],
            "evidence": {"H_B_INPUT": f"{gates.get('H_B_INPUT', {}).get('passed')}/"
                                      f"{gates.get('H_B_INPUT', {}).get('n')}",
                         "failed_subjects": gates.get("H_B_INPUT", {}).get("failed"),
                         "rule": "the battery's measurement schema is current, its "
                                 "half-step twin exists, its repeatable subset "
                                 "clears the lineage floor, and H-B carries an "
                                 "explicit evaluable/withheld disposition"},
            "kind": "runtime"},
    }
    for k, v in items.items():
        v["requirement"] = k.split("_", 1)[1].replace("_", " ")
        v.setdefault("kind", "implementation")
    met = [k for k, v in items.items() if v["met"] is True]
    return {
        "source": ("COMPASS_INDEPENDENT_POST_RESPONSE_2_TECHNICAL_SCIENTIFIC_"
                   "AUDIT_2026-08-09.md section 9, extended by the "
                   "post-response-3 audit sections P0-3 and P0-4D"),
        # HOW AN ITEM IS ESTABLISHED, not only whether it is met.
        # Several items were marked met from the presence of an implementation
        # or a schema string, which reads as runtime acceptance evidence and is
        # not (post-response-3 audit, P0-4D).  `runtime` items are decided from
        # an artifact this build read or a check it executed; `implementation`
        # items are decided from the code being present and correct by review.
        "evidence_kinds": {"runtime": sorted(k for k, v in items.items()
                                             if v["kind"] == "runtime"),
                           "implementation": sorted(k for k, v in items.items()
                                                    if v["kind"] == "implementation")},
        "n_met": len(met), "n_total": len(items),
        "n_met_runtime": len([k for k in met if items[k]["kind"] == "runtime"]),
        "n_runtime": len([k for k, v in items.items() if v["kind"] == "runtime"]),
        "met": met,
        "not_met": [k for k, v in items.items() if v["met"] is False],
        "not_machine_decidable": [k for k, v in items.items() if v["met"] is None],
        # WHAT THE RECEIPT COVERS, not merely that one exists.
        #
        # Gate D of the 2026-08-09 post-response-3 audit asks for "one full clean
        # --force-simulations run". A receipt for the DERIVED half of the
        # pipeline is real evidence and is not that, and the difference has to be
        # legible from the manifest rather than from a paragraph in a response
        # document -- otherwise "a passing receipt check" reads as more than it
        # is. `covers_simulation` is the distinction Gate D turns on.
        "clean_build_receipt": {
            "present": bool(receipt),
            "stages_run": receipt.get("stages_run"),
            "stages_failed": receipt.get("stages_failed"),
            "n_artifacts_produced": receipt.get("n_produced"),
            # INTERSECTION, NOT SUBSET, WAS THE BUG. `&` made this true as soon
            # as EITHER simulation stage ran, so a receipt covering `sobol`
            # alone -- 360 blocks of one design -- reported that the run covered
            # the simulation, i.e. the whole 130k-row corpus. The claim is that
            # every simulation stage ran, so the test is containment, and the
            # stages actually missing are named rather than left to be inferred
            # from two lists.
            #
            # AND the run has to have rebuilt the corpus, not merely visited
            # the stage. `--force` preserves feature shards by design, so a
            # campaign stage that ran and reused every shard covers no
            # simulation at all; only `--force-simulations` does, and the
            # receipt now records which force it was.
            "covers_simulation": bool(
                set(SIMULATION_STAGES).issubset(set(receipt.get("stages_run") or []))
                and receipt.get("force_simulations")),
            "simulation_stages": list(SIMULATION_STAGES),
            "simulation_stages_not_in_receipt": sorted(
                set(SIMULATION_STAGES) - set(receipt.get("stages_run") or [])),
            "force_simulations": receipt.get("force_simulations"),
            # The clean-build question, answered from the artifact list this
            # build just assembled instead of from a manifest on disk. `check`
            # reads the previous manifest, and a `--force` run has quarantined
            # it, so the standalone check could never pass on the one kind of
            # run it exists to validate (P1-1, found running this rebuild).
            "n_artifacts_not_produced_by_this_run": len(unattributed_here),
            "artifacts_not_produced_by_this_run": unattributed_here[:200],
            "note": ("`covers_simulation` false means the feature corpus was "
                     "produced outside this receipt. That is not a gap in the "
                     "corpus -- it is regenerated from immutable parameter "
                     "draws whenever the measurement schema changes, and gate "
                     "REPLAY re-simulates a sample of it from scratch -- but it "
                     "is a gap against Gate D's 'one full clean "
                     "--force-simulations run', and is reported as one."),
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
        PRIMARY_EXCLUSIONS,
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
    # EVIDENCE MUST BELONG TO THIS RELEASE, NOT MERELY CARRY A REVISION FIELD.
    #
    # Every gate record found under results/validation was aggregated regardless
    # of the revision that produced it.  At the audited commit all 5,788 records
    # carried `c736e48...-dirty` while the manifest described `2745fa3...-dirty`,
    # and the confirmatory check tested only that a revision field EXISTED
    # (2026-08-09 post-response-3 audit, P0-4A).  A dirty suffix names neither
    # the content of the uncommitted patch nor its equivalence to any commit, so
    # "the evidence has a revision" is not "the evidence is this release's".
    #
    # Stale records are retained in the manifest as history and are reported,
    # but they do not satisfy a gate: the roll-up counts only records whose
    # `git_tree` equals the manifest's.
    this_tree = manifest["git_tree"]
    stale: dict[str, int] = {}
    _gate_ok_cache: dict[str, bool] = {}
    _gate_diffs: dict[str, list[str]] = {}

    def _gate_code_ok(rev: str | None) -> bool:
        if not rev:
            return False
        if rev not in _gate_ok_cache:
            diff = gate_code_changed_since(rev)
            _gate_diffs[rev] = diff
            _gate_ok_cache[rev] = not diff
        return _gate_ok_cache[rev]

    vdir = ROOT / "results" / "validation"
    if vdir.exists():
        for p in sorted(vdir.rglob("*.json")):
            if p.name == OUT.name:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            file_tree = d.get("git_tree")
            for r in d.get("records", []):
                if not (isinstance(r, dict) and "gate" in r):
                    continue
                rec_tree = r.get("git_tree", file_tree)
                # A record counts when NO GATE-AFFECTING FILE has changed since
                # the revision it was written at -- not when the whole source
                # tree is byte-identical. `_gate_code_ok` memoises the git query
                # per revision. Exact-match remains the fast path.
                if rec_tree != this_tree and not _gate_code_ok(rec_tree):
                    stale[r["gate"]] = stale.get(r["gate"], 0) + 1
                    continue
                records.setdefault(r["gate"], []).append(r)
            if "by_gate" in d and (file_tree == this_tree
                                   or _gate_code_ok(file_tree)):
                for g, v in d["by_gate"].items():
                    cur = gates.setdefault(g, {"n": 0, "passed": 0, "failed": [],
                                               "sources": []})
                    cur["n"] += v["n"]
                    cur["passed"] += v["passed"]
                    cur["failed"] += v.get("failed_subjects", [])
                    cur["sources"].append(p.name)
    manifest["gate_records_rejected_as_stale"] = stale
    manifest["gate_evidence_revision"] = {
        "release_revision": this_tree,
        "revisions_seen_in_records": sorted(_gate_ok_cache) or [this_tree],
        "gate_code_paths": gate_code_paths(),
        "gate_code_changed_since_each": {k: v for k, v in _gate_diffs.items() if v},
        "counted_because_gate_code_unchanged": sorted(
            k for k, v in _gate_ok_cache.items() if v and k != this_tree),
        "_doc": ("A gate record counts when no file that can change a gate "
                 "result has changed since the revision it was written at. The "
                 "whole-tree revision may have moved -- documentation, "
                 "reporting tools, verification tools -- and that does not "
                 "invalidate simulation evidence. Any revision listed under "
                 "`counted_because_gate_code_unchanged` differs from the "
                 "release revision in files outside `gate_code_paths`."),
    }
    manifest["gates"] = gates
    # The second assignment used to overwrite the first, and its `policy` string
    # described the rule this build no longer applies. Merged, with the policy
    # rewritten to what the code actually does.
    manifest["gate_evidence_revision"].update({
        "policy": ("a gate record counts when no file that can change a gate "
                   "result has changed since the revision the record was "
                   "written at -- not when the whole source tree matches. "
                   "Documentation, reporting and verification changes move the "
                   "release revision without invalidating simulation evidence, "
                   "and the difference is enumerated above rather than assumed"),
        "records_rejected_as_stale": dict(sorted(stale.items())),
        "n_records_rejected_as_stale": sum(stale.values()),
    })
    manifest["gate_coverage"] = gate_coverage(inv, records)
    manifest["orphans"] = find_orphans(inv)

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
    if manifest["orphans"]["files"]:
        blockers.append(
            f"obsolete outputs present in the release inventory: "
            f"{len(manifest['orphans']['files'])} file(s) matching "
            f"{manifest['orphans']['patterns']} -- e.g. "
            f"{manifest['orphans']['files'][:3]}. Move them under "
            f"{manifest['orphans']['withdrawn_dir']} or delete them; the "
            f"manifest can prove their bytes are unchanged but not that they "
            f"belong in this release")
    if stale:
        blockers.append(
            f"{sum(stale.values())} gate record(s) were produced at a different "
            f"revision than this release and do not count towards any gate: "
            f"{dict(sorted(stale.items()))}. Re-run the gates on this revision.")
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
    # PER ARM AND PER MODEL, never inferred globally from one filename.
    #
    # This tested `SENSITIVITY_ARMS["primary"]["source_experiment"].endswith
    # ("_n4proj")` and, on that basis alone, declared every N4 failure projected
    # and suppressed the retained-N4 blocker for the whole release -- while
    # `replication_only` was reading the unprojected replication table and its
    # decision artifact still listed tomko_ca1 as retained with three
    # unidentifiable parameters (2026-08-09 post-response-3 audit, P0-5).
    from compass.analysis.cohorts import REPORTED_ARMS, n4_disposition_by_arm

    by_arm = n4_disposition_by_arm()
    manifest["n4_disposition_by_arm"] = by_arm
    reported = REPORTED_ARMS
    unresolved = {a: by_arm[a]["n4_failing_models_present"]
                  for a in reported
                  if by_arm.get(a, {}).get("disposition") == "retained_unresolved"}
    retained_n4 = sorted({m for ms in unresolved.values() for m in ms})
    manifest["n4_disposition"] = (
        "projected out of the sampled space before ensemble construction in "
        + ", ".join(a for a in reported
                    if by_arm.get(a, {}).get("disposition") == "projected")
        if not unresolved else
        "retained with unidentifiable parameters intact in " + ", ".join(unresolved))
    if unresolved:
        blockers.append(
            "N4-failing implementations retained in a REPORTED arm: "
            + "; ".join(f"{a}: " + ", ".join(
                f"{m} ({', '.join(UNIDENTIFIABLE_PARAMETERS[m])})" for m in ms)
                for a, ms in sorted(unresolved.items()))
            + " -- these cohorts are N3-passing, NOT gate-passing. See the "
              "n4_strict, n4_retained and replication_n4_retained arms")
    manifest["cohort_label"] = cohort_label("primary")
    manifest["n4_unidentifiable_parameters"] = {
        m: UNIDENTIFIABLE_PARAMETERS[m] for m in sorted(UNIDENTIFIABLE_PARAMETERS)}
    from compass.adapters.base import (
        MEASUREMENT_SCHEMA_VERSION,
        RHEOBASE_SEARCH_CONTRACT,
    )
    from compass.validation.gates import n2_coverage

    manifest["measurement_schema"] = MEASUREMENT_SCHEMA_VERSION
    # The guarantee travels with the schema version, so a reader of the manifest
    # never has to infer what "probe-4" promises (post-response-3 audit, P0-1).
    manifest["rheobase_search_contract"] = RHEOBASE_SEARCH_CONTRACT
    manifest["n2_coverage"] = n2_coverage()
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
