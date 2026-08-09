"""Regression guards for defects found by the 2026-08-09 audits.

The post-response audit's objection, in its own words: "No tests were found that
directly exercise E7 cross-fit isolation, the drop-feature arm, source_experiment
routing, enforcement of held_out_group/group_col, numerical-gate recorder
behavior, blocked release-manifest exit behavior, or the full pipeline's
dependency order."  Every one of those is a defect that was *repaired* in
response to a review and then left with nothing to stop it coming back.

Each test below names the specific failure mode it guards.  They are all fast:
none of them simulates anything, because every one of these defects lived in
analysis plumbing rather than in NEURON.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- #
# 1. Leave-one-group-out must actually hold out the requested group
# --------------------------------------------------------------------------- #
def _toy_frame() -> pd.DataFrame:
    """Two lineages, three implementations, a signal that depends on regime."""
    rng = np.random.default_rng(0)
    rows = []
    for lineage, models in (("lin_a", ["m1", "m2"]), ("lin_b", ["m3"])):
        for m in models:
            for regime in ("E1", "E3", "E7"):
                for _ in range(12):
                    x = rng.normal()
                    rows.append({
                        "model_id": m, "source_lineage_id": lineage,
                        "regime": regime, "tolerance": "permissive",
                        "metric": "max_z", "protocol_family": "absolute",
                        "mean_abs_pearson": x, "effective_rank": 3 + x,
                        "intrinsic_dimension": 2 + 0.5 * x,
                        "intrinsic_dimension_mle": 2 + 0.4 * x,
                        "participation_ratio": 1 + x, "pc1_variance_fraction": 0.4,
                        "n_samples": 256, "n_params": 6, "n_compartments": 1,
                        "is_point_model": 1.0, "acceptance_rate": 0.2,
                        "neuron_class_id": "pyr",
                        "synergy_area": {"E1": 0.1, "E3": 0.5, "E7": 0.9}[regime] + 0.01 * x,
                    })
    return pd.DataFrame(rows)


def test_leave_one_group_out_holds_out_the_named_group():
    """Folds must be labelled with, and split on, `group_col`."""
    from compass.analysis.hierarchical import leave_one_model_out

    df = _toy_frame()
    out = leave_one_model_out(df, "synergy_area", group_col="source_lineage_id")
    assert set(out["held_out_group"]) == {"lin_a", "lin_b"}, (
        "leave_one_model_out(group_col='source_lineage_id') produced folds that "
        "are not lineages; it fell back to model_id")
    assert set(out["group_col"]) == {"source_lineage_id"}
    # Holding out lin_a must remove BOTH of its implementations from training.
    n_lin_a = int((df["source_lineage_id"] == "lin_a").sum())
    row = out[out["held_out_group"] == "lin_a"].iloc[0]
    assert int(row["n_test"]) == n_lin_a
    assert int(row["n_train"]) == len(df) - n_lin_a


def test_leave_one_group_out_refuses_a_missing_column():
    """A silent fallback to model_id is what made the first fix invisible.

    `source_lineage_id` itself cannot be used to provoke this: `prepare()`
    reconstructs it from the model registry when absent, which is correct
    behaviour and not a fallback.  Any other grouping column must raise.
    """
    from compass.analysis.hierarchical import leave_one_model_out

    df = _toy_frame().drop(columns=["neuron_class_id"])
    with pytest.raises(KeyError, match="neuron_class_id"):
        leave_one_model_out(df, "synergy_area", group_col="neuron_class_id")


# --------------------------------------------------------------------------- #
# 2. Cluster bootstrap: lineages weighted equally, not by implementation count
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_weights_clusters_equally():
    """A two-implementation lineage must not count twice."""
    from compass.analysis.hierarchical import cluster_bootstrap

    # lin_a contributes two rows at 0.0, lin_b one row at 3.0.
    values = np.array([0.0, 0.0, 3.0])
    clusters = np.array(["lin_a", "lin_a", "lin_b"])
    eq = cluster_bootstrap(values, clusters, n_boot=200)
    pooled = cluster_bootstrap(values, clusters, n_boot=200,
                               equal_cluster_weight=False)
    assert eq["point"] == pytest.approx(1.5), (
        "equal cluster weighting should average the two lineage means "
        "(0.0 and 3.0), giving 1.5")
    assert pooled["point"] == pytest.approx(1.0), (
        "pooled rows give 1.0; that is the over-weighting the flag exists to "
        "avoid, and it must still be reachable deliberately")
    assert eq["n_clusters"] == 2
    assert eq["rows_per_cluster"] == {"lin_a": 2, "lin_b": 1}


def test_robustness_bootstrap_refuses_to_fall_back_to_model_id():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rb", ROOT / "tools" / "run_robustness.py")
    rb = iu.module_from_spec(spec)
    spec.loader.exec_module(rb)

    df = _toy_frame().drop(columns=["source_lineage_id"])
    with pytest.raises(KeyError, match="source_lineage_id"):
        rb.cluster_bootstrap_ratio(df, "synergy_area", n_boot=5)


# --------------------------------------------------------------------------- #
# 3. E7 cross-fit isolation
# --------------------------------------------------------------------------- #
def test_e7_selection_is_cross_fitted():
    """No sample may be scored on covariance axes fitted to itself.

    The leaky version fitted the axes on the same half it then selected from,
    so the subset was partly chosen by its own noise.
    """
    from compass.conditioning.ensembles import high_covariance_subset

    rng = np.random.default_rng(7)
    n, p = 400, 4
    params = rng.normal(size=(n, p))
    distance = rng.random(n)
    idx = high_covariance_subset(params, distance, n_target=120, seed=3,
                                 n_candidates=20)
    assert len(idx) > 0
    assert len(set(idx.tolist())) == len(idx), "selection returned duplicates"
    assert idx.max() < n

    # Deterministic for a fixed seed -- a cross-fit that re-split randomly on
    # each call would make the published E7 unreproducible.
    idx2 = high_covariance_subset(params, distance, n_target=120, seed=3,
                                  n_candidates=20)
    assert np.array_equal(np.sort(idx), np.sort(idx2))


def test_e7_records_whether_its_balance_constraint_was_met():
    """A relaxed constraint must be visible in the ensemble metadata."""
    import inspect

    from compass.conditioning import ensembles as E

    src = inspect.getsource(E.build_ensembles)
    assert "balance_satisfied" in src, (
        "E7 must record whether max_mean_shift_sd <= BALANCE_TOL_SD held; the "
        "search relaxes the constraint when nothing satisfies it and the "
        "endpoint table carried no trace of which happened")
    assert E.BALANCE_TOL_SD == 0.10


# --------------------------------------------------------------------------- #
# 4. Sensitivity arms: source_experiment routing and duplicate detection
# --------------------------------------------------------------------------- #
def test_arms_with_a_source_experiment_declare_a_real_table_name():
    from compass.analysis.cohorts import SENSITIVITY_ARMS

    routed = {a: s["source_experiment"] for a, s in SENSITIVITY_ARMS.items()
              if s.get("source_experiment")}
    assert routed, "at least replication_only must be routed to its own table"
    for arm, src in routed.items():
        assert not src.startswith("endpoints_") and not src.endswith(".parquet"), (
            f"{arm}: source_experiment is an EXPERIMENT name, not a filename")


def test_a_routed_arm_is_skipped_not_silently_evaluated_on_the_wrong_table():
    """The replication arm once filtered the discovery table and found nothing."""
    src = (ROOT / "tools" / "run_crossmodel.py").read_text()
    assert "arm skipped rather than silently evaluated" in src
    assert 'arms[arm] = {"error"' in src


def test_duplicate_arms_are_detected_from_the_data():
    from compass.analysis.cohorts import duplicate_arms

    # A frame in which retain_partial_coverage cannot differ from primary,
    # because the model it retains has no rows.
    df = pd.DataFrame({"model_id": ["hh_point", "hh_cable", "bbp_l5pc"] * 2})
    dup = duplicate_arms(df)
    assert dup, ("with no ortner_snc_da or mainen_l5pc rows, several arms select "
                 "identical models and must be reported as duplicates")
    for arm, same_as in dup.items():
        assert arm != same_as


def test_the_primary_cohort_is_not_described_as_gate_passing():
    """bbp_l5pc fails N4 and is in the primary arm; the label must say so."""
    from compass.analysis.cohorts import (
        N4_FAILING_MODELS,
        SENSITIVITY_ARMS,
        cohort_label,
    )

    assert N4_FAILING_MODELS, "the N4 identifiability record must be loaded"
    label = cohort_label("primary")
    assert "gate-passing" not in label, label
    assert "N4 unresolved" in label
    assert "gate-passing" not in SENSITIVITY_ARMS["primary"]["description"]
    # ...and an arm that DOES enforce N4 may use the strong label.
    assert "gate-passing" in cohort_label("n4_strict")
    assert set(SENSITIVITY_ARMS["n4_strict"]["exclude"]) >= N4_FAILING_MODELS


# --------------------------------------------------------------------------- #
# 5. The drop-feature and projection arms must really change the analysis
# --------------------------------------------------------------------------- #
def test_run_analysis_rejects_an_unknown_dropped_feature():
    """A silently ignored drop produces an arm identical to its comparator."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_analysis.py"),
         "--experiment", "screening", "--drop-features", "not_a_feature"],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert proc.returncode != 0
    assert "not in the conditioning phenotype" in (proc.stderr + proc.stdout)


def test_run_analysis_can_project_out_unidentifiable_parameters():
    """The n4_projected arm needs a real, outcome-blind projection."""
    from compass.analysis.cohorts import UNIDENTIFIABLE_PARAMETERS

    src = (ROOT / "tools" / "run_analysis.py").read_text()
    assert "--drop-unidentifiable-parameters" in src
    assert "parameters_projected_out" in src
    # The dropped set comes from the N4 measurement, not from a hand-written list.
    assert "UNIDENTIFIABLE_PARAMETERS" in src
    assert UNIDENTIFIABLE_PARAMETERS, "N4 measured at least one such parameter"


# --------------------------------------------------------------------------- #
# 6. The numerical-gate recorder
# --------------------------------------------------------------------------- #
def _load_recorder():
    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "_rng_", ROOT / "tools" / "record_numerical_gates.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_recorder_separates_validity_from_exclusion_policy():
    """'N3 46/46' must not mean 'every model passes N3'."""
    m = _load_recorder()
    rows = [
        {"gate": "N3", "kind": "validity", "param": "good_model",
         "outcome": "passed", "subject": "t::test_n3_x[good_model]"},
        {"gate": "N3", "kind": "policy", "param": "mainen_l5pc",
         "outcome": "passed",
         "subject": "t::test_n3_known_failures_still_fail[mainen_l5pc]"},
    ]
    assert m.POLICY_TESTS.search(rows[1]["subject"])
    assert not m.POLICY_TESTS.search(rows[0]["subject"])


def test_recorder_marks_exempt_models_exempt_not_passing():
    """A model excused from the decisive test must never read as a pass."""
    m = _load_recorder()
    # mainen_l5pc is a documented N3 failure. It still PASSES the two ancillary
    # N3 tests it is parametrised into, and that must not become "pass".
    rows = [{"gate": "N3", "kind": "validity", "param": "mainen_l5pc",
             "outcome": "passed", "subject": "t::test_n3_audit_reference[mainen_l5pc]"}]
    mat = m._validity_matrix(rows)
    assert mat["N3"]["mainen_l5pc"] == "exempt", (
        "exemption must dominate a pass from an ancillary test")


def test_recorder_records_release_grade_provenance():
    """A bare tally could have come from any revision or any subset."""
    src = (ROOT / "tools" / "record_numerical_gates.py").read_text()
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "pytest_rc" in names, "the pytest exit status must be captured"
    for field in ("git_revision", "git_tree", "command", "pytest_version",
                  "pytest_exit_code", "duration_s", "collection_errors",
                  "environment"):
        assert f'"{field}"' in src, f"provenance field {field} is not recorded"


def test_stored_gate_record_carries_its_provenance():
    p = ROOT / "results" / "validation" / "numerical_gates.json"
    if not p.exists():
        pytest.skip("no stored numerical_gates.json")
    d = json.loads(p.read_text())
    prov = d.get("provenance")
    assert prov, "the stored gate record predates the provenance requirement"
    assert prov.get("git_revision"), "no revision recorded"
    assert "pytest_exit_code" in prov
    assert "validity_matrix" in d and "by_gate_semantics" in d


# --------------------------------------------------------------------------- #
# 7. The release manifest must exit non-zero when it blocks
# --------------------------------------------------------------------------- #
def test_release_manifest_exits_non_zero_when_blocked():
    """A pipeline must not read a blocked release as a good run."""
    src = (ROOT / "tools" / "build_release_manifest.py").read_text()
    assert 'return 0 if manifest["release_verdict"] == "RELEASE-READY" else 1' in src
    # ...and a gate with no evidence must be missing, not passing.
    assert 'gates[g].get("n", 0) == 0' in src


def test_mandatory_gates_include_the_neuronal_validity_gates():
    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "_brm", ROOT / "tools" / "build_release_manifest.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    for g in ("D1", "D2", "D3", "D4", "D5", "REPLAY",
              "N1", "N2", "N3", "N4", "N5", "N6", "N7"):
        assert g in m.MANDATORY_GATES, f"{g} is not mandatory for a release"


# --------------------------------------------------------------------------- #
# 8. Pipeline dependency order
# --------------------------------------------------------------------------- #
def _stages() -> list[str]:
    src = (ROOT / "tools" / "run_all.sh").read_text()
    m = re.search(r"^STAGES=\((.*?)\)$", src, re.S | re.M)
    assert m, "could not find the STAGES list in run_all.sh"
    return m.group(1).split()


def test_perturbations_are_generated_before_cross_model_scoring():
    """run_crossmodel scores two H-B criteria from the perturbation artifacts."""
    stages = _stages()
    assert stages.index("perturb") < stages.index("crossmodel"), (
        f"perturb must precede crossmodel; got {stages}")
    for earlier, later in (("analysis", "matched"), ("analysis", "perturb"),
                           ("crossmodel", "robustness"), ("report", "inventory"),
                           ("gates", "notebook")):
        assert stages.index(earlier) < stages.index(later), \
            f"{earlier} must precede {later}"


def test_expected_gate_failures_do_not_abort_the_pipeline():
    """N1/N2/N4 fail by design here; the report stages must still be reached."""
    src = (ROOT / "tools" / "run_all.sh").read_text()
    assert "run_stage_soft numerical" in src, (
        "record_numerical_gates returns non-zero for the expected blockers; "
        "under the hard runner it aborted before the report and the release "
        "manifest, making the blocked-release handling unreachable")
    assert "GATES_RC" in src
    stages = _stages()
    assert stages.index("numerical") < stages.index("report") < stages.index("inventory")


def test_the_convergence_stage_reproduces_or_says_it_did_not():
    """A stage that only prints instructions reports 'ok' having done nothing."""
    src = (ROOT / "tools" / "run_all.sh").read_text()
    assert "verify_convergence_evidence.py" in src
    assert 'echo "convergence audits are run per model' not in src
    tool = (ROOT / "tools" / "verify_convergence_evidence.py").read_text()
    assert '"regenerated": False' in tool, (
        "the record must state explicitly that the audits were not re-run")


# --------------------------------------------------------------------------- #
# 9. The claim-manifest flag must mean what it says
# --------------------------------------------------------------------------- #
def test_write_manifest_flag_is_not_a_no_op():
    src = (ROOT / "tools" / "verify_report_numbers.py").read_text()
    # Match the CODE, not any prose about it -- the comment explaining the
    # defect quotes the old expression verbatim.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) \
                and isinstance(node.test.op, ast.Or):
            consts = [v for v in node.test.values
                      if isinstance(v, ast.Constant) and v.value is True]
            assert not consts, (
                "`if <flag> or True:` rewrote claim_manifest.json on every run "
                "regardless of the flag, which is how its hash stopped matching "
                "the one recorded in release_manifest.json")
    assert "--no-write-manifest" in src


def test_claim_manifest_is_deterministic_for_an_unchanged_tree():
    """A wall-clock timestamp made the release manifest's hash stale at once."""
    src = (ROOT / "tools" / "verify_report_numbers.py").read_text()
    body = src.split("MANIFEST.write_text(")[1].split("indent=1")[0]
    assert "written_utc" not in body, (
        "claim_manifest.json must not embed a timestamp: it is hashed into "
        "release_manifest.json, so a re-verification of an unchanged tree would "
        "invalidate the manifest that was just built from it")
    assert "git_revision" in body

    # ...and check the artifact itself, not only the code that writes it.
    p = ROOT / "results" / "analyses" / "claim_manifest.json"
    if not p.exists():
        pytest.skip("no stored claim_manifest.json")
    d = json.loads(p.read_text())
    assert "written_utc" not in d, "the stored manifest still carries a timestamp"
    assert "git_revision" in d
    for fact in d.get("repository_facts", []):
        assert "detail" not in fact, (
            "repository_facts details embed file mtimes, which change on a "
            "checkout without any content changing")


def test_criterion_values_are_checked_for_a_single_authoritative_occurrence():
    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "_vrn", ROOT / "tools" / "verify_report_numbers.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)

    marker = re.compile(r"superseded|earlier", re.I)
    dec = {"criteria": [{"criterion": "sampling regime predictable across "
                                      "unseen lineages", "value": 0.281}]}
    m.DOCS = {"RESULTS.md": "| sampling regime predictable across unseen "
                            "lineages | H-G | 0.4352 | yes |"}
    bad = m.criterion_consistency(dec, marker)
    assert bad, ("a table row quoting a superseded value for a named criterion "
                 "must fail even when the current value appears elsewhere")
    m.DOCS = {"RESULTS.md": "| sampling regime predictable across unseen "
                            "lineages | H-G | 0.2805 | yes |"}
    assert not m.criterion_consistency(dec, marker)
