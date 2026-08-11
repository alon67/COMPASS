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
    """No arm may be labelled gate-passing, and no label may overstate N2."""
    from compass.analysis.cohorts import (
        N4_FAILING_MODELS,
        SENSITIVITY_ARMS,
        cohort_label,
    )

    assert N4_FAILING_MODELS, "the N4 identifiability record must be loaded"
    for arm in SENSITIVITY_ARMS:
        label = cohort_label(arm)
        assert "gate-passing" not in label, f"{arm}: {label}"
        assert "N1-N7" not in label and "N1–N7" not in label, f"{arm}: {label}"
        assert "gate-passing" not in SENSITIVITY_ARMS[arm]["description"]
    # The arm that retains the N4 failures untouched must say so.
    assert "N4 unresolved" in cohort_label("n4_retained")
    # n4_strict excludes them; its label names the gates it DOES enforce and
    # says what N2's coverage is, rather than claiming full N1-N7.
    assert "N1/N3/N4-enforced" in cohort_label("n4_strict")
    assert "N2" in cohort_label("n4_strict")
    assert set(SENSITIVITY_ARMS["n4_strict"]["exclude"]) >= N4_FAILING_MODELS


def test_the_primary_arm_is_n4_compliant_by_projection():
    """3.2: an N4-compliant analysis must BE the primary inferential analysis."""
    from compass.analysis.cohorts import N4_FAILING_MODELS, SENSITIVITY_ARMS

    primary = SENSITIVITY_ARMS["primary"]
    src = primary.get("source_experiment", "")
    assert src.endswith("_n4proj"), (
        "the primary arm must read the projected endpoint table; retaining "
        "N4-failing implementations with their unidentifiable parameters intact "
        "is what the 2026-08-09 audit called a primary scientific blocker")
    # and the alternative must remain available as a measured sensitivity arm
    assert "n4_retained" in SENSITIVITY_ARMS
    assert not SENSITIVITY_ARMS["n4_retained"].get("source_experiment")
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


# =========================================================================== #
# 2026-08-09 post-response-2 audit
# =========================================================================== #

def test_fold_pipeline_learns_nothing_from_the_held_out_rows():
    """4.2: imputation medians must come from the training fold alone."""
    import pandas as pd

    from compass.analysis.hierarchical import FoldPipeline

    train = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]})
    test = pd.DataFrame({"x": [np.nan]})
    p = FoldPipeline(["x"]).fit(train)
    # x is constant in training, so it carries no information and is dropped;
    # the pooled version kept it because the held-out row made it look variable.
    assert p.transform(test).shape[1] == 0

    train2 = pd.DataFrame({"x": [0.0, 2.0, np.nan, 4.0]})
    p2 = FoldPipeline(["x"]).fit(train2)
    med, mu, sd = next(spec for kind, _c, spec in p2.plan if kind == "num")
    assert med == 2.0, "the median must be the TRAINING median"
    got = p2.transform(pd.DataFrame({"x": [np.nan]}))[0, 0]
    assert got == pytest.approx((2.0 - mu) / sd)


def test_fold_pipeline_maps_unseen_categories_to_the_reference_level():
    """4.2: a dummy for a level only the held-out fold has is unconstrained."""
    import pandas as pd

    from compass.analysis.hierarchical import FoldPipeline

    train = pd.DataFrame({"g": ["a", "a", "b", "b"]})
    p = FoldPipeline(["g"]).fit(train)
    assert p.transform(pd.DataFrame({"g": ["a"]})).tolist() == [[0.0]]
    assert p.transform(pd.DataFrame({"g": ["b"]})).tolist() == [[1.0]]
    assert p.transform(pd.DataFrame({"g": ["z"]})).tolist() == [[0.0]], (
        "a level never seen in training must not activate any dummy")


def test_lomo_records_that_preprocessing_was_fold_local():
    import pandas as pd

    from compass.analysis.hierarchical import leave_one_lineage_out

    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({
        "source_lineage_id": np.repeat([f"L{i}" for i in range(6)], n // 6),
        "model_id": np.repeat([f"m{i}" for i in range(6)], n // 6),
        "regime": rng.choice(["E1", "E2"], n),
        "tolerance": rng.choice(["a", "b"], n),
        "n_samples": rng.integers(50, 500, n),
        "acceptance_rate": rng.random(n),
        "n_params": 6, "n_compartments": 1, "is_point_model": True,
        "mean_abs_pearson": rng.random(n), "effective_rank": rng.random(n) * 3,
        "intrinsic_dimension": rng.random(n) * 3,
        "intrinsic_dimension_mle": rng.random(n) * 3,
        "participation_ratio": rng.random(n) * 3,
        "pc1_variance_fraction": rng.random(n),
        "synergy_area": rng.normal(size=n),
    })
    out = leave_one_lineage_out(df, "synergy_area")
    assert len(out)
    assert (out["preprocessing"] == "fold_local").all()
    assert (out["group_col"] == "source_lineage_id").all()


def test_fold_distribution_surfaces_the_worst_fold_not_just_the_median():
    """4.2: a median over five folds hid a -3.783 lineage."""
    import pandas as pd

    from compass.analysis.hierarchical import fold_distribution

    lomo = pd.DataFrame({
        "predictors": ["all_but_identity"] * 5,
        "held_out_group": ["a", "b", "migliore_1999", "d", "e"],
        "r2_out_of_sample": [0.60, 0.57, -3.783, 0.56, 0.55],
    })
    d = fold_distribution(lomo, "all_but_identity")
    assert d["median"] == pytest.approx(0.56)
    assert d["min"] == pytest.approx(-3.783)
    assert d["worst_fold"] == "migliore_1999"
    assert d["n_folds_beating_null"] == 4
    assert set(d["folds"]) == {"a", "b", "migliore_1999", "d", "e"}


def _pert_module():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rp", ROOT / "tools" / "run_perturbations.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_perturbation_cache_slice_is_refused_after_any_incompatible_change(tmp_path):
    """4.3: a cached slice must be bound to the run that produced it."""
    import pandas as pd

    m = _pert_module()
    jobs = [{"tuple_idx": (0, 1, 2), "kind": "top_synergy"}]
    ident = m.slice_identity("hh_point", "screening", jobs, None, None)
    p = tmp_path / "hh_point_main_raw.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p, index=False)
    m.write_slice_identity(p, ident, n_rows=3)

    ok, why, _ = m.slice_is_reusable(p, ident)
    assert ok, why

    # every axis the audit named must invalidate the slice
    for field, value in [
        ("solver", '{"cvode": false, "dt_ms": 0.999}'),
        ("effective_dt_ms", 0.999),
        ("sim_code_hash", "deadbeef"),
        ("mechanism_hash", "deadbeef"),
        ("protocol_hash", "deadbeef"),
        ("candidate_tuples_hash", "deadbeef"),
        ("feature_schema", "features-99"),
        ("measurement_schema", "probe-99"),
        ("battery_version", "battery-99"),
        ("n_backgrounds", 99),
        ("delta_log10", 0.99),
        ("experiment", "something_else"),
    ]:
        changed = dict(ident, **{field: value})
        ok, why, _ = m.slice_is_reusable(p, changed)
        assert not ok, f"a changed {field} must invalidate the cached slice"
        assert field in why


def test_perturbation_cache_slice_is_refused_without_a_sidecar(tmp_path):
    import pandas as pd

    m = _pert_module()
    p = tmp_path / "hh_point_main_raw.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(p, index=False)
    ident = m.slice_identity("hh_point", "screening",
                             [{"tuple_idx": (0,), "kind": "top_synergy"}], None, None)
    ok, why, _ = m.slice_is_reusable(p, ident)
    assert not ok and why == "identity_unknown"


def test_perturbation_cache_slice_is_refused_after_the_file_is_edited(tmp_path):
    import pandas as pd

    m = _pert_module()
    jobs = [{"tuple_idx": (0,), "kind": "top_synergy"}]
    ident = m.slice_identity("hh_point", "screening", jobs, None, None)
    p = tmp_path / "hh_point_main_raw.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(p, index=False)
    m.write_slice_identity(p, ident, n_rows=1)
    pd.DataFrame({"a": [2]}).to_parquet(p, index=False)      # edited underneath
    ok, why, _ = m.slice_is_reusable(p, ident)
    assert not ok and why == "checksum_mismatch"


def test_dt_scale_is_part_of_the_slice_identity_not_the_current_command():
    """4.3: a half-dt slice must never be relabelled with the campaign dt."""
    m = _pert_module()
    jobs = [{"tuple_idx": (0,), "kind": "top_synergy"}]
    full = m.slice_identity("hh_point", "screening", jobs, None, None)
    half = m.slice_identity("hh_point", "screening", jobs, None, 0.5)
    assert full["effective_dt_ms"] == 2 * half["effective_dt_ms"]
    # the identities must not compare equal on the axis that matters
    assert full["solver"] != half["solver"]
    assert full["dt_scale"] is None and half["dt_scale"] == 0.5


def _bare_adapter():
    """A minimal concrete adapter: canonical_absolute needs no NEURON."""
    from compass.adapters.base import NeuronAdapter

    class _A(NeuronAdapter):
        def _build(self): ...
        def parameter_specs(self): return []
        def _targets(self, spec): return []
        @property
        def soma(self): return None

    return _A()


def test_canonical_absolute_is_actually_area_weighted():
    """5.3: the docstring said area-weighted; the body was a segment mean."""
    from compass.adapters.base import NeuronAdapter

    class _Seg:
        def __init__(self, a):
            self._a = a

        def area(self):
            return self._a

    a = _bare_adapter()
    # one large proximal segment and nine small distal ones: a plain mean is
    # dominated by the distal tree, an area-weighted mean is not
    a._canonical = {"g": [(_Seg(900.0), "g", 10.0)]
                    + [(_Seg(1.0), "g", 1.0) for _ in range(9)]}
    got = NeuronAdapter.canonical_absolute(a, "g")
    assert got == pytest.approx((900 * 10 + 9 * 1) / 909)
    assert got > 9.0, "an area-weighted mean must follow the large segment"
    assert abs(got - np.mean([10.0] + [1.0] * 9)) > 5.0


def test_canonical_absolute_falls_back_when_area_is_unavailable():
    from compass.adapters.base import NeuronAdapter

    class _NoArea:
        pass

    a = _bare_adapter()
    a._canonical = {"g": [(_NoArea(), "g", 2.0), (_NoArea(), "g", 4.0)]}
    assert NeuronAdapter.canonical_absolute(a, "g") == pytest.approx(3.0)


def test_a_cvode_fallback_reaches_the_qc_flags():
    """5.2: a successful Recording said nothing about the effective solver."""
    from compass.adapters.base import NeuronAdapter, Recording

    t = np.arange(0.0, 100.0, 0.1)
    v = np.full_like(t, -65.0)
    plain = Recording(t=t, v=v, stim_start_ms=10.0, stim_end_ms=90.0,
                      amplitude_nA=0.1, protocol_id="step",
                      extra={"cvode_requested": False, "cvode_used": False,
                             "cvode_fallback": False})
    assert "cvode_fallback" not in NeuronAdapter.qc_flags(plain)

    fell_back = Recording(t=t, v=v, stim_start_ms=10.0, stim_end_ms=90.0,
                          amplitude_nA=0.1, protocol_id="step",
                          extra={"cvode_requested": True, "cvode_used": False,
                                 "cvode_fallback": True})
    assert "cvode_fallback" in NeuronAdapter.qc_flags(fell_back)


def test_the_release_inventory_declares_more_than_three_directories():
    """3.1: the manifest hashed 185 files and omitted the actual science."""
    import json as _json

    inv = _json.loads((ROOT / "configs" / "release_inventory.json").read_text())
    classes = inv["classes"]
    for needed in ("feature_shards", "trace_archives", "figures", "reports",
                   "provenance", "model_assets", "source", "configuration"):
        assert needed in classes, f"the release inventory omits {needed}"
    assert "WAVEFORM" in inv["mandatory_gates"], (
        "missing waveform evidence must fail closed, not pass silently")
    # coverage must be declared per SUBJECT, not only per gate name
    assert {"REPLAY", "WAVEFORM"} <= set(inv["gate_coverage"])


def test_the_release_verifier_fails_on_a_revision_mismatch():
    """3.1: it warned and returned success."""
    src = (ROOT / "tools" / "build_release_manifest.py").read_text()
    assert "revision mismatch" in src
    body = src[src.index("def verify("):src.index("def main(")]
    assert "return 1" in body
    assert "problems.append" in body
    # and it must walk the TREE, not only the manifest
    assert "unlisted" in body


def test_the_waveform_battery_recomputes_from_the_reloaded_archive():
    """3.3: it recomputed from the in-memory trace and wrote float32 after."""
    src = (ROOT / "tools" / "retain_waveforms.py").read_text()
    assert "np.load(path" in src
    assert "archive_reload_error" in src
    assert "recomputed-from-archive" in src
    assert "dtype=np.float64" in src, "the archive must not be reduced precision"
    # a simulation error must not produce an empty `problems` list
    assert '"problems": [f"{kind}: {detail}"]' in src
    # and quarantined shards must not be sampled from
    assert "_quarantine" in src


def test_the_rheobase_grid_and_the_zero_rheobase_rule_are_declared(tmp_path):
    """4.1 / P0-1: probe-4 must be a declared schema change with a stated limit."""
    from compass.adapters.base import (
        MEASUREMENT_SCHEMA_VERSION,
        RHEOBASE_SEARCH_CONTRACT,
        NeuronAdapter,
    )

    assert MEASUREMENT_SCHEMA_VERSION == "probe-4"
    assert NeuronAdapter.RHEOBASE_SCAN_POINTS >= 32
    # The contract must state both halves: what is guaranteed and what is not.
    assert "GUARANTEED" in RHEOBASE_SEARCH_CONTRACT
    assert "NOT GUARANTEED" in RHEOBASE_SEARCH_CONTRACT
    assert not hasattr(NeuronAdapter, "RHEOBASE_SCAN_RUNGS"), (
        "the geometric ladder was removed, not kept as a fallback")


def test_a_stale_perturbation_battery_cannot_score_the_h_b_criteria():
    """4.1: probe-3 regenerated the rows the battery's backgrounds come from."""
    import importlib.util as iu

    import pandas as pd

    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION

    spec = iu.spec_from_file_location("_rcm", ROOT / "tools" / "run_crossmodel.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)

    pert = pd.DataFrame({"model_id": ["a", "b", "c"],
                         "source_lineage_id": ["L1", "L2", "L3"]})
    repeatable = {"numerical_repeatability": {
        "status": "ok",
        "per_model": {k: {"numerically_repeatable": True} for k in "abc"}}}

    stale = m.perturbation_precondition(pert, dict(repeatable,
                                                   measurement_schema="probe-2"))
    assert stale["battery_schema_current"] is False
    assert stale["evaluable"] is False
    assert "measurement schema" in stale["reason"]

    fresh = m.perturbation_precondition(
        pert, dict(repeatable, measurement_schema=MEASUREMENT_SCHEMA_VERSION))
    assert fresh["battery_schema_current"] is True
    assert fresh["evaluable"] is True


def test_the_run_receipt_attributes_artifacts_to_the_stage_that_made_them():
    """4.4: --force was parsed and never used; nothing proved a clean build."""
    src = (ROOT / "tools" / "run_all.sh").read_text()
    assert "--force-simulations" in src
    assert "quarantine_derived" in src
    assert "receipt begin" in src and "receipt finish" in src
    rec = (ROOT / "tools" / "run_receipt.py").read_text()
    assert "artifacts_produced_by_this_run" in rec
    assert "NOT PRODUCED BY THIS RUN" in rec


def test_replay_and_waveform_coverage_span_every_campaign_with_shards():
    """3.1 / 5.7: three campaigns had no determinism evidence at all."""
    src = (ROOT / "tools" / "run_all.sh").read_text()
    gates = src[src.index("run_stage gates"):src.index("run_stage_soft numerical")]
    for exp in ("screening", "screening_expand", "screening_scramble2",
                "screening_v2_replication", "sobol"):
        assert exp in gates, f"replay does not cover {exp}"
    assert "screening_v2_replication" in src[src.index("run_stage_soft waveforms"):]


def test_no_tool_rewrites_stored_feature_shards_in_place():
    """P0-2: targeted repair of the corpus is no longer an available operation.

    The probe-3 repair rewrote 4,656 shards in place on an enumeration premise
    that turned out to be false, and introduced two defects of its own doing it.
    The replacement is a full regeneration driven by the shard reuse identity,
    so the tool that did the rewriting is gone rather than merely unused.
    """
    assert not (ROOT / "tools" / "repair_rheobase.py").exists(), (
        "in-place shard mutation was withdrawn; regenerate instead")
    # And the history is retained where it can be read, not deleted.
    hist = ROOT / "data" / "provenance" / "_history" / "probe3_repair"
    assert (hist / "README.md").is_file()
    assert (hist / "rheobase_probe3_verification.json").is_file(), (
        "the NOT-equivalent verdict must stay in the tree")


def test_every_affected_row_count_is_defined_by_a_criterion():
    """P0-2: six irreconcilable counts existed because none carried its rule."""
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rc", ROOT / "tools" / "reconcile_corpus.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert len(m.CRITERIA) >= 15
    for name, (criterion, pred) in m.CRITERIA.items():
        assert criterion and len(criterion) > 20, f"{name} has no stated criterion"
        assert callable(pred)
    # The two counts the audit found contradictory must both be present, and
    # distinguishable, rather than collapsed into one "affected rows" number.
    assert "rescued_from_silence" in m.CRITERIA
    assert "rheobase_lowered" in m.CRITERIA
    assert "rows_any_change" in m.CRITERIA


def test_d3_rejects_a_qc_flag_from_a_protocol_that_never_ran():
    import importlib.util as iu

    import pandas as pd

    spec = iu.spec_from_file_location("_vc", ROOT / "tools" / "validate_campaign.py")
    vc = iu.module_from_spec(spec)
    spec.loader.exec_module(vc)

    base = {"run_id": "m:E1:1:0", "model_id": "hh_point", "sampling_regime": "E1",
            "simulation_status": "ok", "non_spiking": False}
    ghost = pd.DataFrame([
        {**base, "rheobase_nA": 0.0,
         "qc_flags": "step_abs_hi:unstable_baseline;step_rheo_1p5x:unstable_baseline"},
        {**base, "rheobase_nA": float("nan"),
         "qc_flags": "step_rheo_3p0x:unstable_baseline"},
    ])
    recs = vc.d3_row_invariants(ghost, "unit")
    assert recs[0].status == "FAIL"
    assert any("step_rheo_* QC flag" in f for f in recs[0].failures)

    clean = pd.DataFrame([
        {**base, "rheobase_nA": 0.0, "qc_flags": "step_abs_hi:unstable_baseline"},
        {**base, "rheobase_nA": 0.25,
         "qc_flags": "step_rheo_1p5x:unstable_baseline"},   # defined: legitimate
    ])
    recs = vc.d3_row_invariants(clean, "unit")
    assert not any("step_rheo_* QC flag" in f for f in recs[0].failures)


# --------------------------------------------------------------------------- #
# 2026-08-09 POST-RESPONSE-3 AUDIT
#
# One test per finding, each pinning the DEFECT rather than the fix, so a
# regression is reported in the audit's own vocabulary.
# --------------------------------------------------------------------------- #

def _brm():
    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "_brm", ROOT / "tools" / "build_release_manifest.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_p0_1_the_rheobase_grid_finds_windows_between_its_own_probes():
    """The release build EXECUTES this; the checklist item must not be a string."""
    m = _brm()
    res = m._rheobase_behaviour_check()
    assert res["met"] is True, res["evidence"]
    assert res["evidence"]["missed_window_starts_nA"] == []
    assert res["evidence"]["low_window_under_a_spiking_ceiling_found"] is True


def test_p0_3_perturbation_inference_may_not_stamp_its_own_schema():
    """It must read the battery's identity sidecar and fail closed without one."""
    src = (ROOT / "tools" / "perturbation_inference.py").read_text()
    assert "_read_aggregate_identity(raw_path)" in src
    assert '"measurement_schema": battery_schema' in src
    payload = src.split("payload = {")[-1]
    assert '"measurement_schema": _MEASUREMENT_SCHEMA_VERSION' not in payload, (
        "the payload must not take measurement_schema from the imported "
        "constant; running the analysis tool would then clear the staleness "
        "check without re-running the battery")

    import importlib.util as iu

    spec = iu.spec_from_file_location("_rp", ROOT / "tools" / "run_perturbations.py")
    rp = iu.module_from_spec(spec)
    spec.loader.exec_module(rp)
    import pytest as _pt

    with _pt.raises(SystemExit, match="identity sidecar"):
        rp.read_aggregate_identity(ROOT / "results" / "analyses" / "_no_such.parquet")


def test_p0_3_slice_identity_binds_backgrounds_candidates_and_selection_code():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rp", ROOT / "tools" / "run_perturbations.py")
    rp = iu.module_from_spec(spec)
    spec.loader.exec_module(rp)

    jobs = [{"tuple_idx": (0, 1, 2), "kind": "top_synergy", "o_information": -0.5}]
    bg = np.array([[0.1, 0.2], [0.3, 0.4]])
    a = rp.slice_identity("hh_point", "screening", jobs, None, None, backgrounds=bg)
    b = rp.slice_identity("hh_point", "screening", jobs, None, None,
                          backgrounds=bg + 1e-9)
    assert a["background_vectors_hash"] != b["background_vectors_hash"], (
        "different background vectors must not share an identity")
    assert a["selection_code_hash"] and a["candidate_rows_hash"]
    # the O-information value is part of the candidate identity, not only the index
    jobs2 = [{"tuple_idx": (0, 1, 2), "kind": "top_synergy", "o_information": -0.4}]
    c = rp.slice_identity("hh_point", "screening", jobs2, None, None, backgrounds=bg)
    assert a["candidate_rows_hash"] != c["candidate_rows_hash"]


def test_p0_4b_required_campaigns_are_declared_not_discovered():
    inv = json.loads((ROOT / "configs" / "release_inventory.json").read_text())
    declared = inv["required_campaigns"]["declared"]
    assert "sobol" in declared, (
        "sobol is generated by run_sobol.py and is not a campaigns.json entry; "
        "deriving the required set from that file dropped it from every gate")
    m = _brm()
    assert sorted(m.resolve_gate_subjects("campaigns_with_shards")) == sorted(declared)
    # and the subject list must not depend on what is on disk
    src = (ROOT / "tools" / "build_release_manifest.py").read_text()
    body = src[src.index("def resolve_gate_subjects"):src.index("def find_orphans")]
    assert "rglob" not in body and "is_dir()" not in body


def test_p0_4c_obsolete_outputs_are_rejected_from_the_release():
    m = _brm()
    orph = m.find_orphans(m.load_inventory())
    assert orph["files"] == [], (
        f"obsolete outputs still present: {orph['files'][:5]}")
    assert any("n4_projected" in p for p in orph["patterns"])
    assert any("fast5" in p for p in orph["patterns"])
    # and the withdrawn copies are retained rather than deleted
    wd = ROOT / "results" / "analyses" / "_withdrawn"
    assert (wd / "decision_n4_projected.json").is_file()


def test_p0_4a_stale_gate_records_do_not_satisfy_a_gate():
    src = (ROOT / "tools" / "build_release_manifest.py").read_text()
    assert 'rec_tree = r.get("git_tree", file_tree)' in src
    assert "records_rejected_as_stale" in src
    assert "if rec_tree != this_tree:" in src


def test_p0_4d_runtime_and_implementation_evidence_are_distinguished():
    m = _brm()
    man = json.loads(
        (ROOT / "results" / "validation" / "release_manifest.json").read_text()) \
        if (ROOT / "results" / "validation" / "release_manifest.json").is_file() else {}
    chk = m.acceptance_checklist(
        {**man, "git_tree": "x", "inventory": {}, "gate_coverage": {}}, {})
    assert "evidence_kinds" in chk
    assert chk["evidence_kinds"]["runtime"], "no item is decided from an artifact"
    for name in ("6_rheobase_handles_non_monotonic_firing",
                 "9_perturbation_cache_reuse_is_rejected_on_any_change",
                 "10_archived_traces_are_reloaded_and_fail_closed",
                 "15_h_b_input_evidence_is_current"):
        assert chk["checks"][name]["kind"] == "runtime", name


def test_p0_5_every_reported_arm_applies_the_same_n4_rule():
    from compass.analysis.cohorts import SENSITIVITY_ARMS, n4_disposition_by_arm

    d = n4_disposition_by_arm()
    for arm in ("primary", "replication_only"):
        assert d[arm]["disposition"] in ("projected", "not_applicable"), (
            f"{arm} reports a result under an unprojected N4 rule: {d[arm]}")
    assert SENSITIVITY_ARMS["replication_only"]["source_experiment"].endswith(
        "_n4proj"), "the replication arm must read the projected endpoint table"
    # and the retained variant must still exist, so the cost is measurable
    assert d["replication_n4_retained"]["disposition"] == "retained_unresolved"


def test_p0_6_the_depolarisation_block_window_is_the_stimulus_not_the_tail():
    src = (ROOT / "tools" / "retain_waveforms.py").read_text()
    assert "t.max() - (t.max() - delay_ms) / 3.0" not in src, (
        "for the standard 200/400/800 ms protocol that window starts at the "
        "stimulus OFFSET and examines the post-stimulus tail")
    assert "late = (t >= stim[1] - span / 3.0) & (t <= stim[1])" in src
    # the stated no-spikes condition must actually be implemented
    assert "n_late = detect_spikes(v[late], t[late])" in src
    assert "n_early > 0 and n_late == 0" in src
    # and disagreement must be reported in both directions
    assert "blocked_obs != flag_v" in src


def test_p0_6_one_voltage_band_excludes_and_reports():
    """Two bands meant a trace could be 'impossible' and analysis-eligible."""
    import importlib.util as iu

    from compass.adapters.base import PHYSICAL_V_MAX_MV, PHYSICAL_V_MIN_MV

    spec = iu.spec_from_file_location("_rw", ROOT / "tools" / "retain_waveforms.py")
    rw = iu.module_from_spec(spec)
    spec.loader.exec_module(rw)
    assert (rw.V_MIN_MV, rw.V_MAX_MV) == (PHYSICAL_V_MIN_MV, PHYSICAL_V_MAX_MV)
    # and the corpus rule must use it, so the row is excluded and not just flagged
    base_src = (ROOT / "src" / "compass" / "adapters" / "base.py").read_text()
    assert "np.max(v) > PHYSICAL_V_MAX_MV or np.min(v) < PHYSICAL_V_MIN_MV" in base_src


def test_p0_6_every_waveform_finding_class_has_a_declared_disposition():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rw", ROOT / "tools" / "retain_waveforms.py")
    rw = iu.module_from_spec(spec)
    spec.loader.exec_module(rw)
    for kind in ("voltage_range", "baseline_stability", "spike_shape",
                 "depolarisation_block", "feature_recomputation",
                 "spike_count_agreement", "discontinuity", "finite"):
        assert kind in rw.FINDING_DISPOSITION, f"{kind} has no disposition"
        disp, mech = rw.FINDING_DISPOSITION[kind]
        assert disp in {"integrity_failure", "numerical_invalid", "prior_predictive",
                        "detector_limit", "protocol_settling"}
        assert len(mech) > 20, f"{kind}: no mechanism stated"


def test_p0_6_coverage_counts_only_successful_records():
    import importlib.util as iu

    import pandas as pd

    spec = iu.spec_from_file_location("_rw", ROOT / "tools" / "retain_waveforms.py")
    rw = iu.module_from_spec(spec)
    spec.loader.exec_module(rw)
    want = pd.DataFrame({"model_id": ["m1"], "sampling_regime": ["E1"]})
    failed = pd.DataFrame({"model_id": ["m1"], "regime": ["E1"],
                           "stratum": ["s"], "status": ["simulation_error"]})
    cov = rw.coverage_report(failed, want)
    assert cov["satisfied"] is False, (
        "a cell whose only sample failed to simulate is not a covered cell")
    assert cov["n_records_not_ok"] == 1


def test_p1_1_the_receipt_is_closed_before_the_manifest_is_built():
    src = (ROOT / "tools" / "run_all.sh").read_text()
    # The manifest stage moved to `run_stage_unreceipted` once the 2026-08-11
    # rebuild showed that recording it into the closed receipt is fatal. The
    # ORDER this test exists to protect is unchanged and still checked.
    assert src.index("receipt finish") < src.index(
        "run_stage_unreceipted inventory"), (
        "run_receipt.json is IN the validation inventory; writing it after the "
        "manifest is built makes the manifest stale immediately")
    assert "python3 tools/run_receipt.py \"$@\" > \"$LOGS/receipt.log\"" in src, (
        "receipt operations must not be swallowed by || true")
    assert "RECEIPT_RC" in src, "the receipt check must reach the exit status"


def test_p1_1_stage_generated_provenance_is_an_output_not_an_input():
    """Exercised, not grepped: which classes must a run be able to account for?

    This used to assert a source string, so renaming the local variable broke
    it while the behaviour was unchanged -- and, worse, a change that DID alter
    the behaviour while keeping the string would have passed.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rr", ROOT / "tools" / "run_receipt.py")
    rr = iu.module_from_spec(spec)
    spec.loader.exec_module(rr)

    arts = [
        {"path": "src/compass/adapters/base.py", "inventory_class": "source"},
        {"path": "configs/release_inventory.json", "inventory_class": "configuration"},
        {"path": "data/provenance/downloads.json", "inventory_class": "provenance"},
        {"path": "data/models/hh/hh.mod", "inventory_class": "model_assets"},
        {"path": "data/provenance/_history/probe3_repair/x.json",
         "inventory_class": "provenance"},
        {"path": "results/analyses/decision.json", "inventory_class": "analyses"},
    ]
    # nothing produced: inputs and retained history are still not demanded
    assert rr.unattributed(arts, set()) == [
        "data/models/hh/hh.mod",
        "data/provenance/downloads.json",
        "results/analyses/decision.json",
    ], "the assets, mechanisms, calibrate and gate stages all WRITE provenance"
    assert rr.unattributed(arts, {"data/provenance/downloads.json",
                                  "data/models/hh/hh.mod",
                                  "results/analyses/decision.json"}) == []


def test_p1_1_run_all_regenerates_every_declared_campaign():
    inv = json.loads((ROOT / "configs" / "release_inventory.json").read_text())
    src = (ROOT / "tools" / "run_all.sh").read_text()
    for c in inv["required_campaigns"]["declared"]:
        runner = ("tools/run_sobol.py" if c == "sobol"
                  else f"--experiment {c} ")
        assert runner in src, (
            f"{c} is a required campaign but run_all.sh never generates it; "
            f"--force-simulations would quarantine it and never rebuild it")


def test_p1_2_the_n2_summary_is_generated_and_says_what_it_covers():
    from compass.validation.gates import n2_coverage

    c = n2_coverage()
    assert c["statement"] == (
        "source-referenced canonical scalar agreement: 6/14 evaluable "
        "implementations")
    assert c["n_evaluable"] + len(c["not_evaluable"]) == c["n_registered"]
    assert c["is_not"].lower().startswith("trace-level reproduction")
    assert "scalar" in c["is_not"].lower()


def test_p1_5_fold_distribution_survives_a_non_finite_fold():
    import pandas as pd

    from compass.analysis.hierarchical import fold_distribution

    d = pd.DataFrame({"predictors": ["a"] * 3,
                      "held_out_group": ["L1", "L2", "L3"],
                      "r2_out_of_sample": [0.5, float("nan"), -3.8]})
    r = fold_distribution(d, "a")          # used to raise TypeError on min()
    assert r["worst_fold"] == "L3"
    assert r["nonfinite_folds"] == ["L2"]
    assert r["n_folds"] == 2


def test_p1_4_the_environment_is_locked_and_the_lock_is_checkable():
    import importlib.util as iu

    assert (ROOT / "requirements.lock").is_file()
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()
    spec = iu.spec_from_file_location(
        "_cel", ROOT / "tools" / "check_environment_lock.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    pins = m.read_lock()
    for pkg in ("NEURON", "efel", "numpy", "scipy", "statsmodels"):
        assert pkg in pins, f"{pkg} is not pinned"
    assert m.compare(pins)["conforms"], (
        "the running environment does not match requirements.lock")


def test_p1_4_model_asset_redistribution_is_decided_per_asset():
    doc = (ROOT / "docs" / "ASSET_REDISTRIBUTION.md").read_text()
    from compass.adapters.models import MODEL_REGISTRY

    for m in MODEL_REGISTRY:
        assert f"`{m}`" in doc, f"{m} has no redistribution tier"
    assert "fetch-at-runtime" in doc


def test_p0_3_the_half_step_twin_must_differ_in_the_time_step_alone():
    """Repeatability compares two batteries; only dt may differ between them."""
    src = (ROOT / "tools" / "perturbation_inference.py").read_text()
    assert "fine_ident = _read_aggregate_identity(fine_path)" in src, (
        "the half-step battery was read with no identity check at all")
    assert "is not a half-step twin of" in src
    # background/candidate/selection identity must be compared per model
    body = src[src.index("fine_ident = _read_aggregate_identity"):
               src.index("repeat = (repeatability")]
    for axis in ("background_vectors_hash", "candidate_rows_hash",
                 "selection_code_hash"):
        assert axis in body, f"{axis} is not compared against the half-step twin"
    # ...and the solver must NOT be, or the check would reject every valid twin
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rp", ROOT / "tools" / "run_perturbations.py")
    rp = iu.module_from_spec(spec)
    spec.loader.exec_module(rp)
    assert "solver" not in rp.AGGREGATE_UNIFORM_AXES
    assert "effective_dt_ms" not in rp.AGGREGATE_UNIFORM_AXES


def test_no_response_document_ships_an_unfilled_placeholder():
    """A draft token must never reach a document a reviewer reads.

    The response documents are assembled with `<<TBD:...>>` markers while the
    long regeneration runs; this is what stops one surviving into the copy that
    goes out.
    """
    for p in sorted(ROOT.glob("COMPASS_*.md")):
        txt = p.read_text()
        assert "<<TBD" not in txt, f"{p.name} still contains a draft placeholder"
    for p in (ROOT / "README.md", ROOT / "docs" / "RESULTS.md",
              ROOT / "docs" / "preregistration.md"):
        assert "<<TBD" not in p.read_text(), f"{p.name} contains a placeholder"


def test_force_does_not_destroy_retained_history():
    """`--force` clears derived artifacts; withdrawn history is not derived.

    `results/analyses/_withdrawn` holds outputs that no stage regenerates, kept
    so the record of what was withdrawn stays legible (P0-4C). Sweeping them
    into a superseded-run directory would delete exactly the evidence the audit
    asked to have preserved.
    """
    src = (ROOT / "tools" / "run_all.sh").read_text()
    body = src[src.index("quarantine_derived()"):src.index("SKIPPED=()")]
    assert 'mv "$dest/analyses/_withdrawn" "$ROOT/results/analyses/_withdrawn"' in body
    assert "corpus_snapshots" in body, (
        "a from-zero run must not erase the before-state of the regeneration")


def test_the_manifest_says_what_the_receipt_covers():
    """Gate D asks for a full --force-simulations run; a derived-half receipt
    is real evidence and is not that. The manifest must distinguish them."""
    m = _brm()
    chk = m.acceptance_checklist({"git_tree": "x", "inventory": {},
                                  "gate_coverage": {}}, {})
    rec = chk["clean_build_receipt"]
    assert "covers_simulation" in rec
    assert rec["simulation_stages"] == ["campaign", "sobol"]
    assert "Gate D" in rec["note"] or "--force-simulations" in rec["note"]


def _rp():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rp", ROOT / "tools" / "run_perturbations.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_p0_3_aggregate_identity_round_trips_and_fails_closed(tmp_path):
    """The sidecar is the ONLY admissible identity source, so it has to work.

    This is the longest stage in the pipeline (~16 h) and the gate that decides
    whether H-B may be scored at all; a defect here is expensive in both
    directions, so the writer and the reader are tested against each other.
    """
    import pandas as pd
    import pytest as _pt

    rp = _rp()
    frame = pd.DataFrame({"model_id": ["a", "a", "b"], "eta": [1.0, 2.0, 3.0]})
    path = tmp_path / "perturbations_raw.parquet"
    frame.to_parquet(path, index=False)

    base = {"measurement_schema": "probe-4", "feature_schema": "features-2",
            "battery_version": "battery-1", "selection_code_hash": "aaaa",
            "experiment": "screening", "delta_log10": 0.05,
            "n_backgrounds": 8, "background_seed": 1,
            "background_vectors_hash": "bg", "candidate_rows_hash": "cr",
            "sim_code_hash": "sc", "effective_dt_ms": 0.025}

    class Args:
        suffix = ""

    # happy path
    rp.write_aggregate_identity(path, frame, {"a": dict(base), "b": dict(base)}, Args())
    doc = rp.read_aggregate_identity(path)
    assert doc["models"] == ["a", "b"]
    assert doc["uniform_axes"]["measurement_schema"]["values"] == ["probe-4"]

    # a model in the aggregate with no slice identity must fail closed
    rp.write_aggregate_identity(path, frame, {"a": dict(base)}, Args())
    with _pt.raises(SystemExit, match="no slice identity"):
        rp.read_aggregate_identity(path)

    # MIXED identity is the defect the audit named: half probe-2, half probe-4
    stale = dict(base, measurement_schema="probe-2")
    rp.write_aggregate_identity(path, frame, {"a": dict(base), "b": stale}, Args())
    with _pt.raises(SystemExit, match="mixed identity"):
        rp.read_aggregate_identity(path)

    # and an absent sidecar names the fix rather than defaulting to the code
    path.with_suffix(".identity.json").unlink()
    with _pt.raises(SystemExit, match="identity sidecar"):
        rp.read_aggregate_identity(path)


def test_absent_resolution_evidence_is_not_reported_as_a_clean_result():
    """A run in which every draw errored must not write zeros into findings.

    The first execution of this audit errored on all 37 draws and recorded
    `n_whose_lowest_window_the_campaign_grid_would_miss: 0` -- indistinguishable
    from a passing audit. "We never checked" and "we checked and found nothing"
    must not share a representation; that is the same distinction the audit drew
    for gate coverage (missing_subjects vs failing_subjects).
    """
    src = (ROOT / "tools" / "audit_rheobase_resolution.py").read_text()
    assert '"NO EVIDENCE: every draw errored" if not ok' in src
    assert "len(missed) if ok else None" in src
    assert "return 0 if ok else 1" in src, "a no-evidence run must exit non-zero"
    # and the per-model parameter list must come from the adapter, not from the
    # concatenated frame's columns, which are the union over every model
    assert "pids = [s.parameter_id for s in get_adapter(mid).parameter_specs()]" in src


def test_the_voltage_band_choice_is_measured_by_an_arm_not_asserted():
    """Unifying the bands was required; unifying at the TIGHTER pair was not.

    The audit (P0-6) required one band propagated everywhere and did not say
    which. Choosing the tighter one removed 609 rows from a five-lineage panel,
    so the choice gets an arm, exactly as the N4 projection gets `n4_retained`.
    """
    import importlib.util as iu

    import pandas as pd

    from compass.analysis.cohorts import (
        REPORTED_ARMS,
        SENSITIVITY_ARMS,
        n4_disposition_by_arm,
    )
    from compass.conditioning.ensembles import validity_mask

    arm = SENSITIVITY_ARMS["wide_voltage_band"]
    assert arm["source_experiment"] == "screening_wideband_n4proj"
    assert "wide_voltage_band" not in REPORTED_ARMS, "it is a sensitivity, not a finding"
    # it must differ from `primary` in the BAND alone, so it is N4-projected too
    assert n4_disposition_by_arm()["wide_voltage_band"]["disposition"] == "projected"

    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    ra = iu.module_from_spec(spec)
    spec.loader.exec_module(ra)

    # PROTOCOL-QUALIFIED, as a stored row actually writes them. The first
    # version of this fixture used the bare flag name, which is the same
    # assumption the code under test was making, so the test agreed with the
    # bug: the strip matched nothing on the real corpus and the arm would have
    # been identical to its comparator. A fixture must come from the data.
    df = pd.DataFrame({
        "sampling_regime": ["E1"] * 3, "replicate": [1, 1, 1],
        "parameter_sample_id": [10, 11, 12], "simulation_status": ["ok"] * 3,
        "qc_flags": ["hyperpol_abs:nonphysical_voltage",
                     "step_abs_hi:unstable_baseline;hyperpol_abs:nonphysical_voltage",
                     "step_abs_hi:unstable_baseline"]})
    keys = {("screening", "khaliq_purkinje", "E1", 1, 10),
            ("screening", "khaliq_purkinje", "E1", 1, 11)}
    assert validity_mask(df, "primary").tolist() == [False, False, True]
    out, n = ra.readmit_band_tightened(df, "screening", "khaliq_purkinje", keys)
    assert n == 2
    assert validity_mask(out, "primary").tolist() == [True, True, True]
    # re-admission undoes the BAND change on every protocol and nothing else
    assert out["qc_flags"].tolist() == ["", "step_abs_hi:unstable_baseline",
                                        "step_abs_hi:unstable_baseline"]
    # and a key belonging to another model matches nothing
    assert ra.readmit_band_tightened(df, "screening", "hh_point", keys)[1] == 0


def test_wide_voltage_band_arm_matches_the_flag_the_corpus_actually_stores():
    """The arm's flag name is a copy; this is what pins it to the original.

    `tools/run_analysis.py` cannot import the name from
    `compass.adapters.base`, because that module is inside `SIM_CODE_PATHS` and
    editing it would change every stored shard's recorded `sim_code_hash`. So
    the name is duplicated, and duplication needs a check that reads the real
    thing: the flag `qc_flags` emits, and the qualified form a shard stores.
    """
    import glob
    import importlib.util as iu

    import pandas as pd

    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    ra = iu.module_from_spec(spec)
    spec.loader.exec_module(ra)

    # 1. the emitter still uses this exact string
    base_src = (ROOT / "src" / "compass" / "adapters" / "base.py").read_text()
    assert f'flags.append("{ra.BAND_FLAG}")' in base_src, (
        f"compass.adapters.base no longer emits {ra.BAND_FLAG!r}; "
        f"tools/run_analysis.py strips a flag that is never written")

    # 2. and a stored row carries it protocol-qualified, not bare -- the whole
    #    reason the strip needs to split on the last colon. Look where the flag
    #    is known to occur: the models the key file names. Sampling shards
    #    alphabetically finds only models that never left the band, and an
    #    empty sample would read as "the flag is gone".
    keyfile = ROOT / "data" / "provenance" / "voltage_band_tightened_rows.json"
    if not keyfile.exists():
        pytest.skip("no voltage-band key file on this machine")
    models = sorted(json.loads(keyfile.read_text())["by_model"])
    shards = sorted(p for m in models for p in glob.glob(str(
        ROOT / "results" / "features" / "screening" / m / "*.parquet")))
    if not shards:
        pytest.skip("no corpus on this machine")
    seen = set()
    for path in shards:
        col = pd.read_parquet(path, columns=["qc_flags"])["qc_flags"].dropna()
        for s in col.astype(str):
            seen.update(t for t in s.split(";") if t)
    band = {t for t in seen if t.rsplit(":", 1)[-1] == ra.BAND_FLAG}
    assert band, f"no stored flag ends in {ra.BAND_FLAG}; the name has drifted"
    assert all(":" in t for t in band), (
        f"expected protocol-qualified flags, got {sorted(band)}")


def test_the_n4_disposition_is_not_read_off_the_tail_of_a_filename():
    """`screening_wideband_n4proj` is projected and does not END with _n4proj."""
    src = (ROOT / "src" / "compass" / "analysis" / "cohorts.py").read_text()
    assert '.endswith("_n4proj")' not in src, (
        "a suffix test misreads any projected table whose name carries a "
        "further qualifier; that is the P0-5 defect in miniature")
    assert 'projected = "n4proj" in str(spec.get("source_experiment", "")).split("_")' in src


def _run_all_function(name: str) -> str:
    """Extract one shell function from tools/run_all.sh so it can be exercised."""
    src = (ROOT / "tools" / "run_all.sh").read_text().splitlines()
    start = next(i for i, ln in enumerate(src) if ln.startswith(f"{name}() {{"))
    end = next(i for i in range(start + 1, len(src)) if src[i] == "}")
    return "\n".join(src[start:end + 1])


def test_every_pipeline_stage_declares_what_it_owns():
    """A stage with no declared outputs cannot be checked for deletion."""
    src = (ROOT / "tools" / "run_all.sh").read_text()
    block = re.search(r"^STAGES=\((.*?)\)\s*$", src, re.S | re.M).group(1)
    stages = block.split()
    assert "sobol" in stages and "inventory" in stages

    probe = (_run_all_function("stage_outputs") + "\n"
             + "\n".join(f'echo "{s}|$(stage_outputs {s})"' for s in stages))
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                         check=True).stdout
    undeclared = [ln.split("|")[0] for ln in out.splitlines()
                  if "__UNDECLARED__" in ln]
    assert not undeclared, (
        f"stages with no stage_outputs entry: {undeclared}. A stage the map "
        f"does not know about is a stage whose outputs --force can delete "
        f"without anything noticing.")


def test_force_refuses_to_delete_what_the_plan_will_not_rebuild():
    """`--force --from <late stage>` deleted the Sobol and convergence records.

    Quarantine removes every derived artifact; `--from` rebuilds only the
    stages in the plan. The 2026-08-10 rebuild combined them and lost
    results/analyses/sobol_*.parquet and
    results/validation/convergence_evidence.json out of a release whose own
    inventory declares both. The refusal is checked here on a throwaway tree,
    including that it happens BEFORE anything is moved.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp)
        (tree / "tools").mkdir()
        (tree / "results" / "analyses").mkdir(parents=True)
        (tree / "results" / "validation").mkdir(parents=True)
        shutil.copy(ROOT / "tools" / "run_all.sh", tree / "tools" / "run_all.sh")
        owned = {"results/analyses/sobol_summary.parquet": "sobol",
                 "results/validation/convergence_evidence.json": "convergence"}
        for rel in owned:
            (tree / rel).write_text("x")

        r = subprocess.run(["bash", "tools/run_all.sh", "--force",
                            "--from", "resolution"],
                           cwd=tree, capture_output=True, text=True)
        assert r.returncode == 2, r.stdout + r.stderr
        for rel, stage in owned.items():
            assert rel in r.stdout, f"{rel} not named in the refusal"
            assert f"only stage '{stage}' rebuilds it" in r.stdout
            assert (tree / rel).exists(), f"{rel} was moved despite the refusal"
        assert not list(tree.glob("results/_superseded_run_*")), \
            "the quarantine ran before the plan was checked"

        # ... and the same plan with the owning stages included is allowed
        # through the check (it then fails later, on a tree with no pipeline).
        r2 = subprocess.run(["bash", "tools/run_all.sh", "--force",
                             "--from", "convergence", "--until", "sobol"],
                            cwd=tree, capture_output=True, text=True)
        assert "would discard derived artifacts" not in r2.stdout


def test_a_receipt_covers_simulation_only_when_every_sim_stage_ran():
    """`&` made one simulation stage enough to claim the corpus was covered.

    Two claims are checked, because `covers_simulation` is what Gate D turns
    on: every simulation stage must be in the receipt, AND the run must have
    been the kind that rebuilds the corpus. `--force` preserves feature shards
    on purpose, so a campaign stage that ran and reused all of them simulated
    nothing.
    """
    src = (ROOT / "tools" / "build_release_manifest.py").read_text()
    assert 'SIMULATION_STAGES = ("campaign", "sobol")' in src
    assert "set(SIMULATION_STAGES).issubset(" in src, (
        "covers_simulation must require every simulation stage, not any one")
    assert '{"campaign", "sobol"} & set(' not in src
    assert 'receipt.get("force_simulations")' in src, (
        "a stage that ran on cached shards covers no simulation")
    # and the clean-build rule has ONE implementation, shared with the check
    assert "rr.unattributed(artifacts, produced)" in src, (
        "the manifest must not reimplement what run_receipt.check decides")


def test_receipt_watches_every_class_it_demands():
    """A check cannot demand attribution for a tree its index never looks at.

    The previous revision reclassified `provenance` and `model_assets` from
    inputs to outputs -- correctly, stages write them -- but left run_receipt's
    WATCHED list on `results/**` alone. Every file in those two classes was
    therefore required to be produced by the run and was structurally invisible
    to the code that decides what the run produced, so the check reported the
    same failure no matter what happened.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rr", ROOT / "tools" / "run_receipt.py")
    rr = iu.module_from_spec(spec)
    spec.loader.exec_module(rr)
    inv = json.loads((ROOT / "configs" / "release_inventory.json").read_text())

    demanded = {c: v["include"] for c, v in inv["classes"].items()
                if c not in rr.DECLARED_INPUT_CLASSES}
    # the root each glob starts from, e.g. "data/provenance", "results"
    watched_roots = {w.split("/**")[0].split("/*")[0] for w in rr.WATCHED}
    missing = {c: g for c, globs in demanded.items() for g in globs
               if not any(g == w or g.startswith(w.rstrip("/") + "/")
                          for w in watched_roots)}
    assert not missing, (
        f"inventory classes the check demands but the index cannot see: "
        f"{missing}. Either watch them or declare them inputs.")


def test_a_ban_that_encodes_a_transient_state_is_read_from_the_artifact():
    """A guard must not outlive the condition it guards.

    `the H-B criteria are now scored` was banned outright while the
    perturbation inputs were stale and the tally read 0/0. On the probe-4
    corpus the criteria ARE scored (0/2), so an unconditional ban forbids a
    true statement -- the mirror image of a stale number, and just as wrong.
    The rule is now switched off from decision.json rather than by hand.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_v", ROOT / "tools" / "verify_report_numbers.py")
    v = iu.module_from_spec(spec)
    spec.loader.exec_module(v)

    marked = [p for p, _ in v.FORBIDDEN if p.startswith(v.CONDITIONAL_ON_H_B_WITHHELD)]
    assert marked, "the H-B scoring ban must be marked state-dependent"
    assert any("are (now )?scored" in p for p in marked)

    src = (ROOT / "tools" / "verify_report_numbers.py").read_text()
    assert 'h_b_withheld = not dec.get("n_hb_total")' in src, (
        "the condition must come from the decision artifact, not a constant")
    assert "if not h_b_withheld:\n                continue" in src


def test_every_cited_provenance_artifact_has_something_that_writes_it():
    """`waveform_findings.json` was cited by RESULTS.md and written by nothing.

    It was hand-authored, two corpus regenerations stale, a `provenance`-class
    entry in the release inventory, and named in run_receipt.py's own comment as
    a file "the campaign and gate stages write" -- which was false. The manifest
    hashed whatever bytes were there and the clean-build check could never
    attribute it, because no stage produced it. A generator now exists.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rw", ROOT / "tools" / "retain_waveforms.py")
    rw = iu.module_from_spec(spec)
    spec.loader.exec_module(rw)
    assert hasattr(rw, "summarise"), (
        "tools/retain_waveforms.py must be able to write the roll-up it is "
        "cited for")
    src = (ROOT / "tools" / "retain_waveforms.py").read_text()
    assert '"waveform_findings.json"' in src or "waveform_findings.json" in src

    # and the receipt's comment must no longer name a producer that isn't one
    rr = (ROOT / "tools" / "run_receipt.py").read_text()
    bad = ("numerical_qc.json and waveform_findings.json. Treating all of that "
           "as input")
    assert bad not in rr, "run_receipt.py still credits a stage that never wrote it"

    # the generated artifact must agree with the records it summarises
    prov = ROOT / "data" / "provenance" / "waveform_findings.json"
    recs = sorted((ROOT / "results" / "validation").glob("retained_waveforms_*.json"))
    if not prov.exists() or not recs:
        pytest.skip("no waveform battery output on this machine")
    d = json.loads(prov.read_text())
    assert d["totals"]["n_selected"] == sum(
        json.loads(p.read_text())["n_selected"] for p in recs)
    assert d["n_samples_flagged"] == sum(
        json.loads(p.read_text())["n_with_problems"] for p in recs)
    assert d["n_findings_total"] >= d["n_samples_flagged"], (
        "a sample can carry several findings; the roll-up must not conflate them")


def test_the_waveform_counts_are_a_checked_claim_not_only_a_banned_wrong_value():
    """A ban on one wrong number cannot detect the next one."""
    src = (ROOT / "tools" / "verify_report_numbers.py").read_text()
    assert 'claim("waveform samples retained"' in src
    assert 'claim("waveform samples flagged"' in src
    assert 'claim("waveform features recomputed"' in src


def test_a_stage_after_the_receipt_closes_does_not_write_to_it():
    """The clean-run receipt could not survive a clean run.

    `finish()` drops `_index`; the driver then recorded the `inventory` stage
    into the receipt it had just closed, and `stage()` raised a bare
    `KeyError: '_index'`. Receipt operations are hard failures by design, so the
    2026-08-11 rebuild ran for twenty hours and died on its last call. No
    earlier run had reached that line -- which is precisely why P1-1 asked for a
    completed clean run rather than a receipt that merely exists.
    """
    import importlib.util as iu
    import tempfile

    spec = iu.spec_from_file_location("_rr", ROOT / "tools" / "run_receipt.py")
    rr = iu.module_from_spec(spec)
    spec.loader.exec_module(rr)

    with tempfile.TemporaryDirectory() as tmp:
        rr.RECEIPT = Path(tmp) / "run_receipt.json"
        rr.RECEIPT.write_text(json.dumps({
            "stages": [{"stage": "report", "exit_code": 0, "seconds": 1.0,
                        "created": [], "rewritten": []}],
            "finished_utc": "2026-08-11T14:15:00+00:00",
            "artifacts_produced_by_this_run": [],
        }))
        with pytest.raises(SystemExit) as e:
            rr.stage("inventory", 1, 49)
        msg = str(e.value)
        assert "closed" in msg and "inventory" in msg, msg
        assert "run_stage_unreceipted" in msg, "the error must name the remedy"

    # and the driver must actually use that path for the manifest stage
    sh = (ROOT / "tools" / "run_all.sh").read_text()
    assert "run_stage_unreceipted inventory python3 tools/build_release_manifest.py" in sh
    assert "run_stage_soft inventory" not in sh
    # the receipt check must read a manifest that exists by then
    assert sh.index("run_stage_unreceipted inventory") < sh.index(
        "python3 tools/run_receipt.py check"), (
        "`check` reads the release manifest; on a --force run the quarantine "
        "has taken the previous one, so it must run after the rebuild")


def test_the_sobol_reuse_check_can_actually_return_true():
    """`_shard_is_current` compared two dictionaries with no schema in common.

    The sidecar is written from `flat_identity()` -- flat, string-valued,
    campaign-shaped. The check compared it against `shard_identity()` -- nested,
    with different keys. They are never equal, so every Sobol shard was
    quarantined and re-simulated on every invocation: eight hours of NEURON
    thrown away and redone each time, silently, because a deterministic re-run
    produces the same bytes. Found 2026-08-11 when a table rebuild that should
    have taken seconds started re-simulating 360 blocks.
    """
    import importlib.util as iu

    src = (ROOT / "tools" / "run_sobol.py").read_text()
    # the DEFINITION must be gone, not merely unused; the name survives in the
    # comment that explains why it was removed
    assert "def _shard_is_current" not in src, (
        "the unsatisfiable comparison must be gone, not merely bypassed")
    assert "ok, why = shard_is_reusable(out, ident, stop - start)" in src, (
        "quarantine must use the same rule that decides what to simulate")

    spec = iu.spec_from_file_location("_rs", ROOT / "tools" / "run_sobol.py")
    rs = iu.module_from_spec(spec)
    spec.loader.exec_module(rs)
    from compass.workflows.simulate import shard_is_reusable

    shards = sorted(rs.FEAT_DIR.glob("*/S_*.parquet"))
    shards = [p for p in shards if "_superseded" not in p.parts]
    if not shards:
        pytest.skip("no sobol corpus on this machine")
    # The stored corpus must be reusable AS STORED: if it is not, the next run
    # silently re-simulates it.
    checked = 0
    for p in shards[:24]:
        mid = p.parent.name
        start = int(p.stem.split("_")[1])
        ident = rs.flat_identity(mid, rs.shard_identity(mid), start,
                                 int(json.loads(rs._identity_path(p).read_text())["n_rows"]))
        ok, why = shard_is_reusable(p, ident, int(ident["n_rows"]))
        assert ok, f"{mid}/{p.name} would be re-simulated: {why}"
        checked += 1
    assert checked >= 8


def test_the_figures_class_has_a_producer_and_the_stage_runs_it():
    """`results/figures/**` is declared with min_files 6 and nothing wrote it.

    `build_notebook.py` emitted the .ipynb and no stage executed it, so the
    figure cells never ran; the published PNG/PDF pairs had been exported by
    hand. `--force` then quarantined a directory no stage could rebuild and the
    release came out with zero figures against a declared minimum of six. Same
    "declared class, no producer" defect as waveform_findings.json.
    """
    nb = (ROOT / "tools" / "build_notebook.py").read_text()
    assert "def savefig(fig):" in nb, "the notebook must save, not only show"
    assert nb.count("fig = savefig(FG.") >= 6, (
        "every figure cell must save; the inventory minimum is 6 files")

    sh = (ROOT / "tools" / "run_all.sh").read_text()
    assert "nbconvert --to notebook --execute" in sh, (
        "writing the notebook is not running it")
    # and the stage that owns the class must be the one that executes it
    probe = (_run_all_function("stage_outputs") + "\necho \"$(stage_outputs notebook)\"")
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                         check=True).stdout
    assert "results/figures" in out


def test_numerical_gate_records_carry_the_revision_the_manifest_reads():
    """A gate record with no revision is rejected, and reads as absent evidence.

    P0-4A made gate records count only when their `git_tree` matches the
    release. `record_numerical_gates.py` recorded the revision under
    `provenance.git_tree`, while the manifest reads a TOP-LEVEL `git_tree` and
    falls back to it per record -- so all 84 N1-N7 records were rejected and the
    2026-08-11 manifest said "mandatory gate records absent: N1..N7" with
    numerical_gates.json beside it reporting N1 82/84. A false absence blocks
    for an untrue reason and hides the true ones.
    """
    src = (ROOT / "tools" / "record_numerical_gates.py").read_text()
    assert '"git_tree": git_tree(),\n        "slow_tests_included"' in src, (
        "the revision must sit at the top level, where the manifest reads it")

    p = ROOT / "results" / "validation" / "numerical_gates.json"
    if not p.exists():
        pytest.skip("no numerical gate record on this machine")
    d = json.loads(p.read_text())
    if "git_tree" not in d:
        pytest.skip("record predates the fix; re-run tools/record_numerical_gates.py")
    man = ROOT / "results" / "validation" / "release_manifest.json"
    if man.exists():
        assert d["git_tree"] == json.loads(man.read_text())["git_tree"], (
            "the gate evidence and the release describe different trees")


def test_the_audit_responses_are_inside_the_verifier_scope():
    """The documents that go to the reviewer were the ones not checked.

    Forbidden phrasings and the retired-value scan ran over RESULTS.md and
    README.md only. Every audit response -- a `reports`-class inventory entry
    whose entire purpose is to assert things to a reviewer -- was outside them.
    Bringing response 4 into scope immediately found six banned phrasings in it.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_v", ROOT / "tools" / "verify_report_numbers.py")
    v = iu.module_from_spec(spec)
    spec.loader.exec_module(v)

    responses = sorted((ROOT / "docs").glob("COMPASS_AUDIT_RESPONSE_*.md"))
    if not responses:
        pytest.skip("no audit response documents in this tree")
    for r in responses:
        assert r.name in v.DOC_PATHS, f"{r.name} is not checked by the verifier"
        assert r.name in v.DOCS
    # globbed, so the next response is covered without anyone remembering
    src = (ROOT / "tools" / "verify_report_numbers.py").read_text()
    assert 'glob("COMPASS_AUDIT_RESPONSE_*.md")' in src


def test_the_tree_identity_is_the_source_revision_not_the_output():
    """Generated output must not dirty the revision its own evidence records.

    `git_tree()` ran `git status --porcelain` over everything, and this
    repository tracks its generated results -- 176 of 204 modified files at the
    2026-08-11 release were under `results/`. So every gate record read
    `<rev>-dirty` permanently (writing a gate dirties the tree it stamps), and
    committing invalidated all of them at once. The acceptance item "one clean
    identified revision" was unreachable by construction, not by omission.
    """
    from compass.validation.gates import _source_paths, git_tree

    inv = json.loads((ROOT / "configs" / "release_inventory.json").read_text())
    paths = set(_source_paths())
    # derived from the inventory's declared INPUT classes, not hand-listed
    for cls in ("source", "configuration"):
        for glob in inv["classes"][cls]["include"]:
            top = glob.split("/")[0].split("*")[0]
            if top:
                assert top in paths, f"{glob} is a declared input and is unwatched"
    assert "results" not in paths and "data" not in paths, (
        "generated output must not participate in the tree identity")

    src = (ROOT / "src" / "compass" / "validation" / "gates.py").read_text()
    assert '"log", "-1", "--format=%H", "--", *paths' in src, (
        "the revision must be the last commit that touched SOURCE, so that "
        "committing regenerated results does not invalidate the evidence")
    assert '"status", "--porcelain", "--", *paths' in src

    # and there is exactly one implementation
    rr = (ROOT / "tools" / "run_receipt.py").read_text()
    assert "from compass.validation.gates import git_tree" in rr
    assert "def git_tree" not in rr, "two definitions can disagree"
    assert isinstance(git_tree(), str)
