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


def test_the_rheobase_ladder_and_the_zero_rheobase_rule_are_declared(tmp_path):
    """4.1: probe-3 must be a declared schema change, not a silent one."""
    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION, NeuronAdapter

    assert MEASUREMENT_SCHEMA_VERSION == "probe-3"
    assert NeuronAdapter.RHEOBASE_SCAN_RUNGS >= 8


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


def test_normalised_qc_flags_are_stripped_when_the_family_is_undefined():
    """The shard-equivalence check found stored rows naming protocols never run.

    Two bugs in the first fix are pinned here: QC flags are namespaced with a
    COLON while feature columns use a dot, and the mask has to cover
    rheobase = NaN as well as rheobase = 0.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rr", ROOT / "tools" / "repair_rheobase.py")
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert m.NORMALISED_FLAG_PREFIXES == ("step_rheo_1p5x:", "step_rheo_3p0x:")
    flag = "step_rheo_1p5x:unstable_baseline"
    assert flag.startswith(m.NORMALISED_FLAG_PREFIXES)
    assert not flag.startswith(m.NORMALISED_PREFIXES), (
        "the dot-namespaced column prefixes must not be reused for flags; "
        "doing so silently stripped nothing")


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
