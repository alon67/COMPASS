"""Gate N3: the EXACT production setting must be an accepted setting.

Audit 6.4.1, gate N3:

    Load the convergence audit for every model and require a direct row for the
    exact production ``dt`` and mesh at every prespecified prior point. [...]
    The exact production setting is present, is the declared ``chosen``
    setting, and is accepted for every primary feature and audit point.
    ``chosen=null``, a missing feature or an untested exact setting fails the
    release.  **The current release fails this gate for Mainen.**

The point of this file is that the failure is *enforced*, not footnoted.  Models
that fail N3 are listed in :data:`N3_KNOWN_FAILURES` with their remediation, and
the test asserts that each such model is excluded from primary inference by
``compass.analysis.cohorts.PRIMARY_EXCLUSIONS``.  Adding a model to the
known-failure list therefore does not silence anything: it forces the exclusion
and the dependent sensitivity analysis.

Runs from stored audit JSON, so it is cheap and belongs in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY  # noqa: E402
from compass.workflows.simulate import CAMPAIGN_SOLVER, solver_kwargs  # noqa: E402

PROV = ROOT / "data" / "provenance"
MODELS = sorted(MODEL_REGISTRY)

# Models known to fail N3 at their production setting.  Every entry must also
# appear in compass.analysis.cohorts.PRIMARY_EXCLUSIONS, which is what actually
# keeps them out of the headline numbers.
N3_KNOWN_FAILURES = {
    "mainen_l5pc": (
        "dt=0.0125 ms on the 2304-segment mesh reproduces 12 of 13 conditioning "
        "features on all 7 audit points but fails adaptation_index on 1 point. "
        "Excluded from primary inference; retained in the sensitivity cascade."),
    "miyasho_purkinje": (
        "no tested step converges: ap_amplitude_mV fails at 0.05/0.025/0.0125 ms "
        "and membrane_tau_ms at 0.00625 ms, with the failing feature moving "
        "between steps. Not simulated."),
    "tzilivaki_fsbc": (
        "mean_firing_rate_Hz reproduces on 25-50% of audit points at 0.05 and "
        "0.025 ms on both meshes; a ~145 Hz cell needs a finer step than the "
        "budget allows on a 9,875-segment morphology. Not simulated."),
    "ortner_snc_da": (
        "no tested step converges: ap_amplitude_mV fails on 1-2 of 7 points at "
        "0.025, 0.0125 and 0.00625 ms. Not simulated; excluded from every "
        "campaign, so no dependent artifact exists to regenerate."),
}


def _audit(model_id: str) -> dict | None:
    p = PROV / f"convergence_audit_{model_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_has_a_campaign_solver_entry(model_id):
    assert model_id in CAMPAIGN_SOLVER, (
        f"{model_id} has no production solver contract; solver_kwargs() now "
        f"fails closed, so this model cannot be simulated at all")


@pytest.mark.parametrize("model_id", MODELS)
def test_every_model_has_a_convergence_audit(model_id):
    assert _audit(model_id) is not None, (
        f"no data/provenance/convergence_audit_{model_id}.json; N3 fails closed "
        f"-- run tools/convergence_audit.py {model_id}")


def _production_nseg(model_id) -> int | None:
    """Segment count the adapter actually builds, from its canonical manifest."""
    p = ROOT / "results" / "manifests" / f"canonical_{model_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("n_compartments")


def _production_row(model_id):
    """Match on (dt, total_nseg), not on the audit's mesh LABEL.

    Several adapters apply a d_lambda rule inside `_build`, so their "as-built"
    mesh is a d_lambda multiple and a label match reports them as untested even
    when the exact discretisation was audited. The segment count is what the
    solver sees.
    """
    d = _audit(model_id)
    if d is None:
        return None, None
    dt = float(solver_kwargs(model_id)["dt_ms"])
    nseg = _production_nseg(model_id)
    for key, s in d["settings"].items():
        if abs(float(s["dt_ms"]) - dt) >= 1e-12:
            continue
        if nseg is not None and s.get("total_nseg") is not None:
            if int(s["total_nseg"]) == int(nseg):
                return key, s
        elif s.get("mesh_factor") in (None, "as_built"):
            return key, s
    return None, None


@pytest.mark.parametrize("model_id", MODELS)
def test_n3_exact_production_setting_was_directly_tested(model_id):
    """Acceptance may not be interpolated from a neighbouring setting."""
    key, row = _production_row(model_id)
    dt = solver_kwargs(model_id)["dt_ms"]
    assert row is not None, (
        f"{model_id}: production dt={dt} ms on the as-built mesh has no directly "
        f"tested row in its convergence audit")


@pytest.mark.parametrize("model_id", sorted(set(MODELS) - set(N3_KNOWN_FAILURES)))
def test_n3_production_setting_is_accepted(model_id):
    key, row = _production_row(model_id)
    if row is None:
        pytest.fail(f"{model_id}: production setting untested (see the N3 test above)")
    failing = sorted(f.split(".")[-1] for f, v in
                     row.get("per_feature_agreement", {}).items() if v < 1.0)
    assert row["accepted"], (
        f"{model_id}: production setting {key} is NOT accepted "
        f"(failing features: {failing}). Either fix convergence, or add the "
        f"model to N3_KNOWN_FAILURES *and* to PRIMARY_EXCLUSIONS.")


@pytest.mark.parametrize("model_id", sorted(N3_KNOWN_FAILURES))
def test_n3_known_failures_are_excluded_from_primary_inference(model_id):
    """A documented failure must cost the model its vote, not gain it a footnote."""
    from compass.analysis.cohorts import PRIMARY_EXCLUSIONS

    assert model_id in PRIMARY_EXCLUSIONS, (
        f"{model_id} fails gate N3 but is not in PRIMARY_EXCLUSIONS. The audit's "
        f"release rule 7: an exception requires exclusion of the affected model "
        f"from primary inference and regeneration of dependent artifacts; it "
        f"does not convert the gate to a pass.")
    assert PRIMARY_EXCLUSIONS[model_id]["gate"] == "N3"


@pytest.mark.parametrize("model_id", sorted(N3_KNOWN_FAILURES))
def test_n3_known_failures_still_fail(model_id):
    """If a documented failure is repaired, this test tells you to delist it."""
    key, row = _production_row(model_id)
    if row is not None and row["accepted"]:
        pytest.fail(
            f"{model_id} now PASSES gate N3 at {key}. Remove it from "
            f"N3_KNOWN_FAILURES and PRIMARY_EXCLUSIONS and regenerate the "
            f"dependent artifacts -- a stale exclusion is as wrong as a stale "
            f"inclusion.")


@pytest.mark.parametrize("model_id", MODELS)
def test_n3_audit_reference_is_finer_than_the_production_setting(model_id):
    """A 'convergence' audit whose reference is not finer proves nothing."""
    d = _audit(model_id)
    if d is None:
        pytest.skip("covered by test_every_model_has_a_convergence_audit")
    ref_dt = float(d["reference"]["dt_ms"])
    prod_dt = float(solver_kwargs(model_id)["dt_ms"])
    assert ref_dt < prod_dt, (
        f"{model_id}: reference dt={ref_dt} is not finer than production "
        f"dt={prod_dt}")
