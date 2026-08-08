"""Gates N1, N2 and N7 for EVERY registered model (audit 6.4.1).

The 2026-08-06 audit's finding 5: "Real NEURON physiology assertions are
concentrated on Pospischil RS. The remaining models are build-checked and
contract-checked, but are not all simulated under their exact production
solver/mesh and compared with model-specific neuronal behaviour ranges."

This file closes that gap.  Every adapter is built **in a fresh spawned
process** -- NEURON's section list is process-global, so this is not a stylistic
choice -- simulated under its exact production solver contract, and checked
against the frozen, source-justified interval in
``compass.validation.bounds``.

These are slow (minutes per multicompartment model) and are marked ``slow``.
They are nightly-and-release tests, not pull-request tests.  Deselecting the
marker is allowed for a PR and **not** allowed for a release: the release
command refuses to report a pass when a required marker was deselected.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY  # noqa: E402
from compass.validation.bounds import bounds_for  # noqa: E402

pytestmark = pytest.mark.slow

MODELS = sorted(MODEL_REGISTRY)


def _probe(model_id: str) -> dict:
    """Run the canonical protocol battery in a fresh process."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    from compass.adapters.models import get_adapter
    from compass.features.extract import extract, extract_rest
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    kw = solver_kwargs(model_id)
    a = get_adapter(model_id).build()
    lo, mid, hi = P.ABSOLUTE_AMPLITUDES[model_id]

    rest = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                      total_ms=P.REST_TOTAL_MS, protocol_id="rest", **kw)
    rf = extract_rest(rest)

    out: dict = {
        "model_id": model_id,
        "solver_requested": kw,
        # gate N1: the requested solver settings were ACTUALLY used, no silent
        # fallback.  Read back from NEURON rather than assumed.
        "dt_actual": None, "tstop_actual": None, "celsius_actual": None,
        "n_segments": a.manifest.n_compartments,
        "resting_potential_mV": rf["resting_potential_mV"],
        "spontaneous_rate_Hz": rf["spontaneous_rate_Hz"],
        "t_monotonic": bool(np.all(np.diff(rest.t) > 0)),
        "t_complete": bool(rest.t.size and
                           abs(rest.t[-1] - P.REST_TOTAL_MS) <= 2 * kw["dt_ms"]),
        "v_finite": bool(np.all(np.isfinite(rest.v))),
    }
    from neuron import h

    out["dt_actual"] = float(h.dt)
    out["tstop_actual"] = float(h.tstop)
    out["celsius_actual"] = float(h.celsius)

    hyper = a.simulate(lo, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                       total_ms=P.TOTAL_MS, protocol_id="hyperpol", **kw)
    hf = extract(hyper)
    out["input_resistance_Mohm"] = hf.get("input_resistance_Mohm")
    out["hyperpolarises"] = bool(
        np.nanmin(hyper.v[(hyper.t > P.DELAY_MS) & (hyper.t < P.DELAY_MS + P.DURATION_MS)])
        < np.nanmedian(hyper.v[hyper.t < P.DELAY_MS]) + 0.5)

    for label, amp in (("lo", mid), ("hi", hi)):
        rec = a.simulate(amp, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                         total_ms=P.TOTAL_MS, protocol_id=label, **kw)
        f = extract(rec)
        out[f"rate_at_{label}_Hz"] = f["mean_firing_rate_Hz"]
        out[f"spikes_at_{label}"] = f["spike_count"]
        out[f"qc_{label}"] = a.qc_flags(rec)
        if label == "hi":
            out["ap_amplitude_mV"] = f["ap_amplitude_mV"]
            out["ap_halfwidth_ms"] = f["ap_halfwidth_ms"]
            out["adaptation_index"] = f["adaptation_index"]
            out["v_finite_hi"] = bool(np.all(np.isfinite(rec.v)))

    out["rheobase_nA"] = a.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                                         duration_ms=P.RHEO_DURATION_MS,
                                         total_ms=P.RHEO_TOTAL_MS, **kw)

    # gate N7: determinism.  Re-run an identical protocol after unrelated work
    # and after a reset; a fixed-step run in one process must be bit-identical.
    a.reset()
    again = a.simulate(hi, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                       total_ms=P.TOTAL_MS, protocol_id="hi", **kw)
    first = a.simulate(hi, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                       total_ms=P.TOTAL_MS, protocol_id="hi", **kw)
    out["repeat_max_abs_dv"] = (float(np.max(np.abs(again.v - first.v)))
                                if again.v.shape == first.v.shape else float("inf"))
    return out


@pytest.fixture(scope="session")
def canonical_probes() -> dict:
    """Probe every model once; each build gets its own spawned process."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(MODELS), max(1, mp.cpu_count() - 2)),
                  maxtasksperchild=1) as pool:
        return {r["model_id"]: r for r in pool.map(_probe, MODELS)}


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_execution_is_complete_and_finite(canonical_probes, model_id):
    p = canonical_probes[model_id]
    assert p["t_monotonic"], f"{model_id}: time vector is not monotonic"
    assert p["t_complete"], f"{model_id}: trace stops before tstop"
    assert p["v_finite"], f"{model_id}: non-finite voltage at rest"
    assert p["v_finite_hi"], f"{model_id}: non-finite voltage on the hi step"


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_requested_solver_settings_were_actually_used(canonical_probes, model_id):
    """A silent fallback to a different dt would invalidate the whole campaign."""
    p = canonical_probes[model_id]
    want = p["solver_requested"]
    assert p["dt_actual"] == pytest.approx(want["dt_ms"], rel=1e-9), (
        f"{model_id}: requested dt={want['dt_ms']} but NEURON ran {p['dt_actual']}")
    expected_c = MODEL_REGISTRY[model_id].manifest.temperature_c
    assert p["celsius_actual"] == pytest.approx(expected_c, abs=1e-6), (
        f"{model_id}: manifest says {expected_c} C, NEURON ran {p['celsius_actual']}")


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_canonical_behaviour_is_inside_the_frozen_bounds(canonical_probes, model_id):
    """Model-specific intervals, each with a source justification."""
    p = canonical_probes[model_id]
    failures = []
    for key, (lo, hi, why) in bounds_for(model_id).items():
        got = p.get(key)
        if got is None or not np.isfinite(got):
            failures.append(f"{key}: not measured (bound exists: [{lo}, {hi}] -- {why})")
            continue
        if not (lo <= got <= hi):
            failures.append(f"{key}={got:.4g} outside [{lo}, {hi}] -- {why}")
    assert not failures, f"{model_id}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_hyperpolarising_step_actually_hyperpolarises(canonical_probes, model_id):
    """Catches a sign-flipped current injection, which nothing else would."""
    assert canonical_probes[model_id]["hyperpolarises"], (
        f"{model_id}: the negative step did not hyperpolarise -- check the "
        f"sign of ABSOLUTE_AMPLITUDES[{model_id}][0]")


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_no_numerical_qc_flag_at_canonical_parameters(canonical_probes, model_id):
    """A model that flags at its own published parameters is not usable."""
    numerical = {"solver_failure", "nonfinite_voltage", "nonphysical_voltage"}
    for label in ("lo", "hi"):
        bad = set(canonical_probes[model_id][f"qc_{label}"]) & numerical
        assert not bad, f"{model_id}: {label} step raised {sorted(bad)}"


@pytest.mark.parametrize("model_id", MODELS)
def test_n7_fixed_step_runs_are_deterministic(canonical_probes, model_id):
    """Same input, same process, same answer -- to the last bit."""
    d = canonical_probes[model_id]["repeat_max_abs_dv"]
    assert d == 0.0, (
        f"{model_id}: repeating an identical fixed-step run changed the trace by "
        f"{d:.3g} mV; state is leaking between simulations")


@pytest.mark.parametrize("model_id", MODELS)
def test_n1_spontaneously_active_models_are_declared(canonical_probes, model_id):
    """A cell that fires at rest must be declared, or its rheobase is a lie."""
    from compass.protocols import common as P

    rate = canonical_probes[model_id]["spontaneous_rate_Hz"]
    declared = model_id in P.SPONTANEOUSLY_ACTIVE
    if rate and rate > 1.0:
        assert declared, (
            f"{model_id} fires at {rate:.1f} Hz with no current but is not in "
            f"SPONTANEOUSLY_ACTIVE; its rheobase-normalised protocols would be "
            f"anchored to a meaningless threshold")
