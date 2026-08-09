"""Gate N4: every exposed parameter does something, and reset undoes it.

Audit 6.4.1, gate N4:

    For every exposed parameter, perturb it up and down in a canonical context,
    inspect the target NEURON attribute/mechanism, run the trace, then reset and
    rerun. [...] The intended target changes by the requested amount, reset
    restores the baseline, and at least one prespecified mechanistic observable
    or trace feature changes above numerical tolerance. An inactive,
    misaddressed, sign-reversed or state-leaking parameter fails the model.

This is the gate that would have caught the Repressilator's inactive ``k_p``
(finding C3 of the previous audit) at the moment it was written rather than two
reviews later, and the reason it is worth paying for on every adapter.

The trace-effect half is deliberately two-sided: a parameter is sampled over
[1/4, 4], so a x4 change that moves no feature at all means the parameter is not
part of the model as far as this study can observe, whatever the mechanism
tables say.
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

pytestmark = pytest.mark.slow

MODELS = sorted(MODEL_REGISTRY)

# A parameter must move at least one of these by more than its tolerance when
# scaled up 4x and down 4x.  Tolerances are numerical, not biological: they sit
# just above fixed-step round-off.
EFFECT_FEATURES = {
    "spike_count": 0.5,
    "mean_firing_rate_Hz": 1.0,
    "ap_amplitude_mV": 0.5,
    "ap_halfwidth_ms": 0.01,
    "resting_potential_mV": 0.2,
    "input_resistance_Mohm": 0.5,
    "first_spike_latency_ms": 0.5,
}


def _activity(model_id: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    from compass.adapters.models import get_adapter
    from compass.features.extract import extract, extract_rest
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    kw = solver_kwargs(model_id)
    a = get_adapter(model_id).build()
    _, _, hi = P.ABSOLUTE_AMPLITUDES[model_id]

    def observe() -> dict:
        rec = a.simulate(hi, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                         total_ms=P.TOTAL_MS, protocol_id="n4", **kw)
        f = {k: v for k, v in extract(rec).items()}
        rest = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                          total_ms=P.REST_TOTAL_MS, protocol_id="n4rest", **kw)
        f.update(extract_rest(rest))
        f["_v_hash"] = float(np.nansum(rec.v[::17]))
        return f

    def targets(pid: str) -> list[float]:
        return [float(getattr(seg, attr)) for seg, attr, _ in a._canonical[pid]]

    a.reset()
    base = observe()
    base_targets = {p.parameter_id: targets(p.parameter_id)
                    for p in a.parameter_specs()}

    out = {"model_id": model_id, "parameters": {}}
    for spec in a.parameter_specs():
        pid = spec.parameter_id
        rec: dict = {"canonical": base_targets[pid][:3],
                     "n_targets": len(base_targets[pid])}
        moved_features = []
        target_ok = True
        for scale in (0.25, 4.0):
            a.reset()
            a.apply({pid: scale})
            got = targets(pid)
            want = [v * scale for v in base_targets[pid]]
            # Section-level attributes (Ra) and segment-level attributes alike:
            # the write must land on the intended object with the intended value.
            if not np.allclose(got, want, rtol=1e-9, atol=1e-14):
                target_ok = False
                rec[f"target_mismatch_{scale}"] = [got[:3], want[:3]]
            f = observe()
            for name, tol in EFFECT_FEATURES.items():
                b, g = base.get(name), f.get(name)
                if b is None or g is None:
                    continue
                if np.isfinite(b) and np.isfinite(g) and abs(g - b) > tol:
                    moved_features.append(f"{name}@{scale}")
                elif np.isfinite(b) != np.isfinite(g):
                    moved_features.append(f"{name}@{scale}:definedness")
        rec["target_write_ok"] = target_ok
        rec["moved_features"] = sorted(set(moved_features))

        # reset must restore the canonical profile exactly and reproduce the
        # baseline trace bit-for-bit
        a.reset()
        rec["reset_targets_exact"] = bool(
            np.allclose(targets(pid), base_targets[pid], rtol=0, atol=0))
        after = observe()
        rec["reset_trace_identical"] = bool(after["_v_hash"] == base["_v_hash"])
        out["parameters"][pid] = rec
    return out


@pytest.fixture(scope="session")
def activity() -> dict:
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(MODELS), max(1, mp.cpu_count() - 2)),
                  maxtasksperchild=1) as pool:
        return {r["model_id"]: r for r in pool.map(_activity, MODELS)}


@pytest.mark.parametrize("model_id", MODELS)
def test_n4_parameter_writes_reach_the_intended_target(activity, model_id):
    bad = [pid for pid, r in activity[model_id]["parameters"].items()
           if not r["target_write_ok"]]
    assert not bad, (
        f"{model_id}: scaling did not produce canonical*scale at the intended "
        f"NEURON attribute for {bad}")


@pytest.mark.parametrize("model_id", MODELS)
def test_n4_every_parameter_changes_the_model(activity, model_id):
    """No inactive parameters.  This is the C3 regression guard."""
    dead = [pid for pid, r in activity[model_id]["parameters"].items()
            if not r["moved_features"]]
    assert not dead, (
        f"{model_id}: scaling {dead} by 1/4 and 4x changed no observable "
        f"feature. An inactive, misaddressed or sign-cancelled parameter "
        f"contributes only noise to the dependence geometry -- and so does a "
        f"FOURTH case this check cannot distinguish: a parameter whose effect on "
        f"the membrane potential is real but smaller than the resolution of the "
        f"feature set (spike_count is an integer, ap_halfwidth_ms is quantised). "
        f"Check test_n4_parameter_writes_reach_the_intended_target first: if it "
        f"passes, the value IS reaching NEURON and the cause is not addressing. "
        f"Measured cases are recorded in "
        f"data/provenance/parameter_identifiability.json.")


@pytest.mark.parametrize("model_id", MODELS)
def test_n4_reset_restores_the_canonical_profile_exactly(activity, model_id):
    bad = [pid for pid, r in activity[model_id]["parameters"].items()
           if not r["reset_targets_exact"]]
    assert not bad, (
        f"{model_id}: reset() left {bad} off their canonical values; a single "
        f"built cell serves thousands of samples, so drift compounds")


@pytest.mark.parametrize("model_id", MODELS)
def test_n4_reset_reproduces_the_baseline_trace(activity, model_id):
    bad = [pid for pid, r in activity[model_id]["parameters"].items()
           if not r["reset_trace_identical"]]
    assert not bad, (
        f"{model_id}: after perturbing and resetting {bad} the baseline trace "
        f"changed; some state other than the parameter is leaking")
