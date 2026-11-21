"""Canonical-reproduction audit (blueprint WP1, Task 3).

For every model at its published parameter values this records:
  * resting potential, rheobase, and the full f-I curve;
  * the common-core feature vector;
  * a canonical voltage trace (retained permanently, blueprint 20.1);
  * wall-clock cost per protocol, used to size the campaign.

Run one model per process: NEURON's section list is process-global.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.features.extract import extract, extract_rest  # noqa: E402
from compass.protocols import common as P  # noqa: E402

OUT = ROOT / "results" / "manifests"
TRACES = ROOT / "results" / "traces"

FI_FRACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])


def solver_kwargs(model_id: str) -> dict:
    """The ONE authoritative solver contract: whatever the campaign runs.

    This used to read `chosen_kwargs` from `numerical_qc.json`, the first-pass
    QC sweep, which selected CVODE at dt = 0.025 ms for six of the eight models.
    The campaign, however, runs fixed-step at per-model time steps chosen by the
    later feature-level convergence audit (`CAMPAIGN_SOLVER`).  The repository
    therefore carried two contradictory solver contracts, and the canonical
    manifests -- which define the conditioning target every ensemble is accepted
    against -- were produced under the integrator the campaign does NOT use.

    The consequence was not cosmetic: a target computed under an adaptive-step
    integrator anchors ensembles simulated at a fixed step, so "accepted" meant
    "near a phenotype this model never produces under its own campaign
    settings".  Caught by the 2026-08-06 audit (C2).

    `numerical_qc.json` is retained as the historical record of the first pass;
    it is no longer consulted for production settings.
    """
    from compass.workflows.simulate import CAMPAIGN_SOLVER

    if model_id in CAMPAIGN_SOLVER:
        return dict(CAMPAIGN_SOLVER[model_id])
    raise KeyError(
        f"{model_id} has no entry in CAMPAIGN_SOLVER; refusing to guess a "
        f"solver contract for a canonical manifest.")


def audit(model_id: str) -> dict:
    kw = solver_kwargs(model_id)
    t0 = time.time()
    a = get_adapter(model_id).build()
    build_s = time.time() - t0

    rec_rest = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                          total_ms=P.REST_TOTAL_MS, protocol_id="rest", **kw)
    rest_feats = extract_rest(rec_rest)

    t0 = time.time()
    # Inherit the FULL solver contract (including cvode).  Passing only dt_ms
    # left the rheobase search on fixed steps while the protocols ran under
    # CVODE, so the canonical rheobase and every amplitude derived from it came
    # from a different integrator than the features they anchor.
    rheo = a.find_rheobase(delay_ms=P.RHEO_DELAY_MS, duration_ms=P.RHEO_DURATION_MS,
                           total_ms=P.RHEO_TOTAL_MS, **kw)
    rheo_s = time.time() - t0

    # f-I curve.  For a spontaneously active model rheobase is 0, so scan on an
    # absolute grid anchored to the model's own search envelope instead.
    if rheo and rheo > 0:
        amps = rheo * FI_FRACTIONS
    else:
        amps = a.manifest_rheobase_hi * FI_FRACTIONS / 8.0
    fi = []
    t0 = time.time()
    for amp in amps:
        r = a.simulate(float(amp), delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                       total_ms=P.TOTAL_MS, protocol_id="fi", **kw)
        f = extract(r)
        fi.append({"amplitude_nA": float(amp), "rate_Hz": f["mean_firing_rate_Hz"],
                   "spike_count": f["spike_count"],
                   "depol_block": f["depolarisation_block"],
                   "ap_amplitude_mV": f["ap_amplitude_mV"],
                   "ap_halfwidth_ms": f["ap_halfwidth_ms"],
                   "qc": a.qc_flags(r)})
    fi_s = (time.time() - t0) / len(amps)

    # canonical traces retained for the report and for regression tests
    TRACES.mkdir(parents=True, exist_ok=True)
    keep = {}
    for label, amp in [("rest", 0.0),
                       ("hyperpol", P.ABSOLUTE_AMPLITUDES[model_id][0]),
                       ("step_lo", P.ABSOLUTE_AMPLITUDES[model_id][1]),
                       ("step_hi", P.ABSOLUTE_AMPLITUDES[model_id][2])]:
        r = a.simulate(amp, delay_ms=P.DELAY_MS if label != "rest" else 0.0,
                       duration_ms=P.DURATION_MS if label != "rest" else 0.0,
                       total_ms=P.TOTAL_MS if label != "rest" else P.REST_TOTAL_MS,
                       protocol_id=label, **kw)
        keep[label] = {"t": r.t.astype(np.float32), "v": r.v.astype(np.float32)}
    np.savez_compressed(TRACES / f"canonical_{model_id}.npz",
                        **{f"{k}_{f}": v[f] for k, v in keep.items() for f in ("t", "v")})

    # common-core features at the canonical point
    feats = {}
    for spec in P.common_core(model_id):
        if spec.kind == "rest":
            r = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                           total_ms=P.REST_TOTAL_MS, protocol_id=spec.protocol_id, **kw)
            f = extract_rest(r)
        else:
            if spec.amplitude_nA is None:
                # Rheobase-scaled step: undefined when the canonical model fires
                # spontaneously (rheobase 0), exactly as for individual draws
                # (simulate.normalised_family_defined).  Simulating "3 x 0 nA"
                # used to store spontaneous firing as a valid target.
                from compass.workflows.simulate import normalised_family_defined
                if not normalised_family_defined(rheo):
                    continue
            amp = (spec.amplitude_nA if spec.amplitude_nA is not None
                   else (rheo or 0.0) * spec.amplitude_rheobase_mult)
            r = a.simulate(float(amp), delay_ms=spec.delay_ms, duration_ms=spec.duration_ms,
                           total_ms=spec.total_ms, protocol_id=spec.protocol_id, **kw)
            f = extract(r)
        for k, v in f.items():
            feats[f"{spec.protocol_id}.{k}"] = None if v is None or (
                isinstance(v, float) and not np.isfinite(v)) else float(v)

    specs = a.parameter_specs()
    return {
        "model_id": model_id,
        "manifest": a.manifest.to_row(),
        "solver": kw,
        "build_s": round(build_s, 3),
        "rheobase_nA": rheo,
        "rheobase_search_s": round(rheo_s, 2),
        "mean_protocol_s": round(fi_s, 3),
        "n_parameters": len(specs),
        "n_compartments": a.manifest.n_compartments,
        "resting_potential_mV": rest_feats["resting_potential_mV"],
        "spontaneous_rate_Hz": rest_feats["spontaneous_rate_Hz"],
        "fi_curve": fi,
        "canonical_features": feats,
        "parameters": [s.to_row() for s in specs],
    }


def _audit_one(model_id: str) -> dict:
    """Audit a single model. MUST run in a fresh process -- see main()."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
    return audit(model_id)


def main(argv):
    targets = argv[1:] or list(MODEL_REGISTRY)
    OUT.mkdir(parents=True, exist_ok=True)

    # One model per process, always.
    #
    # NEURON keeps a single global section list per process, so models loaded
    # from published .hoc morphologies must discover their own sections by
    # differencing `h.allsec()`; building a second model in the same process
    # mixes the lists and silently inserts one model's mechanisms into another's
    # sections.  The adapters gained a guard against that, at which point THIS
    # LOOP STOPPED WORKING -- it had always built every model in one process,
    # so from that commit on `canonical_audit.py` could regenerate the first
    # model's manifest and then died.
    #
    # That is why the manifests were stale (audit C2): not because nobody re-ran
    # the tool, but because the tool could not complete.  A regeneration path
    # that cannot be exercised is not a regeneration path, so the failure is
    # worth naming rather than just fixing.
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(targets), 4), maxtasksperchild=1) as pool:
        for r in pool.imap(_audit_one, targets):
            mid = r["model_id"]
            (OUT / f"canonical_{mid}.json").write_text(json.dumps(r, indent=1))
            fi = [f"{d['amplitude_nA']:.3f}:"
                  f"{0 if d['rate_Hz'] is None or not np.isfinite(d['rate_Hz']) else int(d['rate_Hz'])}"
                  for d in r["fi_curve"]]
            print(f"AUDIT {mid:16s} rheo={r['rheobase_nA']} "
                  f"Vrest={r['resting_potential_mV']:.1f} "
                  f"spont={r['spontaneous_rate_Hz']:.1f}Hz ncomp={r['n_compartments']} "
                  f"np={r['n_parameters']} solver={r['solver']} "
                  f"t/protocol={r['mean_protocol_s']}s", flush=True)
            print(f"      f-I (nA:Hz) {' '.join(fi)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
