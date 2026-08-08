"""Measure each model's canonical f-I curve and propose absolute amplitudes.

Blueprint 14.4 requires the absolute-current protocol amplitudes to be frozen
*before* any ensemble is generated, and audit section 9.6 item 6 repeats the
requirement for every model added to the panel.  Freezing them by hand from a
remembered f-I curve is how ``CANONICAL_RHEOBASE`` drifted out of date (audit
5.5), so this tool measures them instead and writes a machine-readable proposal
that a human then pastes into ``compass.protocols.common``.

The proposal rule is fixed in advance and applied identically to every model:

``hyperpol``  -0.5 x rheobase, **floored at -0.1 x the `hi` amplitude**, rounded
              to 4 decimals. The floor exists because -0.5 x rheobase is not a
              measurable stimulus for a cell that sits very close to threshold:
              `dougalis_vlpag_da` has rheobase 0.0039 nA, and at -0.002 nA the
              membrane *depolarises* by +2.8 mV over the step because the
              baseline is still settling, giving an input resistance of
              -223 MOhm. Input resistance is one of the 13 conditioning
              features, so a drift-dominated value would corrupt the phenotype
              for that model. The floor is a fraction of the model's own
              dynamic range rather than a fixed current, so it stays
              model-appropriate; it binds only for the two cells whose rheobase
              is at or near zero (`dougalis_vlpag_da`, `moubarak_snc_da`) and
              changes nothing for the other twelve.
``step_lo``   the smallest measured amplitude whose rate is >= 25% of the peak
              rate and which is not in depolarisation block
``step_hi``   the amplitude at ~70% of the peak rate on the RISING limb, again
              clear of block

Choosing both steps on the rising limb matters: an amplitude past the peak sits
on a non-monotonic part of the f-I curve, where a conductance change that raises
excitability *lowers* the measured rate and the feature stops being a monotone
readout of the parameter.

Nothing here consults an information endpoint, so amplitude selection cannot be
tuned toward a result.

    python tools/calibrate_amplitudes.py miyasho_purkinje tomko_ca1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY  # noqa: E402

OUT = ROOT / "data" / "provenance" / "amplitude_calibration.json"

# Calibration runs at a deliberately fine fixed step: the point is to measure the
# model, not to test a candidate production setting.
CALIB_DT = 0.0125
DELAY_MS, DURATION_MS, TOTAL_MS = 120.0, 400.0, 550.0
REST_TOTAL_MS = 300.0


def _scan(job) -> dict:
    import sys as _s

    _s.path.insert(0, str(ROOT / "src"))
    model_id, amps, dt = job
    from compass.adapters.models import get_adapter
    from compass.features.extract import extract, extract_rest

    kw = {"cvode": False, "dt_ms": dt}
    t0 = time.time()
    a = get_adapter(model_id).build()
    build_s = time.time() - t0

    rec = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0, total_ms=REST_TOTAL_MS,
                     protocol_id="rest", **kw)
    rest = extract_rest(rec)

    t0 = time.time()
    rheo = a.find_rheobase(delay_ms=40.0, duration_ms=140.0, total_ms=200.0, **kw)
    rheo_s = time.time() - t0

    curve = []
    t0 = time.time()
    for amp in amps:
        r = a.simulate(float(amp), delay_ms=DELAY_MS, duration_ms=DURATION_MS,
                       total_ms=TOTAL_MS, protocol_id="fi", **kw)
        f = extract(r)
        curve.append({
            "amplitude_nA": float(amp),
            "rate_Hz": None if f["mean_firing_rate_Hz"] is None else float(f["mean_firing_rate_Hz"]),
            "spike_count": None if f["spike_count"] is None else float(f["spike_count"]),
            "depol_block": bool(f["depolarisation_block"]),
            "ap_amplitude_mV": None if f["ap_amplitude_mV"] is None else float(f["ap_amplitude_mV"]),
            "qc": a.qc_flags(r),
        })
    per_protocol_s = (time.time() - t0) / max(1, len(amps))

    return {
        "model_id": model_id,
        "n_sections": len(a._sections.get("all", [a.soma])),
        "n_segments": a.manifest.n_compartments,
        "n_parameters": len(a.parameter_specs()),
        "build_s": round(build_s, 2),
        "rheobase_nA": None if rheo is None else float(rheo),
        "rheobase_search_s": round(rheo_s, 2),
        "per_protocol_s": round(per_protocol_s, 3),
        "resting_potential_mV": rest["resting_potential_mV"],
        "spontaneous_rate_Hz": rest["spontaneous_rate_Hz"],
        "calibration_dt_ms": dt,
        "fi_curve": curve,
        "canonical_absolute": {p.parameter_id: a.canonical_absolute(p.parameter_id)
                               for p in a.parameter_specs()},
    }


def propose(rec: dict) -> dict:
    """Apply the frozen selection rule to a measured f-I curve."""
    curve = [c for c in rec["fi_curve"] if c["rate_Hz"] is not None]
    firing = [c for c in curve if (c["rate_Hz"] or 0) > 0 and not c["depol_block"]]
    if not firing:
        return {"status": "no_firing", "amplitudes": None}
    peak = max(c["rate_Hz"] for c in firing)
    i_peak = max(range(len(firing)), key=lambda i: firing[i]["rate_Hz"])
    rising = firing[: i_peak + 1]

    def first_at(frac):
        want = frac * peak
        for c in rising:
            if c["rate_Hz"] >= want:
                return c["amplitude_nA"]
        return rising[-1]["amplitude_nA"]

    lo_amp = first_at(0.25)
    hi_amp = first_at(0.70)
    if hi_amp <= lo_amp:
        hi_amp = rising[-1]["amplitude_nA"]
    threshold = rec["rheobase_nA"] or min(c["amplitude_nA"] for c in firing)
    # -0.5 x rheobase, floored at -0.1 x the hi step: see the module docstring.
    # A step below the model's own baseline drift measures the drift, not the
    # membrane.
    hyper = min(-0.5 * abs(threshold), -0.1 * abs(hi_amp))
    return {
        "status": "ok",
        "peak_rate_Hz": peak,
        "peak_amplitude_nA": firing[i_peak]["amplitude_nA"],
        "hyperpol_floor_applied": bool(-0.5 * abs(threshold) > -0.1 * abs(hi_amp)),
        "amplitudes": (round(hyper, 4), round(lo_amp, 4), round(hi_amp, 4)),
        "rate_at_lo_Hz": next(c["rate_Hz"] for c in rising if c["amplitude_nA"] == lo_amp),
        "rate_at_hi_Hz": next(c["rate_Hz"] for c in rising if c["amplitude_nA"] == hi_amp),
    }


def amp_grid(model_id: str) -> np.ndarray:
    """Per-model current scan range, from the adapter's own search envelope."""
    hi = MODEL_REGISTRY[model_id].manifest_rheobase_hi
    return np.round(np.linspace(0.0, hi, 21), 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=sorted(MODEL_REGISTRY))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--dt", type=float, default=CALIB_DT)
    args = ap.parse_args()

    jobs = [(m, amp_grid(m).tolist(), args.dt) for m in args.models]
    out = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(_scan, jobs):
            rec["proposal"] = propose(rec)
            out[rec["model_id"]] = rec
            p = rec["proposal"]
            print(f"{rec['model_id']:20s} nseg={rec['n_segments']:>5} "
                  f"rheo={rec['rheobase_nA']} Vrest={rec['resting_potential_mV']:.1f} "
                  f"spont={rec['spontaneous_rate_Hz']:.1f}Hz "
                  f"t/protocol={rec['per_protocol_s']}s", flush=True)
            print(f"{'':20s} proposal={p.get('amplitudes')} "
                  f"rates=({p.get('rate_at_lo_Hz')}, {p.get('rate_at_hi_Hz')}) "
                  f"peak={p.get('peak_rate_Hz')}Hz@{p.get('peak_amplitude_nA')}nA",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing.update(out)
    existing["_written_utc"] = datetime.now(UTC).isoformat()
    OUT.write_text(json.dumps(existing, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
