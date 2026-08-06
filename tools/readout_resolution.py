#!/usr/bin/env python3
"""Per-model, per-readout measurement floors for the perturbation battery.

Why this exists
---------------
The functional non-additivity statistic eta is reported in units of the accepted
ensemble's SD for that feature.  That normalisation hides an absolute floor: a
readout cannot resolve a difference smaller than its own quantisation step.

* ``mean_firing_rate_Hz`` is a **count** over a fixed window.  With a 400 ms
  step, one extra spike is 2.5 Hz -- an irreducible quantum, no matter how fine
  the integration time step.
* ``first_isi_ms`` and ``ap_halfwidth_ms`` are **times**, quantised by the
  solver step dt of that model.
* ``ap_amplitude_mV`` is a **continuous** voltage; it has no count quantum.

Dividing the quantum by each model's own feature SD gives the floor in the same
units eta is quoted in, so "is this effect resolvable?" becomes a direct
comparison.  The 2026-08-06 audit (M5) flagged the quantisation; this tool makes
the consequence measurable per model instead of asserted globally.

The headline consequence: the two multicompartment cortical pyramidal
reconstructions -- the models the module claim would matter most for -- have
median |eta| BELOW one spike-count quantum, so the panel-level enrichment is
carried by the two Hodgkin-Huxley toy models.

Writes ``results/analyses/readout_resolution.json``.  No simulation required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "results" / "analyses"


def _quantum(feature: str, dt_ms: float) -> float | None:
    """Absolute measurement quantum of a readout, in that readout's own units."""
    from compass.protocols import common as P

    if feature == "mean_firing_rate_Hz":
        # one extra spike in the analysis window
        return 1000.0 / P.DURATION_MS
    if feature in ("first_isi_ms", "ap_halfwidth_ms"):
        # a time measured on the solver grid
        return dt_ms
    if feature == "ap_amplitude_mV":
        return None  # continuous readout: no count/grid quantum
    return None


def main() -> int:
    from run_perturbations import PERT_FEATURES, load_backgrounds

    from compass.workflows.simulate import CAMPAIGN_SOLVER

    pert = pd.read_parquet(OUT / "perturbations.parquet")
    models = sorted(pert["model_id"].unique())

    rows = []
    for mid in models:
        dt = float(CAMPAIGN_SOLVER[mid]["dt_ms"]) if mid in CAMPAIGN_SOLVER else np.nan
        # reuse the battery's own SD definition so the floor is in the same
        # units as the eta values it produced
        param_ids = sorted({p for t in pert[pert["model_id"] == mid]["tuple"]
                            for p in str(t).split("|")})
        _, sd = load_backgrounds("screening", mid, param_ids, n=8, seed=0)
        if not len(sd):
            print(f"  {mid}: no accepted ensemble, skipped", flush=True)
            continue
        for k, feat in enumerate(PERT_FEATURES):
            q = _quantum(feat, dt)
            sd_f = float(sd[k])
            g = pert[(pert["model_id"] == mid) & (pert["feature"] == feat)
                     & (pert["kind"] == "top_synergy")]
            med = float(np.nanmedian(np.abs(
                pd.to_numeric(g["mean_nonadditivity_sd"], errors="coerce"))))
            floor_sd = (q / sd_f) if (q is not None and sd_f > 0) else None
            rows.append({
                "model_id": mid,
                "feature": feat,
                "dt_ms": dt,
                "feature_sd": sd_f,
                "quantum_native_units": q,
                "floor_in_sd": floor_sd,
                "median_abs_eta_sd": med,
                "resolvable": (None if floor_sd is None
                               else bool(np.isfinite(med) and med > floor_sd)),
            })
        print(f"  {mid}: dt={dt} ms, rate SD={sd[0]:.1f} Hz", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "readout_resolution.parquet", index=False)

    rate = df[df["feature"] == "mean_firing_rate_Hz"]
    payload = {
        "note": ("Measurement floors for the perturbation readouts. "
                 "floor_in_sd is the readout's quantisation step expressed in "
                 "units of that model's accepted-ensemble feature SD, i.e. "
                 "directly comparable with median_abs_eta_sd. "
                 "ap_amplitude_mV is continuous and has no count quantum."),
        "step_duration_ms": 400.0,
        "rate_quantum_Hz": 2.5,
        "per_model_rate": {
            r.model_id: {
                "feature_sd_Hz": round(r.feature_sd, 3),
                "floor_in_sd": round(r.floor_in_sd, 4),
                "median_abs_eta_sd": round(r.median_abs_eta_sd, 4),
                "resolvable": r.resolvable,
            } for r in rate.itertuples()
        },
        "n_models_rate_resolvable": int(rate["resolvable"].sum()),
        "n_models_rate_total": int(len(rate)),
        "unresolvable_models": sorted(rate[~rate["resolvable"].astype(bool)]["model_id"]),
    }
    (OUT / "readout_resolution.json").write_text(json.dumps(payload, indent=1,
                                                            default=float))
    print(f"\nrate readout resolvable in "
          f"{payload['n_models_rate_resolvable']}/{payload['n_models_rate_total']} models; "
          f"below floor: {payload['unresolvable_models']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
