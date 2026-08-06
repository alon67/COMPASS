"""Block-based simulation workflow with atomic result shards (blueprint Task 7).

Execution model
---------------
* One **process per model** -- NEURON's section list is process-global, so a
  worker is pinned to a single model and builds the cell exactly once, then
  reuses it for every sample in its block (parameter injection + reset).
* Work is split into **blocks** of samples.  Each block writes one parquet shard
  atomically (temp file + rename), so an interrupted campaign resumes by simply
  skipping shards that already exist -- no duplicated samples, no partial files.
* Failures are recorded, never dropped: ``simulation_status`` and ``qc_flags``
  columns carry solver failures, non-physical voltages and extraction failures
  into the audit table (blueprint 26 "never silently remove numerical failures").
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"


# Campaign solver policy: DETERMINISTIC FIXED TIME STEP.
#
# The convergence audit (tools/numerical_qc.py) selects the cheapest setting that
# reproduces the finest-step result at *canonical* parameters, and for most
# models that is CVODE.  Under a broad prior, however, CVODE's cost is a
# function of how stiff a particular parameter sample happens to be, and the
# tail is severe: profiling mainen_l5pc over six prior samples gave a median of
# 11 s but individual samples of 233 s and 307 s (mean 96 s).  A campaign cannot
# be budgeted against that distribution, and a per-sample cost that depends on
# the sample is also a subtle sampling bias if any sample is ever abandoned.
#
# Each dt below is the largest step the audit accepted for that model
# (data/provenance/numerical_qc.json, "accepted" field), so cost is constant per
# simulation and every result is bit-reproducible.
CAMPAIGN_SOLVER = {
    # Settings accepted by tools/convergence_audit.py: every one of the 13
    # conditioning features reproduced, on 6 parameter points drawn from each
    # model's own prior, against a reference at half the step on twice the mesh.
    # This SUPERSEDES tools/numerical_qc.py, which judged a single canonical
    # point on spike count alone -- a knife-edge criterion that both passed
    # settings that do not converge (mainen, bbp, hh_point, khaliq) and rejected
    # one that does (migliore).
    "hh_point": {"cvode": False, "dt_ms": 0.025},        # 0.05 fails membrane tau
    "hh_cable": {"cvode": False, "dt_ms": 0.05},
    "pospischil_rs": {"cvode": False, "dt_ms": 0.05},
    "pospischil_fs": {"cvode": False, "dt_ms": 0.025},
    # Spontaneously active, resurgent Na + spatial Ca diffusion: the stiffest
    # model in the panel.  0.0125 still misses membrane tau and AP amplitude on
    # 1 of 6 samples; 0.00625 reproduces everything.
    "khaliq_purkinje": {"cvode": False, "dt_ms": 0.00625},
    # Needs BOTH a finer step and a finer mesh; the adapter builds 2x the
    # d_lambda(100 Hz) mesh (2304 segments) for this reason.
    "mainen_l5pc": {"cvode": False, "dt_ms": 0.0125},
    "bbp_l5pc": {"cvode": False, "dt_ms": 0.025},        # 0.05 fails AP amplitude
    # migliore_ca1: dt=0.025 on the d_lambda(100 Hz) mesh (1106 segments).
    # numerical_qc.py originally REJECTED this step on an exact spike-count
    # comparison at the canonical point, where a 0.5 nA step sits on a knife
    # edge (4 vs 5 spikes).  The feature-level audit accepts it: agreement is
    # 1.00 for every feature on every point.
    "migliore_ca1": {"cvode": False, "dt_ms": 0.025},
}


def solver_kwargs(model_id: str) -> dict:
    if model_id in CAMPAIGN_SOLVER:
        return dict(CAMPAIGN_SOLVER[model_id])
    qc_path = ROOT / "data" / "provenance" / "numerical_qc.json"
    if qc_path.exists():
        for r in json.loads(qc_path.read_text()):
            if r["model_id"] == model_id:
                return dict(r["chosen_kwargs"])
    return {"cvode": False, "dt_ms": 0.025}


def _feature_row(adapter, kw, rheobase, protocols, spontaneous: bool) -> tuple[dict, list[str]]:
    """Run the common-core protocol suite for one already-applied parameter set."""
    from compass.features.extract import extract, extract_rest

    row: dict[str, float] = {}
    flags: list[str] = []
    for spec in protocols:
        pid = spec.protocol_id
        if spec.kind == "rest":
            rec = adapter.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                                   total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract_rest(rec)
        else:
            if spec.amplitude_nA is not None:
                amp = spec.amplitude_nA
            elif rheobase is None or (spontaneous and rheobase == 0.0):
                # rheobase-normalised protocols are undefined for a
                # spontaneously active or non-spiking sample
                for k in ("spike_count", "mean_firing_rate_Hz", "first_isi_ms",
                          "adaptation_index", "ap_amplitude_mV", "ap_halfwidth_ms",
                          "ahp_depth_mV", "first_spike_latency_ms", "cv_isi",
                          "ap_threshold_mV", "ap_peak_mV", "last_isi_ms",
                          "depolarisation_block", "resting_potential_mV"):
                    row[f"{pid}.{k}"] = np.nan
                continue
            else:
                amp = rheobase * spec.amplitude_rheobase_mult
            rec = adapter.simulate(float(amp), delay_ms=spec.delay_ms,
                                   duration_ms=spec.duration_ms,
                                   total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract(rec)
        f = adapter.qc_flags(rec)
        if f:
            flags.extend(f"{pid}:{x}" for x in f)
        for k, v in feats.items():
            row[f"{pid}.{k}"] = float(v) if v is not None and np.isfinite(v) else np.nan
    return row, flags


def run_block(
    model_id: str,
    scales: np.ndarray,
    param_ids: list[str],
    sample_ids: np.ndarray,
    regime: str,
    replicate: int,
    seed: int,
    prior_id: str,
    out_path: Path,
) -> dict:
    """Simulate one block of parameter samples and write an atomic parquet shard."""
    from compass.adapters.models import get_adapter
    from compass.protocols import common as P

    kw = solver_kwargs(model_id)
    # Forward the FULL solver settings (including cvode) to the rheobase search.
    # The bisection costs ~9 simulations per sample and dominates the campaign
    # budget; running it fixed-step while the protocols run under CVODE made the
    # search ~10x more expensive than everything else combined.
    rheo_kw = dict(kw)
    spontaneous = model_id in P.SPONTANEOUSLY_ACTIVE
    protocols = P.common_core(model_id)

    adapter = get_adapter(model_id).build()
    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids):
        adapter.reset()
        rec_meta = {
            "run_id": f"{model_id}:{regime}:{replicate}:{int(sid)}",
            "model_id": model_id,
            "model_variant_id": "canonical",
            "sampling_regime": regime,
            "prior_id": prior_id,
            "conditioning_id": "none",
            "replicate": replicate,
            "seed": seed,
            "parameter_sample_id": int(sid),
            "protocol_id": "common_core",
        }
        try:
            adapter.apply(dict(zip(param_ids, scales[i], strict=True)))
            rheo = adapter.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                                         duration_ms=P.RHEO_DURATION_MS,
                                         total_ms=P.RHEO_TOTAL_MS, **rheo_kw)
            feats, flags = _feature_row(adapter, kw, rheo, protocols, spontaneous)
            rec_meta["simulation_status"] = "ok"
            rec_meta["qc_flags"] = ";".join(sorted(set(flags)))
            rec_meta["rheobase_nA"] = np.nan if rheo is None else float(rheo)
            rec_meta["non_spiking"] = bool(rheo is None)
            rec_meta.update(feats)
        except Exception as exc:  # noqa: BLE001
            rec_meta["simulation_status"] = "error"
            rec_meta["qc_flags"] = f"exception:{type(exc).__name__}"
            rec_meta["error"] = f"{exc}\n{traceback.format_exc()[-400:]}"
            rec_meta["rheobase_nA"] = np.nan
            rec_meta["non_spiking"] = True
        rows.append(rec_meta)

    df = pd.DataFrame(rows)
    for j, pid in enumerate(param_ids):
        df[f"param.{pid}"] = scales[:, j]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return {
        "shard": out_path.name,
        "n": len(rows),
        "wall_s": round(time.time() - t_start, 1),
        "n_error": int((df["simulation_status"] == "error").sum()),
        "n_nonspiking": int(df["non_spiking"].sum()),
    }


def _worker(args) -> dict:
    """Entry point executed in a fresh process (one model, one block)."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")
    (model_id, scales, param_ids, sample_ids, regime, replicate, seed,
     prior_id, out_path) = args
    if Path(out_path).exists():
        return {"shard": Path(out_path).name, "n": 0, "wall_s": 0.0,
                "n_error": 0, "n_nonspiking": 0, "skipped": True}
    return run_block(model_id, np.asarray(scales), list(param_ids),
                     np.asarray(sample_ids), regime, replicate, seed,
                     prior_id, Path(out_path))
