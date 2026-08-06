"""Variance-based sensitivity campaign (blueprint Task 12, WP7).

Sobol indices must be computed on an intact Saltelli design drawn from the
declared prior -- never on a phenotype-filtered ensemble (blueprint 16.4).  This
tool therefore runs its own simulations rather than reusing the screening
ensembles.

    python tools/run_sobol.py --models hh_point pospischil_rs
    python tools/run_sobol.py --analyse-only
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import get_adapter  # noqa: E402
from compass.analysis.sensitivity import (  # noqa: E402
    VALID_FRACTION_FLOOR,  # noqa: E402
    saltelli_design,
    sobol_convergence,
    sobol_indices,
)
from compass.sampling.priors import scales_from_unit  # noqa: E402
from compass.workflows.simulate import _worker  # noqa: E402

FEATURES = [
    "rheobase_nA",
    "hyperpol_abs.input_resistance_Mohm",
    "rest.resting_potential_mV",
    "step_abs_hi.mean_firing_rate_Hz",
    "step_abs_hi.ap_amplitude_mV",
    "step_abs_hi.ap_halfwidth_ms",
    "step_abs_hi.adaptation_index",
]

# Base sample size per model, chosen so the Saltelli design N*(2D+2) stays
# affordable at each model's measured per-sample cost.
N_BASE = {
    "hh_point": 256, "hh_cable": 256, "pospischil_rs": 256, "pospischil_fs": 256,
    "khaliq_purkinje": 128, "mainen_l5pc": 32, "bbp_l5pc": 64, "migliore_ca1": 32,
}
BLOCK = 64
SEED = 20260804
FEAT_DIR = ROOT / "results" / "features" / "sobol"
OUT = ROOT / "results" / "analyses"


def simulate(models: list[str], workers: int) -> None:
    jobs = []
    for mid in models:
        specs = get_adapter(mid).parameter_specs()
        pids = [s.parameter_id for s in specs]
        n_base = N_BASE[mid]
        u = saltelli_design(len(specs), n_base, seed=SEED, calc_second_order=True)
        scales = scales_from_unit(u, specs)
        np.save(FEAT_DIR / f"{mid}_design.npy", u) if FEAT_DIR.exists() else None
        for start in range(0, len(scales), BLOCK):
            stop = min(start + BLOCK, len(scales))
            out = FEAT_DIR / mid / f"S_{start:06d}.parquet"
            jobs.append((mid, scales[start:stop], pids, np.arange(start, stop),
                         "SOBOL", 1, SEED, f"saltelli_N{n_base}", str(out)))
    todo = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"sobol: {len(jobs)} blocks, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, _st in enumerate(pool.imap_unordered(_worker, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} blocks  {(time.time()-t0)/60:.1f} min", flush=True)


def analyse(models: list[str]) -> None:
    rows, conv_rows, s2_rows = [], [], []
    for mid in models:
        files = sorted(glob.glob(str(FEAT_DIR / mid / "*.parquet")))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values("parameter_sample_id").reset_index(drop=True)
        specs = get_adapter(mid).parameter_specs()
        names = [s.parameter_id for s in specs]
        n_base = N_BASE[mid]
        expected = n_base * (2 * len(names) + 2)
        if len(df) != expected:
            print(f"  {mid}: have {len(df)}/{expected} rows, skipping "
                  f"(Sobol needs the intact design)", flush=True)
            continue
        for feat in FEATURES:
            if feat not in df.columns:
                continue
            y = df[feat].to_numpy(dtype=float)
            try:
                res = sobol_indices(y, names, n_base, output_name=feat, seed=SEED)
            except ValueError as exc:
                rows.append({"model_id": mid, "feature": feat, "error": str(exc)[:120],
                             "quotable": False, "valid_fraction": float("nan")})
                continue
            rows.append({
                "model_id": mid, "feature": feat, "n_base": n_base,
                "n_valid": res.n_valid, "n_total": len(y),
                "valid_fraction": res.n_valid / len(y),
                "quotable": bool(res.n_valid / len(y) >= VALID_FRACTION_FLOOR),
                "interaction_fraction": res.interaction_fraction,
                "mean_ST_minus_S1": res.mean_total_minus_first,
                "sum_S1": float(np.nansum(res.S1)),
                "sum_ST": float(np.nansum(res.ST)),
                "max_S1_conf": float(np.nanmax(res.S1_conf)),
            })
            for j, nm in enumerate(names):
                s2_rows.append({"model_id": mid, "feature": feat, "parameter": nm,
                                "S1": float(res.S1[j]), "S1_conf": float(res.S1_conf[j]),
                                "ST": float(res.ST[j]), "ST_conf": float(res.ST_conf[j]),
                                "ST_minus_S1": float(res.ST[j] - res.S1[j])})
            if res.S2 is not None:
                s2 = res.S2
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        if np.isfinite(s2[a, b]):
                            s2_rows.append({
                                "model_id": mid, "feature": feat,
                                "parameter": f"{names[a]}*{names[b]}",
                                "S1": np.nan, "S1_conf": np.nan,
                                "ST": np.nan, "ST_conf": np.nan,
                                "S2": float(s2[a, b]),
                                "S2_conf": float(res.S2_conf[a, b])})
            for c in sobol_convergence(y, names, n_base, output_name=feat, seed=SEED):
                c.update({"model_id": mid, "feature": feat})
                conv_rows.append(c)
        print(f"  {mid}: analysed {len(FEATURES)} features (N={n_base}, "
              f"{len(df)} runs)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "sobol_summary.parquet", index=False)
    pd.DataFrame(s2_rows).to_parquet(OUT / "sobol_indices.parquet", index=False)
    pd.DataFrame(conv_rows).to_parquet(OUT / "sobol_convergence.parquet", index=False)
    print(f"wrote sobol tables ({len(rows)} summaries) -> {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(N_BASE))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        simulate(args.models, args.workers)
    analyse(args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
