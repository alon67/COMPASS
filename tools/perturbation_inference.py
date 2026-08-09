#!/usr/bin/env python3
"""Background-aware inference on the raw perturbation outcomes.

Replaces ``tools/readout_resolution.py``, whose conclusions the 2026-08-06 audit
withdrew (section 4).

What was wrong
--------------
The withdrawn tool compared each model's **median over six tuple means of eight
backgrounds** against **one unaveraged single-observation spike quantum**
(2.5 Hz for a 400 ms window).  Three separate errors:

1. the reported statistic is a mean over eight backgrounds, so its native grid
   is 2.5/8 = 0.3125 Hz, and a median over six such means can be finer still;
2. a deterministic quantisation grid is not a stochastic detection limit -- a
   mean below one quantum is exactly observable and exactly repeatable; and
3. the boolean ``resolvable`` it produced tested neither measurement error, nor
   simulation variability, nor background variability, nor the uncertainty of
   the mean.

What replaces it
----------------
"Resolvable" is split into the two questions it was conflating, each with a
prespecified definition (audit 4.4 point 4).

**Numerical repeatability** -- would this number survive a finer integration?
Re-run the battery at half the campaign time step and compare per-background
``eta`` directly.  Repeatable means the paired difference is small relative to
the effect: median |Δeta| below ``REPEAT_TOL_SD`` feature SDs *and* per-tuple
sign agreement of the mean.  This is a property of the solver, not of the data
analysis, and it is measured, not asserted.

**Inferential precision** -- is the background mean distinguishable from zero,
and how large is it?  Estimated from the spread across backgrounds: a
bias-corrected-and-accelerated-free percentile bootstrap over backgrounds, an
exact paired sign test on background-level ``eta`` (which is distribution-free
and therefore safe for a quantised readout), and, for the count readout, a
paired exact resampling test on the integer spike counts themselves.

Counts are analysed as counts
-----------------------------
``mean_firing_rate_Hz`` is a spike count over a fixed window.  The battery now
stores ``spike_count``, so the discrete readout is analysed on the integer
scale, where the quantisation is a property of the sample space rather than a
nuisance: ``eta_count = (N_joint - N_base) - sum_i (N_i - N_base)`` is an
integer, and its exact permutation distribution over sign flips is available in
closed form for eight backgrounds (2^8 = 256 assignments, enumerated exactly).

Writes ``results/analyses/perturbation_inference.json`` and
``results/analyses/perturbation_inference.parquet``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from compass.adapters.base import (  # noqa: E402
    MEASUREMENT_SCHEMA_VERSION as _MEASUREMENT_SCHEMA_VERSION,
)

OUT = ROOT / "results" / "analyses"
SEED = 20260806
N_BOOT = 10000

# Prespecified before looking at the repeatability run.  A per-background eta
# that moves by less than a twentieth of the accepted-ensemble feature SD when
# the time step is halved is numerically repeatable at the resolution this study
# reports effects at (effects of interest are >= 0.25 SD).
REPEAT_TOL_SD = 0.05


def exact_sign_test(x: np.ndarray) -> float:
    """Exact two-sided sign test that the median of ``x`` is zero.

    Distribution-free, so a quantised readout costs it nothing.  Ties (exact
    zeros, which a discrete readout produces often) are dropped, which is the
    conservative convention.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    n = x.size
    if n == 0:
        return float("nan")
    k = int((x > 0).sum())
    from math import comb

    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return float(min(1.0, 2.0 * tail / 2**n))


def exact_signflip_p(x: np.ndarray, max_exact: int = 20) -> float:
    """Two-sided p for mean(x) = 0 under exact sign-flip randomisation.

    With eight backgrounds the full 2^8 set of sign assignments is enumerated,
    so no Monte-Carlo error enters the p-value.  Above ``max_exact`` backgrounds
    it falls back to sampling.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    obs = abs(float(np.mean(x)))
    if n <= max_exact:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        stats = np.abs(signs @ x) / n
    else:
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1.0, 1.0], size=(N_BOOT, n))
        stats = np.abs(signs @ x) / n
    return float((stats >= obs - 1e-12).mean())


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, x.size, replace=True))
                      for _ in range(n_boot)])
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def precision_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per model x feature x tuple: effect, uncertainty and sign stability."""
    rows = []
    for (mid, feat, tup, kind), g in raw.groupby(
            ["model_id", "feature", "tuple", "kind"], sort=True):
        eta = g["nonadditivity_sd"].to_numpy(dtype=float)
        eta_raw = g["nonadditivity_raw"].to_numpy(dtype=float)
        fin = np.isfinite(eta)
        lo, hi = bootstrap_ci(eta)
        rows.append({
            "model_id": mid, "feature": feat, "tuple": tup, "kind": kind,
            "n_backgrounds": int(len(g)),
            "n_finite": int(fin.sum()),
            "mean_eta_sd": float(np.nanmean(eta)) if fin.any() else np.nan,
            "median_eta_sd": float(np.nanmedian(eta)) if fin.any() else np.nan,
            "sd_eta_sd": float(np.nanstd(eta, ddof=1)) if fin.sum() > 1 else np.nan,
            "sem_eta_sd": (float(np.nanstd(eta, ddof=1) / np.sqrt(fin.sum()))
                           if fin.sum() > 1 else np.nan),
            "ci95_lo": lo, "ci95_hi": hi,
            # A mean whose bootstrap interval excludes zero is distinguishable
            # from zero given the background variation -- the question the
            # withdrawn "floor" was pretending to answer.
            "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (lo > 0 or hi < 0)),
            "sign_test_p": exact_sign_test(eta),
            "signflip_p": exact_signflip_p(eta),
            "sign_consistency": (float(max((eta[fin] > 0).mean(), (eta[fin] < 0).mean()))
                                 if fin.any() else np.nan),
            "mean_eta_native": float(np.nanmean(eta_raw)) if fin.any() else np.nan,
            "feature_sd": float(g["feature_sd"].iloc[0]),
        })
    return pd.DataFrame(rows)


def count_inference(raw: pd.DataFrame) -> pd.DataFrame:
    """Exact paired inference on the integer spike-count non-additivity."""
    cnt = raw[raw["feature"] == "spike_count"]
    rows = []
    for (mid, tup, kind), g in cnt.groupby(["model_id", "tuple", "kind"], sort=True):
        eta = g["nonadditivity_raw"].to_numpy(dtype=float)
        fin = eta[np.isfinite(eta)]
        if fin.size == 0:
            continue
        integral = bool(np.all(np.abs(fin - np.round(fin)) < 1e-9))
        rows.append({
            "model_id": mid, "tuple": tup, "kind": kind,
            "n_backgrounds": int(fin.size),
            "eta_counts": ";".join(str(int(round(v))) for v in fin) if integral
            else ";".join(f"{v:.3f}" for v in fin),
            "eta_is_integer": integral,
            "mean_eta_counts": float(np.mean(fin)),
            "median_eta_counts": float(np.median(fin)),
            "n_nonzero": int((np.round(fin) != 0).sum()),
            "exact_signflip_p": exact_signflip_p(fin),
            "exact_sign_test_p": exact_sign_test(fin),
        })
    return pd.DataFrame(rows)


def repeatability(base: pd.DataFrame, fine: pd.DataFrame) -> dict:
    """Compare per-background eta between the campaign dt and a halved dt."""
    keys = ["model_id", "tuple", "feature", "background"]
    merged = base.merge(fine, on=keys, suffixes=("_base", "_fine"))
    if merged.empty:
        return {"status": "no_paired_rows"}
    merged["d_eta_sd"] = merged["nonadditivity_sd_fine"] - merged["nonadditivity_sd_base"]
    per_model = {}
    for mid, g in merged.groupby("model_id"):
        prim = g[g["feature"] == "mean_firing_rate_Hz"]
        d = prim["d_eta_sd"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        by_tuple = prim.groupby("tuple").agg(
            base=("nonadditivity_sd_base", "mean"),
            fine=("nonadditivity_sd_fine", "mean"))
        sign_ok = float((np.sign(by_tuple["base"]) == np.sign(by_tuple["fine"])).mean()) \
            if len(by_tuple) else float("nan")
        per_model[mid] = {
            "n_paired": int(d.size),
            "median_abs_delta_eta_sd": float(np.median(np.abs(d))) if d.size else np.nan,
            "max_abs_delta_eta_sd": float(np.max(np.abs(d))) if d.size else np.nan,
            "tuple_mean_sign_agreement": sign_ok,
            "numerically_repeatable": bool(
                d.size and np.median(np.abs(d)) <= REPEAT_TOL_SD
                and np.isfinite(sign_ok) and sign_ok >= 0.8),
        }
    # Diagnostic: WHERE the sign instability lives.  The verdict above is the
    # prespecified one and is not rescored -- but a failure concentrated
    # entirely in effects that are indistinguishable from zero means something
    # different from one spread across large effects, and the record should say
    # which it is.
    prim_all = merged[merged["feature"] == "mean_firing_rate_Hz"]
    tup = (prim_all.groupby(["model_id", "tuple"])
           .agg(base=("nonadditivity_sd_base", "mean"),
                fine=("nonadditivity_sd_fine", "mean")).reset_index())
    tup["flipped"] = np.sign(tup["base"]) != np.sign(tup["fine"])
    tup["abs_base"] = tup["base"].abs()
    bands, edges = {}, [(0.0, 0.02), (0.02, 0.10), (0.10, np.inf)]
    for lo, hi in edges:
        g = tup[(tup["abs_base"] >= lo) & (tup["abs_base"] < hi)]
        label = f"{lo}-{hi} SD" if np.isfinite(hi) else f">={lo} SD"
        bands[label] = {
            "n_tuples": int(len(g)),
            "n_sign_flipped": int(g["flipped"].sum()),
            "fraction_flipped": round(float(g["flipped"].mean()), 4) if len(g) else None,
            "median_abs_delta_eta_sd": (round(float((g["fine"] - g["base"]).abs().median()), 4)
                                        if len(g) else None),
        }

    n_ok = sum(v["numerically_repeatable"] for v in per_model.values())
    return {
        "status": "ok",
        "sign_stability_by_effect_size": bands,
        "diagnostic_note": (
            "the prespecified verdict is not rescored on this breakdown. It is "
            "reported because a sign instability confined to effects that are "
            "already indistinguishable from zero constrains the conclusion "
            "differently from one that touches large effects. The preregistered "
            "functional-support threshold is 0.25 SD."),
        "definition": (f"per-background eta recomputed at half the campaign dt; "
                       f"repeatable = median |delta eta| <= {REPEAT_TOL_SD} feature "
                       f"SD AND >= 80% of tuple means keep their sign"),
        "tolerance_sd": REPEAT_TOL_SD,
        "per_model": per_model,
        "n_models_repeatable": int(n_ok),
        "n_models": len(per_model),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="perturbations_raw.parquet")
    ap.add_argument("--fine", default="perturbations_raw_dt_half.parquet",
                    help="optional halved-dt rerun for the repeatability check")
    args = ap.parse_args()

    raw_path = OUT / args.raw
    if not raw_path.exists():
        raise SystemExit(
            f"missing {raw_path}. Run tools/run_perturbations.py first -- the raw "
            "per-background outcomes it now stores are what this tool analyses.")
    raw = pd.read_parquet(raw_path)

    prec = precision_table(raw)
    counts = count_inference(raw)
    prec.to_parquet(OUT / "perturbation_inference.parquet", index=False)

    fine_path = OUT / args.fine
    repeat = (repeatability(raw, pd.read_parquet(fine_path)) if fine_path.exists()
              else {"status": "not_run",
                    "note": "run tools/run_perturbations.py --dt <half> "
                            "--suffix _dt_half to populate"})

    prim = prec[(prec["feature"] == "mean_firing_rate_Hz")
                & (prec["kind"] == "top_synergy")]
    by_model = prim.groupby("model_id").agg(
        n_tuples=("tuple", "size"),
        median_abs_mean_eta_sd=("mean_eta_sd", lambda s: float(np.nanmedian(np.abs(s)))),
        median_sem=("sem_eta_sd", "median"),
        n_ci_excludes_zero=("ci_excludes_zero", "sum"),
        median_signflip_p=("signflip_p", "median"))

    payload = {
        "supersedes": "results/analyses/readout_resolution.json",
        "withdrawn_claim": (
            "'the firing-rate readout cannot resolve the effect in 4/8 models, "
            "so the panel-level enrichment is carried by two toy models'. "
            "Withdrawn: it compared a median of eight-background means with one "
            "unaveraged 2.5 Hz single-observation quantum, and a deterministic "
            "grid is not a stochastic detection limit."),
        "quantisation_arithmetic": {
            "window_ms": 400.0,
            "single_observation_quantum_Hz": 2.5,
            "n_backgrounds": int(raw["background"].nunique()),
            "grid_of_the_reported_mean_Hz": 2.5 / max(1, raw["background"].nunique()),
            "note": "stated as arithmetic only; it licenses no conclusion",
        },
        "inferential_precision": {
            "definition": ("bootstrap CI and exact sign-flip test over the "
                           "background distribution of eta; an effect is "
                           "'distinguishable' when the 95% CI excludes zero"),
            "per_model_primary_feature": {
                m: {
                    "n_top_tuples": int(r.n_tuples),
                    "median_abs_mean_eta_sd": round(float(r.median_abs_mean_eta_sd), 4),
                    "median_sem_sd": (round(float(r.median_sem), 4)
                                      if np.isfinite(r.median_sem) else None),
                    "n_tuples_ci_excludes_zero": int(r.n_ci_excludes_zero),
                    "median_signflip_p": (round(float(r.median_signflip_p), 4)
                                          if np.isfinite(r.median_signflip_p) else None),
                } for m, r in by_model.iterrows()
            },
            "n_top_tuples_total": int(len(prim)),
            "n_top_tuples_ci_excludes_zero": int(prim["ci_excludes_zero"].sum()),
        },
        "discrete_count_analysis": {
            "definition": ("eta recomputed on integer spike counts and tested by "
                           "exact enumeration of all 2^n background sign flips"),
            "n_tuples": int(len(counts)),
            "n_integer_valued": int(counts["eta_is_integer"].sum()) if len(counts) else 0,
            "n_significant_p05": int((counts["exact_signflip_p"] < 0.05).sum())
            if len(counts) else 0,
            "per_kind_mean_abs_eta_counts": (
                counts.assign(a=counts["mean_eta_counts"].abs())
                      .groupby("kind")["a"].mean().round(4).to_dict()
                if len(counts) else {}),
        },
        "numerical_repeatability": repeat,
        # The measurement schema the battery's inputs were produced under.
        # run_crossmodel refuses to score the perturbation-derived H-B criteria
        # when this does not match the current campaign schema: the battery's
        # backgrounds and candidate tuples are derived from feature rows, so a
        # measurement change makes stored eta a probe of points that no longer
        # exist in the ensemble it claims to test (2026-08-09 audit, 4.1).
        "measurement_schema": _MEASUREMENT_SCHEMA_VERSION,
    }
    (OUT / "perturbation_inference.json").write_text(
        json.dumps(payload, indent=1, default=float))

    print(json.dumps(payload, indent=1, default=float)[:3000])
    print(f"\nwrote {OUT/'perturbation_inference.json'} and "
          f"{OUT/'perturbation_inference.parquet'} ({len(prec)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
