"""Estimator benchmarks and the non-neuronal comparator experiment.

Part A (blueprint 17.2, WP5) -- estimator validation on systems with known
structure, including the sample-size dependence of the estimate.

Part B (blueprint 17.3) -- the decisive generic-geometry test.  Two nonlinear
non-neuronal dynamical systems (the Elowitz-Leibler repressilator and a
three-species Lotka-Volterra community) are sampled from *independent* priors
and then conditioned on their own canonical phenotype using exactly the same
machinery as the neuron models.  If O-information moves from ~0 to strongly
negative here, higher-order synergy in phenotype-conditioned ensembles cannot be
read as a fact about neurons.

Output: results/analyses/benchmarks.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.analysis.benchmarks import (  # noqa: E402
    SYNTHETIC_SYSTEMS,
    lotka_volterra_features,
    repressilator_features,
    sample_synthetic,
)
from compass.analysis.geometry import describe_geometry  # noqa: E402
from compass.analysis.oinfo import o_information, rsi  # noqa: E402
from compass.analysis.tuples import order_curve  # noqa: E402
from compass.conditioning.distances import TOLERANCES, condition  # noqa: E402

SEED = 20260804
OUT = ROOT / "results" / "analyses"


# --------------------------------------------------------------------------- #
def part_a_estimators() -> list[dict]:
    """Sign and magnitude recovery on known systems, across estimators and N."""
    rows = []
    for name, (_, expect, sign) in SYNTHETIC_SYSTEMS.items():
        for n in (300, 1000, 4000, 16000):
            # d = 4 and the whole matrix is analysed: every system places its
            # structure inside the analysed columns, so nothing is diluted by
            # independent padding variables.
            x = sample_synthetic(name, n, 4, SEED)
            for est in ("gaussian", "copula", "knn"):
                try:
                    val = o_information(x, est)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"system": name, "n": n, "estimator": est,
                                 "error": str(exc)[:80]})
                    continue
                classified = ("redundancy" if val > 0.02 else
                              "synergy" if val < -0.02 else "none")
                rows.append({
                    "system": name, "expected": expect, "expected_sign": sign,
                    "n": n, "estimator": est, "o_information": float(val),
                    "classified": classified, "correct": bool(classified == expect),
                })
    return rows


# --------------------------------------------------------------------------- #
def _conditioning_experiment(name: str, feature_fn, n_params: int, n: int,
                             prior_decades: float, feature_names: list[str],
                             workers: int = 10) -> dict:
    """Sample an independent prior, condition on the canonical phenotype, compare.

    This is the exact protocol applied to the neuron models, run on a system
    with no neurons in it.
    """
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(SEED)
    log_p = rng.uniform(-prior_decades / 2, prior_decades / 2, size=(n, n_params))
    params = 10.0**log_p

    with ProcessPoolExecutor(workers) as ex:
        feats = np.array(list(ex.map(feature_fn, params, chunksize=32)))

    canonical = feature_fn(np.ones(n_params))
    finite = np.isfinite(feats).all(axis=1)
    disp = np.nanstd(feats[finite], axis=0)
    target = np.where(np.isfinite(canonical), canonical,
                      np.nanmedian(feats[finite], axis=0))

    out = {
        "system": name, "n_prior": int(n), "n_params": n_params,
        "prior_decades": prior_decades,
        "n_valid": int(finite.sum()),
        "feature_names": feature_names,
        "canonical_features": [None if not np.isfinite(v) else float(v) for v in canonical],
        "ensembles": [],
    }

    def record(label, x, extra=None):
        if len(x) < 100:
            return
        oc = order_curve(x, orders=(3, 4, 5), max_tuples=800, seed=SEED, n_surrogate=20)
        syn, red = oc.normalised_areas()
        geo = describe_geometry(x)
        rec = {
            "label": label, "n": int(len(x)),
            "o_information_full": float(o_information(x)),
            "synergy_area": syn, "redundancy_area": red,
            "frac_synergy_o3": oc.frac_synergy[0],
            "frac_synergy_sig_o3": oc.frac_synergy_sig[0],
            "median_omega_o3": oc.median[0],
            "mean_abs_pearson": geo.mean_abs_pearson,
            "effective_rank": geo.effective_rank,
            "intrinsic_dimension": geo.intrinsic_dimension,
        }
        rec.update(extra or {})
        out["ensembles"].append(rec)

    # E1: the unconditioned independent prior -- ground truth is zero dependence
    record("E1_unconditioned_prior", log_p)
    # E3/E4: phenotype-conditioned at each tolerance
    for tol in TOLERANCES:
        res = condition(feats, target, disp, metric="max_z", tolerance_name=tol)
        record(f"conditioned_{tol}", log_p[res.accepted],
               {"tolerance": tol, "acceptance_rate": res.acceptance_rate,
                "threshold": res.threshold})
    return out


def part_b_comparators(n: int, workers: int) -> list[dict]:
    return [
        _conditioning_experiment(
            "repressilator", repressilator_features, 6, n, 1.0,
            ["period", "amplitude", "mean_level", "n_peaks", "cv_peak_interval",
             "log_protein_mean"], workers),
        _conditioning_experiment(
            "lotka_volterra", lotka_volterra_features, 6, n, 0.8,
            ["mean_x1", "mean_x2", "mean_x3", "total_biomass",
             "oscillation_amplitude", "n_surviving"], workers),
    ]


# --------------------------------------------------------------------------- #
def part_c_rsi_signs() -> list[dict]:
    """Explicit RSI sign-convention validation report (blueprint WP5)."""
    rng = np.random.default_rng(SEED)
    rows = []
    n = 20000
    # synergistic: target is the sum of independent inputs
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=n)
    r = rsi(x, y)
    rows.append({"case": "sum_of_independent_inputs", "expected": "synergy",
                 "rsi": r.value, "joint": r.joint, "sum_individual": r.sum_individual,
                 "correct": bool(r.value > 0)})
    # redundant: two noisy copies of the same latent predict it
    z = rng.normal(size=n)
    x = np.column_stack([z + 0.1 * rng.normal(size=n) for _ in range(2)])
    y = z + 0.1 * rng.normal(size=n)
    r = rsi(x, y)
    rows.append({"case": "duplicated_predictors", "expected": "redundancy",
                 "rsi": r.value, "joint": r.joint, "sum_individual": r.sum_individual,
                 "correct": bool(r.value < 0)})
    # independent/additive: neither
    x = rng.normal(size=(n, 2))
    y = 2.0 * x[:, 0] + rng.normal(size=n)
    r = rsi(x, y)
    rows.append({"case": "single_informative_predictor", "expected": "none",
                 "rsi": r.value, "joint": r.joint, "sum_individual": r.sum_individual,
                 "correct": bool(abs(r.value) < 0.02)})
    return rows


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    print("part A: estimator validation on known systems ...", flush=True)
    a = part_a_estimators()
    n_ok = sum(r.get("correct", False) for r in a)
    print(f"  {n_ok}/{len(a)} estimator x system x N cases classified correctly")

    print("part C: RSI sign conventions ...", flush=True)
    c = part_c_rsi_signs()
    for r in c:
        print(f"  {r['case']:32s} RSI={r['rsi']:+.4f} expected={r['expected']:10s} "
              f"{'OK' if r['correct'] else 'FAIL'}")

    print(f"part B: non-neuronal comparators (n={args.n}) ...", flush=True)
    b = part_b_comparators(args.n, args.workers)
    for sysrec in b:
        print(f"  {sysrec['system']}  valid={sysrec['n_valid']}/{sysrec['n_prior']}")
        for e in sysrec["ensembles"]:
            print(f"    {e['label']:28s} n={e['n']:6d} "
                  f"Omega_full={e['o_information_full']:+.4f} "
                  f"synergy_area={e['synergy_area']:+.4f} "
                  f"frac_syn_o3={e['frac_synergy_o3']:.3f} "
                  f"|r|={e['mean_abs_pearson']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmarks.json").write_text(json.dumps(
        {"estimator_validation": a, "rsi_sign_validation": c, "comparators": b},
        indent=1, default=float))
    print(f"\nwrote {OUT / 'benchmarks.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
