#!/usr/bin/env python3
"""Selection effect measured on populations drawn with independent random draws.

Review of manuscript V17 ("second-order vs higher-order dependence"). The main
simulation run drew the unconditioned prior with a scrambled Sobol design. A
selected population keeps every design point inside a region, and therefore the
design's even spacing; a random same-size thinning of the design loses most of
it. The nearest-neighbour entropy estimator responds to spacing, so on the Sobol
design the comparison "selected versus same-size unselected" mixes dependence
with spacing, and shuffled copies or Gaussian-copula surrogates (which are
pseudo-random) are not valid references either.

The campaign ``screening_random`` redraws the prior for the five models whose
selected populations are large enough for the nearest-neighbour estimator with
independent pseudo-random draws. On those draws every reference below is valid:

* unselected samples of the same size (random subsets of the random draws);
* shuffled copies of the selected population (each parameter permuted
  independently: marginals kept, all dependence removed);
* Gaussian-copula surrogates with the selected population's rank correlations
  and marginals (dependence kept up to second order in normal scores, anything
  beyond removed).  Selected minus surrogate is the dependence beyond pairwise
  correlation that the Gaussian estimator cannot see.

Outputs (results/analyses/review_v17/):

random_null.parquet
    Gaussian summaries (share of synergy-dominated triplets, synergy part,
    net O-information, mean |r|) of every selected population of the random
    draws and of 100 unselected random subsets of the same size -- the Fig. 3
    comparison repeated without the Sobol design.
random_knn.parquet
    Per-triplet nearest-neighbour O-information (normal scores) and Gaussian
    O-information of every selected population with at least KNN_MIN_N members,
    with 10 unselected same-size subsets, 10 shuffled copies and 10 copula
    surrogates.  Also the unselected random population itself against its own
    shuffled copies and surrogates (calibration: should show no difference).
random_pairs.parquet
    Pairwise mutual information (nearest-neighbour, normal scores, and Gaussian)
    for the same copies, so the excess of the non-parametric pairwise
    information over the Gaussian one can be compared with the copula surrogate.
"""
from __future__ import annotations

import importlib.util as iu
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os  # noqa: E402

# Overridable only for smoke tests; the stored results use the defaults.
EXPERIMENT = os.environ.get("COMPASS_REVIEW_EXPERIMENT", "screening_random")
MODELS = os.environ.get("COMPASS_REVIEW_MODELS",
                        "hh_point hh_cable pospischil_rs pospischil_fs khaliq_purkinje").split()
OUT = ROOT / "results" / "analyses" / os.environ.get("COMPASS_REVIEW_OUT", "review_v17")
N_NULL = int(os.environ.get("COMPASS_REVIEW_N_NULL", 100))
N_KNN_REF = int(os.environ.get("COMPASS_REVIEW_N_REF", 10))
KNN_MIN_N = 2000


def _ra():
    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(x, seed, ra):
    from compass.analysis.geometry import describe_geometry
    from compass.analysis.tuples import order_curve
    oc = order_curve(x, orders=ra.ORDERS, max_tuples=ra.MAX_TUPLES, seed=seed,
                     copula=False, bias_correct=True, n_surrogate=0)
    syn, red = oc.normalised_areas()
    return {"frac_synergy_o3": oc.frac_synergy[0], "synergy_area": syn,
            "redundancy_area": red, "net_o_information": red - syn,
            "mean_abs_pearson": describe_geometry(x).as_dict()["mean_abs_pearson"]}


def _omegas(x, tuples, estimator):
    from compass.analysis.oinfo import o_information
    return np.array([o_information(x[:, list(t)], estimator) for t in tuples])


def _pair_mi(x, pairs, estimator):
    from compass.analysis.oinfo import mutual_information
    return np.array([mutual_information(x[:, a], x[:, b], estimator) for a, b in pairs])


def references(x, e1, seed, n_ref):
    from compass.sampling.priors import gaussian_copula_control
    out = [("selected", -1, x)]
    for d in range(n_ref):
        if e1 is not None:
            rng = np.random.default_rng(seed + 104729 * d + len(x))
            out.append(("unselected_same_size", d, e1[rng.choice(len(e1), size=len(x), replace=False)]))
        rng = np.random.default_rng(seed + 7 * d + 3)
        out.append(("shuffled", d, np.column_stack([rng.permutation(c) for c in x.T])))
        out.append(("copula_surrogate", d, gaussian_copula_control(x, seed=seed + 1000 + d)))
    return out


def run_model(mid: str) -> dict[str, list[dict]]:
    from compass.adapters.models import get_adapter
    from compass.analysis.cohorts import UNIDENTIFIABLE_PARAMETERS
    from compass.analysis.entropy import copula_transform
    from compass.analysis.tuples import generate_tuples
    from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles

    ra = _ra()
    seed = ra.SEED
    df = ra.load_model_table(EXPERIMENT, mid)
    drop = set(UNIDENTIFIABLE_PARAMETERS.get(mid, []))
    pids = [s.parameter_id for s in get_adapter(mid).parameter_specs() if s.parameter_id not in drop]
    feats = [f for f in CONDITIONING_FEATURES if f in df.columns]
    ens = build_ensembles(df, mid, pids, feats, target=ra.canonical_target(mid, feats), seed=seed,
                          protocol_family="absolute", qc_policy="primary")
    e1 = [e for e in ens if e.regime == "E1"][0].params
    sel = [e for e in ens if e.regime in ("E3", "E3b", "E4", "E4b")]
    res = {"null": [], "knn": [], "pairs": []}

    # ---- the Fig. 3 comparison on random draws (Gaussian estimator) ---------
    for e in sel:
        row = {"model_id": mid, "metric": e.metric, "tolerance": e.tolerance, "n": e.n,
               "n_unselected": len(e1), "kind": "selected", "draw": -1}
        row.update(stats(e.params, seed, ra))
        res["null"].append(row)
        for d in range(N_NULL):
            rng = np.random.default_rng(seed + 104729 * d + e.n)
            r = dict(row, kind="unselected_same_size", draw=d)
            r.update(stats(e1[rng.choice(len(e1), size=e.n, replace=False)], seed, ra))
            res["null"].append(r)

    # ---- nearest-neighbour estimator against valid references ---------------
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=seed).tuples
    pairs = list(itertools.combinations(range(len(pids)), 2))
    pops = [("unselected", "none", "none", e1, None)]
    pops += [("selected", e.metric, e.tolerance, e.params, e1) for e in sel if e.n >= KNN_MIN_N]
    # Fig. 5B: the subset chosen for strong covariance and its size-matched random
    # subset.  Both are subsets of a selected population of random draws, so
    # their spacing is the same and the comparison is valid for this estimator.
    for e in ens:
        if e.regime in ("E7", "E3m") and e.n >= KNN_MIN_N // 2:
            pops.append(({"E7": "high_covariance_subset", "E3m": "matched_random_subset"}[e.regime],
                         e.metric, e.tolerance, e.params, None))
    # Fig. 5A: the correlated prior and the clustered population of the main run
    # were drawn pseudo-randomly (no Sobol design), so they can be tested too.
    if EXPERIMENT == "screening_random":
        dfm = ra.load_model_table("screening", mid)
        ens_m = build_ensembles(dfm, mid, pids, feats, target=ra.canonical_target(mid, feats),
                                seed=seed, protocol_family="absolute", qc_policy="primary")
        for e in ens_m:
            if e.regime in ("E2", "E6"):
                pops.append(({"E2": "correlated_prior", "E6": "clustered"}[e.regime],
                             "none", "none", e.params, None))
    for label, metric, tol, x, ref_e1 in pops:
        for kind, d, y in references(x, ref_e1, seed, N_KNN_REF):
            z = copula_transform(y)
            vk = _omegas(z, ts, "knn")
            vg = _omegas(y, ts, "gaussian")
            base = {"model_id": mid, "population": label, "metric": metric, "tolerance": tol,
                    "n": len(x), "kind": kind, "draw": d}
            for t, a, g in zip(ts, vk, vg, strict=True):
                res["knn"].append(dict(base, tuple="-".join(map(str, t)), omega_knn=a, omega_gaussian=g))
            mk = _pair_mi(z, pairs, "knn")
            mg = _pair_mi(y, pairs, "gaussian")
            for (a, b), u, v in zip(pairs, mk, mg, strict=True):
                res["pairs"].append(dict(base, pair=f"{a}-{b}", mi_knn=u, mi_gaussian=v))
    print(f"done {mid}: {len(sel)} selected populations, "
          f"{sum(1 for p in pops if p[0] == 'selected')} large enough for the nearest-neighbour check",
          flush=True)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    allres = {"null": [], "knn": [], "pairs": []}
    with ProcessPoolExecutor(max_workers=len(MODELS)) as ex:
        for r in ex.map(run_model, MODELS):
            for k in allres:
                allres[k] += r[k]
    for k, name in (("null", "random_null"), ("knn", "random_knn"), ("pairs", "random_pairs")):
        pd.DataFrame(allres[k]).to_parquet(OUT / f"{name}.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
