#!/usr/bin/env python3
"""Analyses for the review of manuscript V22.

Two parts, written to results/analyses/review_v22/.

``stored`` (stored simulations only, no new runs)

effect_with_chance_ranges.parquet / table3_conditions.csv
    For every selected population in the stored two-stage table: the synergy
    part, the redundancy part and the net O-information of the selected
    population, the mean and the 2.5th-97.5th percentile range of the same
    quantity over 100 unselected samples of the same size (the chance range),
    the share of those samples at or above the selected value, and the same
    for 100 same-size feature-complete samples.  The redundancy part is the
    net O-information plus the synergy part (net = redundancy - synergy, both
    with the same normalisation; tuples.OrderCurve.normalised_areas).
    table3_conditions.csv keeps one condition per row (run, protocol set,
    distance rule, tolerance), so no column mixes conditions.
ratios.csv
    Per model and condition: synergy part of the selected population divided by
    the mean of the same-size feature-complete samples and of the same-size
    unselected samples, and the synergy-to-redundancy ratio of the selected
    population and of the unselected samples.  ratio_summary.csv gives, per
    condition, the median, range and geometric mean of these ratios over models
    (not the ratio of means).
geometry_excluded.parquet
    The same Gaussian summaries recomputed with the parameters that set cell
    size and passive cable properties (membrane capacitance, diameter and axial
    resistance) left out of the parameter vector, for every selected population
    of the main run and of the random-draw run and for the same 100 same-size
    unselected samples (identical rows, identical seeds as review_v18).  The Hay
    model has none of these parameters and is not included.  This removes the
    parameters from the measure; it does not fix them in the simulation.
top_triplets.csv / triplet_shares.csv
    In the selected populations at 3 and 0.5 SD (root-mean-square distance,
    fixed-amplitude protocol set, main run): the five most synergistic triplets
    with their parameter names and correlations, and the share of the order-3
    synergy area carried by triplets that contain a size or passive-cable
    parameter, against the share of triplets that contain one.
rank_one.json / rank_one_check.csv
    Numerical check of the analytical result for a single linear constraint
    (see rank_one_formula): independent standard normal parameters, keep the
    draws whose projection on a unit vector u lies in a band.  The covariance of
    the kept draws is exactly I - k u u^T, and the Gaussian O-information of a
    triplet is exactly
        0.5 [log(1 - k s) + sum_i log(1 - k a_i) - sum_i log(1 - k (s - a_i))],
    a_i = u_i^2, s = a_1 + a_2 + a_3, whose leading term is -k^3 a_1 a_2 a_3,
    i.e. the product of the three pairwise correlations (each about -k u_i u_j).
    A second constraint with a different sign pattern produces redundant
    triplets as well.

``hay`` (after the campaign screening_random has the Hay model)

hay_two_stage.parquet, hay_pools.parquet, hay_pairs_calibrated.parquet, hay_ksg_pairs.parquet,
hay_gauss_common_scale.parquet, hay_random_null.parquet, hay_random_knn.parquet, hay_random_pairs.parquet
    The review_v18 two-stage comparison, the review_v17 random-draw comparison
    and the nearest-neighbour checks, for the Hay model on 8,192 independent
    random draws, with the same code, seeds and reference sizes as the other
    models.

    python tools/run_review_v22.py stored
    python tools/run_review_v22.py hay [--workers N]
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "results" / "analyses" / "review_v22"
V18 = ROOT / "results" / "analyses" / "review_v18"
SEED = 20260804
KEY = ["experiment", "model_id", "protocol_family", "metric", "tolerance"]
TOL_SD = {"permissive": 3.0, "moderate": 2.0, "strict": 1.0, "very_strict": 0.5}
# parameters that set cell size or passive cable properties
GEOMETRIC = {
    "hh_point": ["cm_soma", "diam_soma"],
    "hh_cable": ["cm_all", "diam_dend"],
    "pospischil_rs": ["cm_soma", "diam_soma"],
    "pospischil_fs": ["cm_soma", "diam_soma"],
    "khaliq_purkinje": ["cm_soma", "diam_soma"],
    "migliore_ca1": ["cm_all", "ra_all"],
}
RANDOM = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje"]


def _load(name):
    spec = iu.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- stored 1
def chance_ranges() -> pd.DataFrame:
    ts = pd.read_parquet(V18 / "two_stage.parquet")
    ts = ts.assign(redundancy_area=ts.net_o_information + ts.synergy_area)
    qs = ["synergy_area", "redundancy_area", "net_o_information", "mean_abs_pearson"]
    sel = ts[ts.kind == "selected"].set_index(KEY)
    rows = []
    for k, s in sel.iterrows():
        r = dict(zip(KEY, k, strict=True))
        r.update(n=int(s.n), n_unselected=int(s.n_unselected), n_feature_complete=int(s.n_feature_complete))
        sub = ts[(ts[KEY] == pd.Series(k, index=KEY)).all(axis=1)]
        for ref in ("unselected_same_size", "feature_complete_same_size"):
            g = sub[sub.kind == ref]
            tag = "uns" if ref.startswith("unselected") else "fc"
            r[f"n_{tag}_samples"] = len(g)
            for q in qs:
                v = g[q].to_numpy()
                r[f"{q}_{tag}_mean"] = float(v.mean())
                r[f"{q}_{tag}_lo"] = float(np.quantile(v, 0.025))
                r[f"{q}_{tag}_hi"] = float(np.quantile(v, 0.975))
                r[f"{q}_{tag}_share_ge_selected"] = float(np.mean(v >= s[q]))
        for q in qs:
            r[f"{q}_selected"] = float(s[q])
            r[f"{q}_change"] = float(s[q]) - r[f"{q}_uns_mean"]
        rows.append(r)
    return pd.DataFrame(rows)


def ratios(cr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cr.copy()
    d["ratio_selected_to_feature_complete"] = d.synergy_area_selected / d.synergy_area_fc_mean
    d["ratio_selected_to_unselected"] = d.synergy_area_selected / d.synergy_area_uns_mean
    d["synergy_to_redundancy_selected"] = d.synergy_area_selected / d.redundancy_area_selected
    d["synergy_to_redundancy_unselected"] = d.synergy_area_uns_mean / d.redundancy_area_uns_mean
    cols = ["ratio_selected_to_feature_complete", "ratio_selected_to_unselected",
            "synergy_to_redundancy_selected", "synergy_to_redundancy_unselected"]
    out = d[KEY + ["n"] + cols]
    summ = []
    for k, g in out.groupby(["experiment", "protocol_family", "metric", "tolerance"]):
        r = dict(zip(["experiment", "protocol_family", "metric", "tolerance"], k, strict=True), n_models=len(g))
        for c in cols:
            v = g[c].to_numpy()
            r[f"{c}_median"] = float(np.median(v))
            r[f"{c}_min"] = float(v.min())
            r[f"{c}_max"] = float(v.max())
            r[f"{c}_geomean"] = float(np.exp(np.mean(np.log(v)))) if (v > 0).all() else float("nan")
            r[f"{c}_model_min"] = g.model_id.iloc[int(np.argmin(v))]
            r[f"{c}_model_max"] = g.model_id.iloc[int(np.argmax(v))]
        summ.append(r)
    return out, pd.DataFrame(summ)


# --------------------------------------------------------------------- stored 2
def geometry_excluded(task):
    experiment, mid = task
    v18 = _load("run_review_v18")
    ra, ens, pids, feats, target, f1, x1 = v18.setup(experiment, mid, "absolute")
    keep = [i for i, p in enumerate(pids) if p not in GEOMETRIC[mid]]
    rows = []
    for e in ens:
        if e.regime not in ("E3", "E3b", "E4", "E4b"):
            continue
        base = {"experiment": experiment, "model_id": mid, "protocol_family": "absolute",
                "metric": e.metric, "tolerance": e.tolerance, "n": e.n,
                "n_parameters_kept": len(keep), "parameters_removed": ",".join(GEOMETRIC[mid])}
        for which, cols in (("all", list(range(len(pids)))), ("geometry_excluded", keep)):
            rows.append(dict(base, parameters=which, kind="selected", draw=-1, **v18.stats(e.params[:, cols], ra)))
            for d in range(100):
                rng = np.random.default_rng(SEED + 104729 * d + e.n)  # the review_v18 rows
                u = x1[rng.choice(len(x1), e.n, replace=False)]
                rows.append(dict(base, parameters=which, kind="unselected_same_size", draw=d,
                                 **v18.stats(u[:, cols], ra)))
    print("geometry_excluded", experiment, mid, flush=True)
    return rows


# --------------------------------------------------------------------- stored 3
def top_triplets(mid):
    from compass.analysis.oinfo import GaussianInfoEngine
    v18 = _load("run_review_v18")
    ra, ens, pids, feats, target, f1, x1 = v18.setup("screening", mid, "absolute")
    geo = set(GEOMETRIC.get(mid, []))
    trip = list(itertools.combinations(range(len(pids)), 3))
    has_geo = np.array([any(pids[i] in geo for i in t) for t in trip])
    tops, shares = [], []
    for e in ens:
        if e.regime not in ("E3", "E4b") or e.metric != "weighted_euclidean":
            continue
        eng = GaussianInfoEngine(e.params, copula=False, bias_correct=True)
        om = eng.o_information_batch(trip)
        c = np.corrcoef(np.asarray(e.params, float), rowvar=False)
        neg = np.minimum(om, 0.0)
        shares.append({"model_id": mid, "tolerance": e.tolerance, "n": e.n, "n_triplets": len(trip),
                       "share_triplets_with_geometric": float(has_geo.mean()),
                       "share_synergy_area_with_geometric": float(neg[has_geo].sum() / neg.sum()) if neg.sum() < 0 else float("nan"),
                       "frac_synergy_o3": float(np.mean(om < 0))})
        for rank, j in enumerate(np.argsort(om)[:5], 1):
            a, b, cc = trip[j]
            tops.append({"model_id": mid, "tolerance": e.tolerance, "n": e.n, "rank": rank,
                         "triplet": " / ".join(pids[i] for i in trip[j]),
                         "o_information_nats": float(om[j]),
                         "r_ab": float(c[a, b]), "r_ac": float(c[a, cc]), "r_bc": float(c[b, cc]),
                         "product_of_correlations": float(c[a, b] * c[a, cc] * c[b, cc]),
                         "contains_geometric": bool(has_geo[j])})
    print("top_triplets", mid, flush=True)
    return tops, shares


# --------------------------------------------------------------------- stored 4
def rank_one_formula(a, k):
    a = np.asarray(a, float)
    s = a.sum()
    return 0.5 * (np.log1p(-k * s) + np.log1p(-k * a).sum() - np.log1p(-k * (s - a)).sum())


def gaussian_o(x):
    c = np.cov(x, rowvar=False)
    n = c.shape[0]
    ld = lambda m: np.linalg.slogdet(np.atleast_2d(m))[1]  # noqa: E731
    h = 0.5 * ld(c)
    o = (n - 2) * h
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        o += 0.5 * ld(c[i, i]) - 0.5 * ld(c[np.ix_(rest, rest)])
    return o


def rank_one():
    from scipy.stats import norm
    rng = np.random.default_rng(SEED)
    p, n = 8, 400_000
    u = rng.normal(size=p)
    u /= np.linalg.norm(u)
    x = rng.normal(size=(n, p))
    z = x @ u
    rows = []
    trip = list(itertools.combinations(range(p), 3))
    for tau in (3.0, 2.0, 1.0, 0.5, 0.25):
        keep = np.abs(z) < tau
        k_th = 2 * tau * norm.pdf(tau) / (2 * norm.cdf(tau) - 1)  # 1 - Var(z | |z| < tau)
        xs = x[keep]
        c = np.corrcoef(xs, rowvar=False)
        for t in trip:
            a = u[list(t)] ** 2
            exact = rank_one_formula(a, k_th)
            r = [c[t[0], t[1]], c[t[0], t[2]], c[t[1], t[2]]]
            rows.append({"band_halfwidth_sd": tau, "k": k_th, "n_kept": int(keep.sum()),
                         "triplet": t, "o_simulated": gaussian_o(xs[:, list(t)]),
                         "o_exact": exact, "o_leading": -k_th ** 3 * a.prod(),
                         "product_of_correlations": float(np.prod(r)),
                         "mean_abs_r": float(np.mean(np.abs(r)))})
    df = pd.DataFrame(rows)
    # the formula is negative for every k in (0, 1) and every direction
    kk = np.linspace(1e-3, 0.999, 400)
    aa = rng.dirichlet(np.ones(p), size=2000)[:, :3]
    worst = max(rank_one_formula(a, k) for a in aa[:200] for k in kk)
    # scaling of the synergy part with the mean absolute correlation
    g = df.groupby("band_halfwidth_sd").agg(k=("k", "first"), mean_abs_r=("mean_abs_r", "mean"),
                                            synergy=("o_simulated", lambda v: -np.minimum(v, 0).mean()))
    slope = float(np.polyfit(np.log(g.mean_abs_r), np.log(g.synergy), 1)[0])
    # two constraints: the sign pattern of the correlations now varies, and some
    # triplets become redundant
    v = rng.normal(size=p)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    keep2 = (np.abs(z) < 0.5) & (np.abs(x @ v) < 0.5)
    o2 = np.array([gaussian_o(x[keep2][:, list(t)]) for t in trip])
    summary = {
        "set_up": "8 independent standard normal parameters; keep draws whose projection on a random unit "
                  "vector lies within +/- tau; 400,000 draws",
        "max_abs_error_exact_vs_simulated": float((df.o_simulated - df.o_exact).abs().max()),
        "max_o_exact_over_grid": float(worst),
        "share_triplets_synergistic_one_constraint": float((df.o_simulated < 0).mean()),
        "log_log_slope_synergy_vs_mean_abs_r": slope,
        "corr_leading_vs_exact_at_tau_0p5": float(np.corrcoef(df[df.band_halfwidth_sd == 0.5].o_leading,
                                                              df[df.band_halfwidth_sd == 0.5].o_exact)[0, 1]),
        "two_constraints_share_synergistic": float((o2 < 0).mean()),
        "two_constraints_share_redundant": float((o2 > 0).mean()),
        "by_band": g.reset_index().to_dict(orient="records"),
    }
    return df, summary


def run_stored(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    cr = chance_ranges()
    cr.to_parquet(OUT / "effect_with_chance_ranges.parquet", index=False)
    cr.to_csv(OUT / "table3_conditions.csv", index=False, float_format="%.6g")
    r, s = ratios(cr)
    r.to_csv(OUT / "ratios.csv", index=False, float_format="%.6g")
    s.to_csv(OUT / "ratio_summary.csv", index=False, float_format="%.6g")
    print("chance ranges and ratios written", flush=True)
    df, summ = rank_one()
    df.to_csv(OUT / "rank_one_check.csv", index=False, float_format="%.6g")
    (OUT / "rank_one.json").write_text(json.dumps(summ, indent=1))
    print("rank one written", flush=True)
    tasks = [("screening", m) for m in GEOMETRIC] + [("screening_random", m) for m in RANDOM]
    geo, tops, shares = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        fg = [ex.submit(geometry_excluded, t) for t in tasks]
        ft = [ex.submit(top_triplets, m) for m in list(GEOMETRIC) + ["bbp_l5pc"]]
        for f in ft:
            a, b = f.result()
            tops += a
            shares += b
        pd.DataFrame(tops).to_csv(OUT / "top_triplets.csv", index=False, float_format="%.6g")
        pd.DataFrame(shares).to_csv(OUT / "triplet_shares.csv", index=False, float_format="%.6g")
        for f in fg:
            geo += f.result()
    pd.DataFrame(geo).to_parquet(OUT / "geometry_excluded.parquet", index=False)
    print("stored part done", flush=True)


# ------------------------------------------------------------------------ hay
def _call(module: str, func: str, arg):
    """Load a tools/ script inside the worker (loaded modules cannot be pickled)."""
    return getattr(_load(module), func)(arg)


def run_hay(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    mid = "bbp_l5pc"
    with ProcessPoolExecutor(workers) as ex:
        f_ts = ex.submit(_call, "run_review_v18", "two_stage", ("screening_random", mid, "absolute"))
        f_pr = ex.submit(_call, "run_review_v18", "pairs_random", mid)
        f_rd = ex.submit(_call, "run_review_random_draws", "run_model", mid)
        ts, pools = f_ts.result()
        pd.DataFrame(ts).to_parquet(OUT / "hay_two_stage.parquet", index=False)
        pd.DataFrame(pools).to_parquet(OUT / "hay_pools.parquet", index=False)
        cal, common, ksg = f_pr.result()
        pd.DataFrame(cal).to_parquet(OUT / "hay_pairs_calibrated.parquet", index=False)
        pd.DataFrame(common).to_parquet(OUT / "hay_gauss_common_scale.parquet", index=False)
        pd.DataFrame(ksg).to_parquet(OUT / "hay_ksg_pairs.parquet", index=False)
        rd = f_rd.result()
        for k, name in (("null", "hay_random_null"), ("knn", "hay_random_knn"), ("pairs", "hay_random_pairs")):
            pd.DataFrame(rd[k]).to_parquet(OUT / f"{name}.parquet", index=False)
    print("hay part done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["stored", "hay"])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    (run_stored if a.part == "stored" else run_hay)(a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
