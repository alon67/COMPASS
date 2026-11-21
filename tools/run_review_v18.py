#!/usr/bin/env python3
"""Analyses for the review of manuscript V18 (stored simulations only, no new runs).

Parts (results/analyses/review_v18/):

two_stage.parquet
    Every selected population (both protocol families, both distance criteria,
    all tolerances) of the main Sobol run (seven implementations of the primary
    panel) and of the random-draw run (five implementations), with Gaussian
    summaries of: the selected population; 100 same-size samples of the
    unselected sample after numerical quality control; 100 same-size samples of
    the feature-complete pool (rows of the unselected sample whose conditioning
    features are all finite).  Separates the two stages of the acceptance rule:
    "all features measurable" and "distance to the target below the tolerance".
    Also used for the matched-size version of Fig. 4C.
pools.parquet
    Sizes of the unselected sample, the feature-complete pool and each selected
    population, and the Gaussian summaries of the full feature-complete pool.
dropout_decomposed.parquet
    Leave-one-feature-out at 3 and 2 SD (absolute family, Sobol run), split into
    rows newly eligible because the left-out feature was undefined and rows whose
    acceptance changed through the distance alone, with the synergy part of the
    population re-selected on the fixed complete-case rows.
pairs_calibrated.parquet, gauss_common_scale.parquet, ksg_pairs.parquet
    Random-draw run.  Excess of nearest-neighbour pairwise information over
    Gaussian-copula surrogates in each selected population and in 10 same-size
    unselected samples (each with its own surrogates); Gaussian triplet shift and
    Gaussian pairwise information on normal scores (the scale used by the
    nearest-neighbour estimator); KSG mutual information at 3 SD with 39
    surrogates.
sobol_box.parquet
    Spacing control on the Sobol run: nearest-neighbour O-information of every
    triplet in the five large selected populations, in 5 axis-aligned boxes of
    the design with the same number of points, and in 10 random same-size
    subsamples.
clustered_interior.parquet
    Clustered populations with every row that touches a prior bound removed,
    against 5 shuffled copies and 5 Gaussian-copula surrogates.
attribution.json
    Variance partition and leave-one-source-model-out prediction of the synergy
    part for all 166 populations, without the clustered populations, and for the
    phenotype-selected populations only; predictor sets fixed in advance and the
    post hoc sampling-only variant labelled as such; bootstrap over source models
    that skips draws containing fewer than two distinct original source models.
run_record.json
    Features, transforms, seeds and sizes used.
"""
from __future__ import annotations

import importlib.util as iu
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "analyses" / "review_v18"
SEED = 20260804
N_REF = 100
PRIMARY = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje",
           "migliore_ca1", "bbp_l5pc"]
RANDOM = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje"]
TOLS = ["permissive", "moderate", "strict", "very_strict"]


def _ra():
    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(x, ra):
    from compass.analysis.geometry import describe_geometry
    from compass.analysis.tuples import order_curve
    oc = order_curve(x, orders=ra.ORDERS, max_tuples=ra.MAX_TUPLES, seed=SEED,
                     copula=False, bias_correct=True, n_surrogate=0)
    syn, red = oc.normalised_areas()
    return {"frac_synergy_o3": oc.frac_synergy[0], "synergy_area": syn,
            "net_o_information": red - syn,
            "mean_abs_pearson": describe_geometry(x).as_dict()["mean_abs_pearson"]}


def setup(experiment, mid, family):
    """Exactly the rows, features and target that build_ensembles uses; strict inputs."""
    from compass.adapters.models import get_adapter
    from compass.analysis.cohorts import UNIDENTIFIABLE_PARAMETERS
    from compass.conditioning.ensembles import (
        CONDITIONING_FEATURES,
        CONDITIONING_FEATURES_NORMALISED,
        _log_params,
        build_ensembles,
        validity_mask,
    )
    ra = _ra()
    df = ra.load_model_table(experiment, mid)
    if df is None:
        raise SystemExit(f"{experiment}/{mid}: no simulations")
    feats = CONDITIONING_FEATURES if family == "absolute" else CONDITIONING_FEATURES_NORMALISED
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"{experiment}/{mid}: feature columns missing {missing}")
    target = ra.canonical_target(mid, feats)
    if target is None:
        raise SystemExit(f"{mid}: canonical target missing; refusing the median fallback")
    drop = set(UNIDENTIFIABLE_PARAMETERS.get(mid, []))
    pids = [s.parameter_id for s in get_adapter(mid).parameter_specs() if s.parameter_id not in drop]
    ens = build_ensembles(df, mid, pids, feats, target=target, seed=SEED,
                          protocol_family=family, qc_policy="primary")
    base = df[validity_mask(df, "primary")].reset_index(drop=True)
    e1 = (base["sampling_regime"] == "E1").to_numpy()
    f1 = base[feats].to_numpy(float)[e1]
    x1 = _log_params(base, pids)[e1]
    return ra, ens, pids, feats, target, f1, x1


# --------------------------------------------------------------------------- 1
def two_stage(task):
    experiment, mid, family = task
    ra, ens, pids, feats, target, f1, x1 = setup(experiment, mid, family)
    complete = np.isfinite(f1).all(axis=1)
    pool = x1[complete]
    rows = []
    prow = {"experiment": experiment, "model_id": mid, "protocol_family": family,
            "n_unselected": len(x1), "n_feature_complete": int(complete.sum())}
    prow.update({f"pool_{k}": v for k, v in stats(pool, ra).items()})
    for e in ens:
        if e.regime not in ("E3", "E3b", "E4", "E4b"):
            continue
        base = {"experiment": experiment, "model_id": mid, "protocol_family": family,
                "metric": e.metric, "tolerance": e.tolerance, "n": e.n,
                "n_unselected": len(x1), "n_feature_complete": int(complete.sum())}
        rows.append(dict(base, kind="selected", draw=-1, **stats(e.params, ra)))
        for d in range(N_REF):
            rng = np.random.default_rng(SEED + 104729 * d + e.n)
            rows.append(dict(base, kind="unselected_same_size", draw=d,
                             **stats(x1[rng.choice(len(x1), e.n, replace=False)], ra)))
            if e.n < len(pool):
                rng = np.random.default_rng(72341 + 7919 * d + e.n)
                rows.append(dict(base, kind="feature_complete_same_size", draw=d,
                                 **stats(pool[rng.choice(len(pool), e.n, replace=False)], ra)))
        if e.n >= len(pool):
            rows.append(dict(base, kind="feature_complete_same_size", draw=0, **stats(pool, ra)))
    print("two_stage", experiment, mid, family, flush=True)
    return rows, [prow]


# --------------------------------------------------------------------------- 2
def dropout(mid):
    from compass.conditioning.distances import condition
    ra, ens, pids, feats, target, f1, x1 = setup("screening", mid, "absolute")
    disp = np.nanstd(f1, axis=0)
    disp = np.where(np.isfinite(disp) & (disp > 0), disp, np.nan)
    rows = []
    for metric in ("weighted_euclidean", "max_z"):
        for tol in ("permissive", "moderate"):
            full = condition(f1, target, disp, metric=metric, tolerance_name=tol)
            a = full.accepted
            if a.sum() < 60:
                continue
            complete13 = np.isfinite(f1).all(axis=1)
            for j, f in enumerate(feats):
                keep = [k for k in range(len(feats)) if k != j]
                sub = condition(f1[:, keep], target[keep], disp[keep], metric=metric, tolerance_name=tol)
                b = sub.accepted
                newly = b & ~complete13
                fixed = b & complete13
                row = {"model_id": mid, "metric": metric, "tolerance": tol, "dropped_feature": f,
                       "n_full": int(a.sum()), "n_dropped": int(b.sum()),
                       "frac_changed_total": float((a ^ b).sum() / a.sum()),
                       "frac_newly_eligible": float(newly.sum() / a.sum()),
                       "frac_changed_by_distance": float((a ^ fixed).sum() / a.sum()),
                       "dispersion_defined": bool(np.isfinite(disp[j]))}
                s_full = stats(x1[a], ra)["synergy_area"]
                row["synergy_full"] = s_full
                row["synergy_dropped"] = stats(x1[b], ra)["synergy_area"] if b.sum() >= 60 else np.nan
                row["synergy_fixed_complete_case"] = stats(x1[fixed], ra)["synergy_area"] if fixed.sum() >= 60 else np.nan
                rows.append(row)
    print("dropout", mid, flush=True)
    return rows


# --------------------------------------------------------------------------- 3
def ksg_mi(x, y, k=4, rng=None):
    """KSG estimator 1 (Kraskov et al. 2004), maximum norm, tiny tie-breaking jitter."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    rng = rng or np.random.default_rng(0)
    x = x + rng.normal(0, 1e-10, len(x))
    y = y + rng.normal(0, 1e-10, len(y))
    xy = np.column_stack([x, y])
    eps = cKDTree(xy).query(xy, k=k + 1, p=np.inf)[0][:, -1]
    tx, ty = cKDTree(x[:, None]), cKDTree(y[:, None])
    nx = tx.query_ball_point(x[:, None], eps - 1e-15, p=np.inf, return_length=True) - 1
    ny = ty.query_ball_point(y[:, None], eps - 1e-15, p=np.inf, return_length=True) - 1
    n = len(x)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def pairs_random(mid):
    from compass.analysis.entropy import copula_transform
    from compass.analysis.oinfo import mutual_information, o_information
    from compass.analysis.tuples import generate_tuples
    from compass.sampling.priors import gaussian_copula_control
    ra, ens, pids, feats, target, f1, x1 = setup("screening_random", mid, "absolute")
    pairs = list(itertools.combinations(range(len(pids)), 2))
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=SEED).tuples

    def knn_pairs(y):
        z = copula_transform(y)
        return float(np.mean([mutual_information(z[:, a], z[:, b], "knn") for a, b in pairs]))

    def excess(y, seed0, n_sur=5):
        s = [knn_pairs(gaussian_copula_control(y, seed=seed0 + i)) for i in range(n_sur)]
        return knn_pairs(y) - float(np.mean(s))

    cal, common, ksg = [], [], []
    for e in [e for e in ens if e.regime in ("E3", "E3b", "E4", "E4b") and e.n >= 2000]:
        base = {"model_id": mid, "metric": e.metric, "tolerance": e.tolerance, "n": e.n}
        cal.append(dict(base, kind="selected", draw=-1, excess=excess(e.params, SEED + 1000)))
        zs = copula_transform(e.params)
        g_sel = np.mean([o_information(zs[:, list(t)], "gaussian") for t in ts])
        gp_sel = np.mean([mutual_information(zs[:, a], zs[:, b], "gaussian") for a, b in pairs])
        for d in range(10):
            rng = np.random.default_rng(SEED + 104729 * d + e.n)
            u = x1[rng.choice(len(x1), e.n, replace=False)]
            cal.append(dict(base, kind="unselected_same_size", draw=d, excess=excess(u, SEED + 5000 + 10 * d)))
            zu = copula_transform(u)
            common.append(dict(base, draw=d,
                               gauss_triplet_shift_normal_scores=g_sel - np.mean([o_information(zu[:, list(t)], "gaussian") for t in ts]),
                               gauss_pair_mi_selected_normal_scores=gp_sel))
        if e.metric == "weighted_euclidean" and e.tolerance == "permissive":
            rng = np.random.default_rng(SEED + 77)
            sel = np.mean([ksg_mi(zs[:, a], zs[:, b], rng=rng) for a, b in pairs])
            sur = []
            for i in range(39):
                zc = copula_transform(gaussian_copula_control(e.params, seed=SEED + 9000 + i))
                sur.append(np.mean([ksg_mi(zc[:, a], zc[:, b], rng=rng) for a, b in pairs]))
            ksg.append(dict(base, ksg_selected=sel, ksg_surrogate_mean=float(np.mean(sur)),
                            ksg_surrogates_below=int(np.sum(np.array(sur) < sel)), n_surrogates=39))
    print("pairs", mid, flush=True)
    return cal, common, ksg


# --------------------------------------------------------------------------- 4
def sobol_box(mid):
    from compass.analysis.entropy import copula_transform
    from compass.analysis.oinfo import o_information
    from compass.analysis.tuples import generate_tuples
    ra, ens, pids, feats, target, f1, x1 = setup("screening", mid, "absolute")
    e = [e for e in ens if e.regime == "E3" and e.metric == "weighted_euclidean"][0]
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=SEED).tuples
    u = (np.argsort(np.argsort(x1, 0), 0) + 0.5) / len(x1)
    rng = np.random.default_rng(11)
    copies = [("selected", 0, e.params)]
    for b in range(5):
        flip = rng.random(x1.shape[1]) < 0.5
        m = np.where(flip, 1 - u, u).max(1)
        copies.append(("box_same_size", b, x1[np.argsort(m)[:e.n]]))
    for d in range(10):
        r2 = np.random.default_rng(SEED + 104729 * d + e.n)
        copies.append(("unselected_same_size", d, x1[r2.choice(len(x1), e.n, replace=False)]))
    rows = []
    for kind, d, x in copies:
        z = copula_transform(x)
        for t in ts:
            rows.append({"model_id": mid, "n": e.n, "kind": kind, "draw": d, "tuple": "-".join(map(str, t)),
                         "omega_knn": o_information(z[:, list(t)], "knn")})
    print("box", mid, flush=True)
    return rows


def clustered_interior(mid):
    from compass.analysis.entropy import copula_transform
    from compass.analysis.oinfo import o_information
    from compass.analysis.tuples import generate_tuples
    from compass.sampling.priors import gaussian_copula_control
    ra, ens, pids, feats, target, f1, x1 = setup("screening", mid, "absolute")
    x = [e for e in ens if e.regime == "E6"][0].params
    # jitter is clipped to the prior bounds, which piles values onto each
    # column's extremes; drop every row that sits on a column extreme
    touch = (np.isclose(x, x.min(axis=0), atol=1e-9) | np.isclose(x, x.max(axis=0), atol=1e-9)).any(axis=1)
    ties = float(np.mean([1 - len(np.unique(c)) / len(c) for c in x.T]))
    xi = x[~touch]
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=SEED).tuples
    rng = np.random.default_rng(SEED + 31)
    refs = [("interior", 0, xi)] + [("shuffled", d, np.column_stack([rng.permutation(c) for c in xi.T])) for d in range(5)] \
        + [("copula_surrogate", d, gaussian_copula_control(xi, seed=SEED + 600 + d)) for d in range(5)]
    rows = []
    for kind, d, y in refs:
        z = copula_transform(y)
        rows.append({"model_id": mid, "n_all": len(x), "n_interior": len(xi), "mean_tied_fraction": ties,
                     "kind": kind, "draw": d, "omega_knn_mean": float(np.mean([o_information(z[:, list(t)], "knn") for t in ts]))})
    print("clustered", mid, flush=True)
    return rows


# --------------------------------------------------------------------------- 5
def attribution():
    spec = iu.spec_from_file_location("rv", ROOT / "tools" / "run_review_regression_variants.py")
    rv = iu.module_from_spec(spec)
    spec.loader.exec_module(rv)
    from compass.analysis.hierarchical import (
        PREDICTOR_BLOCKS,
        leave_one_lineage_out,
        variance_partition,
    )
    g = rv.load()
    SEL = ["E3", "E3b", "E4", "E4b"]
    fits = {"all_populations": g, "without_clustered": g[g.regime != "E6"],
            "phenotype_selected_only": g[g.regime.isin(SEL)]}
    blocks = {"sampling": PREDICTOR_BLOCKS["sampling"], "geometry": rv.NONCOV, "lineage": ["source_lineage_id"]}
    design_only = {"design": ["regime", "tolerance"], "achieved": ["log_n_samples", "acceptance_rate"],
                   "geometry": rv.NONCOV, "lineage": ["source_lineage_id"]}

    def boot(d0, bl, n_boot=2000):
        rng = np.random.default_rng(SEED)
        L = d0.source_lineage_id.unique()
        us = []
        skipped = 0
        for _ in range(n_boot):
            pick = rng.choice(L, len(L), replace=True)
            if len(set(pick)) < 2:          # check ORIGINAL lineages, before relabelling
                skipped += 1
                continue
            parts = []
            for k, lin in enumerate(pick):
                s = d0[d0.source_lineage_id == lin].copy()
                s["source_lineage_id"] = f"{lin}__{k}"
                s["model_id"] = s["model_id"] + f"__{k}"
                parts.append(s)
            try:
                v = variance_partition(pd.concat(parts, ignore_index=True), "synergy_area", bl).set_index("block")["r2_unique"]
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            us.append(v.reindex(list(bl)).to_numpy())
        us = np.array(us)
        return ({b: [float(np.nanpercentile(us[:, i], 2.5)), float(np.nanpercentile(us[:, i], 97.5))]
                 for i, b in enumerate(bl)}, int(len(us)), skipped)

    out = {}
    for name, d0 in fits.items():
        vp = variance_partition(d0, "synergy_area", blocks).set_index("block")
        ci, n_ok, skipped = boot(d0, blocks)
        lo = leave_one_lineage_out(d0, "synergy_area", {
            "sampling+intrinsic_dimension (fixed in advance)": blocks["sampling"] + blocks["geometry"],
            "sampling_only (post hoc)": blocks["sampling"]})
        vd = variance_partition(d0, "synergy_area", design_only).set_index("block")
        out[name] = {"rows": int(len(d0)), "regimes": d0.regime.value_counts().to_dict(),
                     "alone": vp["r2_alone"].round(4).to_dict(), "unique": vp["r2_unique"].round(4).to_dict(),
                     "full": float(vp["r2_full_model"].iloc[0]), "unique_ci": ci,
                     "bootstrap_fits": n_ok, "bootstrap_skipped": skipped,
                     "lolo": {p: s.set_index("held_out_group")["r2_out_of_sample"].round(3).to_dict()
                              for p, s in lo.groupby("predictors")},
                     "design_vs_achieved_unique": vd["r2_unique"].round(4).to_dict()}
        print("attribution", name, out[name]["unique"], flush=True)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [("screening", m, f) for m in PRIMARY for f in ("absolute", "normalised")] + \
            [("screening_random", m, "absolute") for m in RANDOM]
    ts_rows, pool_rows, drop_rows, cal, common, ksg, box, clus = [], [], [], [], [], [], [], []
    with ProcessPoolExecutor(12) as ex:
        f_ts = [ex.submit(two_stage, t) for t in tasks]
        f_dr = [ex.submit(dropout, m) for m in PRIMARY]
        f_pr = [ex.submit(pairs_random, m) for m in RANDOM]
        f_bx = [ex.submit(sobol_box, m) for m in RANDOM]
        f_cl = [ex.submit(clustered_interior, m) for m in RANDOM]
        for f in f_ts:
            r, p = f.result()
            ts_rows += r
            pool_rows += p
        for f in f_dr:
            drop_rows += f.result()
        for f in f_pr:
            a, b, c = f.result()
            cal += a
            common += b
            ksg += c
        for f in f_bx:
            box += f.result()
        for f in f_cl:
            clus += f.result()
    for name, rows in [("two_stage", ts_rows), ("pools", pool_rows), ("dropout_decomposed", drop_rows),
                       ("pairs_calibrated", cal), ("gauss_common_scale", common), ("ksg_pairs", ksg),
                       ("sobol_box", box), ("clustered_interior", clus)]:
        pd.DataFrame(rows).to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / "attribution.json").write_text(json.dumps(attribution(), indent=1))
    from compass.conditioning.ensembles import (
        CONDITIONING_FEATURES,
        CONDITIONING_FEATURES_NORMALISED,
    )
    (OUT / "run_record.json").write_text(json.dumps({
        "seed": SEED, "n_reference_samples": N_REF, "feature_complete_seed_base": 72341,
        "features_absolute": CONDITIONING_FEATURES, "features_normalised": CONDITIONING_FEATURES_NORMALISED,
        "parameter_scale": "log10 of the multiplicative factor (Gaussian estimates)",
        "nearest_neighbour_scale": "normal scores by rank; fourth neighbour; maximum norm",
        "ksg": "KSG estimator 1, k=4, maximum norm, normal scores, jitter 1e-10, 39 surrogates",
        "qc_policy": "primary", "experiments": {"screening": PRIMARY, "screening_random": RANDOM}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
