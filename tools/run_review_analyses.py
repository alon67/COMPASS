#!/usr/bin/env python3
"""Analyses requested by the four informal reviews of manuscript V4 (Nov 2026).

No simulation is run: every analysis re-uses the stored feature tables of the
main simulation run.  The primary panel is used (Mainen excluded, Hay with its
two inert parameters removed before analysis), Gaussian estimator, the same
group sizes, group sampling and seed as the main analysis.

Outputs (results/analyses/review_v5/):

size_matched_null.parquet
    For every selected population: its own statistics, and the same statistics
    in 100 random subsamples of the unselected population of exactly the same
    size (the size-matched unselected comparison asked for by reviewer 4).
feature_dropout.parquet
    Every selected population at 3 and 2 SD rebuilt with each of the 13
    features left out of the acceptance rule in turn (which features drive the
    effect; and a bound on the rheobase-definition concern).
knn_check.parquet
    Nearest-neighbour O-information of every parameter triplet in the five
    largest selected populations, with size-matched unselected and shuffled
    comparisons (does a non-Gaussian estimator change the conclusion).
top_triplets.parquet
    Overlap of the ten most synergistic triplets between tolerances.
"""

from __future__ import annotations

import importlib.util as iu
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "analyses" / "review_v5"
PANEL = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs",
         "khaliq_purkinje", "migliore_ca1", "bbp_l5pc"]
N_NULL = 100
N_KNN_NULL = 10
KNN_MIN_N = 2000


def _ra():
    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(x: np.ndarray, seed: int, ra) -> dict:
    from compass.analysis.geometry import describe_geometry
    from compass.analysis.tuples import order_curve

    oc = order_curve(x, orders=ra.ORDERS, max_tuples=ra.MAX_TUPLES, seed=seed,
                     copula=False, bias_correct=True, n_surrogate=0)
    syn, red = oc.normalised_areas()
    geo = describe_geometry(x).as_dict()
    return {"frac_synergy_o3": oc.frac_synergy[0],
            "frac_synergy_o345": float(np.mean(oc.frac_synergy)),
            "synergy_area": syn, "redundancy_area": red,
            "net_o_information": red - syn,
            "mean_abs_pearson": geo["mean_abs_pearson"]}


def model_data(mid: str, ra):
    from compass.adapters.models import get_adapter
    from compass.analysis.cohorts import UNIDENTIFIABLE_PARAMETERS
    from compass.conditioning.ensembles import validity_mask

    df = ra.load_model_table("screening", mid)
    specs = get_adapter(mid).parameter_specs()
    drop = set(UNIDENTIFIABLE_PARAMETERS.get(mid, []))
    pids = [s.parameter_id for s in specs if s.parameter_id not in drop]
    return df, pids, validity_mask


def triplet_omegas(x, tuples, estimator):
    from compass.analysis.oinfo import o_information
    return np.array([o_information(x[:, list(t)], estimator) for t in tuples])


def run_model(mid: str) -> dict[str, list[dict]]:
    from compass.analysis.oinfo import GaussianInfoEngine
    from compass.analysis.tuples import generate_tuples
    from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles

    ra = _ra()
    df, pids, _ = model_data(mid, ra)
    seed = ra.SEED
    res = {"null": [], "drop": [], "knn": [], "top": []}

    for family, feats in (("absolute", CONDITIONING_FEATURES),):
        feats = [f for f in feats if f in df.columns]
        target = ra.canonical_target(mid, feats)
        ens = build_ensembles(df, mid, pids, feats, target=target, seed=seed,
                              protocol_family=family, qc_policy="primary")
        e1 = [e for e in ens if e.regime == "E1"][0].params
        sel = [e for e in ens if e.regime in ("E3", "E3b", "E4", "E4b")]

        # ---- A: size-matched unselected comparison --------------------------
        for e in sel:
            n = e.n
            row = {"model_id": mid, "protocol_family": family, "metric": e.metric,
                   "tolerance": e.tolerance, "regime": e.regime, "n": n,
                   "n_unselected": len(e1), "kind": "selected", "draw": -1}
            row.update(stats(e.params, seed, ra))
            res["null"].append(row)
            for d in range(N_NULL):
                rng = np.random.default_rng(seed + 104729 * d + n)
                idx = rng.choice(len(e1), size=n, replace=False)
                r = dict(row, kind="unselected_same_size", draw=d)
                r.update(stats(e1[idx], seed, ra))
                res["null"].append(r)
            # shuffled copies of the selected population at the same size
            for d in range(20):
                rng = np.random.default_rng(seed + 7 * d + 3)
                xs = np.column_stack([rng.permutation(c) for c in e.params.T])
                r = dict(row, kind="shuffled_same_size", draw=d)
                r.update(stats(xs, seed, ra))
                res["null"].append(r)

        # ---- D: stability of the ten most synergistic triplets --------------
        for metric in ("weighted_euclidean", "max_z"):
            tops = {}
            for e in sel:
                if e.metric != metric:
                    continue
                eng = GaussianInfoEngine(e.params, copula=False, bias_correct=True)
                ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=seed)
                v = eng.o_information_batch(ts.tuples)
                order = np.argsort(v)[:10]
                tops[e.tolerance] = {tuple(ts.tuples[i]) for i in order if v[i] < 0}
            base = tops.get("permissive", set())
            for tol, s in tops.items():
                res["top"].append({"model_id": mid, "metric": metric, "tolerance": tol,
                                   "n_top": len(s), "overlap_with_3sd": len(s & base),
                                   "jaccard_with_3sd": (len(s & base) / len(s | base))
                                   if (s | base) else np.nan})

        # ---- B: leave one feature out of the acceptance rule ----------------
        full = {(e.metric, e.tolerance): e for e in sel}
        for f in feats:
            sub = [g for g in feats if g != f]
            tgt = ra.canonical_target(mid, sub)
            if tgt is None:
                continue
            ens_f = build_ensembles(df, mid, pids, sub, target=tgt, seed=seed,
                                    protocol_family=family, qc_policy="primary")
            for e in ens_f:
                if e.regime not in ("E3", "E3b") or (e.metric, e.tolerance) not in full:
                    continue
                ref = full[(e.metric, e.tolerance)]
                a = ref.conditioning.accepted
                b = e.conditioning.accepted
                row = {"model_id": mid, "metric": e.metric, "tolerance": e.tolerance,
                       "dropped_feature": f, "n": e.n, "n_full": ref.n,
                       "jaccard_accepted": float((a & b).sum() / max((a | b).sum(), 1)),
                       "frac_changed": float((a ^ b).sum() / max(a.sum(), 1))}
                row.update(stats(e.params, seed, ra))
                res["drop"].append(row)

        # ---- C: nearest-neighbour estimator on large selected populations ---
        for e in sel:
            if e.metric != "weighted_euclidean" or e.tolerance != "permissive":
                continue
            if e.n < KNN_MIN_N:
                continue
            ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=seed).tuples
            copies = [("selected", -1, e.params)]
            for d in range(N_KNN_NULL):
                rng = np.random.default_rng(seed + 104729 * d + e.n)
                copies.append(("unselected_same_size", d,
                               e1[rng.choice(len(e1), size=e.n, replace=False)]))
                rng = np.random.default_rng(seed + 7 * d + 3)
                copies.append(("shuffled_same_size", d,
                               np.column_stack([rng.permutation(c) for c in e.params.T])))
            for kind, d, x in copies:
                vk = triplet_omegas(x, ts, "knn")
                vg = triplet_omegas(x, ts, "gaussian")
                for t, a, g in zip(ts, vk, vg, strict=True):
                    res["knn"].append({"model_id": mid, "kind": kind, "draw": d,
                                       "n": e.n, "tuple": "-".join(map(str, t)),
                                       "omega_knn": a, "omega_gaussian": g})
    print(f"done {mid}", flush=True)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    allres = {"null": [], "drop": [], "knn": [], "top": []}
    with ProcessPoolExecutor(max_workers=len(PANEL)) as ex:
        for r in ex.map(run_model, PANEL):
            for k in allres:
                allres[k].extend(r[k])
    names = {"null": "size_matched_null", "drop": "feature_dropout",
             "knn": "knn_check", "top": "top_triplets"}
    for k, rows in allres.items():
        pd.DataFrame(rows).to_parquet(OUT / f"{names[k]}.parquet", index=False)
        print(k, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
