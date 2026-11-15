#!/usr/bin/env python3
"""Nearest-neighbour O-information after converting every parameter to normal scores
by rank, so the estimate no longer depends on the shape of each parameter's
distribution (only on the dependence between parameters).  Same populations,
triplets and comparison samples as knn_check in run_review_analyses.py."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_review_analyses as R  # noqa: E402

from compass.analysis.entropy import copula_transform  # noqa: E402

MODELS = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje"]


def run(mid):
    from compass.analysis.tuples import generate_tuples
    from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles
    ra = R._ra()
    df, pids, _ = R.model_data(mid, ra)
    seed = ra.SEED
    feats = [f for f in CONDITIONING_FEATURES if f in df.columns]
    ens = build_ensembles(df, mid, pids, feats, target=ra.canonical_target(mid, feats), seed=seed,
                          protocol_family="absolute", qc_policy="primary")
    e1 = [e for e in ens if e.regime == "E1"][0].params
    e = [e for e in ens if e.regime == "E3" and e.metric == "weighted_euclidean"][0]
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=seed).tuples
    copies = [("selected", -1, e.params)]
    for d in range(R.N_KNN_NULL):
        rng = np.random.default_rng(seed + 104729 * d + e.n)
        copies.append(("unselected_same_size", d, e1[rng.choice(len(e1), size=e.n, replace=False)]))
        rng = np.random.default_rng(seed + 7 * d + 3)
        copies.append(("shuffled_same_size", d, np.column_stack([rng.permutation(c) for c in e.params.T])))
    rows = []
    for kind, d, x in copies:
        z = copula_transform(x)
        v = R.triplet_omegas(z, ts, "knn")
        for t, a in zip(ts, v, strict=True):
            rows.append({"model_id": mid, "kind": kind, "draw": d, "n": e.n,
                         "tuple": "-".join(map(str, t)), "omega_knn_ranked": a})
    print("done", mid, flush=True)
    return rows


if __name__ == "__main__":
    out = []
    with ProcessPoolExecutor(5) as ex:
        for r in ex.map(run, MODELS):
            out += r
    pd.DataFrame(out).to_parquet(R.OUT / "knn_ranked_check.parquet", index=False)
