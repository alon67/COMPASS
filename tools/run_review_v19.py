#!/usr/bin/env python3
"""Analyses for the review of manuscript V20 (stored tables and simulations only).

Outputs (results/analyses/review_v19/):

estimator_agreement.json
    Correlation between Gaussian and Gaussian-copula triplet estimates in the
    primary panel, (a) as previously counted, (b) without the two Khaliq
    selections of the undefined rheobase-normalised family, and (c) also
    counting unselected populations stored under both protocol labels once (the
    convention of the attribution analysis); plus the correlation between the
    populations' shares of synergy-dominated triplets.
lolo_two_baselines.json
    Leave-one-source-model-out prediction of the synergy part for the three sets
    of populations and three predictor sets, scored against two baselines: the
    training-model mean (the published score) and the held-out model's own mean.
top_triplet_thinning.parquet
    Overlap between the (up to) ten most synergistic triplets at 3 SD and at each
    tighter tolerance, compared with 200 random subsets of the 3-SD population of
    the same size as the tighter population (fixed-amplitude protocols, RMS
    criterion, primary panel).
"""
from __future__ import annotations

import importlib.util as iu
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "analyses" / "review_v19"
SEED = 20260804
PRIMARY = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje",
           "migliore_ca1", "bbp_l5pc"]
N_THIN = 200


def _load(name):
    spec = iu.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def estimator_agreement():
    from compass.analysis.cohorts import drop_duplicate_scientific_rows
    ra = _load("run_analysis")
    t = pd.read_parquet(ROOT / "results" / "analyses" / "tuple_scores_screening_n4proj.parquet")
    t = t[t.model_id.isin(PRIMARY) & (t.tuple_idx.str.count("-") == 2)]
    key = ["model_id", "regime", "label", "protocol_family", "metric", "tuple_idx"]
    w = t.pivot_table(index=key, columns="estimator", values="o_information").dropna(subset=["gaussian", "copula"]).reset_index()

    def summary(d):
        share = d.groupby(key[:-1]).apply(lambda g: pd.Series({"g": (g.gaussian < 0).mean(), "c": (g.copula < 0).mean()}),
                                         include_groups=False)
        return {"n_triplet_estimates": int(len(d)), "n_populations": int(len(share)),
                "r_triplets": float(np.corrcoef(d.gaussian, d.copula)[0, 1]),
                "r_shares": float(np.corrcoef(share.g, share.c)[0, 1])}

    a = w
    b = ra.drop_undefined_family_rows(a)
    c = drop_duplicate_scientific_rows(b)
    return {"as_previously_counted": summary(a), "without_undefined_khaliq_family": summary(b),
            "unselected_counted_once": summary(c)}


def lolo_two_baselines():
    rv = _load("run_review_regression_variants")
    from compass.analysis.hierarchical import PREDICTOR_BLOCKS, leave_one_lineage_out
    g = rv.load()
    SEL = ["E3", "E3b", "E4", "E4b"]
    fits = {"all_populations": g, "without_clustered": g[g.regime != "E6"],
            "phenotype_selected_only": g[g.regime.isin(SEL)]}
    sets = {"sampling+geometry, preregistered geometry block": PREDICTOR_BLOCKS["sampling"] + PREDICTOR_BLOCKS["geometry"],
            "sampling+intrinsic_dimension (geometry revised after review)": PREDICTOR_BLOCKS["sampling"] + rv.NONCOV,
            "sampling_only (post hoc)": PREDICTOR_BLOCKS["sampling"]}
    out = {}
    for name, d0 in fits.items():
        lo = leave_one_lineage_out(d0, "synergy_area", sets)
        out[name] = {"rows": int(len(d0))}
        for p, s in lo.groupby("predictors"):
            s = s.set_index("held_out_group")
            out[name][p] = {"vs_training_mean": s["r2_out_of_sample"].round(3).to_dict(),
                            "vs_heldout_mean": s["r2_vs_heldout_mean"].round(3).to_dict(),
                            "n_positive_vs_training_mean": int((s["r2_out_of_sample"] > 0).sum()),
                            "n_positive_vs_heldout_mean": int((s["r2_vs_heldout_mean"] > 0).sum())}
    return out


def _tops(x, ts, eng_cls):
    v = eng_cls(x, copula=False, bias_correct=True).o_information_batch(ts)
    order = np.argsort(v)[:10]
    return {tuple(ts[i]) for i in order if v[i] < 0}


def thinning(mid):
    from compass.analysis.oinfo import GaussianInfoEngine
    from compass.analysis.tuples import generate_tuples
    rr = _load("run_review_v18")
    ra, ens, pids, feats, target, f1, x1 = rr.setup("screening", mid, "absolute")
    ts = generate_tuples(len(pids), 3, max_tuples=ra.MAX_TUPLES, seed=SEED).tuples
    pops = {e.tolerance: e.params for e in ens if e.regime in ("E3", "E3b", "E4", "E4b") and e.metric == "weighted_euclidean"}
    if "permissive" not in pops:
        return []
    base = pops["permissive"]
    top3 = _tops(base, ts, GaussianInfoEngine)
    rows = []
    for tol in ("moderate", "strict", "very_strict"):
        if tol not in pops:
            continue
        n = len(pops[tol])
        obs = len(_tops(pops[tol], ts, GaussianInfoEngine) & top3)
        rng = np.random.default_rng(SEED + n)
        thin = [len(_tops(base[rng.choice(len(base), n, replace=False)], ts, GaussianInfoEngine) & top3)
                for _ in range(N_THIN)]
        rows.append({"model_id": mid, "tolerance": tol, "n": n, "n_3sd": len(base),
                     "n_top_3sd": len(top3), "overlap_observed": obs,
                     "overlap_thinning_mean": float(np.mean(thin)),
                     "overlap_thinning_p05": float(np.percentile(thin, 5))})
    print("thinning", mid, flush=True)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "estimator_agreement.json").write_text(json.dumps(estimator_agreement(), indent=1))
    (OUT / "lolo_two_baselines.json").write_text(json.dumps(lolo_two_baselines(), indent=1))
    rows = []
    with ProcessPoolExecutor(7) as ex:
        for r in ex.map(thinning, PRIMARY):
            rows += r
    pd.DataFrame(rows).to_parquet(OUT / "top_triplet_thinning.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
