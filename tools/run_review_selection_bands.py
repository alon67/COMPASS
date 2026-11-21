#!/usr/bin/env python3
"""Estimator test systems that look like selection on positive parameters.

Review of manuscript V17. The product systems of Fig. 2 multiply zero-mean normal
variables, so they have no pairwise correlation at all; that, not the
multiplication, is why the Gaussian and copula estimators return zero. Model
parameters are positive factors analysed as logarithms, where a product is a
sum. This script adds two kinds of test system:

* selection bands: three parameters drawn log-uniformly over one quarter to four
  times a reference value, and only the draws whose score lies in a narrow band
  around its median are kept (as a tolerance band around a target feature
  does).  Scores: none (independent control), the sum of the log-parameters (a
  multiplicative constraint), the raw sum (additive), the sum of squared
  log-parameters (a shell: non-monotone) and x1 + |x2 - x3| (non-monotone in
  two parameters).  Analysed on the log scale.
* positive products: four variables, the fourth equal to the product of the
  first three (small multiplicative noise), the first three log-uniform as
  above, analysed on the raw and on the log scale.

Each system is sampled at 300, 1,000, 4,000 and 16,000 points with five seeds,
and analysed with the Gaussian, Gaussian-copula and nearest-neighbour
estimators exactly as in Fig. 2 (nearest-neighbour on the analysed values,
fourth neighbour, maximum-coordinate distance), plus the nearest-neighbour value
minus the mean of five shuffled copies.

Output: results/analyses/review_v17/selection_bands.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compass.analysis.oinfo import o_information  # noqa: E402

OUT = ROOT / "results" / "analyses" / "review_v17"
LO, HI = np.log(0.25), np.log(4.0)
KEEP = 0.10            # fraction of the pool kept by a band
SIZES = (300, 1000, 4000, 16000)
SEEDS = range(5)

BANDS = {
    "independent (no selection)": None,
    "band on the sum of log-parameters": lambda x: x.sum(1),
    "band on the raw sum": lambda x: np.exp(x).sum(1),
    "band on the sum of squared log-parameters": lambda x: (x ** 2).sum(1),
    "band on x1 + |x2 - x3|": lambda x: x[:, 0] + np.abs(x[:, 1] - x[:, 2]),
}


def band_system(score, n, rng):
    if score is None:
        return rng.uniform(LO, HI, (n, 3))
    pool = rng.uniform(LO, HI, (int(n / KEEP), 3))
    s = score(pool)
    keep = np.argsort(np.abs(s - np.median(s)))[:n]
    return pool[keep]


def product_system(n, rng, log_scale):
    x = np.exp(rng.uniform(LO, HI, (n, 3)))
    y = x.prod(1) * np.exp(0.05 * rng.normal(size=n))
    z = np.column_stack([x, y])
    return np.log(z) if log_scale else z


def classify(v):
    return "redundancy" if v > 0.02 else "synergy" if v < -0.02 else "none"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    systems = [(name, (lambda n, rng, f=f: band_system(f, n, rng)),
                "none" if f is None else "synergy") for name, f in BANDS.items()]
    systems += [("product of three positive variables, raw scale",
                 lambda n, rng: product_system(n, rng, False), "synergy"),
                ("product of three positive variables, log scale",
                 lambda n, rng: product_system(n, rng, True), "synergy")]
    for name, make, expected in systems:
        for n in SIZES:
            for s in SEEDS:
                rng = np.random.default_rng(20260804 + 1000 * s + n)
                x = make(n, rng)
                vals = {est: o_information(x, est) for est in ("gaussian", "copula", "knn")}
                sh = [o_information(np.column_stack([rng.permutation(c) for c in x.T]), "knn")
                      for _ in range(5)]
                vals["knn_minus_shuffled"] = vals["knn"] - float(np.mean(sh))
                r = np.corrcoef(x.T)[np.triu_indices(x.shape[1], 1)]
                for est, v in vals.items():
                    rows.append({"system": name, "expected": expected, "n": n, "seed": s,
                                 "estimator": est, "o_information": float(v),
                                 "classified": classify(v), "correct": classify(v) == expected,
                                 "max_abs_pearson": float(np.abs(r).max())})
            print(name, n, flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "selection_bands.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
