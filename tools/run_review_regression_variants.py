#!/usr/bin/env python3
"""Fig. 5C-D without the covariance-based geometry predictors.

Review of manuscript V17: the Gaussian synergy part is a function of the
correlation matrix, and four of the six "population geometry" predictors (mean
|r|, effective rank, participation ratio, share of variance on the first
principal component) are functions of the same covariance, so part of the
geometry R^2 is mechanical. This repeats the variance partition (with the same
bootstrap over source lineages as manuscript/variance_partition_bootstrap.py)
and the leave-one-source-lineage-out prediction with

* the published blocks,
* geometry restricted to the two intrinsic-dimension estimates, which are
  nearest-neighbour based and not functions of the covariance,
* no geometry block at all.

Output: results/analyses/review_v17/regression_variants.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compass.analysis.hierarchical import (  # noqa: E402
    PREDICTOR_BLOCKS,
    leave_one_lineage_out,
    variance_partition,
)

spec = importlib.util.spec_from_file_location("rc", ROOT / "tools" / "run_crossmodel.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

OUT = ROOT / "results" / "analyses" / "review_v17"
NONCOV = ["intrinsic_dimension_norm", "intrinsic_dimension_mle_norm"]
VARIANTS = {
    "published": {"sampling": PREDICTOR_BLOCKS["sampling"], "geometry": PREDICTOR_BLOCKS["geometry"],
                  "lineage": ["source_lineage_id"]},
    "geometry_without_covariance": {"sampling": PREDICTOR_BLOCKS["sampling"], "geometry": NONCOV,
                                    "lineage": ["source_lineage_id"]},
    "no_geometry": {"sampling": PREDICTOR_BLOCKS["sampling"], "lineage": ["source_lineage_id"]},
}


def load():
    arm = "discovery_only"
    sp = rc.SENSITIVITY_ARMS[arm]
    src = sp.get("source_experiment", "screening")
    ep = pd.read_parquet(ROOT / f"results/analyses/endpoints_annotated_{src}_n4proj.parquet")
    ep = ep[rc.primary_mask(ep, arm)].reset_index(drop=True)
    fit = rc.drop_duplicate_scientific_rows(ep)
    return fit[fit.estimator == "gaussian"].reset_index(drop=True)


def boot(g, blocks, endpoint, n_boot=2000):
    rng = np.random.default_rng(20260804)
    lineages = g.source_lineage_id.unique()
    us = []
    for _ in range(n_boot):
        pick = rng.choice(lineages, len(lineages), replace=True)
        parts = []
        for k, lin in enumerate(pick):
            s = g[g.source_lineage_id == lin].copy()
            s["source_lineage_id"] = f"{lin}__{k}"
            s["model_id"] = s["model_id"] + f"__{k}"
            parts.append(s)
        d = pd.concat(parts, ignore_index=True)
        if d.source_lineage_id.nunique() < 2:
            continue
        try:
            v = variance_partition(d, endpoint, blocks).set_index("block")["r2_unique"]
        except Exception:  # noqa: BLE001
            continue
        us.append(v.reindex(list(blocks)).to_numpy())
    us = np.array(us)
    return {b: [float(np.nanpercentile(us[:, i], 2.5)), float(np.nanpercentile(us[:, i], 97.5))]
            for i, b in enumerate(blocks)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load()
    out = {"n_rows": len(g), "n_lineages": int(g.source_lineage_id.nunique()), "variants": {}}
    for endpoint in ("synergy_area", "net_o_information"):
        for name, blocks in VARIANTS.items():
            vp = variance_partition(g, endpoint, blocks).set_index("block")
            pred_sets = {"sampling+geometry": blocks["sampling"] + blocks.get("geometry", []),
                         "sampling_only": blocks["sampling"]}
            lo = leave_one_lineage_out(g, endpoint, pred_sets)
            rec = {"alone": vp["r2_alone"].round(4).to_dict(), "unique": vp["r2_unique"].round(4).to_dict(),
                   "full": float(vp["r2_full_model"].iloc[0]),
                   "unique_ci": boot(g, blocks, endpoint) if endpoint == "synergy_area" else None,
                   "lolo": {p: s.set_index("held_out_group")["r2_out_of_sample"].round(3).to_dict()
                            for p, s in lo.groupby("predictors")}}
            out["variants"][f"{endpoint}:{name}"] = rec
            print(endpoint, name, json.dumps({k: rec[k] for k in ("alone", "unique", "full")}), flush=True)
    (OUT / "regression_variants.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
