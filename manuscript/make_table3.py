#!/usr/bin/env python3
"""Supplementary Table S3 (main-text Table 3 up to manuscript V23): the effect of selection in each implementation.

Reads stored analysis tables only (no simulation):
results/analyses/review_v18/two_stage.parquet (main run) and
results/analyses/review_v22/hay_two_stage.parquet (Hay model, 8,192 random draws).

One condition throughout: populations selected with the root-mean-square
(weighted Euclidean) distance and the fixed-amplitude protocol set, at 3 SD
(permissive) and 0.5 SD (very strict).  For each population: the number of
accepted parameter sets; the synergy part, the redundancy part and the net
O-information (all in 1e-5 nats) of the selected population; the mean and the
2.5th-97.5th percentile range of 100 same-size unselected samples; the mean of
100 same-size feature-complete samples (synergy part only).  The redundancy part
is the net O-information plus the synergy part (net = redundancy - synergy).

    python manuscript/make_table3.py [OUT_CSV]
    default OUT_CSV: results/analyses/manuscript/table3_per_implementation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje", "migliore_ca1", "bbp_l5pc"]
TOLS = {"permissive": 3.0, "very_strict": 0.5}


def _rows(ts: pd.DataFrame, run: str, models) -> list[dict]:
    ts = ts[(ts.protocol_family == "absolute") & (ts.metric == "weighted_euclidean")]
    ts = ts.assign(redundancy_area=ts.net_o_information + ts.synergy_area)
    out = []
    for m in models:
        for tol, sd in TOLS.items():
            g = ts[(ts.model_id == m) & (ts.tolerance == tol)]
            if g.empty:
                out.append({"model_id": m, "run": run, "tolerance_sd": sd, "n_accepted": float("nan")})
                continue
            s = g[g.kind == "selected"].iloc[0]
            u = g[g.kind == "unselected_same_size"]
            f = g[g.kind == "feature_complete_same_size"]
            r = {"model_id": m, "run": run, "tolerance_sd": sd, "n_accepted": int(s.n)}
            for q, tag in (("synergy_area", "synergy"), ("redundancy_area", "redundancy"),
                           ("net_o_information", "net")):
                r[f"{tag}_selected"] = s[q] * 1e5
                r[f"{tag}_unselected_mean"] = u[q].mean() * 1e5
                r[f"{tag}_unselected_lo"] = u[q].quantile(0.025) * 1e5
                r[f"{tag}_unselected_hi"] = u[q].quantile(0.975) * 1e5
            r["synergy_feature_complete_mean"] = f.synergy_area.mean() * 1e5
            out.append(r)
    return out


def table3() -> pd.DataFrame:
    main = pd.read_parquet(ROOT / "results" / "analyses" / "review_v18" / "two_stage.parquet")
    main = main[main.experiment == "screening"]
    hay = pd.read_parquet(ROOT / "results" / "analyses" / "review_v22" / "hay_two_stage.parquet")
    return pd.DataFrame(_rows(main, "main (Sobol design)", ORDER) + _rows(hay, "random draws", ["bbp_l5pc"]))


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "analyses" / "manuscript" / "table3_per_implementation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = table3()
    t.to_csv(out, index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
