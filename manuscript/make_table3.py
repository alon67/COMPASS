#!/usr/bin/env python3
"""Table 3 of the main text: the effect of selection in each implementation.

Reads results/analyses/review_v18/two_stage.parquet only (no simulation).

Columns 2-7: populations selected with the root-mean-square (weighted Euclidean)
distance and the fixed-amplitude protocol set on the Sobol design, at 3 SD
(permissive) and 0.5 SD (very strict): accepted sizes, and the synergy part of the
net O-information (in 1e-5 nats) of the selected population and of the means of
100 same-size unselected and feature-complete samples.

Columns 8-9: selected minus the mean of the same-size unselected samples, in
millinats, averaged over tolerances, both distance criteria and both protocol sets
(the rheobase-normalised set is undefined for the Khaliq model and absent from the
stored table), exactly as in Fig. 4C.

    python manuscript/make_table3.py [OUT_CSV]
    default OUT_CSV: results/analyses/manuscript/table3_per_implementation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs", "khaliq_purkinje", "migliore_ca1", "bbp_l5pc"]
KEY = ["model_id", "protocol_family", "metric", "tolerance"]


def table3() -> pd.DataFrame:
    ts = pd.read_parquet(ROOT / "results" / "analyses" / "review_v18" / "two_stage.parquet")
    sc = ts[ts.experiment == "screening"]
    sel = sc[sc.kind == "selected"].set_index(KEY)
    uns = sc[sc.kind == "unselected_same_size"].groupby(KEY)[["synergy_area", "net_o_information"]].mean()
    d_syn = (sel.synergy_area - uns.synergy_area.reindex(sel.index)).groupby(level=0).mean() * 1e3
    d_net = (sel.net_o_information - uns.net_o_information.reindex(sel.index)).groupby(level=0).mean() * 1e3

    rms = sc[(sc.protocol_family == "absolute") & (sc.metric == "weighted_euclidean")]
    g = rms.groupby(["model_id", "tolerance", "kind"]).agg(n=("n", "first"), s=("synergy_area", "mean"))
    rows = []
    for m in ORDER:
        def val(tol, kind, col="s", m=m):
            k = (m, tol, kind)
            return g.loc[k, col] if k in g.index else float("nan")
        rows.append({
            "model_id": m,
            "n_accepted_3sd": val("permissive", "selected", "n"),
            "n_accepted_0p5sd": val("very_strict", "selected", "n"),
            "synergy_3sd_selected_1e-5": val("permissive", "selected") * 1e5,
            "synergy_3sd_unselected_1e-5": val("permissive", "unselected_same_size") * 1e5,
            "synergy_3sd_feature_complete_1e-5": val("permissive", "feature_complete_same_size") * 1e5,
            "synergy_0p5sd_selected_1e-5": val("very_strict", "selected") * 1e5,
            "synergy_0p5sd_feature_complete_1e-5": val("very_strict", "feature_complete_same_size") * 1e5,
            "change_synergy_part_millinats": d_syn[m],
            "change_net_o_information_millinats": d_net[m],
        })
    out = pd.DataFrame(rows)
    mean = out.drop(columns="model_id").mean(numeric_only=True)
    mean[["n_accepted_3sd", "n_accepted_0p5sd", "change_synergy_part_millinats",
          "change_net_o_information_millinats"]] = float("nan")
    return pd.concat([out, pd.DataFrame([{"model_id": "mean", **mean.to_dict()}])], ignore_index=True)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "analyses" / "manuscript" / "table3_per_implementation.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = table3()
    t.to_csv(out, index=False, float_format="%.6g")
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
