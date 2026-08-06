"""Extract the headline numbers used in docs/RESULTS.md.

Every number in the results document comes from here, so each claim can be
traced to a derived table (blueprint Task 15: "every claim links to a result
table and experiment manifest").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analyses"
EXP = "screening"


def _load(name, kind="parquet"):
    p = AN / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if kind == "parquet" else json.loads(p.read_text())


def main() -> int:
    out: dict = {}

    # NB: `a or b` on DataFrames raises (ambiguous truth value) -- test for None.
    ep = _load(f"endpoints_annotated_{EXP}.parquet")
    if ep is None:
        ep = _load(f"endpoints_{EXP}.parquet")
    cu = _load(f"order_curves_{EXP}.parquet")
    ts = _load(f"tuple_scores_{EXP}.parquet")
    rsi = _load(f"rsi_{EXP}.parquet")
    cond = _load("conditioning_shift.parquet")
    cov = _load("covariance_shift.parquet")
    vp = _load("variance_partition.parquet")
    lomo = _load("lomo.parquet")
    clf = _load("classification.parquet")
    stab = _load("module_stability.parquet")
    sob = _load("sobol_summary.parquet")
    sobi = _load("sobol_indices.parquet")
    pert = _load("perturbations.parquet")
    res = _load("readout_resolution.parquet")
    bench = _load("benchmarks.json", "json")
    dec = _load("decision.json", "json")
    mixed = _load("mixed_effects.json", "json")

    # ---- campaign scale ------------------------------------------------- #
    import glob
    tot, per_model = 0, {}
    for d in sorted((ROOT / "results" / "features" / EXP).glob("*")):
        fs = glob.glob(str(d / "*.parquet"))
        if not fs:
            continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        per_model[d.name] = {
            "n": len(df),
            "n_error": int((df["simulation_status"] == "error").sum()),
            "n_nonspiking": int(df["non_spiking"].sum()),
            "pct_nonspiking": round(100 * float(df["non_spiking"].mean()), 1),
            "regimes": sorted(df["sampling_regime"].unique().tolist()),
        }
        tot += len(df)
    sob_n = sum(len(pd.read_parquet(f)) for f in
                glob.glob(str(ROOT / "results" / "features" / "sobol" / "*" / "*.parquet")))
    out["campaign"] = {"total_simulated_samples": tot, "per_model": per_model,
                       "sobol_samples": sob_n,
                       "grand_total": tot + sob_n}

    if ep is not None:
        g = ep[(ep.estimator == "gaussian") & (ep.protocol_family == "absolute") &
               (ep.metric.isin(["none", "max_z"]))]
        out["regime_means"] = (g.groupby("regime")[["net_o_information", "synergy_area",
                                                    "frac_synergy_o3", "mean_abs_pearson",
                                                    "n_samples"]]
                               .mean().round(5).to_dict("index"))
        out["variability"] = {
            "sd_within_model_across_regimes": float(
                g.groupby("model_id")["net_o_information"].std().mean()),
            "sd_within_regime_across_models": float(
                g.groupby("regime")["net_o_information"].std().mean()),
        }
        out["variability"]["ratio"] = (out["variability"]["sd_within_model_across_regimes"] /
                                       out["variability"]["sd_within_regime_across_models"])
        e1 = g[g.regime == "E1"]
        out["e1_baseline"] = {
            "mean_abs_pearson": float(e1["mean_abs_pearson"].mean()),
            "net_o_information": float(e1["net_o_information"].mean()),
            "n_models": int(e1["model_id"].nunique()),
        }
        # geometry correlation with the endpoint
        d = g[["mean_abs_pearson", "net_o_information"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        out["geometry_correlation"] = {
            "r_meanabs_pearson_vs_net_oinfo": float(np.corrcoef(
                d["mean_abs_pearson"], d["net_o_information"])[0, 1]),
            "n": int(len(d)),
        }

    if cond is not None and len(cond):
        c = cond[(cond.estimator == "gaussian") & (cond.protocol_family == "absolute") &
                 (cond.metric == "max_z")]
        per = c.groupby("model_id")[["d_net_o_information", "d_mean_abs_pearson"]].mean()
        out["conditioning"] = {
            "n_models": int(len(per)),
            "n_toward_synergy": int((per["d_net_o_information"] < 0).sum()),
            "n_gaining_correlation": int((per["d_mean_abs_pearson"] > 0).sum()),
            "by_tolerance": c.groupby("tolerance")[
                ["d_net_o_information", "d_mean_abs_pearson", "acceptance_rate"]
            ].mean().round(5).to_dict("index"),
            "per_model": per.round(5).to_dict("index"),
        }

    if cov is not None and len(cov):
        cc = cov[(cov.estimator == "gaussian") & (cov.protocol_family == "absolute")]
        out["covariance_contrast"] = cc.groupby("contrast")[
            ["d_net_o_information", "d_frac_synergy_o3", "d_mean_abs_pearson"]
        ].mean().round(5).to_dict("index")
        e7 = cc[cc.regime == "E7"].groupby("model_id")["d_net_o_information"].mean()
        out["covariance_contrast"]["_n_models_toward_redundancy"] = int((e7 > 0).sum())
        out["covariance_contrast"]["_n_models"] = int(len(e7))

    if cu is not None:
        c = cu[cu.estimator == "gaussian"]
        e8 = c[c.regime == "E8"]
        e3 = c[c.regime == "E3"]
        out["surrogate_control"] = {
            "E8_naive_frac_synergy": float(e8["frac_synergy"].mean()),
            "E8_surrogate_controlled_frac_synergy": float(e8["frac_synergy_sig"].mean()),
            "E3_naive_frac_synergy": float(e3["frac_synergy"].mean()),
            "E3_surrogate_controlled_frac_synergy": float(e3["frac_synergy_sig"].mean()),
        }
        out["order_curve_by_regime"] = (
            c[c.regime.isin(["E1", "E3", "E4", "E7", "E8"])]
            .groupby(["regime", "order"])[["median", "frac_synergy", "frac_synergy_sig",
                                           "frac_redundancy_sig"]]
            .mean().round(4).reset_index().to_dict("records"))

    if vp is not None and len(vp):
        out["variance_partition"] = vp.round(4).to_dict("records")
    if mixed:
        out["mixed_effects"] = mixed
    if lomo is not None and len(lomo):
        out["lomo"] = (lomo.groupby(["endpoint", "predictors"])["r2_out_of_sample"]
                       .median().round(4).reset_index().to_dict("records"))
    if clf is not None and len(clf):
        out["classification"] = clf.round(4).to_dict("records")
    if stab is not None and len(stab):
        out["module_stability"] = stab.round(4).to_dict("records")

    if ts is not None and len(ts):
        piv = ts[ts.regime == "E3"].pivot_table(
            index=["model_id", "tuple"], columns="estimator",
            values="o_information").dropna()
        if {"gaussian", "copula"}.issubset(piv.columns):
            out["estimator_agreement"] = {
                "pearson_r_gaussian_vs_copula": float(
                    np.corrcoef(piv["gaussian"], piv["copula"])[0, 1]),
                "n_tuples": int(len(piv)),
            }

    if sob is not None and len(sob) and "interaction_fraction" in sob:
        # The `quotable` gate MUST be applied here.  It marks analyses whose
        # valid fraction clears VALID_FRACTION_FLOOR; below that floor the
        # missing evaluations are mean-imputed, which inflates the apparent
        # interaction in exactly the direction of the withdrawn 0.992
        # adaptation-index claim.  Until the 2026-08-06 audit (M8) this file
        # reported the ungated 0.4812 over 41 analyses while the prose reported
        # the gated 0.3635 over 24 -- two different numbers for one quantity,
        # in the same repository.
        s = sob.dropna(subset=["interaction_fraction"])
        q = s[s["quotable"].fillna(False).astype(bool)]
        out["sobol"] = {
            "mean_interaction_fraction": float(q["interaction_fraction"].mean()),
            "by_model": q.groupby("model_id")["interaction_fraction"].mean().round(3).to_dict(),
            "by_feature": q.groupby("feature")["interaction_fraction"].mean().round(3).to_dict(),
            "n_analyses": int(len(q)),
            "gate": "quotable == True (valid fraction above the preregistered floor)",
            "n_analyses_before_gate": int(len(s)),
            "mean_interaction_fraction_ungated": float(s["interaction_fraction"].mean()),
            "excluded_by_gate": sorted(
                f"{r.model_id}.{r.feature}" for r in
                s[~s["quotable"].fillna(False).astype(bool)].itertuples()),
        }
        if sobi is not None and "ST_minus_S1" in sobi:
            d = sobi.dropna(subset=["ST_minus_S1"])
            if "quotable" in d:
                d = d[d["quotable"].fillna(False).astype(bool)]
            out["sobol"]["mean_ST_minus_S1"] = float(d["ST_minus_S1"].mean())

    if pert is not None and len(pert):
        # Headline numbers use the PRIMARY readout only.  The battery evaluates
        # four features from the same simulations, so pooling counts each tuple
        # four times and averages over quantities with different noise floors --
        # the same defect the decision code was already corrected for.
        prim = pert[pert["primary_feature"]] if "primary_feature" in pert else pert
        gp = prim.groupby("kind")
        out["perturbation"] = {
            k: {"n": int(len(g)),
                "mean_abs_eta_sd": float(np.nanmean(np.abs(g["mean_nonadditivity_sd"]))),
                "median_abs_eta_sd": float(np.nanmedian(np.abs(g["mean_nonadditivity_sd"]))),
                "frac_supported": float(g["functionally_supported"].mean()),
                "mean_sign_consistency": float(g["sign_consistency"].mean())}
            for k, g in gp}
        out["perturbation"]["primary_feature"] = (
            str(prim["feature"].iloc[0]) if "feature" in prim else "unknown")
        # Readout dependence is a finding, not noise: report it rather than
        # averaging it away.
        if "feature" in pert:
            out["perturbation"]["by_feature_median_abs_eta_sd"] = {
                f: g.groupby("kind")["mean_nonadditivity_sd"]
                    .apply(lambda s: float(np.nanmedian(np.abs(s)))).round(4).to_dict()
                for f, g in pert.groupby("feature")}
        d = prim.dropna(subset=["o_information", "mean_nonadditivity_sd"])
        if len(d) > 5:
            out["perturbation"]["r_omega_vs_eta"] = float(np.corrcoef(
                d["o_information"], np.abs(d["mean_nonadditivity_sd"]))[0, 1])

    if res is not None and len(res):
        # Measurement floors (audit M5): an effect below its readout's
        # quantisation step is not resolvable however it is normalised.
        r = res[res["feature"] == "mean_firing_rate_Hz"]
        out["readout_resolution"] = {
            "rate_quantum_Hz": 2.5,
            "n_models_resolvable": int(r["resolvable"].sum()),
            "n_models": int(len(r)),
            "below_floor": sorted(r[~r["resolvable"].astype(bool)]["model_id"]),
            "floor_in_sd": r.set_index("model_id")["floor_in_sd"].round(4).to_dict(),
            "median_abs_eta_sd": r.set_index("model_id")["median_abs_eta_sd"].round(4).to_dict(),
        }

    if bench:
        ev = pd.DataFrame(bench["estimator_validation"])
        ev = ev[ev.n == ev.n.max()]
        out["estimator_validation"] = {
            "by_estimator_correct_rate":
                pd.DataFrame(bench["estimator_validation"]).groupby(
                    "estimator")["correct"].mean().round(3).to_dict(),
            "largest_n_values": ev.pivot_table(
                index="system", columns="estimator", values="o_information"
            ).round(4).to_dict("index"),
        }
        out["rsi_sign_validation"] = bench["rsi_sign_validation"]
        out["comparators"] = [{
            "system": s["system"], "n_valid": s["n_valid"], "n_prior": s["n_prior"],
            "ensembles": [{k: e.get(k) for k in
                           ("label", "n", "o_information_full", "frac_synergy_o3",
                            "mean_abs_pearson", "acceptance_rate")}
                          for e in s["ensembles"]],
        } for s in bench["comparators"]]

    if rsi is not None and len(rsi):
        r = rsi[rsi.estimator == "gaussian"]
        out["rsi"] = (r.groupby("regime")[["frac_synergistic", "median_rsi"]]
                      .mean().round(4).to_dict("index"))

    if dec:
        out["decision"] = dec

    path = AN / "key_numbers.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])
    print(f"\n... wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
