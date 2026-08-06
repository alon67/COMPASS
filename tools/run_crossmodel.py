"""Cross-model inference and the frozen hypothesis decision (blueprint 18, 19).

Consumes results/analyses/endpoints_*.parquet and produces:

``variance_partition.parquet``   unique / shared R^2 per predictor block
``mixed_effects.json``           model-identity intraclass correlation
``lomo.parquet``                 leave-one-model-family-out generalisation
``classification.parquet``       regime vs identity recoverability
``decision.json``                the preregistered H-G / H-B / H-M decision table

The decision rules are the ones written down in blueprint 19.4 *before* any
result was inspected; this tool only evaluates them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.analysis.hierarchical import (  # noqa: E402
    classification_test,
    leave_one_model_out,
    mixed_effects_components,
    variance_partition,
)

OUT = ROOT / "results" / "analyses"

PRIMARY_ENDPOINTS = [
    "synergy_area",
    "redundancy_area",
    "net_o_information",
    "frac_synergy_o3",
]

INFO_FEATURES = [
    "synergy_area", "redundancy_area", "net_o_information",
    "frac_synergy_o3", "frac_synergy_o4", "frac_synergy_o5",
    "median_omega_o3", "median_omega_o4", "median_omega_o5",
    "frac_synergy_sig_o3", "frac_redundancy_sig_o3",
]

# Preregistered thresholds (blueprint 19.4).
THRESHOLDS = {
    "cross_model_direction_agreement": 0.80,
    "sampling_geometry_over_identity_ratio": 2.0,
    "module_selection_probability": 0.80,
    "perturbation_effect_sd": 0.25,
    "lomo_r2_floor": 0.0,
}


def conditioning_shift(ep: pd.DataFrame) -> pd.DataFrame:
    """Endpoint 3: change from unconditioned E1 to each conditioned ensemble."""
    rows = []
    for (mid, est, fam), g in ep.groupby(["model_id", "estimator", "protocol_family"]):
        base = g[g["regime"] == "E1"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in g[g["regime"].isin(["E3", "E3b", "E4", "E4b"])].iterrows():
            rows.append({
                "model_id": mid, "estimator": est, "protocol_family": fam,
                "regime": r["regime"], "tolerance": r["tolerance"],
                "metric": r["metric"],
                "d_synergy_area": r["synergy_area"] - b["synergy_area"],
                "d_net_o_information": r["net_o_information"] - b["net_o_information"],
                "d_frac_synergy_o3": r["frac_synergy_o3"] - b["frac_synergy_o3"],
                "d_mean_abs_pearson": r["mean_abs_pearson"] - b["mean_abs_pearson"],
                "d_effective_rank": (r["effective_rank"] - b["effective_rank"]) / r["n_params"],
                "acceptance_rate": r["acceptance_rate"],
                "n_conditioned": r["n_samples"], "n_unconditioned": b["n_samples"],
            })
    return pd.DataFrame(rows)


def covariance_shift(ep: pd.DataFrame) -> pd.DataFrame:
    """Endpoint 4: change from a diverse accepted set to matched high-covariance/clustered."""
    rows = []
    for (mid, est, fam), g in ep.groupby(["model_id", "estimator", "protocol_family"]):
        ref = g[g["regime"] == "E3m"]
        ref = ref if not ref.empty else g[g["regime"] == "E3"]
        if ref.empty:
            continue
        b = ref.iloc[0]
        for regime in ("E7", "E6", "E8", "E9"):
            r = g[g["regime"] == regime]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "model_id": mid, "estimator": est, "protocol_family": fam,
                "contrast": f"{regime}_vs_{b['regime']}",
                "regime": regime,
                "d_synergy_area": r["synergy_area"] - b["synergy_area"],
                "d_net_o_information": r["net_o_information"] - b["net_o_information"],
                "d_frac_synergy_o3": r["frac_synergy_o3"] - b["frac_synergy_o3"],
                "d_mean_abs_pearson": r["mean_abs_pearson"] - b["mean_abs_pearson"],
                "n_a": r["n_samples"], "n_b": b["n_samples"],
            })
    return pd.DataFrame(rows)


def decide(ep, cond, cov, vp, lomo, clf, mixed, perturb, stability) -> dict:
    """Evaluate the preregistered decision framework (blueprint 19)."""
    ev: dict = {"criteria": [], "descriptive": [], "thresholds": THRESHOLDS}

    def add(name, hypothesis, value, passed, detail=""):
        ev["criteria"].append({"criterion": name, "supports": hypothesis,
                               "value": value, "passed": bool(passed),
                               "detail": detail})

    def note(name, value, detail=""):
        """Record a quantity for description without letting it cast a vote.

        Anything in `descriptive` is excluded from the H-G/H-B tally by
        construction, so a withdrawn criterion cannot leak back into the
        decision by being re-read from the same file.
        """
        ev["descriptive"].append({"quantity": name, "value": value,
                                  "detail": detail})

    # 1. direction of the conditioning effect is consistent across models
    if len(cond):
        g = cond[cond["estimator"] == "gaussian"].groupby("model_id")["d_net_o_information"].mean()
        if len(g):
            frac = float(max((g < 0).mean(), (g > 0).mean()))
            direction = "toward synergy" if (g < 0).mean() >= 0.5 else "toward redundancy"
            add("conditioning effect direction agreement across models", "H-G",
                frac, frac >= THRESHOLDS["cross_model_direction_agreement"],
                f"{direction}; n_models={len(g)}")

    # 2. direction of the covariance/clustering effect
    if len(cov):
        sub = cov[(cov["regime"] == "E7") & (cov["estimator"] == "gaussian")]
        g = sub.groupby("model_id")["d_net_o_information"].mean()
        if len(g):
            frac = float(max((g > 0).mean(), (g < 0).mean()))
            direction = "toward redundancy" if (g > 0).mean() >= 0.5 else "toward synergy"
            add("high-covariance subset effect direction agreement", "H-G",
                frac, frac >= THRESHOLDS["cross_model_direction_agreement"],
                f"{direction}; n_models={len(g)}")

    # 3. sampling+geometry variance vs model identity
    for endpoint in PRIMARY_ENDPOINTS[:1]:
        v = vp[vp["endpoint"] == endpoint]
        if len(v):
            samp = float(v[v["block"] == "sampling"]["r2_unique"].iloc[0])
            geo = float(v[v["block"] == "geometry"]["r2_unique"].iloc[0])
            ident = float(v[v["block"] == "model_identity"]["r2_unique"].iloc[0])
            ratio = (samp + geo) / ident if ident > 1e-6 else np.inf
            add(f"unique R2(sampling+geometry) / unique R2(model identity) [{endpoint}]",
                "H-G", float(ratio),
                ratio >= THRESHOLDS["sampling_geometry_over_identity_ratio"],
                f"sampling={samp:.3f} geometry={geo:.3f} identity={ident:.3f}")

    # 4. transfer to unseen model families
    if len(lomo):
        sub = lomo[(lomo["predictors"] == "sampling+geometry") &
                   (lomo["endpoint"] == PRIMARY_ENDPOINTS[0])]
        if len(sub):
            med = float(sub["r2_out_of_sample"].median())
            add("out-of-sample R2 for a held-out model family (sampling+geometry)",
                "H-G", med, med > THRESHOLDS["lomo_r2_floor"],
                f"n_folds={len(sub)}")

    # 5. classifier: regime generalises, identity does not
    if len(clf):
        reg = clf[clf["task"] == "sampling_regime"]
        ide = clf[clf["task"] == "model_identity"]
        if len(reg):
            r = reg.iloc[0]
            add("sampling regime predictable across unseen model families", "H-G",
                float(r["accuracy"]), r["accuracy"] > r["baseline"] * 1.5,
                f"baseline={r['baseline']:.3f}, {int(r['n_classes'])} classes")
        if len(ide):
            r = ide.iloc[0]
            add("model identity predictable from information summaries", "H-B",
                float(r["accuracy"]), r["accuracy"] > r["baseline"] * 1.5,
                f"baseline={r['baseline']:.3f}, {int(r['n_classes'])} classes")

    # 6. model-identity intraclass correlation -- REPORTED, NOT SCORED.
    #
    # Withdrawn as a decision criterion by the 2026-08-06 audit (M7).  Two
    # reasons.  (a) The quantity that was being scored was computed with a
    # double-counted group variance, so the 0.048 that voted for H-G was not an
    # ICC.  (b) Even computed correctly, an adjusted ICC measures how much
    # between-family variance survives the fixed covariates, and those
    # covariates (regime, geometry, sample size) are themselves the H-G
    # hypothesis.  Conditioning on the alternative and then testing whether the
    # residual favours H-B loads the die.  The value is still written to
    # mixed_effects.json for description; it no longer casts a vote.
    if mixed and "icc_adjusted" in mixed:
        note("adjusted ICC of model identity (mixed model, descriptive only)",
             float(mixed["icc_adjusted"]),
             "not scored: the fixed covariates are the H-G hypothesis, so the "
             "residual between-family share is not a fair test of H-B")

    # 7. module stability across estimators / replicates
    if stability:
        j = float(stability.get("mean_jaccard", np.nan))
        add("top-tuple module stability (mean per-model Jaccard across estimators)",
            "H-B", j, np.isfinite(j) and j >= 0.5,
            f"{stability.get('n_models')} models over {stability.get('n_cells')} "
            f"ensemble cells, min per-model {stability.get('min_jaccard', float('nan')):.2f}. "
            f"NOTE blueprint 19.3 requires modules that are stable AND functionally "
            f"validated; the perturbation criteria below test the second half.")

    # 8. perturbation validation of statistical modules
    #
    # Restricted to the PRIMARY feature: the battery now evaluates four features
    # from the same simulations, and pooling them would count one tuple four
    # times and mix quantities with different noise floors.  The control group
    # is the matched middle-Omega family; the uniformly random family is
    # reported alongside because it is the comparator the earlier write-up
    # wrongly claimed to have used.
    if perturb is not None and len(perturb):
        prim = perturb[perturb["primary_feature"]] if "primary_feature" in perturb \
            else perturb
        top = prim[prim["kind"] == "top_synergy"]
        ctrl = prim[prim["kind"].isin(["control_middle", "control"])]
        rand = prim[prim["kind"] == "control_random"]
        if len(top):
            frac = float(top["functionally_supported"].mean())
            frac_c = float(ctrl["functionally_supported"].mean()) if len(ctrl) else float("nan")
            add("statistical synergy modules validated by direct perturbation",
                "H-B", frac, frac >= 0.5,
                f"n_top={len(top)}, n_control={len(ctrl)}; "
                f"supported {frac:.3f} vs control {frac_c:.3f}")
            for group, label in ((ctrl, "matched middle-Omega"), (rand, "random")):
                if not len(group):
                    continue
                a = float(np.nanmean(np.abs(top["mean_nonadditivity_sd"])))
                b = float(np.nanmean(np.abs(group["mean_nonadditivity_sd"])))
                add(f"top-synergy tuples more non-additive than {label} controls",
                    "H-B", float(a / b) if b > 0 else float("nan"), b > 0 and a > b * 1.5,
                    f"top={a:.3f} SD vs control={b:.3f} SD (ratio {a/b:.2f}x)"
                    if b > 0 else "control SD is zero")

    hg = [c for c in ev["criteria"] if c["supports"] == "H-G"]
    hb = [c for c in ev["criteria"] if c["supports"] == "H-B"]
    ev["n_hg_passed"] = sum(c["passed"] for c in hg)
    ev["n_hg_total"] = len(hg)
    ev["n_hb_passed"] = sum(c["passed"] for c in hb)
    ev["n_hb_total"] = len(hb)

    hg_frac = ev["n_hg_passed"] / max(1, ev["n_hg_total"])
    hb_frac = ev["n_hb_passed"] / max(1, ev["n_hb_total"])
    if hg_frac >= 0.6 and hb_frac < 0.4:
        ev["conclusion"] = "H-G"
    elif hb_frac >= 0.6 and hg_frac < 0.4:
        ev["conclusion"] = "H-B"
    elif hg_frac >= 0.5 and hb_frac >= 0.4:
        ev["conclusion"] = "H-M"
    else:
        ev["conclusion"] = "inconclusive"
    ev["hg_fraction_passed"] = hg_frac
    ev["hb_fraction_passed"] = hb_frac
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    args = ap.parse_args()

    ep = pd.read_parquet(OUT / f"endpoints_{args.experiment}.parquet")
    manifests = {}
    for p in (ROOT / "results" / "manifests").glob("canonical_*.json"):
        r = json.loads(p.read_text())
        manifests[r["model_id"]] = r
    ep["n_compartments"] = ep["model_id"].map(lambda m: manifests.get(m, {}).get("n_compartments", 1))
    ep["is_point_model"] = ep["model_id"].map(
        lambda m: float(manifests.get(m, {}).get("manifest", {}).get("is_point_model", False)))
    ep["neuron_class"] = ep["model_id"].map(
        lambda m: manifests.get(m, {}).get("manifest", {}).get("neuron_class", "?"))
    ep["lab"] = ep["model_id"].map(
        lambda m: manifests.get(m, {}).get("manifest", {}).get("lab", "?"))
    ep["pipeline"] = ep["model_id"].map(
        lambda m: manifests.get(m, {}).get("manifest", {}).get("pipeline", "?"))
    ep.to_parquet(OUT / f"endpoints_annotated_{args.experiment}.parquet", index=False)

    cond = conditioning_shift(ep)
    cov = covariance_shift(ep)
    cond.to_parquet(OUT / "conditioning_shift.parquet", index=False)
    cov.to_parquet(OUT / "covariance_shift.parquet", index=False)
    print(f"conditioning contrasts={len(cond)} covariance contrasts={len(cov)}")

    gauss = ep[ep["estimator"] == "gaussian"]
    vps = [variance_partition(gauss, e) for e in PRIMARY_ENDPOINTS]
    vp = pd.concat([v for v in vps if len(v)], ignore_index=True) if any(len(v) for v in vps) else pd.DataFrame()
    vp.to_parquet(OUT / "variance_partition.parquet", index=False)

    mixed = mixed_effects_components(gauss, PRIMARY_ENDPOINTS[0])
    (OUT / "mixed_effects.json").write_text(json.dumps(mixed, indent=1, default=float))

    lomos = [leave_one_model_out(gauss, e) for e in PRIMARY_ENDPOINTS]
    lomo = pd.concat([x for x in lomos if len(x)], ignore_index=True) if any(len(x) for x in lomos) else pd.DataFrame()
    lomo.to_parquet(OUT / "lomo.parquet", index=False)

    clf = classification_test(gauss, INFO_FEATURES, seed=20260804)
    clf.to_parquet(OUT / "classification.parquet", index=False)

    # module stability across estimators (Gaussian vs copula) per model
    ts_path = OUT / f"tuple_scores_{args.experiment}.parquet"
    stability = {}
    if ts_path.exists():
        from compass.analysis.tuples import module_stability

        ts = pd.read_parquet(ts_path)
        per_model = []
        # Jaccard is only meaningful WITHIN a model: tuple labels are parameter
        # names, so lists from two different models can never overlap and a
        # pooled comparison would report ~0 stability by construction.
        # Group by the EXACT ensemble, not just the model.  A model contributes
        # several E3 ensembles (absolute/normalised protocol family x max-z /
        # weighted-Euclidean metric); pooling them let one tuple carry several
        # scores from different conditioned ensembles, so the "top ten" was not a
        # like-for-like estimator comparison.
        for (mid, lab, fam, met), g in ts[ts["regime"] == "E3"].groupby(
                ["model_id", "label", "protocol_family", "metric"]):
            runs = [list(x.sort_values("o_information")["tuple"])
                    for _, x in g.groupby("estimator")]
            if len(runs) >= 2:
                st = module_stability(runs, top_k=10)
                if np.isfinite(st["mean_jaccard"]):
                    per_model.append({"model_id": mid, "label": lab,
                                      "protocol_family": fam, "metric": met,
                                      **{k: v for k, v in st.items()
                                         if k != "selection_probability"}})
        pd.DataFrame(per_model).to_parquet(OUT / "module_stability.parquet", index=False)
        if per_model:
            # Average WITHIN each model first, then across models.  The rows are
            # ensemble x protocol-family x metric cells, and models contribute
            # unequal numbers of them, so a flat mean silently weights the
            # models with more surviving ensembles more heavily -- and the
            # criterion string used to call that row count "models" (it was
            # reporting 31 models from a panel of 8).  Audit 2026-08-06.
            pm = pd.DataFrame(per_model)
            pm = pm[np.isfinite(pm["mean_jaccard"])]
            by_model = pm.groupby("model_id")["mean_jaccard"].mean()
            stability = {"mean_jaccard": float(by_model.mean()) if len(by_model) else float("nan"),
                         "n_cells": int(len(pm)),
                         "n_models": int(by_model.size),
                         "mean_jaccard_flat_over_cells": float(pm["mean_jaccard"].mean())
                         if len(pm) else float("nan"),
                         "min_jaccard": float(by_model.min()) if len(by_model) else float("nan")}

    pert = None
    if (OUT / "perturbations.parquet").exists():
        pert = pd.read_parquet(OUT / "perturbations.parquet")

    decision = decide(ep, cond, cov, vp, lomo, clf, mixed, pert, stability)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=1, default=float))

    print("\n=== DECISION TABLE ===")
    for c in decision["criteria"]:
        mark = "PASS" if c["passed"] else "----"
        print(f"[{mark}] ({c['supports']}) {c['criterion']}: "
              f"{c['value']:.3f}  {c['detail']}")
    for q in decision.get("descriptive", []):
        print(f"[desc] {q['quantity']}: {q['value']:.3f}  {q['detail']}")
    print(f"\nH-G {decision['n_hg_passed']}/{decision['n_hg_total']}   "
          f"H-B {decision['n_hb_passed']}/{decision['n_hb_total']}   "
          f"-> conclusion: {decision['conclusion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
