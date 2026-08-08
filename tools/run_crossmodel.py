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

from compass.analysis.cohorts import (  # noqa: E402
    PRIMARY_EXCLUSIONS,
    SENSITIVITY_ARMS,
    annotate_lineage,
    drop_duplicate_scientific_rows,
    effective_sample_summary,
    primary_mask,
)
from compass.analysis.hierarchical import (  # noqa: E402
    classification_test,
    cluster_bootstrap,
    leave_one_lineage_out,
    leave_one_model_out,
    mixed_effects_components,
    nested_variance_components,
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
                "source_lineage_id": r.get("source_lineage_id", mid),
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
                "source_lineage_id": r.get("source_lineage_id", mid),
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

    # 1. direction of the conditioning effect is consistent across LINEAGES.
    #
    # Counting model ids let hh_point and hh_cable -- one lineage, one extra
    # compartment -- vote twice, as did pospischil_rs/fs.  Under a fair null the
    # exact two-sided sign probability for 8 independent units is 0.0078; for
    # the 6 lineages those 8 ids actually represent it is 0.03125 (audit 9.1).
    if len(cond):
        g = (cond[cond["estimator"] == "gaussian"]
             .groupby("source_lineage_id")["d_net_o_information"].mean())
        by_model = (cond[cond["estimator"] == "gaussian"]
                    .groupby("model_id")["d_net_o_information"].mean())
        if len(g):
            frac = float(max((g < 0).mean(), (g > 0).mean()))
            direction = "toward synergy" if (g < 0).mean() >= 0.5 else "toward redundancy"
            k = int(max((g < 0).sum(), (g > 0).sum()))
            add("conditioning effect direction agreement across LINEAGES", "H-G",
                frac, frac >= THRESHOLDS["cross_model_direction_agreement"],
                f"{direction}; {k}/{len(g)} lineages "
                f"({len(by_model)} implementations); exact two-sided sign "
                f"p={_sign_p(k, len(g)):.4g}")

    # 2. direction of the covariance/clustering effect
    if len(cov):
        sub = cov[(cov["regime"] == "E7") & (cov["estimator"] == "gaussian")]
        g = sub.groupby("source_lineage_id")["d_net_o_information"].mean()
        if len(g):
            frac = float(max((g > 0).mean(), (g < 0).mean()))
            direction = "toward redundancy" if (g > 0).mean() >= 0.5 else "toward synergy"
            k = int(max((g > 0).sum(), (g < 0).sum()))
            add("high-covariance subset effect direction agreement (lineages)", "H-G",
                frac, frac >= THRESHOLDS["cross_model_direction_agreement"],
                f"{direction}; {k}/{len(g)} lineages; exact two-sided sign "
                f"p={_sign_p(k, len(g)):.4g}")

    # 3. sampling+geometry variance vs model identity
    for endpoint in PRIMARY_ENDPOINTS[:1]:
        v = vp[vp["endpoint"] == endpoint]
        if len(v):
            samp = float(v[v["block"] == "sampling"]["r2_unique"].iloc[0])
            geo = float(v[v["block"] == "geometry"]["r2_unique"].iloc[0])
            ident = float(v[v["block"] == "model_identity"]["r2_unique"].iloc[0])
            # `ident` is the unique (semi-partial) R2 of SOURCE LINEAGE, which
            # can be numerically zero once sampling and geometry are in the
            # model.  Reporting inf would be a non-quotable headline, so the
            # ratio is floored at the resolution of the R2 computation and
            # reported as a LOWER BOUND when identity is unresolvable.
            floor = 1e-4
            unresolved = ident < floor
            ratio = (samp + geo) / max(ident, floor)
            add(f"unique R2(sampling+geometry) / unique R2(model identity) [{endpoint}]",
                "H-G", float(ratio),
                ratio >= THRESHOLDS["sampling_geometry_over_identity_ratio"],
                f"sampling={samp:.3f} geometry={geo:.3f} identity={ident:.4f}"
                + (f"; identity is below the {floor} resolution floor, so the "
                   f"ratio is a LOWER BOUND (>{ratio:.0f}x), not a point estimate"
                   if unresolved else ""))

    # 4. transfer to an unseen SOURCE LINEAGE
    if len(lomo):
        sub = lomo[(lomo["predictors"] == "sampling+geometry")
                   & (lomo["endpoint"] == PRIMARY_ENDPOINTS[0])
                   & (lomo["fold_level"] == "source_lineage_id")]
        if len(sub):
            med = float(sub["r2_out_of_sample"].median())
            add("out-of-sample R2 for a held-out LINEAGE (sampling+geometry)",
                "H-G", med, med > THRESHOLDS["lomo_r2_floor"],
                f"n_folds={len(sub)}; leave-one-lineage-out, so no relative of the "
                f"held-out model stays in training")

    # 5. classifier: regime generalises across lineages; identity does not.
    #
    # The identity task is NOT scored.  It uses random within-table splits of
    # derived rows from a single campaign, so it measures within-campaign
    # dataset fingerprinting, not biological identifiability: two rows from the
    # same model share a campaign, a seed and a parameter matrix, and a random
    # split puts near-duplicates on both sides (audit 7.5).
    if len(clf):
        reg = clf[clf["task"] == "sampling_regime"]
        ide = clf[clf["task"] == "model_identity"]
        if len(reg):
            r = reg.iloc[0]
            add("sampling regime predictable across unseen lineages", "H-G",
                float(r["accuracy"]), r["accuracy"] > r["baseline"] * 1.5,
                f"baseline={r['baseline']:.3f}, {int(r['n_classes'])} classes, "
                f"cv={r['cv']}")
        if len(ide):
            r = ide.iloc[0]
            note("within-campaign implementation identifiability (descriptive only)",
                 float(r["accuracy"]),
                 f"baseline={r['baseline']:.3f}, {int(r['n_classes'])} classes, "
                 f"cv={r['cv']}. Withdrawn as an H-B criterion by the 2026-08-06 "
                 f"audit (7.5): random within-table splits of derived rows from "
                 f"one campaign measure dataset fingerprinting, not independent "
                 f"biological identity.")

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

    # 7. module stability across estimators -- REPORTED, NOT SCORED.
    if stability:
        j = float(stability.get("mean_jaccard", np.nan))
        note("top-tuple Jaccard, Gaussian vs Gaussian-copula (descriptive only)", j,
             f"{stability.get('n_models')} models over {stability.get('n_cells')} "
             f"ensemble cells, min per-model "
             f"{stability.get('min_jaccard', float('nan')):.2f}. Withdrawn as an "
             f"H-B criterion by the 2026-08-06 audit (M6): both rankings come "
             f"from Gaussian estimators on the SAME samples -- the copula "
             f"transform only removes marginal non-Gaussianity -- so agreement "
             f"between them measures estimator similarity, not module stability. "
             f"Stability across independent samples is what blueprint 19.3 asks "
             f"for and what the campaign-scramble replicate now supplies.")

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
            # The two point ratios are NOT two votes.  They come from the same
            # perturbation result on the same top tuples, differing only in
            # which control family sits in the denominator, so scoring both
            # double-counts one measurement (audit M6, 7.7).  One paired,
            # lineage-clustered test decides; the other ratio is described.
            for group, label, scored in ((ctrl, "matched middle-Omega", True),
                                         (rand, "random", False)):
                if not len(group):
                    continue
                a = float(np.nanmean(np.abs(top["mean_nonadditivity_sd"])))
                b = float(np.nanmean(np.abs(group["mean_nonadditivity_sd"])))
                ratio = float(a / b) if b > 0 else float("nan")
                # Pair within lineage, then resample lineages: the eight model
                # ids are not eight independent units.
                paired = (prim.pivot_table(index=["source_lineage_id", "model_id"],
                                           columns="kind",
                                           values="mean_nonadditivity_sd",
                                           aggfunc=lambda s: float(np.nanmean(np.abs(s))))
                          if "source_lineage_id" in prim else None)
                ci = None
                if paired is not None and "top_synergy" in paired and \
                        group["kind"].iloc[0] in paired:
                    d = (paired["top_synergy"] - paired[group["kind"].iloc[0]]).to_numpy(float)
                    lin = paired.index.get_level_values("source_lineage_id").to_numpy()
                    ci = cluster_bootstrap(d, lin)
                detail = (f"top={a:.3f} SD vs control={b:.3f} SD (ratio {ratio:.2f}x)"
                          if b > 0 else "control SD is zero")
                if ci:
                    detail += (f"; paired difference {ci['point']:+.3f} SD, "
                               f"lineage-clustered 95% CI "
                               f"[{ci['ci95'][0]:+.3f}, {ci['ci95'][1]:+.3f}] "
                               f"over {ci['n_clusters']} lineages")
                if scored:
                    passed = bool(ci and ci["ci95"][0] > 0)
                    add("top-synergy tuples more non-additive than matched controls",
                        "H-B", ratio, passed,
                        detail + ". Scored on the lineage-clustered paired interval, "
                        "not on the point ratio.")
                else:
                    note(f"top vs {label} control point ratio (descriptive only)",
                         ratio, detail + ". Not scored: same measurement as the "
                         "criterion above with a different denominator.")

    hg = [c for c in ev["criteria"] if c["supports"] == "H-G"]
    hb = [c for c in ev["criteria"] if c["supports"] == "H-B"]
    ev["n_hg_passed"] = sum(c["passed"] for c in hg)
    ev["n_hg_total"] = len(hg)
    ev["n_hb_passed"] = sum(c["passed"] for c in hb)
    ev["n_hb_total"] = len(hb)

    # A side with NO computable criterion must not be scored as a failing side.
    # `max(1, total)` was only a ZeroDivisionError guard, but it turned "no H-B
    # criterion could be evaluated" into hb_frac = 0.0 -- indistinguishable from
    # "every H-B criterion was tested and failed".  The replication arm then
    # concluded H-G from a cohort that contains no perturbation data at all,
    # i.e. absence of evidence scored as evidence.  This is the same principle
    # the release manifest already applies to gates: a missing record is a
    # failure to evaluate, not a pass (post-revision audit).
    if ev["n_hg_total"] == 0 or ev["n_hb_total"] == 0:
        absent = [s for s, n in (("H-G", ev["n_hg_total"]),
                                 ("H-B", ev["n_hb_total"])) if n == 0]
        ev["conclusion"] = (f"not evaluable: {' and '.join(absent)} has no "
                            f"computable criterion in this arm")
        ev["hg_fraction_passed"] = (ev["n_hg_passed"] / ev["n_hg_total"]
                                    if ev["n_hg_total"] else None)
        ev["hb_fraction_passed"] = (ev["n_hb_passed"] / ev["n_hb_total"]
                                    if ev["n_hb_total"] else None)
        return ev

    hg_frac = ev["n_hg_passed"] / ev["n_hg_total"]
    hb_frac = ev["n_hb_passed"] / ev["n_hb_total"]
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


def _sign_p(k: int, n: int) -> float:
    """Exact two-sided sign probability of k/n agreeing under a fair null."""
    from math import comb

    if n == 0:
        return float("nan")
    k = max(k, n - k)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2.0 * tail / 2**n))


def run_arm(ep_all: pd.DataFrame, arm: str, experiment: str,
            pert_all, stability, write: bool) -> dict:
    """Run the whole cross-model analysis under one sensitivity arm."""
    spec = SENSITIVITY_ARMS[arm]
    ep = ep_all[primary_mask(ep_all, arm)].reset_index(drop=True)
    if ep.empty or ep["model_id"].nunique() < 3:
        return {"arm": arm, "error": "too few models", "spec": spec}
    # Duplicate scientific rows are valid for display but are not independent
    # observations for regression or classification (audit 6.2).
    ep_fit = drop_duplicate_scientific_rows(ep)

    cond = conditioning_shift(ep)
    cov = covariance_shift(ep)
    gauss = ep_fit[ep_fit["estimator"] == "gaussian"]

    vps = [variance_partition(gauss, e) for e in PRIMARY_ENDPOINTS]
    vp = (pd.concat([v for v in vps if len(v)], ignore_index=True)
          if any(len(v) for v in vps) else pd.DataFrame())
    mixed = mixed_effects_components(gauss, PRIMARY_ENDPOINTS[0])
    nested = nested_variance_components(gauss, PRIMARY_ENDPOINTS[0])

    lomos = []
    for e in PRIMARY_ENDPOINTS:
        a = leave_one_lineage_out(gauss, e)
        if len(a):
            a["fold_level"] = "source_lineage_id"
            lomos.append(a)
        b = leave_one_model_out(gauss, e)
        if len(b):
            b["fold_level"] = "model_id"
            lomos.append(b)
    lomo = pd.concat(lomos, ignore_index=True) if lomos else pd.DataFrame()

    clf = classification_test(gauss, INFO_FEATURES, seed=20260804)
    pert = pert_all
    if pert is not None and len(pert):
        pert = annotate_lineage(pert)
        pert = pert[primary_mask(pert, arm)]

    decision = decide(ep, cond, cov, vp, lomo, clf, mixed, pert, stability)
    # An arm can only test what it has data for.  `retain_mainen` exists to show
    # what changes when mainen_l5pc is put back, but mainen's perturbation
    # battery was not regenerated (it is excluded from primary inference and
    # cost ~16 h), so the perturbation criteria in that arm are computed WITHOUT
    # it.  Silently reporting the arm's conclusion would attribute a change to
    # the analysis that is really an absence of data.
    endpoint_models = sorted(ep["model_id"].unique())
    pert_models = sorted(pert["model_id"].unique()) if pert is not None and len(pert) else []
    missing_pert = [m for m in endpoint_models if m not in pert_models]
    decision["perturbation_coverage"] = {
        "models_in_endpoints": endpoint_models,
        "models_with_perturbation_data": pert_models,
        "missing_from_perturbation": missing_pert,
        "complete": not missing_pert,
        "caveat": ("" if not missing_pert else
                   f"the perturbation criteria in this arm are computed without "
                   f"{missing_pert}; their conclusion cannot be compared with an "
                   f"arm that has complete coverage"),
    }
    decision["arm"] = arm
    decision["arm_description"] = spec["description"]
    decision["excluded_models"] = spec["exclude"]
    decision["effective_sample"] = effective_sample_summary(ep, arm)
    decision["nested_variance_components"] = nested

    if write:
        sfx = "" if arm == "primary" else f"_{arm}"
        cond.to_parquet(OUT / f"conditioning_shift{sfx}.parquet", index=False)
        cov.to_parquet(OUT / f"covariance_shift{sfx}.parquet", index=False)
        vp.to_parquet(OUT / f"variance_partition{sfx}.parquet", index=False)
        lomo.to_parquet(OUT / f"lomo{sfx}.parquet", index=False)
        clf.to_parquet(OUT / f"classification{sfx}.parquet", index=False)
        (OUT / f"mixed_effects{sfx}.json").write_text(
            json.dumps({"flat": mixed, "nested": nested}, indent=1, default=float))
        (OUT / f"decision{sfx}.json").write_text(
            json.dumps(decision, indent=1, default=float))
    return decision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--arms", nargs="*", default=sorted(SENSITIVITY_ARMS))
    args = ap.parse_args()

    manifests = {}
    for p in (ROOT / "results" / "manifests").glob("canonical_*.json"):
        r = json.loads(p.read_text())
        manifests[r["model_id"]] = r

    def load_annotated(experiment: str) -> pd.DataFrame:
        """Endpoint table for one experiment, with registry annotation applied.

        An arm may declare ``source_experiment`` to be evaluated on a DIFFERENT
        endpoint table rather than on a filtered view of the primary one.  Before
        this existed, ``replication_only`` filtered the discovery table for models
        that are not in it and therefore reported "too few models" while
        ``endpoints_screening_v2_replication.parquet`` sat unread on disk.
        """
        t = pd.read_parquet(OUT / f"endpoints_{experiment}.parquet")
        t["n_compartments"] = t["model_id"].map(
            lambda m: manifests.get(m, {}).get("n_compartments", 1))
        t["is_point_model"] = t["model_id"].map(
            lambda m: float(manifests.get(m, {}).get("manifest", {}).get("is_point_model", False)))
        t["neuron_class"] = t["model_id"].map(
            lambda m: manifests.get(m, {}).get("manifest", {}).get("neuron_class", "?"))
        t["lab"] = t["model_id"].map(
            lambda m: manifests.get(m, {}).get("manifest", {}).get("lab", "?"))
        t["pipeline"] = t["model_id"].map(
            lambda m: manifests.get(m, {}).get("manifest", {}).get("pipeline", "?"))
        t = annotate_lineage(t)
        t.to_parquet(OUT / f"endpoints_annotated_{experiment}.parquet", index=False)
        return t

    ep = load_annotated(args.experiment)

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

    arms = {}
    _sources: dict[str, pd.DataFrame] = {args.experiment: ep}
    for arm in args.arms:
        src = SENSITIVITY_ARMS[arm].get("source_experiment")
        if src:
            path = OUT / f"endpoints_{src}.parquet"
            if not path.exists():
                print(f"\n### arm {arm}: source table {path.name} is absent; "
                      f"arm skipped rather than silently evaluated on "
                      f"{args.experiment}")
                arms[arm] = {"error": f"source table {path.name} absent"}
                continue
            if src not in _sources:
                _sources[src] = load_annotated(src)
            ep_arm = _sources[src]
        else:
            ep_arm = ep
        d = run_arm(ep_arm, arm, args.experiment, pert, stability, write=True)
        arms[arm] = d
        if "error" in d:
            print(f"\n### arm {arm}: {d['error']}")
            continue
        print(f"\n=== DECISION TABLE [arm: {arm}] ===")
        print(f"    {d['arm_description']}")
        es = d["effective_sample"]
        print(f"    {es['n_model_ids_primary']} implementations / "
              f"{es['n_source_lineages_primary']} source lineages / "
              f"{es['n_neuron_classes']} classes; "
              f"{es['n_classes_with_2plus_lineages']} classes have >=2 lineages")
        for c in d["criteria"]:
            mark = "PASS" if c["passed"] else "----"
            print(f"[{mark}] ({c['supports']}) {c['criterion']}: "
                  f"{c['value']:.3f}  {c['detail']}")
        for q in d.get("descriptive", []):
            print(f"[desc] {q['quantity']}: {q['value']:.3f}  {q['detail'][:110]}")
        pc = d.get("perturbation_coverage", {})
        if pc and not pc.get("complete", True):
            print(f"    ! partial perturbation coverage: {pc['caveat']}")
        print(f"H-G {d['n_hg_passed']}/{d['n_hg_total']}   "
              f"H-B {d['n_hb_passed']}/{d['n_hb_total']}   "
              f"-> conclusion: {d['conclusion']}")

    (OUT / "decision_sensitivity.json").write_text(json.dumps(
        {"primary_arm": "primary",
         "exclusions": PRIMARY_EXCLUSIONS,
         "arms": {a: {"conclusion": d.get("conclusion"),
                      "description": d.get("arm_description"),
                      "n_hg_passed": d.get("n_hg_passed"),
                      "n_hg_total": d.get("n_hg_total"),
                      "n_hb_passed": d.get("n_hb_passed"),
                      "n_hb_total": d.get("n_hb_total"),
                      "effective_sample": d.get("effective_sample"),
                      "perturbation_coverage": d.get("perturbation_coverage"),
                      "error": d.get("error")}
                  for a, d in arms.items()}},
        indent=1, default=float))

    print("\n=== SENSITIVITY CASCADE ===")
    for a, d in arms.items():
        print(f"  {a:26s} -> {d.get('conclusion', d.get('error'))}"
              f"  (H-G {d.get('n_hg_passed')}/{d.get('n_hg_total')}, "
              f"H-B {d.get('n_hb_passed')}/{d.get('n_hb_total')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
