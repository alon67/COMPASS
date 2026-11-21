"""Turn simulated feature tables into the frozen analysis tables.

Outputs (all under results/analyses/):

``endpoints.parquet``     one row per (model, ensemble, estimator, protocol family):
                          the primary endpoints of blueprint 18.1 plus the
                          statistical-geometry covariates of 18.2.
``order_curves.parquet``  the full O-information order curve behind each row,
                          with per-order surrogate bounds.
``rsi.parquet``           target-specific (RSI) summaries per electrical feature.
``tuple_scores.parquet``  per-tuple O-information for module-stability analysis.

Nothing here re-runs a simulation, so estimator changes are cheap -- which is
precisely the separation blueprint 22.2 asks for.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.analysis.geometry import describe_geometry  # noqa: E402
from compass.analysis.oinfo import GaussianInfoEngine  # noqa: E402
from compass.analysis.tuples import generate_tuples, order_curve  # noqa: E402
from compass.conditioning.ensembles import (  # noqa: E402
    BALANCE_TOL_SD,
    CONDITIONING_FEATURES,
    CONDITIONING_FEATURES_NORMALISED,
    build_ensembles,
    validity_mask,  # noqa: E402
)

OUT = ROOT / "results" / "analyses"

# The QC flag `compass.adapters.base.qc_flags` raises for a trace outside
# [PHYSICAL_V_MIN_MV, PHYSICAL_V_MAX_MV].  Copied rather than imported: that
# module is inside `SIM_CODE_PATHS`, so adding a constant to it would change
# every stored shard's recorded `sim_code_hash` and make the corpus's own
# provenance record disagree with the tree that produced it.  The copy is
# pinned to the original by test_wide_voltage_band_arm_matches_the_stored_flag,
# which reads the flag off a real stored row rather than trusting either name.
BAND_FLAG = "nonphysical_voltage"
ORDERS = (3, 4, 5)
MAX_TUPLES = 1200
N_SURROGATE = 25
SEED = 20260804

# Features used as RSI targets (blueprint 16.3: "for every common electrical
# feature").  These are outputs, not conditioning variables.
RSI_TARGETS = [
    "rheobase_nA",
    "hyperpol_abs.input_resistance_Mohm",
    "step_abs_hi.mean_firing_rate_Hz",
    "step_abs_hi.ap_amplitude_mV",
    "step_abs_hi.ap_halfwidth_ms",
    "step_abs_hi.adaptation_index",
    "step_abs_hi.first_spike_latency_ms",
    "rest.resting_potential_mV",
]


def load_model_table(experiment: str, model_id: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ROOT / "results" / "features" / experiment / model_id / "*.parquet")))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def readmit_band_tightened(df: pd.DataFrame, experiment: str, model_id: str,
                           keys: set[tuple]) -> tuple[pd.DataFrame, int]:
    """Undo the voltage-band tightening for exactly the listed rows.

    Unifying the corpus and waveform bands at ``[-150, 80] mV`` excluded 609
    rows that the previous corpus band ``[-160, 120]`` admitted.  That was a
    judgement call -- the 2026-08-09 post-response-3 audit required ONE band
    propagated everywhere, not which one -- so this arm measures its cost the
    way ``n4_retained`` measures the N4 projection's.

    Re-admission is done by removing ``nonphysical_voltage`` from those rows'
    ``qc_flags``, which is exactly what the wider band would have produced:
    ``validity_mask`` reads the flags, and every one of these rows gained that
    flag and only that flag from the change.  Other flags are left alone, so a
    row excluded for a second reason stays excluded.

    A stored flag is PROTOCOL-QUALIFIED -- ``hyperpol_abs:nonphysical_voltage``,
    not the bare name -- because one row carries a trace per protocol and the
    band is checked on each.  The first version of this function compared whole
    tokens against the bare name, matched nothing, and stripped nothing; the
    arm would have been byte-identical to its comparator and "the wider band
    changes nothing" would have been read off an edit that never happened.  The
    protocol prefix is kept and the FLAG part is what is matched, here and in
    the check below -- one rule, so the check cannot pass a strip it disagrees
    with.  Every protocol's copy goes: the band moved for all of them.
    """
    if "qc_flags" not in df.columns:
        return df, 0
    regime = df["sampling_regime"] if "sampling_regime" in df else df.get("regime")
    hit = np.array([
        (experiment, model_id, str(r), int(rep), int(sid)) in keys
        for r, rep, sid in zip(regime, df["replicate"],
                               df["parameter_sample_id"], strict=True)])
    if not hit.any():
        return df, 0
    is_band = lambda tok: tok.rsplit(":", 1)[-1] == BAND_FLAG      # noqa: E731
    out = df.copy()
    flags = out.loc[hit, "qc_flags"].astype("string").fillna("")
    out.loc[hit, "qc_flags"] = (
        flags.str.split(";")
             .apply(lambda xs: ";".join(x for x in xs if x and not is_band(x))))
    still = int(out.loc[hit, "qc_flags"].apply(
        lambda s: any(is_band(x) for x in str(s).split(";") if x)).sum())
    if still:
        raise SystemExit(f"{model_id}: {still} re-admitted rows still carry "
                         f"{BAND_FLAG}; the strip did not apply")
    return out, int(hit.sum())


def family_defined(model_id: str, feature_names: list[str], rec: dict | None = None) -> bool:
    """Is the protocol family of ``feature_names`` defined for the canonical model?

    Rheobase-scaled features (``step_rheo_*``) are undefined when the canonical
    rheobase is 0 (a spontaneously active model; preregistration section 3).  The
    same rule is applied to every draw by ``normalised_family_defined``; applying
    it to the target too stops draws stimulated at a multiple of their own
    rheobase from being matched against spontaneous firing at 0 nA.
    """
    from compass.workflows.simulate import normalised_family_defined

    if not any(f.startswith("step_rheo_") for f in feature_names):
        return True
    if rec is None:
        path = ROOT / "results" / "manifests" / f"canonical_{model_id}.json"
        if not path.exists():
            return False
        rec = json.loads(path.read_text())
    return bool(normalised_family_defined(rec.get("rheobase_nA")))


SELECTION_DERIVED = ("E3", "E3b", "E4", "E4b", "E3m", "E7", "E8", "E9")


def drop_undefined_family_rows(df):
    """Remove stored rows selected against an undefined canonical target.

    Tables written before ``family_defined`` existed contain the
    rheobase-normalised selections of khaliq_purkinje (canonical rheobase 0),
    which matched draws stimulated at 3x their own rheobase against spontaneous
    firing at 0 nA.  Unselected regimes (E1/E2/E6) do not depend on the target
    and are kept.
    """
    if "protocol_family" not in df.columns:
        return df
    bad_models = {m for m in df["model_id"].unique()
                  if not family_defined(m, ["step_rheo_3p0x.spike_count"])}
    bad = ((df["protocol_family"] == "normalised") & df["regime"].isin(SELECTION_DERIVED)
           & df["model_id"].isin(bad_models))
    return df[~bad].reset_index(drop=True)


def canonical_target(model_id: str, feature_names: list[str]) -> np.ndarray | None:
    path = ROOT / "results" / "manifests" / f"canonical_{model_id}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if not family_defined(model_id, feature_names, rec):
        return None
    feats = rec["canonical_features"]
    out = []
    for name in feature_names:
        if name == "rheobase_nA":
            out.append(rec.get("rheobase_nA"))
        else:
            out.append(feats.get(name))
    arr = np.array([np.nan if v is None else float(v) for v in out], dtype=float)
    return arr if np.isfinite(arr).sum() >= len(arr) - 2 else None


def analyse_ensemble(ens, estimator: str, seed: int) -> tuple[dict, list[dict], list[dict]]:
    """Geometry + order curve + per-tuple scores for one ensemble."""
    copula = estimator == "copula"
    x = ens.params
    geo = describe_geometry(x)
    oc = order_curve(x, orders=ORDERS, max_tuples=MAX_TUPLES, seed=seed,
                     copula=copula, bias_correct=True, n_surrogate=N_SURROGATE)
    syn_area, red_area = oc.normalised_areas()

    row = {
        "model_id": ens.model_id,
        "regime": ens.regime,
        "label": ens.label,
        "estimator": estimator,
        "tolerance": ens.tolerance,
        "metric": ens.metric,
        "source_regime": ens.source_regime,
        "protocol_family": ens.meta.get("protocol_family", "absolute"),
        "n_samples": ens.n,
        "n_params": len(ens.param_ids),
        "synergy_area": syn_area,
        "redundancy_area": red_area,
        "net_o_information": red_area - syn_area,
        "acceptance_rate": ens.meta.get("acceptance_rate", np.nan),
    }
    # E7's achieved marginal balance travels WITH the endpoint.
    #
    # `high_covariance_subset` enforces max_mean_shift_sd <= 0.10 SD but relaxes
    # the constraint when no candidate satisfies it, and the stored endpoint
    # table carried no trace of which of those two happened. The absolute
    # migliore_ca1 cell is a relaxed one at ~0.131 SD, and nothing downstream
    # could see it (post-response audit, section 4). Recorded per row, so any
    # consumer can filter on `e7_balance_satisfied` instead of trusting that the
    # constraint held.
    for k in ("max_mean_shift_sd", "mean_mean_shift_sd", "max_ks",
              "min_sd_ratio", "max_sd_ratio"):
        if k in ens.meta:
            row[f"e7_{k}"] = float(ens.meta[k])
    if "max_mean_shift_sd" in ens.meta:
        tol = float(ens.meta.get("balance_tol_sd", BALANCE_TOL_SD))
        row["e7_balance_tol_sd"] = tol
        row["e7_balance_satisfied"] = bool(
            float(ens.meta["max_mean_shift_sd"]) <= tol)
    for i, o in enumerate(oc.orders):
        row[f"frac_synergy_o{o}"] = oc.frac_synergy[i]
        row[f"frac_synergy_sig_o{o}"] = oc.frac_synergy_sig[i]
        row[f"frac_redundancy_sig_o{o}"] = oc.frac_redundancy_sig[i]
        row[f"median_omega_o{o}"] = oc.median[i]
    row.update(geo.as_dict())

    curves = []
    for i, o in enumerate(oc.orders):
        curves.append({
            "model_id": ens.model_id, "regime": ens.regime, "label": ens.label,
            "estimator": estimator,
            "protocol_family": ens.meta.get("protocol_family", "absolute"),
            "metric": ens.metric, "order": o, "n_tuples": oc.n_tuples[i],
            "n_possible": oc.n_possible[i], "median": oc.median[i],
            "q05": oc.q05[i], "q95": oc.q95[i],
            "frac_synergy": oc.frac_synergy[i], "frac_redundancy": oc.frac_redundancy[i],
            "frac_synergy_sig": oc.frac_synergy_sig[i],
            "frac_redundancy_sig": oc.frac_redundancy_sig[i],
            "mean_negative": oc.mean_negative[i], "mean_positive": oc.mean_positive[i],
            "area_negative": oc.area_negative[i], "area_positive": oc.area_positive[i],
            "surrogate_lo": oc.surrogate_lo[i], "surrogate_hi": oc.surrogate_hi[i],
            "n_samples": ens.n,
        })

    # per-tuple scores at order 3 for module-stability analysis
    eng = GaussianInfoEngine(x, copula=copula, bias_correct=True)
    ts = generate_tuples(len(ens.param_ids), 3, max_tuples=MAX_TUPLES, seed=seed)
    vals = eng.o_information_batch(ts.tuples)
    scores = [{
        "model_id": ens.model_id, "regime": ens.regime, "label": ens.label,
        "estimator": estimator,
        "protocol_family": ens.meta.get("protocol_family", "absolute"),
        "metric": ens.metric,
        "tuple": "|".join(ens.param_ids[i] for i in t),
        "tuple_idx": "-".join(map(str, t)),
        "o_information": float(v), "n_samples": ens.n,
    } for t, v in zip(ts.tuples, vals, strict=True) if np.isfinite(v)]
    return row, curves, scores


def analyse_rsi(ens, df_accepted: pd.DataFrame, estimator: str, seed: int) -> list[dict]:
    """RSI of order-3 parameter tuples about each electrical feature."""
    copula = estimator == "copula"
    out = []
    d = len(ens.param_ids)
    ts = generate_tuples(d, 3, max_tuples=400, seed=seed).tuples
    for target in RSI_TARGETS:
        if target not in df_accepted.columns:
            continue
        y = df_accepted[target].to_numpy(dtype=float)
        good = np.isfinite(y)
        if good.sum() < 100:
            continue
        xy = np.column_stack([ens.params[good], y[good]])
        eng = GaussianInfoEngine(xy, copula=copula, bias_correct=True)
        vals = np.array([eng.rsi(t, d).value for t in ts])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out.append({
            "model_id": ens.model_id, "regime": ens.regime, "label": ens.label,
            "estimator": estimator, "target": target,
            "n_samples": int(good.sum()), "n_tuples": int(vals.size),
            "median_rsi": float(np.median(vals)),
            "frac_synergistic": float(np.mean(vals > 0)),
            "mean_rsi": float(np.mean(vals)),
            "q95_rsi": float(np.quantile(vals, 0.95)),
            "q05_rsi": float(np.quantile(vals, 0.05)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY))
    ap.add_argument("--estimators", nargs="*", default=["gaussian", "copula"])
    ap.add_argument("--qc-policy", default="primary",
                    help="scientific-validity policy for QC flags (primary|strict|none)")
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help="conditioning features to remove from the phenotype for "
                         "EVERY model before ensembles are built. Used by the "
                         "drop_failed_feature sensitivity arm; the phenotype is "
                         "fixed at ensemble construction, so this cannot be "
                         "applied by filtering endpoints afterwards.")
    ap.add_argument("--drop-unidentifiable-parameters", action="store_true",
                    help="remove the parameters gate N4 measured as "
                         "unidentifiable (data/provenance/parameter_"
                         "identifiability.json) from the sampled space before "
                         "ensembles are built. Projects the N4-failing models "
                         "onto their observable dimensions instead of excluding "
                         "them; used by the n4_projected sensitivity arm.")
    ap.add_argument("--readmit-keys", default=None, metavar="JSON",
                    help="re-admit the rows listed in this key file by removing "
                         "the `nonphysical_voltage` flag they gained when the "
                         "corpus and waveform voltage bands were unified at the "
                         "tighter pair. Used by the wide_voltage_band arm to "
                         "MEASURE that choice instead of asserting it; the file "
                         "is produced by tools/reconcile_corpus.py --write-keys.")
    ap.add_argument("--suffix", default="",
                    help="extra output suffix, e.g. '_dropfeat', so a deviating "
                         "analysis never overwrites the primary tables")
    args = ap.parse_args()

    # Rows to re-admit, keyed exactly as the corpus snapshots are keyed.
    readmit: set[tuple] = set()
    if args.readmit_keys:
        rec = json.loads(Path(args.readmit_keys).read_text())
        if rec.get("criterion") != "left_the_analysis_set":
            ap.error(f"--readmit-keys expects a key file for criterion "
                     f"'left_the_analysis_set', got {rec.get('criterion')!r}")
        readmit = {(r["experiment"], r["model_id"], r["regime"],
                    int(r["replicate"]), int(r["parameter_sample_id"]))
                   for r in rec["rows"]}
        if not readmit:
            ap.error("--readmit-keys names no rows. Refusing to run an arm that "
                     "would be identical to its comparator.")
        print(f"re-admitting {len(readmit)} rows excluded by the voltage-band "
              f"unification ({rec['by_model']})", flush=True)

    # Which parameters to project out, per model.  Empty unless asked for, so
    # the primary analysis is untouched.
    drop_params: dict[str, list[str]] = {}
    if args.drop_unidentifiable_parameters:
        from compass.analysis.cohorts import UNIDENTIFIABLE_PARAMETERS

        drop_params = dict(UNIDENTIFIABLE_PARAMETERS)
        if not drop_params:
            ap.error("--drop-unidentifiable-parameters was requested but the "
                     "identifiability record lists none. Refusing to run an arm "
                     "that would be identical to its comparator.")
        print(f"projecting out N4-unidentifiable parameters: {drop_params}",
              flush=True)

    drop = set(args.drop_features)
    if drop:
        known = set(CONDITIONING_FEATURES) | set(CONDITIONING_FEATURES_NORMALISED)
        unknown = sorted(drop - known)
        if unknown:
            ap.error(f"--drop-features names features that are not in the "
                     f"conditioning phenotype: {unknown}. Refusing to run: a "
                     f"silently ignored drop would produce an arm identical to "
                     f"its comparator while claiming to differ from it.")
        print(f"dropping from the conditioning phenotype: {sorted(drop)}", flush=True)

    endpoints, curves, scores, rsis = [], [], [], []
    readmitted: dict[str, int] = {}
    for model_id in args.models:
        df = load_model_table(args.experiment, model_id)
        if df is None or len(df) < 100:
            print(f"skip {model_id}: no data", flush=True)
            continue
        if readmit:
            df, k = readmit_band_tightened(df, args.experiment, model_id, readmit)
            if k:
                readmitted[model_id] = k
                print(f"  {model_id}: re-admitted {k} rows", flush=True)
        specs = get_adapter(model_id).parameter_specs()
        param_ids = [s.parameter_id for s in specs]
        dropped_params: list[str] = []
        if model_id in drop_params:
            want = set(drop_params[model_id])
            missing = sorted(want - set(param_ids))
            if missing:
                raise SystemExit(
                    f"{model_id}: identifiability record names {missing}, which "
                    f"the adapter does not expose. The record and the adapter "
                    f"have drifted; refusing to run a projection that silently "
                    f"drops nothing.")
            specs = [s for s in specs if s.parameter_id not in want]
            param_ids = [s.parameter_id for s in specs]
            dropped_params = sorted(want)
            print(f"  {model_id}: projected out {dropped_params} "
                  f"({len(param_ids)} identifiable parameters remain)", flush=True)
        tier = specs[0].evidence_tier.value
        prior_width = float(np.mean([s.log_width for s in specs]))

        for family, feats in (("absolute", CONDITIONING_FEATURES),
                              ("normalised", CONDITIONING_FEATURES_NORMALISED)):
            feats = [f for f in feats if f in df.columns and f not in drop]
            if len(feats) < 6:
                continue
            target = canonical_target(model_id, feats)
            if target is None:
                # undefined family for this model (or no canonical record):
                # never fall back to the median of the draws
                print(f"  {model_id}: {family} family undefined for the canonical "
                      f"model; skipped", flush=True)
                continue
            ens_list = build_ensembles(df, model_id, param_ids, feats, target=target,
                                       seed=SEED, protocol_family=family,
                                       qc_policy=args.qc_policy)
            # Must use the SAME validity mask build_ensembles applied, otherwise
            # the boolean `accepted` mask (length = post-QC rows) does not align
            # with this frame and RSI indexing raises.
            base_ok = df[validity_mask(df, args.qc_policy)].reset_index(drop=True)
            e1_rows = base_ok[base_ok["sampling_regime"] == "E1"].reset_index(drop=True)
            for ens in ens_list:
                for est in args.estimators:
                    row, cs, sc = analyse_ensemble(ens, est, SEED)
                    row["evidence_tier"] = tier
                    row["qc_policy"] = args.qc_policy
                    row["mean_prior_width_decades"] = prior_width
                    row["n_parameters_analysed"] = len(param_ids)
                    row["parameters_projected_out"] = ";".join(dropped_params)
                    endpoints.append(row)
                    curves.extend(cs)
                    scores.extend(sc)
                    # RSI only on ensembles drawn from E1 rows (aligned indices)
                    if ens.regime in ("E1", "E3", "E3b", "E4") and ens.n == len(ens.params):
                        if ens.regime == "E1" and ens.n == len(e1_rows):
                            rsis.extend(analyse_rsi(ens, e1_rows, est, SEED))
                        elif ens.conditioning is not None:
                            sub = e1_rows[ens.conditioning.accepted].reset_index(drop=True)
                            if len(sub) == ens.n:
                                rsis.extend(analyse_rsi(ens, sub, est, SEED))
        print(f"{model_id:16s} ensembles={len(ens_list)} rows={len(endpoints)}", flush=True)

    # A SENSITIVITY ARM THAT CHANGED NOTHING IS NOT A PASSING ARM.
    #
    # The key file spans every campaign, so a run on one campaign re-admits only
    # that campaign's share of it -- but if the share is ZERO the arm's tables
    # are byte-identical to its comparator's, and "the wider band changes
    # nothing" would be read off a join that never matched. That is the same
    # confusion as counting an unmeasured gate as a passing one, so it fails
    # here instead of being reported as a null result. The counts are written
    # beside the tables so the arm's claim is checkable without rerunning it.
    if readmit:
        n_readmitted = sum(readmitted.values())
        if not n_readmitted:
            keyed_models = sorted({k[1] for k in readmit})
            raise SystemExit(
                f"--readmit-keys named {len(readmit)} rows but NONE of them "
                f"landed in experiment '{args.experiment}' over models "
                f"{args.models}. The key file's rows are in {keyed_models}. "
                f"This arm would be identical to its comparator; refusing to "
                f"write it as a measured result.")
        rec = {
            "generated_utc": datetime.now(UTC).isoformat(),
            "arm": f"{args.experiment}{args.suffix}",
            "key_file": os.path.relpath(Path(args.readmit_keys).resolve(), ROOT),
            "criterion": "left_the_analysis_set",
            "n_rows_in_key_file": len(readmit),
            "n_rows_readmitted_here": n_readmitted,
            "by_model": dict(sorted(readmitted.items())),
            "_doc": ("rows re-admitted by removing the `nonphysical_voltage` "
                     "flag the voltage-band unification gave them; the key file "
                     "spans every campaign, this arm covers one"),
        }
        out = ROOT / "data" / "provenance" / "voltage_band_arm.json"
        out.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"re-admitted {n_readmitted}/{len(readmit)} keyed rows "
              f"({rec['by_model']}) -> {out.relative_to(ROOT)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # SEPARATE NAMESPACE per QC policy (2026-08-06 audit, M1).
    #
    # `--qc-policy strict` used to write the primary filenames, so running the
    # sensitivity analysis silently destroyed the primary result and there was
    # no way to compare the two.  A non-primary policy now gets its own suffix,
    # and primary artifacts can never be overwritten by a sensitivity run.
    tag = ("" if args.qc_policy == "primary" else f"_qc_{args.qc_policy}") + args.suffix
    for name, rows in (("endpoints", endpoints), ("order_curves", curves),
                       ("tuple_scores", scores), ("rsi", rsis)):
        pd.DataFrame(rows).to_parquet(
            OUT / f"{name}_{args.experiment}{tag}.parquet", index=False)
    print(f"\nendpoints={len(endpoints)} curves={len(curves)} "
          f"tuples={len(scores)} rsi={len(rsis)} "
          f"[qc_policy={args.qc_policy}] -> {OUT}/*_{args.experiment}{tag}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
