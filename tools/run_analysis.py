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
    CONDITIONING_FEATURES,
    CONDITIONING_FEATURES_NORMALISED,
    build_ensembles,
    validity_mask,  # noqa: E402
)

OUT = ROOT / "results" / "analyses"
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


def canonical_target(model_id: str, feature_names: list[str]) -> np.ndarray | None:
    path = ROOT / "results" / "manifests" / f"canonical_{model_id}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
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
    args = ap.parse_args()

    endpoints, curves, scores, rsis = [], [], [], []
    for model_id in args.models:
        df = load_model_table(args.experiment, model_id)
        if df is None or len(df) < 100:
            print(f"skip {model_id}: no data", flush=True)
            continue
        specs = get_adapter(model_id).parameter_specs()
        param_ids = [s.parameter_id for s in specs]
        tier = specs[0].evidence_tier.value
        prior_width = float(np.mean([s.log_width for s in specs]))

        for family, feats in (("absolute", CONDITIONING_FEATURES),
                              ("normalised", CONDITIONING_FEATURES_NORMALISED)):
            feats = [f for f in feats if f in df.columns]
            if len(feats) < 6:
                continue
            target = canonical_target(model_id, feats)
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

    OUT.mkdir(parents=True, exist_ok=True)
    # SEPARATE NAMESPACE per QC policy (2026-08-06 audit, M1).
    #
    # `--qc-policy strict` used to write the primary filenames, so running the
    # sensitivity analysis silently destroyed the primary result and there was
    # no way to compare the two.  A non-primary policy now gets its own suffix,
    # and primary artifacts can never be overwritten by a sensitivity run.
    tag = "" if args.qc_policy == "primary" else f"_qc_{args.qc_policy}"
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
