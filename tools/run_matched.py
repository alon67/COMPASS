"""Size-matched cross-model endpoint recomputation (blueprint 26; review F4).

The screening campaign deliberately gives cheap point models ~32x more samples
than the expensive reconstructions (8192 vs 256).  Every information estimator
has sample-size-dependent variance, so a cross-model comparison run on the raw
endpoint table confounds "model identity" with "how many samples that model
got".  The preregistration and the campaign comments both promised a
size-matched recomputation; this tool is that recomputation.

Method
------
For each (model, regime, estimator, protocol family) ensemble:

1. take ``n_match`` = the smallest ensemble size available for that regime
   across all models (so every model contributes the same N);
2. draw ``n_draws`` independent random subsamples of that size;
3. recompute the endpoints on each draw.

The output therefore carries the *resampling* variability explicitly, and any
cross-model statistic can be computed per draw and reported with a spread rather
than as a single number.

Surrogate thresholds are not recomputed per draw (they are a significance
device, not an endpoint, and cost ~25 extra order curves each); the matched
table reports the endpoint quantities only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.analysis.cohorts import PRIMARY_EXCLUSIONS  # noqa: E402
from compass.analysis.geometry import describe_geometry  # noqa: E402
from compass.analysis.tuples import order_curve  # noqa: E402
from compass.conditioning.ensembles import (  # noqa: E402
    CONDITIONING_FEATURES,
    CONDITIONING_FEATURES_NORMALISED,
    build_ensembles,
)

OUT = ROOT / "results" / "analyses"
ORDERS = (3, 4, 5)
MAX_TUPLES = 400
N_DRAWS = 20
MIN_MATCH_N = 120          # below this an order-5 estimate is not worth reporting
BASE_SEED = 20260805


def endpoints_for(x: np.ndarray, estimator: str, seed: int) -> dict:
    """Endpoint quantities for one parameter matrix (no surrogate thresholds)."""
    oc = order_curve(x, orders=ORDERS, max_tuples=MAX_TUPLES, seed=seed,
                     copula=(estimator == "copula"), bias_correct=True,
                     n_surrogate=0)
    syn, red = oc.normalised_areas()
    geo = describe_geometry(x)
    row = {
        "synergy_area": syn, "redundancy_area": red,
        "net_o_information": red - syn,
        "n_samples": len(x), "n_params": x.shape[1],
    }
    for i, o in enumerate(oc.orders):
        row[f"frac_synergy_o{o}"] = oc.frac_synergy[i]
        row[f"median_omega_o{o}"] = oc.median[i]
    row.update({k: v for k, v in geo.as_dict().items()
                if k in ("mean_abs_pearson", "effective_rank", "intrinsic_dimension",
                         "participation_ratio", "pc1_variance_fraction")})
    return row


def _analysis_helpers():
    """Reuse run_analysis.py's table loader and canonical-target reader."""
    import importlib.util as iu

    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_model_table, mod.canonical_target


def collect(experiments: list[str], models: list[str], qc_policy: str) -> dict:
    """Gather every ensemble of every model, keyed by (regime, family, metric).

    Rows are pooled across ``experiments`` before ensembles are built.  An
    expansion campaign writes to its own shard namespace so the frozen
    screening campaign is never mutated, which means its extra draws reach a
    matched design only if they are concatenated here.

    Pooling is a concatenation and must never be a merge or a de-duplication on
    ``parameter_sample_id``: that column is not a global key.  screening and
    screening_expand draw migliore_ca1 from different seeds (20262703 vs
    20262589), so id *n* denotes a different parameter vector in each campaign
    (verified: max |delta param| = 3.7 across the 256 shared ids).  Treating the
    ids as shared would silently discard 256 independent draws.
    """
    load_model_table, canonical_target = _analysis_helpers()
    store: dict = {}
    for mid in models:
        frames = [t for t in (load_model_table(e, mid) for e in experiments)
                  if t is not None]
        if not frames:
            continue
        df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        specs = get_adapter(mid).parameter_specs()
        pids = [s.parameter_id for s in specs]
        for family, feats in (("absolute", CONDITIONING_FEATURES),
                              ("normalised", CONDITIONING_FEATURES_NORMALISED)):
            target = canonical_target(mid, feats)
            if target is None:
                continue
            for ens in build_ensembles(df, mid, pids, feats, target=target,
                                       seed=1, protocol_family=family,
                                       qc_policy=qc_policy):
                key = (ens.regime, family, ens.metric, ens.tolerance)
                store.setdefault(key, {})[mid] = ens
    return store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--pool", nargs="*", default=[],
                    help="additional experiments concatenated onto --experiment "
                         "before ensembles are built (e.g. screening_expand). "
                         "Output filenames still follow --experiment.")
    ap.add_argument("--keep-gate-failures", action="store_true",
                    help="retain models that fail numerical acceptance "
                         "(compass.analysis.cohorts.PRIMARY_EXCLUSIONS) in the "
                         "matched design. Off by default: n_match is a min() "
                         "across models, so one gate-failed model silently "
                         "binds every cell it appears in.")
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--qc-policy", default="primary")
    ap.add_argument("--estimators", nargs="*", default=["gaussian"])
    args = ap.parse_args()

    models = list(args.models)
    if not args.keep_gate_failures:
        dropped = sorted(m for m in models if m in PRIMARY_EXCLUSIONS)
        models = [m for m in models if m not in PRIMARY_EXCLUSIONS]
        if dropped:
            print(f"excluded on numerical-acceptance failure (gate N3): {dropped}",
                  flush=True)
    experiments = [args.experiment, *args.pool]
    if args.pool:
        print(f"pooling rows across experiments: {experiments}", flush=True)
    store = collect(experiments, models, args.qc_policy)
    rows, summary = [], []
    for key, per_model in sorted(store.items(), key=lambda kv: str(kv[0])):
        regime, family, metric, tolerance = key
        if len(per_model) < 2:
            continue
        n_match = min(e.n for e in per_model.values())
        if n_match < MIN_MATCH_N:
            summary.append({"regime": regime, "protocol_family": family,
                            "metric": metric, "tolerance": tolerance,
                            "n_models": len(per_model), "n_match": n_match,
                            "skipped": "below MIN_MATCH_N"})
            continue
        summary.append({"regime": regime, "protocol_family": family,
                        "metric": metric, "tolerance": tolerance,
                        "n_models": len(per_model), "n_match": int(n_match),
                        "skipped": ""})
        for mid, ens in sorted(per_model.items()):
            for estimator in args.estimators:
                for draw in range(args.draws):
                    # Same draw index -> same seed for every model, so a draw
                    # is a paired comparison across models rather than an
                    # independent one.
                    rng = np.random.default_rng(BASE_SEED + 7919 * draw)
                    idx = rng.choice(ens.n, size=n_match, replace=False)
                    r = endpoints_for(ens.params[idx], estimator,
                                      seed=BASE_SEED + draw)
                    r.update({"model_id": mid, "regime": regime,
                              "protocol_family": family, "metric": metric,
                              "tolerance": tolerance, "estimator": estimator,
                              "draw": draw, "n_match": int(n_match),
                              "n_available": ens.n, "qc_policy": args.qc_policy})
                    rows.append(r)
        print(f"  {regime:4s} {family:10s} {metric:18s} {tolerance:11s} "
              f"models={len(per_model)} N={n_match}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"endpoints_matched_{args.experiment}.parquet", index=False)
    # Carry the design's own provenance: which campaigns were pooled and which
    # models were withheld.  n_match is a min() across models, so a cell's value
    # is unreadable without knowing the membership that produced it.
    design = pd.DataFrame(summary)
    design["pooled_experiments"] = "|".join(experiments)
    design["excluded_models"] = "|".join(sorted(set(args.models) - set(models)))
    design.to_parquet(
        OUT / f"matched_design_{args.experiment}.parquet", index=False)
    print(f"\nwrote {len(df)} matched endpoint rows "
          f"({df['model_id'].nunique() if len(df) else 0} models, "
          f"{args.draws} draws) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
