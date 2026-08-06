"""Uncertainty and balance diagnostics for the cross-model claims (review F6, F7).

Three things the headline cross-model numbers were missing.

1. **Cluster-aware uncertainty.**  The endpoint table has ~200 Gaussian rows but
   they are not independent experiments: E3/E4/E7/E8/E9 are all *derived* from
   the same E1 campaign, and E1/E2/E6 parameter O-information is identical
   across the two protocol families.  The inferential unit is the *model*, so
   the variance ratio is re-estimated with a bootstrap over models.

2. **All held-out folds.**  The reported leave-one-model-family-out R^2 was a
   median over eight folds, two of which are strongly negative.  Reporting the
   median alone hides that transfer fails badly for two families.

3. **E7 marginal balance.**  E7 is described as matched on marginals.  It is
   matched on size and on the target-distance distribution, but the per-parameter
   marginals are not constrained; this quantifies how far apart they actually are.
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

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.analysis.hierarchical import variance_partition  # noqa: E402
from compass.conditioning.ensembles import (  # noqa: E402
    CONDITIONING_FEATURES,
    build_ensembles,
)

OUT = ROOT / "results" / "analyses"
N_BOOT = 2000
SEED = 20260805

SAMPLING = ["regime", "tolerance", "metric", "protocol_family"]
GEOMETRY = ["mean_abs_pearson", "effective_rank", "intrinsic_dimension",
            "participation_ratio", "pc1_variance_fraction"]


def cluster_bootstrap_ratio(ep: pd.DataFrame, endpoint: str,
                            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap the sampling+geometry vs identity variance ratio over MODELS.

    Resampling models (not rows) respects the fact that all rows belonging to
    one model share a single simulated campaign.
    """
    models = sorted(ep["model_id"].unique())
    rng = np.random.default_rng(seed)
    ratios, samp_u, geo_u, id_u = [], [], [], []
    for _ in range(n_boot):
        pick = rng.choice(models, size=len(models), replace=True)
        # need >= 2 distinct models for identity to be estimable
        if len(set(pick)) < 2:
            continue
        parts = []
        for k, m in enumerate(pick):
            sub = ep[ep["model_id"] == m].copy()
            sub["model_id"] = f"{m}__{k}"        # distinct cluster per draw
            parts.append(sub)
        boot = pd.concat(parts, ignore_index=True)
        try:
            vp = variance_partition(boot, endpoint, {
                "sampling": SAMPLING,
                "geometry": GEOMETRY,
                "model_identity": ["model_id"],
            })
        except Exception:  # noqa: BLE001 - singular designs happen in resamples
            continue
        u = vp.set_index("block")["r2_unique"]
        if not {"sampling", "geometry", "model_identity"} <= set(u.index):
            continue
        s, g, i = float(u["sampling"]), float(u["geometry"]), float(u["model_identity"])
        samp_u.append(s)
        geo_u.append(g)
        id_u.append(i)
        ratios.append((s + g) / i if i > 1e-9 else np.inf)
    finite = [r for r in ratios if np.isfinite(r)]
    return {
        "endpoint": endpoint,
        "n_boot_effective": len(ratios),
        "n_infinite": len(ratios) - len(finite),
        "ratio_median": float(np.median(finite)) if finite else float("nan"),
        "ratio_q025": float(np.quantile(finite, 0.025)) if finite else float("nan"),
        "ratio_q975": float(np.quantile(finite, 0.975)) if finite else float("nan"),
        "ratio_gt_2_fraction": float(np.mean([r > 2 for r in ratios])) if ratios else float("nan"),
        "unique_sampling_median": float(np.median(samp_u)) if samp_u else float("nan"),
        "unique_geometry_median": float(np.median(geo_u)) if geo_u else float("nan"),
        "unique_identity_median": float(np.median(id_u)) if id_u else float("nan"),
    }


def e7_marginal_balance(experiment: str, models: list[str]) -> pd.DataFrame:
    """Per-parameter KS distance and SD ratio between E7 and its E3m comparator."""
    import importlib.util as iu

    from scipy.stats import ks_2samp

    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    ra = iu.module_from_spec(spec)
    spec.loader.exec_module(ra)

    rows = []
    for mid in models:
        df = ra.load_model_table(experiment, mid)
        if df is None:
            continue
        pids = [s.parameter_id for s in get_adapter(mid).parameter_specs()]
        target = ra.canonical_target(mid, CONDITIONING_FEATURES)
        if target is None:
            continue
        ens = build_ensembles(df, mid, pids, CONDITIONING_FEATURES, target=target,
                              seed=1, protocol_family="absolute")
        e7 = next((e for e in ens if e.regime == "E7"), None)
        e3m = next((e for e in ens if e.regime == "E3m"), None)
        if e7 is None or e3m is None:
            continue
        for j, pid in enumerate(pids):
            a, b = e7.params[:, j], e3m.params[:, j]
            rows.append({
                "model_id": mid, "parameter": pid,
                "ks_distance": float(ks_2samp(a, b).statistic),
                "ks_p": float(ks_2samp(a, b).pvalue),
                "sd_ratio": float(np.std(a) / np.std(b)) if np.std(b) > 0 else np.nan,
                "mean_shift_sd": float((np.mean(a) - np.mean(b)) / np.std(b))
                if np.std(b) > 0 else np.nan,
                "n_e7": e7.n, "n_e3m": e3m.n,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    out: dict = {}

    # ---- 1. cluster bootstrap on the raw and the size-matched tables -------- #
    raw = pd.read_parquet(OUT / f"endpoints_annotated_{args.experiment}.parquet")
    raw = raw[raw["estimator"] == "gaussian"]
    boot_raw = {e: cluster_bootstrap_ratio(raw, e, args.boot)
                for e in ("synergy_area", "frac_synergy_o3")}
    out["cluster_bootstrap_raw"] = boot_raw

    matched_path = OUT / f"endpoints_matched_{args.experiment}.parquet"
    if matched_path.exists():
        m = pd.read_parquet(matched_path)
        m = m[m["estimator"] == "gaussian"]
        # one row per (model, ensemble, draw); average draws so each model
        # contributes one row per ensemble, then bootstrap over models
        keys = ["model_id", "regime", "tolerance", "metric", "protocol_family"]
        agg = (m.groupby(keys, as_index=False)
                 [["synergy_area", "redundancy_area", "net_o_information",
                   "frac_synergy_o3", *GEOMETRY, "n_params", "n_samples"]].mean())
        out["cluster_bootstrap_matched"] = {
            e: cluster_bootstrap_ratio(agg, e, args.boot)
            for e in ("synergy_area", "frac_synergy_o3")
        }
        # draw-to-draw spread of the per-model endpoint
        spread = (m.groupby(keys)["net_o_information"].std().reset_index()
                    .rename(columns={"net_o_information": "sd_over_draws"}))
        out["matched_draw_spread"] = {
            "median_sd_over_draws": float(spread["sd_over_draws"].median()),
            "q95_sd_over_draws": float(spread["sd_over_draws"].quantile(0.95)),
        }

    # ---- 2. all held-out folds -------------------------------------------- #
    lomo = pd.read_parquet(OUT / "lomo.parquet")
    sub = lomo[(lomo["predictors"] == "sampling+geometry") &
               (lomo["endpoint"] == "synergy_area")]
    out["lomo_all_folds"] = {
        "folds": {r.held_out_model: round(float(r.r2_out_of_sample), 3)
                  for r in sub.itertuples()},
        "median": float(sub["r2_out_of_sample"].median()),
        "mean": float(sub["r2_out_of_sample"].mean()),
        "n_negative": int((sub["r2_out_of_sample"] < 0).sum()),
        "n_folds": len(sub),
    }

    # ---- 3. E7 marginal balance ------------------------------------------- #
    bal = e7_marginal_balance(args.experiment, list(MODEL_REGISTRY))
    if len(bal):
        bal.to_parquet(OUT / "e7_marginal_balance.parquet", index=False)
        out["e7_marginal_balance"] = {
            "n_parameters_compared": int(len(bal)),
            "mean_ks_distance": float(bal["ks_distance"].mean()),
            "max_ks_distance": float(bal["ks_distance"].max()),
            "frac_ks_significant_p05": float((bal["ks_p"] < 0.05).mean()),
            "max_sd_ratio": float(bal["sd_ratio"].max()),
            "min_sd_ratio": float(bal["sd_ratio"].min()),
            "max_abs_mean_shift_sd": float(bal["mean_shift_sd"].abs().max()),
            "per_model_mean_ks": bal.groupby("model_id")["ks_distance"].mean()
                                    .round(4).to_dict(),
        }

    (OUT / "robustness.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])
    print(f"\nwrote {OUT / 'robustness.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
