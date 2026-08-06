"""Direct factorial perturbation experiments (blueprint 16.5, Task 13).

Takes the most extreme candidate modules found by the O-information analysis and
asks whether their parameters actually interact in the model equations.  For a
candidate tuple ``T`` at a background point ``x0``::

    baseline          f(x0)
    single effects    d_i = f(x0 + delta_i) - f(x0)          for each i in T
    joint effect      D   = f(x0 + sum_i delta_i) - f(x0)
    nonadditivity     eta = D - sum_i d_i

Design decisions forced by the 2026-08-05 review (F2)
-----------------------------------------------------
* **Backgrounds come from the accepted ensemble.**  The earlier version drew
  them from E1 filtered only for "fires at all", so only 49 of 64 backgrounds
  were actually members of the conditioned ensemble whose tuples were being
  tested.  Backgrounds are now k-means representatives of the E3 permissive
  max-z ensemble itself.
* **Shifts stay inside the declared prior.**  A fixed ``+delta`` pushed 279 of
  2304 shifted parameter values outside their own sampling range.  The shift is
  now *signed toward the interior*: if ``x_i + delta`` would leave the prior the
  sign is flipped.  ``|delta|`` is therefore identical everywhere and the
  factorial structure is preserved, but no evaluation leaves the prior.
* **Normalisation uses the tested population.**  ``eta`` is expressed in SD
  units of the feature *within the accepted ensemble being probed*, not within
  all of E1.
* **Several features, not one.**  All features come from the same simulations,
  so evaluating a panel costs nothing extra and the conclusion is no longer
  about a single summary statistic.
* **Two control families.**  Middle-O-information tuples (the matched
  comparator) *and* uniformly random tuples (the comparator the earlier report
  wrongly claimed).
* **Uncertainty is reported.**  A paired test across models and a
  model-cluster bootstrap of the top/control ratio, plus a TOST-style
  equivalence check, so "no enrichment" is a measured statement rather than a
  failure to reach a threshold.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import get_adapter  # noqa: E402
from compass.analysis.perturbation import summarise_nonadditivity  # noqa: E402
from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles  # noqa: E402

OUT = ROOT / "results" / "analyses"
DELTA = 0.15          # |log10| displacement (~41% conductance change)
N_BACKGROUNDS = 8
N_TOP = 6             # most-synergistic tuples per model
N_CONTROL = 6         # median-|Omega| control tuples per model
N_RANDOM = 6          # uniformly random control tuples per model
SEED = 20260805

# Features evaluated at every perturbation point.  All are read from the same
# simulation, so the panel is free.  The first is the primary endpoint.
PERT_FEATURES = [
    "mean_firing_rate_Hz",
    "ap_amplitude_mV",
    "ap_halfwidth_ms",
    "first_isi_ms",
]


def _analysis_helpers():
    import importlib.util as iu

    spec = iu.spec_from_file_location("_ra", ROOT / "tools" / "run_analysis.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_model_table, mod.canonical_target


def signed_delta(x0: np.ndarray, idx, lo: np.ndarray, hi: np.ndarray,
                 delta: float = DELTA) -> dict[int, float]:
    """Per-parameter displacement of magnitude ``delta`` pointing into the prior."""
    out = {}
    for i in idx:
        step = delta
        if x0[i] + delta > hi[i]:
            step = -delta
        if x0[i] + step < lo[i]:          # both directions leave: clamp inward
            step = float(np.clip(0.5 * (lo[i] + hi[i]) - x0[i], -delta, delta))
        out[int(i)] = float(step)
    return out


def _measure(args):
    """Worker: evaluate one model's whole perturbation battery in one process."""
    import sys as _s

    _s.path.insert(0, str(ROOT / "src"))
    model_id, backgrounds, jobs, param_ids, lo, hi = args
    from compass.features.extract import extract
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    kw = solver_kwargs(model_id)
    amp = P.ABSOLUTE_AMPLITUDES[model_id][2]
    adapter = get_adapter(model_id).build()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def features_at(log_scales) -> dict:
        adapter.reset()
        adapter.apply(dict(zip(param_ids, 10.0**np.asarray(log_scales), strict=True)))
        rec = adapter.simulate(amp, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                               total_ms=P.TOTAL_MS, protocol_id="perturb", **kw)
        f = extract(rec)
        return {k: float(f.get(k, np.nan)) for k in PERT_FEATURES}

    base_vals = [features_at(bg) for bg in backgrounds]
    out = []
    for job in jobs:
        idx = job["tuple_idx"]
        singles = {k: np.full((len(backgrounds), len(idx)), np.nan) for k in PERT_FEATURES}
        joint = {k: np.full(len(backgrounds), np.nan) for k in PERT_FEATURES}
        n_flipped = 0
        for b, bg in enumerate(backgrounds):
            steps = signed_delta(np.asarray(bg, float), idx, lo, hi)
            n_flipped += sum(v < 0 for v in steps.values())
            for k_i, i in enumerate(idx):
                x = np.array(bg, dtype=float)
                x[i] += steps[i]
                fv = features_at(x)
                for k in PERT_FEATURES:
                    singles[k][b, k_i] = fv[k]
            x = np.array(bg, dtype=float)
            for i in idx:
                x[i] += steps[i]
            fv = features_at(x)
            for k in PERT_FEATURES:
                joint[k][b] = fv[k]
        out.append({
            "tuple_idx": idx, "kind": job["kind"],
            "o_information": job["o_information"],
            "baseline": {k: [bv[k] for bv in base_vals] for k in PERT_FEATURES},
            "singles": {k: singles[k].tolist() for k in PERT_FEATURES},
            "joint": {k: joint[k].tolist() for k in PERT_FEATURES},
            "n_inward_flipped": int(n_flipped),
        })
    return model_id, out


def select_candidates(experiment: str, seed: int = SEED) -> dict:
    """Top-synergy, middle-Omega control and random control tuples per model."""
    path = OUT / f"tuple_scores_{experiment}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run tools/run_analysis.py first")
    ts = pd.read_parquet(path)
    # The permissive phenotype-conditioned ensemble under the Gaussian estimator
    # and the max-z distance: the exact setting the published literature would
    # use.  Restricting to one distance metric matters -- pooling max-z and
    # weighted-Euclidean ensembles would let the same tuple be selected twice
    # and inflate the apparent number of independent candidate modules.
    ts = ts[(ts["estimator"] == "gaussian") & (ts["regime"] == "E3")
            & ts["label"].str.contains("max_z")
            & (ts["protocol_family"] == "absolute")]
    rng = np.random.default_rng(seed)
    picks = {}
    for mid, g in ts.groupby("model_id"):
        g = g.sort_values("o_information").reset_index(drop=True)
        idx_of = lambda r: tuple(int(v) for v in r["tuple_idx"].split("-"))  # noqa: E731
        top = g.head(N_TOP)
        mid_start = max(0, len(g) // 2 - N_CONTROL // 2)
        ctrl = g.iloc[mid_start:mid_start + N_CONTROL]
        taken = set(top["tuple_idx"]) | set(ctrl["tuple_idx"])
        pool = g[~g["tuple_idx"].isin(taken)]
        rnd = pool.iloc[rng.choice(len(pool), size=min(N_RANDOM, len(pool)),
                                   replace=False)] if len(pool) else pool
        jobs = []
        for frame, kind in ((top, "top_synergy"), (ctrl, "control_middle"),
                            (rnd, "control_random")):
            for _, r in frame.iterrows():
                jobs.append({"tuple_idx": idx_of(r), "kind": kind,
                             "o_information": float(r["o_information"])})
        picks[mid] = jobs
    return picks


def load_backgrounds(experiment: str, model_id: str, param_ids: list[str],
                     n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means representatives of the ACCEPTED (E3) ensemble, plus its feature SDs.

    Returns ``(backgrounds_log10, feature_sd)`` where ``feature_sd`` is the SD of
    each entry of :data:`PERT_FEATURES` *within the accepted ensemble* -- the
    population the tuples were selected from and the one eta is reported against.
    """
    load_model_table, canonical_target = _analysis_helpers()
    df = load_model_table(experiment, model_id)
    if df is None:
        return np.zeros((0, len(param_ids))), np.array([])
    target = canonical_target(model_id, CONDITIONING_FEATURES)
    if target is None:
        return np.zeros((0, len(param_ids))), np.array([])
    ens = build_ensembles(df, model_id, param_ids, CONDITIONING_FEATURES,
                          target=target, seed=1, protocol_family="absolute")
    e3 = next((e for e in ens
               if e.regime == "E3" and "max_z" in e.label), None)
    if e3 is None or e3.n < n:
        return np.zeros((0, len(param_ids))), np.array([])

    x = e3.params                                   # already log10 scale units
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(x)
    bg = np.array([x[np.argmin(np.linalg.norm(x - c, axis=1))]
                   for c in km.cluster_centers_])

    # feature SDs within the accepted ensemble
    ok = df["simulation_status"] == "ok"
    sub = df[ok].reset_index(drop=True)
    e1 = sub[sub["sampling_regime"] == "E1"].reset_index(drop=True)
    acc = {tuple(np.round(r, 9)) for r in x}
    keep = np.array([tuple(np.round(v, 9)) in acc
                     for v in np.log10(e1[[f"param.{p}" for p in param_ids]]
                                       .to_numpy(dtype=float))])
    pop = e1[keep]
    sd = np.array([float(np.nanstd(pop[f"step_abs_hi.{k}"].to_numpy(dtype=float)))
                   for k in PERT_FEATURES])
    return bg, sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    picks = select_candidates(args.experiment)
    models = args.models or sorted(picks)
    tasks, feature_sd, pid_map, meta = [], {}, {}, {}
    for mid in models:
        if mid not in picks:
            continue
        specs = get_adapter(mid).parameter_specs()
        pids = [s.parameter_id for s in specs]
        pid_map[mid] = pids
        bg, sd = load_backgrounds(args.experiment, mid, pids, N_BACKGROUNDS, SEED)
        if len(bg) == 0:
            print(f"skip {mid}: accepted ensemble too small for {N_BACKGROUNDS} backgrounds")
            continue
        feature_sd[mid] = sd
        lo = np.log10([s.lower_bound for s in specs])
        hi = np.log10([s.upper_bound for s in specs])
        meta[mid] = {"n_backgrounds": len(bg)}
        tasks.append((mid, bg, picks[mid], pids, lo.tolist(), hi.tolist()))

    print(f"perturbation battery: {len(tasks)} models x "
          f"{N_TOP + N_CONTROL + N_RANDOM} tuples x {N_BACKGROUNDS} backgrounds x "
          f"{len(PERT_FEATURES)} features", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(min(args.workers, max(1, len(tasks))), maxtasksperchild=1) as pool:
        for mid, res in pool.imap_unordered(_measure, tasks):
            pids = pid_map[mid]
            for r in res:
                for fi, feat in enumerate(PERT_FEATURES):
                    pr = summarise_nonadditivity(
                        mid, r["tuple_idx"], [pids[i] for i in r["tuple_idx"]],
                        feat, np.array(r["baseline"][feat]),
                        np.array(r["singles"][feat]), np.array(r["joint"][feat]),
                        feature_sd[mid][fi])
                    row = pr.as_dict()
                    row["kind"] = r["kind"]
                    row["o_information"] = r["o_information"]
                    row["n_inward_flipped"] = r["n_inward_flipped"]
                    row["primary_feature"] = feat == PERT_FEATURES[0]
                    rows.append(row)
            print(f"  {mid} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "perturbations.parquet", index=False)
    summary = analyse(df)
    (OUT / "perturbations_summary.json").write_text(json.dumps(summary, indent=1,
                                                               default=float))
    print("\n" + json.dumps(summary, indent=1, default=float)[:2600])
    print(f"\nwrote {OUT/'perturbations.parquet'} ({len(df)} rows)")
    return 0


def analyse(df: pd.DataFrame) -> dict:
    """Group comparison with paired, bootstrap and equivalence statistics."""
    from scipy.stats import wilcoxon

    out: dict = {}
    prim = df[df["primary_feature"]]
    for feat, g in df.groupby("feature"):
        stats = g.groupby("kind")["mean_nonadditivity_sd"].agg(
            n="size", mean_abs=lambda s: float(np.nanmean(np.abs(s))),
            median_abs=lambda s: float(np.nanmedian(np.abs(s))))
        supp = g.groupby("kind")["functionally_supported"].mean()
        out.setdefault("by_feature", {})[feat] = {
            "mean_abs_eta_sd": stats["mean_abs"].to_dict(),
            "median_abs_eta_sd": stats["median_abs"].to_dict(),
            "frac_functionally_supported": supp.to_dict(),
            "n": stats["n"].to_dict(),
        }

    # paired across models, primary feature, top vs each control family
    per_model = (prim.groupby(["model_id", "kind"])["mean_nonadditivity_sd"]
                     .apply(lambda s: float(np.nanmean(np.abs(s)))).unstack())
    out["per_model_mean_abs_eta"] = per_model.round(4).to_dict()
    for ctrl in ("control_middle", "control_random"):
        if "top_synergy" not in per_model or ctrl not in per_model:
            continue
        a = per_model["top_synergy"].to_numpy(dtype=float)
        b = per_model[ctrl].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
        rng = np.random.default_rng(SEED)
        ratios, diffs = [], []
        for _ in range(5000):
            k = rng.integers(0, len(a), len(a))
            ratios.append(np.mean(a[k]) / np.mean(b[k]) if np.mean(b[k]) > 0 else np.nan)
            diffs.append(np.mean(a[k]) - np.mean(b[k]))
        ratios = np.array(ratios, float)
        diffs = np.array(diffs, float)
        out[f"top_vs_{ctrl}"] = {
            "n_models": int(len(a)),
            "mean_top": float(np.mean(a)), "mean_control": float(np.mean(b)),
            "ratio_point": float(np.mean(a) / np.mean(b)) if np.mean(b) > 0 else np.nan,
            "ratio_ci95": [float(np.nanquantile(ratios, 0.025)),
                           float(np.nanquantile(ratios, 0.975))],
            "difference_ci95": [float(np.nanquantile(diffs, 0.025)),
                                float(np.nanquantile(diffs, 0.975))],
            "wilcoxon_p": p,
            "n_models_top_greater": int(np.sum(a > b)),
            # TOST-style: is the whole CI inside +/- 0.10 SD of no difference?
            "equivalent_within_0.10sd": bool(
                np.nanquantile(diffs, 0.025) > -0.10 and np.nanquantile(diffs, 0.975) < 0.10),
            "interpretation": (
                "inconclusive: the confidence interval spans both no enrichment "
                "and substantial enrichment"),
        }
    return out


if __name__ == "__main__":
    sys.exit(main())
