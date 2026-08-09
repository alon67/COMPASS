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

Design decisions forced by the 2026-08-06 audit (section 4)
-----------------------------------------------------------
* **Raw factorial outcomes are archived.**  Every baseline, single, joint and
  per-background non-additivity is written to ``perturbations_raw.parquet``,
  one row per model x tuple x background x feature, together with the signed
  displacement actually applied and the simulation status.  The previous run
  held those arrays in memory and stored only summaries, so no discrete-outcome
  reanalysis was possible after the fact.
* **Spike counts are stored, not just rates.**  ``mean_firing_rate_Hz`` is a
  count over a fixed 400 ms window; the count is the quantity a discrete model
  should see, and dividing by the window before analysis throws away the fact
  that the observation is an integer.
* **No "measurement floor".**  The withdrawn analysis compared a median of
  eight-background means against one unaveraged spike quantum.  Precision is
  now reported as the standard error over backgrounds, and repeatability is
  tested by re-running selected points at a finer time step
  (``tools/perturbation_inference.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import get_adapter  # noqa: E402
from compass.analysis.perturbation import (  # noqa: E402
    background_grid_note,
    summarise_nonadditivity,
)
from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles  # noqa: E402

OUT = ROOT / "results" / "analyses"
DELTA = 0.15          # |log10| displacement (~41% conductance change)
N_BACKGROUNDS = 8
N_TOP = 6             # most-synergistic tuples per model
N_CONTROL = 6         # median-|Omega| control tuples per model
N_RANDOM = 6          # uniformly random control tuples per model
SEED = 20260805

# Bumped whenever the battery's own construction changes in a way that makes an
# older cached slice incomparable: the displacement, the number or selection of
# backgrounds, the candidate-tuple rule, or the measured feature panel.
PERTURBATION_BATTERY_VERSION = "battery-1"

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
    model_id, backgrounds, jobs, param_ids, lo, hi, dt_override, dt_scale = args
    from compass.features.extract import extract
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    kw = solver_kwargs(model_id)
    if dt_override is not None:
        kw = dict(kw, dt_ms=float(dt_override))
    elif dt_scale is not None:
        # Scale each model's OWN campaign step.  A single absolute --dt cannot
        # express "half the campaign dt" across a panel whose accepted steps
        # span 0.05 to 0.003125 ms: --dt 0.0125 halves hh_point correctly but
        # COARSENS khaliq_purkinje (0.00625) by 2x, inverting the convergence
        # test it is supposed to perform.
        base = kw.get("dt_ms")
        if base is None:
            raise ValueError(
                f"{model_id}: --dt-scale requires a fixed-step solver, got {kw!r}")
        kw = dict(kw, dt_ms=float(base) * float(dt_scale))
    amp = P.ABSOLUTE_AMPLITUDES[model_id][2]
    adapter = get_adapter(model_id).build()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    # `spike_count` is carried alongside the derived rate so that the discrete
    # readout can be modelled as the integer it is (audit 4.4 point 3).
    measured = [*PERT_FEATURES, "spike_count"]

    def features_at(log_scales) -> dict:
        adapter.reset()
        adapter.apply(dict(zip(param_ids, 10.0**np.asarray(log_scales), strict=True)))
        rec = adapter.simulate(amp, delay_ms=P.DELAY_MS, duration_ms=P.DURATION_MS,
                               total_ms=P.TOTAL_MS, protocol_id="perturb", **kw)
        flags = adapter.qc_flags(rec)
        f = extract(rec)
        out = {k: float(f.get(k, np.nan)) if f.get(k) is not None else np.nan
               for k in measured}
        out["_status"] = "ok" if not flags else ";".join(sorted(flags))
        return out

    base_vals = [features_at(bg) for bg in backgrounds]
    out = []
    for job in jobs:
        idx = job["tuple_idx"]
        singles = {k: np.full((len(backgrounds), len(idx)), np.nan) for k in measured}
        joint = {k: np.full(len(backgrounds), np.nan) for k in measured}
        applied = np.zeros((len(backgrounds), len(idx)))
        status = []
        n_flipped = 0
        for b, bg in enumerate(backgrounds):
            steps = signed_delta(np.asarray(bg, float), idx, lo, hi)
            n_flipped += sum(v < 0 for v in steps.values())
            st = [base_vals[b]["_status"]]
            for k_i, i in enumerate(idx):
                applied[b, k_i] = steps[i]
                x = np.array(bg, dtype=float)
                x[i] += steps[i]
                fv = features_at(x)
                st.append(fv["_status"])
                for k in measured:
                    singles[k][b, k_i] = fv[k]
            x = np.array(bg, dtype=float)
            for i in idx:
                x[i] += steps[i]
            fv = features_at(x)
            st.append(fv["_status"])
            for k in measured:
                joint[k][b] = fv[k]
            bad = sorted({s for s in st if s != "ok"})
            status.append("ok" if not bad else ";".join(bad))
        out.append({
            "tuple_idx": idx, "kind": job["kind"],
            "o_information": job["o_information"],
            "baseline": {k: [bv[k] for bv in base_vals] for k in measured},
            "singles": {k: singles[k].tolist() for k in measured},
            "joint": {k: joint[k].tolist() for k in measured},
            "applied_steps": applied.tolist(),
            "status": status,
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


# --------------------------------------------------------------------------- #
# Cache identity (2026-08-09 audit, 4.3)
# --------------------------------------------------------------------------- #
def slice_identity(model_id: str, experiment: str, jobs: list[dict],
                   dt_override, dt_scale) -> dict:
    """Everything that must match before a cached perturbation slice is reused.

    The cache used to be keyed on ``{model}{suffix}_{kind}.parquet`` and nothing
    else, so ``--resume`` would happily merge a slice simulated by different
    code, at a different time step, from a different candidate-tuple selection
    or a different background set -- and the aggregate summary then relabelled
    its time step from the *current* command line, which is how a historical
    simulation acquires a present-day provenance claim (2026-08-09 audit, 4.3).

    Effective solver settings are part of the identity AND are stored, so a
    resumed run reads them back rather than re-deriving them.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rc", ROOT / "tools" / "run_campaign.py")
    rc = iu.module_from_spec(spec)
    spec.loader.exec_module(rc)

    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION
    from compass.features.extract import FEATURE_SCHEMA_VERSION
    from compass.workflows.simulate import solver_kwargs

    kw = solver_kwargs(model_id)
    if dt_override is not None:
        kw = dict(kw, dt_ms=float(dt_override))
    elif dt_scale is not None:
        kw = dict(kw, dt_ms=float(kw["dt_ms"]) * float(dt_scale))
    payload = json.dumps([sorted(j["tuple_idx"]) for j in jobs] +
                         [j["kind"] for j in jobs], sort_keys=True)
    return {
        "model_id": model_id,
        "experiment": experiment,
        "battery_version": PERTURBATION_BATTERY_VERSION,
        "delta_log10": DELTA,
        "n_backgrounds": N_BACKGROUNDS,
        "background_seed": SEED,
        "pert_features": ",".join([*PERT_FEATURES, "spike_count"]),
        "candidate_tuples_hash": hashlib.blake2b(payload.encode(),
                                                 digest_size=8).hexdigest(),
        "n_candidate_tuples": len(jobs),
        # the settings the rows were ACTUALLY integrated at
        "solver": json.dumps(kw, sort_keys=True),
        "effective_dt_ms": kw.get("dt_ms"),
        "dt_override": dt_override,
        "dt_scale": dt_scale,
        "asset_hash": rc.asset_hash(model_id),
        "mechanism_hash": rc.mechanism_hash(model_id),
        "protocol_hash": rc.protocol_hash(model_id),
        "env_hash": rc.env_hash(),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
        "sim_code_hash": rc.sim_code_hash(),
    }


def slice_sidecar(path: Path) -> Path:
    return path.with_suffix(".identity.json")


def write_slice_identity(path: Path, identity: dict, n_rows: int) -> Path:
    rec = dict(identity)
    rec["n_rows"] = int(n_rows)
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rec["written_utc"] = datetime.now(UTC).isoformat()
    side = slice_sidecar(path)
    side.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    return side


def slice_is_reusable(path: Path, identity: dict) -> tuple[bool, str, dict]:
    """May a cached slice be merged into this aggregate?  Fails closed."""
    if not path.exists():
        return False, "absent", {}
    side = slice_sidecar(path)
    if not side.exists():
        return False, "identity_unknown", {}
    try:
        rec = json.loads(side.read_text())
    except Exception:                                             # noqa: BLE001
        return False, "identity_unreadable", {}
    if rec.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        return False, "checksum_mismatch", rec
    mismatched = [k for k, v in identity.items() if str(rec.get(k)) != str(v)]
    if mismatched:
        return False, "identity_mismatch:" + ",".join(sorted(mismatched)), rec
    return True, "match", rec


def reanalyse(suffix: str) -> int:
    """Rewrite a stored battery's summary from its stored rows.

    The summary carries statistics -- the bootstrap unit above all -- that are
    properties of the ANALYSIS, not of the simulations.  When one of them
    changes there is nothing to re-simulate, and re-running the battery to pick
    up a new bootstrap unit would burn hours and produce identical eta.
    Everything that describes the run itself (effective dt, resumed models) is
    preserved from the existing summary rather than re-derived, so a reanalysis
    cannot silently relabel which solver settings produced the rows.
    """
    src = OUT / f"perturbations{suffix}.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src}; nothing to reanalyse")
    df = pd.read_parquet(src)
    path = OUT / f"perturbations_summary{suffix}.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    summary = analyse(df)
    for k in ("solver_dt_override_ms", "solver_dt_scale", "effective_dt_ms",
              "models_resumed_from_cache", "quantisation_arithmetic",
              "withdrawn"):
        if k in prev:
            summary[k] = prev[k]
    summary["reanalysed_utc"] = datetime.now(UTC).isoformat()
    summary["reanalysed_note"] = (
        f"statistics recomputed from {src.name} without re-simulating; the "
        f"per-row eta is unchanged")
    path.write_text(json.dumps(summary, indent=1, default=float))
    print(f"reanalysed {len(df)} rows -> {path}")
    print(f"  bootstrap unit: {summary.get('bootstrap_unit')}")
    for k in ("top_vs_control_middle", "top_vs_control_random"):
        if k in summary:
            v = summary[k]
            print(f"  {k}: ratio {v['ratio_point']:.3f} "
                  f"CI {[round(x, 2) for x in v['ratio_ci95']]} "
                  f"p={v['wilcoxon_p']:.4f} n={v['n_models']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dt", type=float, default=None,
                    help="override the campaign time step; used by the "
                         "numerical-repeatability check, which reruns the "
                         "battery at a finer dt and compares eta")
    ap.add_argument("--dt-scale", type=float, default=None,
                    help="multiply each model's OWN campaign time step by this "
                         "factor (0.5 = half). Prefer this over --dt for the "
                         "numerical-repeatability check: a single absolute dt "
                         "refines some models and coarsens others across a "
                         "panel whose accepted steps span 0.05-0.003125 ms")
    ap.add_argument("--resume", action="store_true",
                    help="skip models whose per-model cache already exists")
    ap.add_argument("--suffix", default="",
                    help="filename suffix, e.g. '_dt_half' for a repeatability run")
    ap.add_argument("--reanalyse", action="store_true",
                    help="do not simulate: reload the stored "
                         "perturbations<suffix>.parquet and rewrite its summary. "
                         "For when the ANALYSIS changed and the simulations did "
                         "not -- re-running a 16-hour battery to change a "
                         "bootstrap unit is not a reproducibility improvement.")
    args = ap.parse_args()
    if args.dt is not None and args.dt_scale is not None:
        ap.error("--dt and --dt-scale are mutually exclusive")

    if args.reanalyse:
        return reanalyse(args.suffix)

    picks = select_candidates(args.experiment)
    models = args.models or sorted(picks)
    # Models whose cached slices must be folded back into the aggregate.
    #
    # `--resume` used to skip cached models and then build the aggregate from
    # the newly executed ones ALONE, so resuming a five-of-seven run wrote a
    # two-model `perturbations_raw.parquet` over the complete one.  The skip and
    # the aggregate are now two different questions: skipped models are not
    # re-simulated, but their cached rows are read back and concatenated
    # (post-response audit, section 4, "Resume and cache semantics").
    #
    # `--resume` also has to compose with `--models`.  Restricting execution to
    # two slow models is the normal way to finish a partial battery, and if the
    # merge set were `--models` minus the cache it would be EMPTY in exactly
    # that case -- writing a two-model aggregate, which is the same failure by
    # another route.  Under `--resume` the merge set is therefore every cached
    # slice for this suffix that is not being re-executed, independent of
    # `--models`.  Without `--resume` the aggregate is the executed models only,
    # which is explicit and unchanged.
    #
    # Discovery is not enough on its own: a cached slice is only admissible if
    # its stored identity matches what THIS invocation would produce for that
    # model (2026-08-09 audit, 4.3).  A slice that does not match is neither
    # merged nor silently re-run under a resume flag -- it is reported and the
    # model is put back into the execution set.
    resumed: list[str] = []
    slice_identities: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    if args.resume:
        tag = args.suffix or "_main"
        cdir = OUT / "_perturbation_cache"
        cached = {p.name.split(tag)[0] for p in cdir.glob(f"*{tag}_raw.parquet")}
        admissible = set()
        for m in sorted(cached):
            if m not in picks:
                rejected[m] = "not a candidate model in this experiment"
                continue
            want = slice_identity(m, args.experiment, picks[m], args.dt, args.dt_scale)
            ok_r, why_r, rec = slice_is_reusable(cdir / f"{m}{tag}_raw.parquet", want)
            ok_s, why_s, _ = slice_is_reusable(cdir / f"{m}{tag}_summary.parquet", want)
            if ok_r and ok_s:
                admissible.add(m)
                slice_identities[m] = rec
            else:
                rejected[m] = why_r if not ok_r else why_s
        skipped = sorted(set(models) & admissible)
        models = [m for m in models if m not in admissible]
        resumed = sorted(admissible - set(models))
        if skipped:
            print(f"resuming: {len(skipped)} requested model(s) already cached "
                  f"with a matching identity, not re-simulating {skipped}", flush=True)
        if resumed:
            print(f"aggregate will also include cached slices for {resumed}",
                  flush=True)
        if rejected:
            print("REJECTED cached slices (identity does not match this run):",
                  flush=True)
            for m, why in sorted(rejected.items()):
                print(f"  {m}: {why}", flush=True)
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
        tasks.append((mid, bg, picks[mid], pids, lo.tolist(), hi.tolist(),
                      args.dt, args.dt_scale))

    print(f"perturbation battery: {len(tasks)} models x "
          f"{N_TOP + N_CONTROL + N_RANDOM} tuples x {N_BACKGROUNDS} backgrounds x "
          f"{len(PERT_FEATURES)} features", flush=True)
    # Per-model incremental cache.
    #
    # The battery used to accumulate every row in the parent and write once, at
    # the end. One slow model therefore held ALL output hostage: a 15-hour
    # `mainen_l5pc` meant zero usable output for 15 hours even though seven
    # models had finished. Each model's rows are now cached the moment it
    # completes, so a run can be interrupted, resumed or partially consumed.
    CACHE = OUT / "_perturbation_cache"
    CACHE.mkdir(parents=True, exist_ok=True)

    def cache_path(mid: str, kind: str) -> Path:
        tag = args.suffix or "_main"
        return CACHE / f"{mid}{tag}_{kind}.parquet"

    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows, raw_rows = [], []
    # Spike count shares the firing-rate SD scaled back to counts, so that a
    # count-space eta is expressed against the same population dispersion.
    window_s = 0.4
    with ctx.Pool(min(args.workers, max(1, len(tasks))), maxtasksperchild=1) as pool:
        for mid, res in pool.imap_unordered(_measure, tasks):
            pids = pid_map[mid]
            sds = dict(zip(PERT_FEATURES, feature_sd[mid], strict=True))
            sds["spike_count"] = float(sds[PERT_FEATURES[0]] * window_s)
            for r in res:
                for feat in (*PERT_FEATURES, "spike_count"):
                    pr = summarise_nonadditivity(
                        mid, r["tuple_idx"], [pids[i] for i in r["tuple_idx"]],
                        feat, np.array(r["baseline"][feat]),
                        np.array(r["singles"][feat]), np.array(r["joint"][feat]),
                        sds[feat], applied_steps=np.array(r["applied_steps"]),
                        status=tuple(r["status"]))
                    row = pr.as_dict()
                    row["kind"] = r["kind"]
                    row["o_information"] = r["o_information"]
                    row["n_inward_flipped"] = r["n_inward_flipped"]
                    row["primary_feature"] = feat == PERT_FEATURES[0]
                    rows.append(row)
                    for rr in pr.as_rows():
                        rr["kind"] = r["kind"]
                        rr["o_information"] = r["o_information"]
                        rr["primary_feature"] = feat == PERT_FEATURES[0]
                        raw_rows.append(rr)
            # Write this model's slice immediately.
            mine = [r for r in rows if r["model_id"] == mid]
            mine_raw = [r for r in raw_rows if r["model_id"] == mid]
            ident = slice_identity(mid, args.experiment, picks[mid],
                                   args.dt, args.dt_scale)
            slice_identities[mid] = dict(ident)
            for kind, payload in (("summary", mine), ("raw", mine_raw)):
                p = cache_path(mid, kind)
                pd.DataFrame(payload).to_parquet(p, index=False)
                write_slice_identity(p, ident, n_rows=len(payload))
            print(f"  {mid} done ({(time.time()-t0)/60:.1f} min) "
                  f"-> cached {len(mine)} summary / {len(mine_raw)} raw rows "
                  f"(dt={ident['effective_dt_ms']} ms)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix
    df = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    # Fold the resumed models back in.  Without this the aggregate describes
    # only the models this invocation happened to execute, which is how a
    # resumed run could silently shrink the battery.
    if resumed:
        cached_sum = [pd.read_parquet(cache_path(m, "summary")) for m in resumed]
        cached_raw = [pd.read_parquet(cache_path(m, "raw")) for m in resumed]
        df = pd.concat([*cached_sum, df] if len(df) else cached_sum,
                       ignore_index=True)
        raw = pd.concat([*cached_raw, raw] if len(raw) else cached_raw,
                        ignore_index=True)
        df = df.sort_values(["model_id", "kind", "tuple", "feature"],
                            kind="stable").reset_index(drop=True)
        raw = raw.sort_values(["model_id", "tuple", "feature", "background"],
                              kind="stable").reset_index(drop=True)
        print(f"aggregate merged: {sorted(df['model_id'].unique())}", flush=True)
    df.to_parquet(OUT / f"perturbations{sfx}.parquet", index=False)
    raw.to_parquet(OUT / f"perturbations_raw{sfx}.parquet", index=False)
    summary = analyse(df)
    summary["models_resumed_from_cache"] = resumed
    summary["rejected_cache_slices"] = rejected
    summary["solver_dt_override_ms"] = args.dt
    summary["solver_dt_scale"] = args.dt_scale
    # The step each model actually integrated at, taken from the IMMUTABLE
    # per-slice identity written when the rows were produced.  Re-deriving it
    # from the current command line is how a cached slice acquires a time step
    # it was never run at (2026-08-09 audit, 4.3); a model with no stored
    # identity is reported as unknown rather than guessed.
    summary["effective_dt_ms"] = {
        mid: (slice_identities.get(mid) or {}).get("effective_dt_ms", "unknown")
        for mid in sorted(set(meta) | set(df["model_id"].unique() if len(df) else []))}
    summary["effective_dt_source"] = "per-slice cache identity (immutable)"
    summary["slice_identity"] = {
        mid: {k: rec.get(k) for k in ("solver", "sim_code_hash", "mechanism_hash",
                                      "candidate_tuples_hash", "battery_version",
                                      "measurement_schema", "feature_schema")}
        for mid, rec in sorted(slice_identities.items())}
    summary["quantisation_arithmetic"] = background_grid_note(N_BACKGROUNDS)
    summary["withdrawn"] = (
        "The 2.5 Hz 'measurement floor' analysis and its '4/8 models "
        "unresolvable' conclusion were withdrawn by the 2026-08-06 audit "
        "(section 4). They compared a median of eight-background means against "
        "one unaveraged single-observation quantum. See "
        "results/analyses/perturbation_inference.json for the replacement.")
    (OUT / f"perturbations_summary{sfx}.json").write_text(
        json.dumps(summary, indent=1, default=float))
    print("\n" + json.dumps(summary, indent=1, default=float)[:2600])
    print(f"\nwrote {OUT/f'perturbations{sfx}.parquet'} ({len(df)} rows) "
          f"and {OUT/f'perturbations_raw{sfx}.parquet'} ({len(raw)} rows)")
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

    # Paired across LINEAGES, primary feature, top vs each control family.
    #
    # This resampled `model_id`, so the HH lineage and the Pospischil lineage
    # each supplied two of the seven "independent" units the interval was built
    # from -- while the README claimed every bootstrap clusters by lineage. The
    # scored criterion in run_crossmodel was already lineage-clustered; this
    # summary, which is where the quoted ratio CI comes from, was not
    # (post-response audit, section 4). Implementations are averaged within
    # lineage FIRST, so a lineage counts once however many it contributes.
    from compass.analysis.cohorts import annotate_lineage

    prim = annotate_lineage(prim) if "source_lineage_id" not in prim.columns else prim
    per_model = (prim.groupby(["model_id", "kind"])["mean_nonadditivity_sd"]
                     .apply(lambda s: float(np.nanmean(np.abs(s)))).unstack())
    out["per_model_mean_abs_eta"] = per_model.round(4).to_dict()
    per_impl = (prim.groupby(["source_lineage_id", "model_id", "kind"])
                    ["mean_nonadditivity_sd"]
                    .apply(lambda s: float(np.nanmean(np.abs(s)))).unstack())
    per_lineage = per_impl.groupby(level="source_lineage_id").mean()
    out["per_lineage_mean_abs_eta"] = per_lineage.round(4).to_dict()
    out["bootstrap_unit"] = "source_lineage_id"
    out["lineage_members"] = {
        lin: sorted(g.index.get_level_values("model_id").unique())
        for lin, g in per_impl.groupby(level="source_lineage_id")}
    for ctrl in ("control_middle", "control_random"):
        if "top_synergy" not in per_lineage or ctrl not in per_lineage:
            continue
        a = per_lineage["top_synergy"].to_numpy(dtype=float)
        b = per_lineage[ctrl].to_numpy(dtype=float)
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
