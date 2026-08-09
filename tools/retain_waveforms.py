#!/usr/bin/env python3
"""Retain and inspect a stratified sample of actual voltage waveforms.

Why this exists
---------------
``tools/validate_replay.py`` proves that a stored feature row can be reproduced
bit-for-bit by re-running the simulation.  That is a strong check of
determinism, and it is *not* a check that the underlying trace is a plausible
neuronal recording: a trace that is wrong in the same way twice reproduces
perfectly.  The post-response audit put it directly -- replay verifies derived
features rather than archived voltage trajectories, and only a small canonical
subset has retained waveforms.

This tool closes that gap for a *stratified sample* rather than for all 130,098
simulations, which is neither storable nor necessary.  For every
model x protocol x regime x campaign-block cell it re-simulates a small number
of stored samples, keeps the voltage trajectory, and runs a battery of
trace-level checks that features cannot express:

``voltage_range``        membrane potential inside a physically possible band
``baseline_stability``   pre-stimulus drift small relative to the response
``spike_shape``          detected spikes have a plausible amplitude and width
``depolarisation_block`` a depolarised plateau is DETECTED, not silently scored
                         as a low firing rate
``discontinuity``        no step in V larger than a fixed-step integrator can
                         produce in one dt
``finite``               no NaN or Inf anywhere in the trajectory
``feature_recomputation`` features recomputed from the retained waveform match
                         the stored row

Half-step agreement is deliberately NOT part of this battery.  It is a property
of a whole model rather than of one trace and is already measured, per model, by
the convergence audits (gate N3) and by the perturbation repeatability run;
re-deriving it here from a handful of traces would be a weaker version of a
check that already exists.

The waveforms themselves are written to ``results/traces/retained`` as
compressed ``.npz`` (float32 voltage plus the time base), so the sample can be
plotted and re-examined without re-simulating.

    python tools/retain_waveforms.py --experiment screening --workers 12
    python tools/retain_waveforms.py --per-cell 1 --limit 40   # quick pass
"""

from __future__ import annotations

import argparse
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

from compass.validation.gates import gate_record, write_gate_records  # noqa: E402

FEATURES = ROOT / "results" / "features"
OUT = ROOT / "results" / "validation"
TRACES = ROOT / "results" / "traces" / "retained"

# Physically possible band for a somatic membrane potential in these models.
# Deliberately generous: this is a unit-error and blown-up-integration detector,
# not a goodness-of-fit test.  A model that leaves this band is not a model with
# an unusual phenotype, it is a broken simulation.
V_MIN_MV, V_MAX_MV = -150.0, 80.0

# A spike detected by the extractor should have a physically sensible shape.
AP_AMPLITUDE_MIN_MV = 10.0
AP_HALFWIDTH_MAX_MS = 10.0

# Baseline drift over the last 50 ms before stimulus onset, as a fraction of the
# response excursion.  A cell still settling when the step arrives makes every
# step-derived feature a measurement of the settling, not of the response --
# which is exactly what gate N6 found for ortner_snc_da's input resistance.
BASELINE_DRIFT_FRAC = 0.10
BASELINE_WINDOW_MS = 50.0

# Largest single-sample jump in V the integrator may produce.  A fixed-step
# solver stepping through a spike upstroke moves fast, but a jump of this size
# in one dt is a discontinuity, not a spike.
MAX_DV_PER_STEP_MV = 50.0


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def check_trace(v: np.ndarray, t: np.ndarray, delay_ms: float,
                stored: dict, protocol_id: str) -> dict:
    """The trace-level battery for one waveform.  Returns problems, not scores."""
    problems: list[str] = []
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)

    n_nonfinite = int((~np.isfinite(v)).sum())
    if n_nonfinite:
        problems.append(f"finite: {n_nonfinite} non-finite samples")
    fv = _finite(v)
    if fv.size == 0:
        return {"problems": ["finite: trace is entirely non-finite"],
                "v_min": None, "v_max": None}

    vmin, vmax = float(fv.min()), float(fv.max())
    if vmin < V_MIN_MV or vmax > V_MAX_MV:
        problems.append(f"voltage_range: [{vmin:.1f}, {vmax:.1f}] mV outside "
                        f"[{V_MIN_MV}, {V_MAX_MV}]")

    dv = np.abs(np.diff(fv))
    max_dv = float(dv.max()) if dv.size else 0.0
    if max_dv > MAX_DV_PER_STEP_MV:
        problems.append(f"discontinuity: max |dV| per step = {max_dv:.1f} mV")

    # Baseline stability, only where there IS a pre-stimulus window.
    drift = None
    if delay_ms and delay_ms > BASELINE_WINDOW_MS and t.size == v.size:
        pre = (t < delay_ms) & (t >= delay_ms - BASELINE_WINDOW_MS)
        early = (t < delay_ms - BASELINE_WINDOW_MS)
        if pre.sum() > 2 and early.sum() > 2:
            drift = float(abs(np.nanmean(v[pre]) - np.nanmean(v[early])))
            excursion = max(1e-9, vmax - vmin)
            if drift > BASELINE_DRIFT_FRAC * excursion:
                problems.append(
                    f"baseline_stability: pre-stimulus mean moved {drift:.2f} mV, "
                    f"{drift / excursion:.0%} of the {excursion:.1f} mV excursion")

    # Spike shape and depolarisation block, read from the STORED features for
    # this protocol so the check speaks the same language as the campaign.
    amp = stored.get(f"{protocol_id}.ap_amplitude_mV")
    hw = stored.get(f"{protocol_id}.ap_halfwidth_ms")
    n_sp = stored.get(f"{protocol_id}.spike_count")
    try:
        n_sp = float(n_sp) if n_sp is not None else 0.0
    except (TypeError, ValueError):
        n_sp = 0.0
    if n_sp > 0:
        for name, val, lo, hi in (("ap_amplitude_mV", amp, AP_AMPLITUDE_MIN_MV, None),
                                  ("ap_halfwidth_ms", hw, None, AP_HALFWIDTH_MAX_MS)):
            try:
                x = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x):
                continue
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                problems.append(f"spike_shape: {name}={x:.3g} with {n_sp:.0f} spikes")

    # Depolarisation block must be DETECTED rather than showing up as a quiet
    # cell.  A trace whose last third sits above -20 mV with no spikes there is
    # blocked; the stored flag has to agree.
    blocked_obs = False
    if t.size == v.size and t.size > 30:
        tail = v[t >= (t.max() - (t.max() - delay_ms) / 3.0)] if t.max() > delay_ms else v[-10:]
        tail = _finite(tail)
        blocked_obs = bool(tail.size and float(np.median(tail)) > -20.0)
    flag = stored.get(f"{protocol_id}.depolarisation_block")
    try:
        flag_v = bool(float(flag)) if flag is not None and np.isfinite(float(flag)) else None
    except (TypeError, ValueError):
        flag_v = None
    if blocked_obs and flag_v is False:
        problems.append("depolarisation_block: trace plateaus above -20 mV but "
                        "the stored flag is 0")

    return {"problems": problems, "v_min": vmin, "v_max": vmax,
            "max_abs_dv_mV": max_dv, "baseline_drift_mV": drift,
            "n_nonfinite": n_nonfinite, "plateau_observed": blocked_obs}


def _retain(job) -> dict:
    """Re-simulate one stored sample, keep its waveforms, and check them."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    (model_id, param_ids, scales, sample_id, regime, replicate, shard, stored,
     stratum, save) = job

    import importlib.util as _iu

    import numpy as _np

    from compass.adapters.models import get_adapter
    from compass.features.extract import extract, extract_rest
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    # The features-1 -> features-2 `cv_isi` change is a DOCUMENTED, PREDICTED
    # difference, not a waiver: validate_replay verifies that each such
    # difference equals sqrt(n/(n-1)) exactly, or becomes undefined below three
    # spikes. Reuse that predicate rather than reimplementing it, so the two
    # gates cannot drift apart on what counts as reproduction.
    _spec = _iu.spec_from_file_location("_vr", ROOT / "tools" / "validate_replay.py")
    _vr = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_vr)

    key = f"{model_id}__{regime}__r{replicate}__s{sample_id}"
    kw = solver_kwargs(model_id)
    try:
        a = get_adapter(model_id).build()
        a.reset()
        a.apply(dict(zip(param_ids, scales, strict=True)))
        rheo = a.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                               duration_ms=P.RHEO_DURATION_MS,
                               total_ms=P.RHEO_TOTAL_MS, **kw)
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "model_id": model_id, "regime": regime,
                "stratum": stratum, "status": "simulate_error",
                "detail": f"{type(exc).__name__}: {exc}", "problems": [],
                "n_problems": 1, "n_protocols": 0}

    spontaneous = model_id in P.SPONTANEOUSLY_ACTIVE
    problems: list[str] = []
    per_protocol: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    n_recomp_checked = n_recomp_bad = n_schema_drift = 0

    for spec in P.common_core(model_id):
        pid = spec.protocol_id
        if spec.kind == "rest":
            amp, delay = 0.0, 0.0
            rec = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                             total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract_rest(rec)
        else:
            if spec.amplitude_nA is not None:
                amp = float(spec.amplitude_nA)
            elif rheo is None or (spontaneous and rheo == 0.0):
                continue          # undefined for this sample, as in the campaign
            else:
                amp = float(rheo) * spec.amplitude_rheobase_mult
            delay = spec.delay_ms
            rec = a.simulate(amp, delay_ms=spec.delay_ms,
                             duration_ms=spec.duration_ms,
                             total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract(rec)

        chk = check_trace(rec.v, rec.t, delay, stored, pid)
        for p in chk["problems"]:
            problems.append(f"{pid}: {p}")
        per_protocol[pid] = {k: v for k, v in chk.items() if k != "problems"}
        per_protocol[pid]["amplitude_nA"] = amp

        # Feature recomputation: the features derived from the RETAINED waveform
        # must equal the stored ones. This is what makes the archived trace
        # evidence for the stored row rather than a separate artifact.
        for fk, fv in feats.items():
            want = stored.get(f"{pid}.{fk}")
            if want is None:
                continue
            try:
                w, h = float(want), float(fv)
            except (TypeError, ValueError):
                continue
            if not (_np.isfinite(w) or _np.isfinite(h)):
                continue
            n_recomp_checked += 1
            if _np.isfinite(w) != _np.isfinite(h) or \
                    abs(w - h) > max(1e-6, 1e-6 * abs(w)):
                sc = stored.get(f"{pid}.spike_count")
                try:
                    sc = float(sc) if sc is not None else None
                except (TypeError, ValueError):
                    sc = None
                if _vr.explained_by_schema_drift(f"{pid}.{fk}", w, h, sc):
                    n_schema_drift += 1
                    continue
                n_recomp_bad += 1
                if n_recomp_bad <= 4:
                    problems.append(
                        f"{pid}: feature_recomputation {fk}: stored {w:.9g} vs "
                        f"recomputed {h:.9g}")

        if save:
            arrays[f"v__{pid}"] = np.asarray(rec.v, dtype=np.float32)
            arrays[f"t__{pid}"] = np.asarray(rec.t, dtype=np.float32)

    path = None
    if save and arrays:
        TRACES.mkdir(parents=True, exist_ok=True)
        path = TRACES / f"{key}.npz"
        np.savez_compressed(path, stratum=stratum, model_id=model_id,
                            regime=regime, sample_id=sample_id,
                            replicate=replicate, shard=str(shard), **arrays)

    return {
        "key": key, "model_id": model_id, "regime": regime, "replicate": replicate,
        "sample_id": int(sample_id), "stratum": stratum, "shard": str(shard),
        "status": "ok" if not problems else "problems",
        "n_protocols": len(per_protocol),
        "n_features_recomputed": n_recomp_checked,
        "n_features_mismatched": n_recomp_bad,
        "n_features_schema_drift": n_schema_drift,
        "n_problems": len(problems),
        "problems": problems[:10],
        "trace_file": str(path.relative_to(ROOT)) if path else None,
        "per_protocol": per_protocol,
    }


def select_cells(df: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    """A stratified sample over model x regime x campaign block, plus risk strata.

    Two selections, unioned.  The first is the *coverage* requirement the audit
    states -- every model, protocol, regime and campaign block represented --
    which a purely risk-driven sample does not give, because a healthy cell in
    an ordinary block is exactly what a risk sampler skips.  The second reuses
    the risky strata that ``validate_replay`` already defines, so the retained
    set is not a sample of easy cases.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("_vr", ROOT / "tools" / "validate_replay.py")
    vr = iu.module_from_spec(spec)
    spec.loader.exec_module(vr)

    rng = np.random.default_rng(seed)
    picks = []
    block = (df["parameter_sample_id"].astype(int) // 256).rename("block")
    for _key, g in df.groupby(["model_id", "sampling_regime", block]):
        take = min(per_cell, len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        chunk = g.iloc[idx].copy()
        chunk["stratum"] = "coverage"
        picks.append(chunk)
    cover = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    risky = vr.risky_strata(df, per_stratum=max(1, per_cell), seed=seed)
    both = [x for x in (cover, risky) if len(x)]
    if not both:
        return pd.DataFrame()
    out = pd.concat(both, ignore_index=True)
    return out.drop_duplicates(
        subset=["model_id", "sampling_regime", "replicate", "parameter_sample_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--per-cell", type=int, default=1,
                    help="samples per model x regime x campaign block")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of retained samples (0 = no cap)")
    ap.add_argument("--no-save", action="store_true",
                    help="run the checks without writing the .npz waveforms")
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    from compass.adapters.models import get_adapter

    paths = sorted((FEATURES / args.experiment).rglob("*.parquet"))
    if not paths:
        raise SystemExit(f"no feature shards under {FEATURES / args.experiment}")
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        d["_shard"] = p.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"{len(df)} stored rows in {len(paths)} shards", flush=True)

    sel = select_cells(df, args.per_cell, args.seed)
    if args.limit:
        sel = sel.iloc[:args.limit]
    print(f"selected {len(sel)} samples across "
          f"{sel['model_id'].nunique()} models x "
          f"{sel['sampling_regime'].nunique()} regimes; strata "
          f"{sorted(sel['stratum'].unique())}", flush=True)

    pid_cache: dict[str, list[str]] = {}
    jobs = []
    for _, row in sel.iterrows():
        mid = row["model_id"]
        if mid not in pid_cache:
            pid_cache[mid] = [s.parameter_id
                              for s in get_adapter(mid).parameter_specs()]
        pids = pid_cache[mid]
        stored = {k: v for k, v in row.items()
                  if not str(k).startswith(("_", "identity."))}
        jobs.append((mid, pids, [float(row[f"param.{p}"]) for p in pids],
                     int(row["parameter_sample_id"]), row["sampling_regime"],
                     int(row["replicate"]), row["_shard"], stored,
                     row["stratum"], not args.no_save))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(min(args.workers, max(1, len(jobs))), maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_retain, jobs), 1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({(time.time()-t0)/60:.1f} min)",
                      flush=True)

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "per_protocol"}
                        for r in results])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / f"retained_waveforms_{args.experiment}.parquet", index=False)

    ok = res[res["status"] == "ok"]
    bad = res[res["status"] != "ok"]
    by_problem: dict[str, int] = {}
    for r in results:
        for p in r.get("problems", []):
            kind = p.split(":")[1].strip() if ":" in p else p
            by_problem[kind.split()[0]] = by_problem.get(kind.split()[0], 0) + 1

    record = {
        "tool": "retain_waveforms.py",
        "written_utc": datetime.now(UTC).isoformat(),
        "experiment": args.experiment,
        "n_selected": int(len(res)),
        "n_clean": int(len(ok)),
        "n_with_problems": int(len(bad)),
        "n_features_recomputed": int(res["n_features_recomputed"].sum()),
        "n_features_mismatched": int(res["n_features_mismatched"].sum()),
        "problems_by_kind": by_problem,
        "coverage": {
            "models": sorted(res["model_id"].unique()),
            "regimes": sorted(res["regime"].unique()),
            "strata": sorted(res["stratum"].unique()),
        },
        "checks": ["voltage_range", "baseline_stability", "spike_shape",
                   "depolarisation_block", "discontinuity", "finite",
                   "feature_recomputation"],
        "not_checked_here": (
            "half-step agreement: a per-model property already measured by the "
            "convergence audits (gate N3) and the perturbation repeatability "
            "battery, not re-derived from a trace sample"),
        "traces_dir": str(TRACES.relative_to(ROOT)) if not args.no_save else None,
        "worst": [{k: r[k] for k in ("key", "stratum", "problems")}
                  for r in results if r.get("problems")][:20],
    }
    (OUT / f"retained_waveforms_{args.experiment}.json").write_text(
        json.dumps(record, indent=1, default=str))

    # Gate WAVEFORM: one record per retained trace, so the roll-up counts
    # waveforms and a single bad trace is a named failed subject rather than a
    # number in a summary.
    write_gate_records([
        gate_record(
            "WAVEFORM", r["key"], failures=list(r.get("problems", [])),
            measurements={"n_protocols": r["n_protocols"],
                          "n_features_recomputed": r["n_features_recomputed"],
                          "n_features_mismatched": r["n_features_mismatched"]},
            inputs={"model_id": r["model_id"], "regime": r["regime"],
                    "stratum": r["stratum"], "shard": r.get("shard"),
                    "trace_file": r.get("trace_file")},
            detail=f"{args.experiment}: {', '.join(record['checks'])}")
        for r in results
    ], OUT / f"waveform_gate_{args.experiment}.json")

    print(f"\n{len(ok)}/{len(res)} clean; problems by kind: {by_problem}")
    print(f"features recomputed {record['n_features_recomputed']}, "
          f"mismatched {record['n_features_mismatched']}")
    for r in record["worst"][:5]:
        print(f"  ! {r['key']} [{r['stratum']}]: {r['problems'][:2]}")
    print(f"-> {OUT / f'retained_waveforms_{args.experiment}.json'}")
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
