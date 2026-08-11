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
compressed ``.npz`` (**float64** voltage plus the time base), so the sample can
be plotted, re-examined and *recomputed from* without re-simulating.  Reduced
precision was the earlier behaviour and was withdrawn: a recomputation from a
float32 archive can only ever agree "to within rounding", which is weaker than
the claim this gate makes.

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

from compass.adapters.base import PHYSICAL_V_MAX_MV, PHYSICAL_V_MIN_MV  # noqa: E402
from compass.validation.gates import gate_record, write_gate_records  # noqa: E402

FEATURES = ROOT / "results" / "features"
OUT = ROOT / "results" / "validation"
TRACES = ROOT / "results" / "traces" / "retained"

# Physically possible band for a somatic membrane potential in these models.
# Deliberately generous: this is a unit-error and blown-up-integration detector,
# not a goodness-of-fit test.  A model that leaves this band is not a model with
# an unusual phenotype, it is a broken simulation.
#
# Aliased, not restated: the same constant that makes `qc_flags` raise
# `nonphysical_voltage` on every simulation in the corpus, which is what removes
# the row from every downstream analysis.  A trace this battery calls impossible
# is therefore a trace the corpus has already excluded, which is the propagation
# the 2026-08-09 post-response-3 audit (P0-6) required.  Two separate bands --
# [-160, 120] excluding and [-150, 80] flagging -- is what it replaced.
V_MIN_MV, V_MAX_MV = PHYSICAL_V_MIN_MV, PHYSICAL_V_MAX_MV

# A spike detected by the extractor should have a physically sensible shape.
AP_AMPLITUDE_MIN_MV = 10.0
AP_HALFWIDTH_MAX_MS = 10.0

# Baseline drift over the last 50 ms before stimulus onset, as a fraction of the
# response excursion.  A cell still settling when the step arrives makes every
# step-derived feature a measurement of the settling, not of the response --
# which is exactly what gate N6 found for ortner_snc_da's input resistance.
BASELINE_DRIFT_FRAC = 0.10
BASELINE_WINDOW_MS = 50.0

# Median membrane potential over the last third of the stimulus above which a
# silent cell counts as blocked rather than merely quiet.  Deliberately equal to
# the threshold in `compass.features.extract.depolarisation_block`: this
# detector is an independent IMPLEMENTATION of the same definition (own spike
# times, median rather than mean, run on the reloaded float64 array), so a
# disagreement between it and the stored flag is a real disagreement about the
# trace and not an artefact of two different definitions.
BLOCK_PLATEAU_MV = -45.0

# Largest single-sample jump in V the integrator may produce.  A fixed-step
# solver stepping through a spike upstroke moves fast, but a jump of this size
# in one dt is a discontinuity, not a spike.
MAX_DV_PER_STEP_MV = 50.0


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def detect_spikes(v: np.ndarray, t: np.ndarray, threshold_mV: float = -20.0) -> dict:
    """Find spikes in the ARCHIVED trace, independently of the stored features.

    The battery used to read ``ap_amplitude_mV`` and ``ap_halfwidth_ms`` out of
    the stored feature row and check *those* numbers for plausibility, which
    makes the trace-level check a second opinion on eFEL's arithmetic rather than
    on the waveform (2026-08-09 audit, 3.3).  This is a deliberately simple,
    independent detector -- upward threshold crossings, peak between crossings,
    width at half amplitude above the preceding local minimum -- so that
    agreement between it and the stored features is evidence, and disagreement
    is a finding rather than a tautology.
    """
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    v, t = v[ok], t[ok]
    if v.size < 3:
        return {"n_spikes": 0, "amplitudes_mV": [], "halfwidths_ms": [],
                "peaks_mV": []}
    above = v > threshold_mV
    up = np.flatnonzero((~above[:-1]) & above[1:]) + 1
    amps, hws, peaks = [], [], []
    for k, i0 in enumerate(up):
        i1 = up[k + 1] if k + 1 < len(up) else v.size
        seg = v[i0:i1]
        if seg.size < 2:
            continue
        pk = int(np.argmax(seg)) + i0
        # foot: the lowest point in the 5 ms before the crossing
        back = np.flatnonzero(t >= t[i0] - 5.0)
        j0 = int(back[0]) if back.size else max(0, i0 - 50)
        foot = float(np.min(v[j0:i0 + 1])) if i0 > j0 else float(v[i0])
        amp = float(v[pk]) - foot
        if amp <= 0:
            continue
        # Width at half amplitude, measured on the CONTIGUOUS excursion that
        # contains the peak.  Taking the first and last sample above the half
        # level anywhere in the inter-spike window measures the interval, not
        # the spike: on a depolarised plateau that reported 143 ms "action
        # potentials".
        half = foot + 0.5 * amp
        lo_i = pk
        while lo_i > j0 and v[lo_i - 1] >= half:
            lo_i -= 1
        hi_i = pk
        while hi_i < i1 - 1 and v[hi_i + 1] >= half:
            hi_i += 1
        hw = float(t[hi_i] - t[lo_i]) if hi_i > lo_i else np.nan
        amps.append(amp)
        hws.append(hw)
        peaks.append(float(v[pk]))
    return {"n_spikes": len(amps), "amplitudes_mV": amps,
            "halfwidths_ms": hws, "peaks_mV": peaks}


def check_trace(v: np.ndarray, t: np.ndarray, delay_ms: float,
                stored: dict, protocol_id: str,
                stim: tuple[float, float] | None = None) -> dict:
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

    # Spike shape, measured from THIS trace by an independent detector.  Reading
    # the stored ap_amplitude_mV back and checking it for plausibility tested
    # eFEL's arithmetic, not the waveform (2026-08-09 audit, 3.3).
    det = detect_spikes(v, t)
    n_sp_stored = stored.get(f"{protocol_id}.spike_count")
    try:
        n_sp_stored = (float(n_sp_stored) if n_sp_stored is not None
                       and np.isfinite(float(n_sp_stored)) else None)
    except (TypeError, ValueError):
        n_sp_stored = None
    if det["n_spikes"]:
        worst_amp = float(np.nanmin(det["amplitudes_mV"]))
        hws = [h for h in det["halfwidths_ms"] if np.isfinite(h)]
        worst_hw = float(np.nanmax(hws)) if hws else np.nan
        if worst_amp < AP_AMPLITUDE_MIN_MV:
            problems.append(f"spike_shape: smallest detected AP is {worst_amp:.3g} mV "
                            f"over {det['n_spikes']} spikes measured on the trace")
        if np.isfinite(worst_hw) and worst_hw > AP_HALFWIDTH_MAX_MS:
            problems.append(f"spike_shape: widest detected AP is {worst_hw:.3g} ms "
                            f"over {det['n_spikes']} spikes measured on the trace")
    # Independent detection must also AGREE with the stored count, within one
    # spike at the window edges.  A large disagreement means the archived trace
    # and the stored row are not describing the same simulation.
    #
    # Counted inside the SAME analysis window the extractor uses
    # (`strict_stiminterval=True`), or a spontaneously firing draw disagrees by
    # construction: its pre- and post-stimulus spikes are real but are not what
    # `spike_count` measures.
    if n_sp_stored is not None and stim is not None:
        w = (t >= stim[0]) & (t <= stim[1])
        in_window = detect_spikes(v[w], t[w])["n_spikes"] if w.any() else 0
        if abs(in_window - n_sp_stored) > 1:
            problems.append(
                f"spike_count_agreement: {in_window} spikes detected inside the "
                f"stimulus window, stored row says {n_sp_stored:.0f}")

    # Depolarisation block must be DETECTED rather than showing up as a quiet
    # cell, and the observation must be comparable with the stored flag in both
    # directions (2026-08-09 post-response-3 audit, P0-6).
    #
    # Three defects are fixed here.  The window was computed as
    # `t.max() - (t.max() - delay)/3`, which for the standard 200 ms delay,
    # 400 ms stimulus and 800 ms total begins at 600 ms -- the stimulus OFFSET --
    # so the check examined the post-stimulus tail rather than the last third of
    # the stimulus.  The stated "with no spikes there" condition was described
    # but never implemented, so an actively firing plateau counted as blocked.
    # And only `observed block / stored false` was reported, so the opposite
    # disagreement -- the row claims block, the trace shows firing -- was
    # invisible.
    #
    # The rule below is the same DEFINITION the feature extractor uses (early
    # third fires, late third silent, late third depolarised above
    # BLOCK_PLATEAU_MV) implemented independently: this detector's spike times,
    # a median rather than a mean, computed on the reloaded float64 array.  A
    # disagreement is therefore a real disagreement rather than a definitional
    # one, and is reported as a finding to adjudicate rather than repaired.
    blocked_obs = None
    if stim is not None and t.size == v.size and t.size > 30:
        span = stim[1] - stim[0]
        if span > 0:
            late = (t >= stim[1] - span / 3.0) & (t <= stim[1])
            early = (t >= stim[0]) & (t < stim[0] + span / 3.0)
            tail = _finite(v[late])
            if tail.size and early.any() and late.any():
                n_early = detect_spikes(v[early], t[early])["n_spikes"]
                n_late = detect_spikes(v[late], t[late])["n_spikes"]
                blocked_obs = bool(n_early > 0 and n_late == 0
                                   and float(np.median(tail)) > BLOCK_PLATEAU_MV)
    flag = stored.get(f"{protocol_id}.depolarisation_block")
    try:
        flag_v = bool(float(flag)) if flag is not None and np.isfinite(float(flag)) else None
    except (TypeError, ValueError):
        flag_v = None
    if blocked_obs is not None and flag_v is not None and blocked_obs != flag_v:
        problems.append(
            f"depolarisation_block: trace says blocked={blocked_obs}, stored row "
            f"says blocked={flag_v} (last third of the stimulus, "
            f"{stim[1] - (stim[1] - stim[0]) / 3.0:.0f}-{stim[1]:.0f} ms)")

    return {"problems": problems, "v_min": vmin, "v_max": vmax,
            "max_abs_dv_mV": max_dv, "baseline_drift_mV": drift,
            "n_nonfinite": n_nonfinite, "plateau_observed": blocked_obs,
            "n_spikes_detected": det["n_spikes"]}


def _retain(job) -> dict:
    """Re-simulate one stored sample, keep its waveforms, and check them."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    (model_id, param_ids, scales, sample_id, regime, replicate, shard, stored,
     stratum, save, experiment) = job

    import importlib.util as _iu

    import numpy as _np

    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION, Recording
    from compass.adapters.models import get_adapter
    from compass.features.extract import FEATURE_SCHEMA_VERSION, extract, extract_rest
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

    key = f"{experiment}/{model_id}__{regime}__r{replicate}__s{sample_id}"
    fname = f"{experiment}__{model_id}__{regime}__r{replicate}__s{sample_id}"
    base = {"key": key, "model_id": model_id, "regime": regime,
            "replicate": int(replicate), "sample_id": int(sample_id),
            "experiment": experiment, "stratum": stratum, "shard": str(shard)}

    def failed(kind: str, detail: str) -> dict:
        """Any failure to produce, save, reload or check a trace is a FAILURE.

        The previous version returned ``problems: []`` with ``n_problems: 1`` for
        a simulation error, so the gate record it produced carried an empty
        failure list -- and ``gate_record`` derives PASS from exactly that
        (2026-08-09 audit, 3.3: "simulation-error records also contain an empty
        problems list, creating a fail-open gate path").
        """
        return {**base, "status": kind, "detail": detail,
                "problems": [f"{kind}: {detail}"], "n_problems": 1,
                "n_protocols": 0, "n_features_recomputed": 0,
                "n_features_mismatched": 0, "n_features_schema_drift": 0,
                "trace_file": None, "per_protocol": {}}

    kw = solver_kwargs(model_id)
    try:
        a = get_adapter(model_id).build()
        a.reset()
        a.apply(dict(zip(param_ids, scales, strict=True)))
        rheo = a.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                               duration_ms=P.RHEO_DURATION_MS,
                               total_ms=P.RHEO_TOTAL_MS, **kw)
    except Exception as exc:  # noqa: BLE001
        return failed("simulate_error", f"{type(exc).__name__}: {exc}")

    from compass.workflows.simulate import normalised_family_defined

    problems: list[str] = []
    per_protocol: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    protocol_meta: dict[str, dict] = {}

    # ---- 1. simulate and ARCHIVE, at full precision ------------------------ #
    for spec in P.common_core(model_id):
        pid = spec.protocol_id
        if spec.kind == "rest":
            amp, delay = 0.0, 0.0
            rec = a.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                             total_ms=spec.total_ms, protocol_id=pid, **kw)
        else:
            if spec.amplitude_nA is not None:
                amp = float(spec.amplitude_nA)
            elif not normalised_family_defined(rheo):
                continue          # undefined for this sample, as in the campaign
            else:
                amp = float(rheo) * spec.amplitude_rheobase_mult
            delay = spec.delay_ms
            rec = a.simulate(amp, delay_ms=spec.delay_ms,
                             duration_ms=spec.duration_ms,
                             total_ms=spec.total_ms, protocol_id=pid, **kw)
        # float64, not float32.  The archive is the evidence: storing it at
        # reduced precision means a recomputation from it can only ever agree
        # "to within rounding", which is not the claim the report makes.
        arrays[f"v__{pid}"] = np.asarray(rec.v, dtype=np.float64)
        arrays[f"t__{pid}"] = np.asarray(rec.t, dtype=np.float64)
        protocol_meta[pid] = {
            "kind": spec.kind, "amplitude_nA": amp,
            "amplitude_rheobase_mult": spec.amplitude_rheobase_mult,
            "delay_ms": delay, "duration_ms": spec.duration_ms,
            "total_ms": spec.total_ms, "n_samples": int(rec.v.size),
        }

    if not arrays:
        return failed("no_protocols", "no protocol produced a trace for this sample")

    # Everything needed to reconstruct and interpret the trace WITHOUT the
    # feature table: the applied amplitudes, the rheobase they were derived
    # from, the solver contract, the schema versions and the parameter draw.
    meta = {
        **base,
        "rheobase_nA": None if rheo is None else float(rheo),
        "normalised_family_defined": bool(normalised_family_defined(rheo)),
        "solver": kw,
        "protocols": protocol_meta,
        "parameters": dict(zip(param_ids, [float(s) for s in scales], strict=True)),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
        "written_utc": datetime.now(UTC).isoformat(),
    }

    path = meta_path = None
    if save:
        TRACES.mkdir(parents=True, exist_ok=True)
        path = TRACES / f"{fname}.npz"
        meta_path = TRACES / f"{fname}.meta.json"
        try:
            np.savez_compressed(path, metadata_json=json.dumps(meta, default=str),
                                **arrays)
            meta_path.write_text(json.dumps(meta, indent=1, default=str))
        except Exception as exc:  # noqa: BLE001
            return failed("archive_write_error", f"{type(exc).__name__}: {exc}")

    # ---- 2. RELOAD the archive and work only from what it contains --------- #
    #
    # This is the point of the whole exercise.  The previous version checked and
    # recomputed from the in-memory, full-precision `rec` and only then wrote a
    # float32 copy to disk, so nothing ever demonstrated that the ARCHIVED
    # arrays reproduce the stored feature row -- while the report claimed 23,413
    # features had been recomputed "from archived voltage".
    if save:
        try:
            with np.load(path, allow_pickle=False) as z:
                loaded = {k: np.asarray(z[k], dtype=float) for k in z.files
                          if k.startswith(("v__", "t__"))}
                stored_meta = json.loads(str(z["metadata_json"]))
        except Exception as exc:  # noqa: BLE001
            return failed("archive_reload_error", f"{type(exc).__name__}: {exc}")
        if stored_meta.get("key") != key:
            return failed("archive_schema_error",
                          f"archive says key={stored_meta.get('key')!r}, expected {key!r}")
        missing = [p for p in protocol_meta
                   if f"v__{p}" not in loaded or f"t__{p}" not in loaded]
        if missing:
            return failed("archive_schema_error",
                          f"archive is missing arrays for {missing}")
    else:
        loaded = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
        stored_meta = meta

    n_recomp_checked = n_recomp_bad = n_schema_drift = 0
    for pid, pmeta in protocol_meta.items():
        v = loaded[f"v__{pid}"]
        t = loaded[f"t__{pid}"]
        delay = float(pmeta["delay_ms"])

        stim = ((delay, delay + float(pmeta["duration_ms"]))
                if pmeta["kind"] != "rest" else None)
        chk = check_trace(v, t, delay, stored, pid, stim)
        for p in chk["problems"]:
            problems.append(f"{pid}: {p}")
        per_protocol[pid] = {k: val for k, val in chk.items() if k != "problems"}
        per_protocol[pid]["amplitude_nA"] = pmeta["amplitude_nA"]

        # Features recomputed FROM THE RELOADED ARRAYS, through the same
        # extractor the campaign used, and compared to the stored row.
        try:
            rebuilt = Recording(
                t=t, v=v, stim_start_ms=delay,
                stim_end_ms=delay + float(pmeta["duration_ms"]),
                amplitude_nA=float(pmeta["amplitude_nA"]), protocol_id=pid)
            feats = extract_rest(rebuilt) if pmeta["kind"] == "rest" else extract(rebuilt)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{pid}: recomputation_error {type(exc).__name__}: {exc}")
            continue

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
                        f"recomputed-from-archive {h:.9g}")

    return {
        **base,
        "status": "ok" if not problems else "problems",
        "n_protocols": len(per_protocol),
        "n_features_recomputed": n_recomp_checked,
        "n_features_mismatched": n_recomp_bad,
        "n_features_schema_drift": n_schema_drift,
        "n_problems": len(problems),
        "problems": problems[:10],
        "recomputed_from": "reloaded archive" if save else "in-memory (--no-save)",
        "trace_file": str(path.relative_to(ROOT)) if path else None,
        "trace_metadata_file": str(meta_path.relative_to(ROOT)) if meta_path else None,
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


# The stratification the retained set must satisfy, DECLARED before selection
# (2026-08-09 audit, 3.3: "define stratified coverage requirements before trace
# selection").  Coverage is checked against this after the run and reported as a
# gate failure if unmet, so an imbalanced sample is visible rather than implicit
# in whatever the sampler happened to draw.
COVERAGE_POLICY = {
    "every_model_in_the_campaign": 1,          # >= 1 retained trace per model
    "every_model_regime_cell": 1,              # >= 1 per (model, regime)
    "every_absolute_protocol_per_model": 1,    # rest/hyperpol/lo/hi always run
    "risk_strata_represented": 1,
}


# What each finding class MEANS and what it does to the row that produced it.
#
# The 2026-08-09 post-response-3 audit (P0-6) found the findings enumerated but
# not adjudicated: 142 flagged samples with no statement of which ones make a
# row unusable, so "we found 142 problems" carried no consequence.  Every class
# is assigned here, before the run, to one of five dispositions, and each
# disposition names the mechanism that carries it into the analysis -- because a
# class judged invalid has to be excluded by a rule that applies to all 130,098
# rows, not repaired in a report that covers 914.
#
#   integrity_failure  the evidence is unusable; the GATE fails and the release
#                      is blocked. Nothing is silently dropped.
#   numerical_invalid  the simulation is broken. Already excluded corpus-wide by
#                      `qc_flags` -> execution_valid=False -> not analysis
#                      eligible. The waveform battery reports it; the corpus rule
#                      is what removes it.
#   prior_predictive   a real, deterministic property of a broad-prior draw. NOT
#                      excluded: dropping these biases the cohort towards
#                      canonical-looking cells, which is the phenomenon under
#                      study. Reported with prevalence.
#   detector_limit     the finding is a property of the screening heuristic, not
#                      of the trace. Reported, never excluded.
#   protocol_settling  the measurement window and the cell's time constant are
#                      mismatched. Reported and carried into the strict-QC
#                      sensitivity arm, which already drops `unstable_baseline`.
FINDING_DISPOSITION = {
    "simulation_error": ("integrity_failure",
                         "gate record fails; the release manifest lists WAVEFORM as a blocker"),
    "archive_reload_error": ("integrity_failure", "as simulation_error"),
    "archive_schema_error": ("integrity_failure", "as simulation_error"),
    "feature_recomputation": ("integrity_failure",
                              "archive and stored row disagree; determinism claim fails"),
    "spike_count_agreement": ("integrity_failure",
                              "archive and stored row describe different simulations"),
    "no_protocols": ("integrity_failure", "as simulation_error"),
    "finite": ("numerical_invalid",
               "qc_flags raises nonfinite_voltage -> execution_valid=False corpus-wide"),
    "voltage_range": ("numerical_invalid",
                      "qc_flags raises nonphysical_voltage on the same band "
                      "(PHYSICAL_V_MIN_MV/PHYSICAL_V_MAX_MV) -> excluded corpus-wide"),
    "discontinuity": ("numerical_invalid",
                      "a step larger than one dt of the integrator can produce; "
                      "co-occurs with nonfinite/nonphysical voltage in every "
                      "observed case and is reported if it ever does not"),
    "depolarisation_block": ("prior_predictive",
                             "two detectors disagree about one trace; recorded as a "
                             "finding to adjudicate, never used to drop a row"),
    "spike_shape": ("detector_limit",
                    "a 10 mV / 10 ms screening bound, not a validity criterion; a "
                    "broad-prior draw may legitimately have a wide, small AP"),
    "baseline_stability": ("protocol_settling",
                           "the strict-QC sensitivity arm drops rows flagged "
                           "unstable_baseline; the primary arm keeps them and says so"),
}


def disposition_summary(res: pd.DataFrame) -> dict:
    """Count findings by class and by prespecified disposition."""
    import collections

    by_class: collections.Counter = collections.Counter()
    listed = 0
    for probs in res.get("problems", pd.Series(dtype=object)):
        for p in (probs or []):
            listed += 1
            parts = str(p).split(": ")
            by_class[parts[1] if len(parts) > 2 else parts[0].split(":")[0]] += 1
    # `problems` is truncated to the first ten per sample so the record stays
    # readable; `n_problems` is the true count. Report both rather than letting
    # a class breakdown quietly under-count a badly broken sample.
    total = int(pd.to_numeric(res.get("n_problems", pd.Series(dtype=float)),
                              errors="coerce").fillna(0).sum())
    by_disp: collections.Counter = collections.Counter()
    unassigned = []
    for k, n in by_class.items():
        if k in FINDING_DISPOSITION:
            by_disp[FINDING_DISPOSITION[k][0]] += n
        else:
            unassigned.append(k)
    return {
        "policy": {k: {"disposition": d, "mechanism": m}
                   for k, (d, m) in FINDING_DISPOSITION.items()},
        "findings_by_class": dict(sorted(by_class.items())),
        "findings_by_disposition": dict(sorted(by_disp.items())),
        "unassigned_classes": sorted(unassigned),
        "n_findings_total": total,
        "n_findings_classified": listed,
        "n_findings_truncated_from_the_record": max(0, total - listed),
        "note": ("counts are FINDINGS, not samples: one sample can raise several "
                 "and they must not be summed to a sample count. Each record "
                 "lists at most ten; n_findings_total is the true sum."),
    }


def coverage_report(res: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Did the retained set meet the DECLARED strata, model by model?

    Only records with ``status == "ok"`` count towards coverage.  Reading the
    labels off every returned record meant that a cell whose only sample failed
    to simulate, failed to reload or failed its schema check was still reported
    as covered (2026-08-09 post-response-3 audit, P0-6).  Those records do fail
    individually, so this never produced a false release pass -- but "coverage
    satisfied" has to mean archived, reloadable evidence exists, not that the
    battery returned something for that cell.
    """
    want_models = sorted(df["model_id"].unique())
    want_cells = sorted({(m, r) for m, r in
                         zip(df["model_id"], df["sampling_regime"], strict=True)})
    ok = res[res["status"] == "ok"] if "status" in res.columns else res.iloc[0:0]
    got_models = set(ok["model_id"])
    got_cells = {(m, r) for m, r in zip(ok["model_id"], ok["regime"], strict=True)}
    missing_models = [m for m in want_models if m not in got_models]
    missing_cells = [f"{m}/{r}" for m, r in want_cells if (m, r) not in got_cells]
    n_not_ok = int(len(res) - len(ok))
    return {
        "policy": COVERAGE_POLICY,
        "counts_only_status_ok": True,
        "n_records": int(len(res)),
        "n_records_not_ok": n_not_ok,
        "n_models_required": len(want_models),
        "n_models_covered": len(want_models) - len(missing_models),
        "missing_models": missing_models,
        "n_model_regime_cells_required": len(want_cells),
        "n_model_regime_cells_covered": len(want_cells) - len(missing_cells),
        "missing_model_regime_cells": missing_cells,
        "strata_present": sorted(ok["stratum"].unique()) if len(ok) else [],
        "satisfied": not (missing_models or missing_cells),
    }


def summarise() -> int:
    """Roll the per-campaign records up into the provenance artifact.

    `data/provenance/waveform_findings.json` is cited by docs/RESULTS.md as the
    source of the measured prevalences and is a `provenance`-class entry in the
    release inventory -- and until this function existed, **nothing in the
    repository wrote it**. It was a hand-authored file carrying numbers from a
    2026-08-09 run, two corpus regenerations out of date, and no check could
    notice: the release manifest hashed whatever bytes were there, and the
    clean-build receipt could never attribute a file that no stage produced.
    A comment in tools/run_receipt.py even named it as an example of what "the
    campaign and gate stages write", which was not true of this file.

    It is generated now, from the records the battery actually wrote, so it
    cannot disagree with them.
    """
    recs = []
    for p in sorted(OUT.glob("retained_waveforms_*.json")):
        recs.append(json.loads(p.read_text()))
    if not recs:
        raise SystemExit(
            f"no retained_waveforms_*.json under {OUT.relative_to(ROOT)}; "
            f"run the battery before summarising it")

    by_kind: dict[str, int] = {}
    for r in recs:
        for k, v in (r.get("problems_by_kind") or {}).items():
            by_kind[k] = by_kind.get(k, 0) + int(v)
    totals = {k: sum(int(r[k]) for r in recs) for k in
              ("n_selected", "n_clean", "n_with_problems",
               "n_features_recomputed", "n_features_mismatched")}

    doc = {
        "_doc": ("Cross-campaign roll-up of the trace-level battery, GENERATED "
                 "by tools/retain_waveforms.py --summarise from the per-campaign "
                 "records in results/validation/. Do not edit by hand: a "
                 "hand-maintained version of this file went two corpus "
                 "regenerations stale while docs/RESULTS.md cited it."),
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_records": [str((OUT / f"retained_waveforms_{r['experiment']}.json")
                               .relative_to(ROOT)) for r in recs],
        "campaigns": {r["experiment"]: {
            k: r[k] for k in ("n_selected", "n_clean", "n_with_problems",
                              "n_features_recomputed", "n_features_mismatched")}
            for r in recs},
        "totals": totals,
        "sample_arithmetic": (f"{totals['n_clean']} clean + "
                              f"{totals['n_with_problems']} flagged = "
                              f"{totals['n_selected']} samples"),
        # A SAMPLE IS NOT A FINDING. One sample can fail several checks on
        # several protocols; conflating the two is the arithmetic error the
        # 2026-08-09 post-response-3 audit named (P0-6).
        "findings_by_class": dict(sorted(by_kind.items(),
                                         key=lambda kv: -kv[1])),
        "n_findings_total": sum(by_kind.values()),
        "n_samples_flagged": totals["n_with_problems"],
        "disposition_policy": {k: {"disposition": v[0], "mechanism": v[1]}
                               for k, v in FINDING_DISPOSITION.items()},
        "what_this_does_not_show": (
            "prevalence in the CORPUS. These are stratified samples chosen to "
            "over-represent risky cells, so a rate here is an upper bound on "
            "the campaign-wide rate, not an estimate of it."),
    }
    out = ROOT / "data" / "provenance" / "waveform_findings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(recs)} campaign records -> {doc['sample_arithmetic']}, "
          f"{doc['n_findings_total']} findings")
    print(f"-> {out.relative_to(ROOT)}")
    return 0


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
    ap.add_argument("--summarise", action="store_true",
                    help="write the cross-campaign roll-up "
                         "data/provenance/waveform_findings.json from the "
                         "per-campaign records already on disk, and exit")
    args = ap.parse_args()

    if args.summarise:
        return summarise()

    from compass.adapters.models import get_adapter

    # QUARANTINED AND SUPERSEDED SHARDS ARE NOT EVIDENCE.
    #
    # This was a plain `rglob`, so `results/features/<campaign>/<model>/
    # _quarantine/` came with it: 18,944 duplicate (model, regime, replicate,
    # sample) keys in the replication campaign alone, and the battery could pick
    # a superseded row and then report the CURRENT simulation as disagreeing
    # with it.  Found while adding the archive-reload check -- sample
    # dougalis_vlpag_da/E1/s1857 was compared against a shard quarantined on
    # 2026-08-06 for an identity mismatch, whose input resistance is 699.7 MOhm
    # against the live 760.9.  `validate_campaign.load_experiment` already
    # excluded both directories; this tool did not.
    paths = sorted(p for p in (FEATURES / args.experiment).rglob("*.parquet")
                   if "_quarantine" not in p.parts and "_superseded" not in p.parts
                   and not p.name.endswith(".tmp.parquet"))
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
                     row["stratum"], not args.no_save, args.experiment))

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

    cov = coverage_report(res, df)
    disp = disposition_summary(res)
    record = {
        "tool": "retain_waveforms.py",
        "written_utc": datetime.now(UTC).isoformat(),
        "experiment": args.experiment,
        "n_selected": int(len(res)),
        "n_clean": int(len(ok)),
        "n_with_problems": int(len(bad)),
        # n_selected = n_clean + n_with_problems, always.  The findings counts
        # below are FINDINGS and are larger, because one sample can raise
        # several; the two were conflated in the previous revision's prose.
        "sample_arithmetic": (f"{len(ok)} clean + {len(bad)} flagged = "
                              f"{len(res)} samples"),
        "disposition": disp,
        "n_features_recomputed": int(res["n_features_recomputed"].sum()),
        "n_features_mismatched": int(res["n_features_mismatched"].sum()),
        "features_recomputed_from": (
            "the reloaded .npz archive, not the in-memory simulation"
            if not args.no_save else "in-memory simulation (--no-save)"),
        "trace_precision": "float64",
        "problems_by_kind": by_problem,
        "coverage": {
            "models": sorted(res["model_id"].unique()),
            "regimes": sorted(res["regime"].unique()),
            "strata": sorted(res["stratum"].unique()),
        },
        "declared_coverage": cov,
        "checks": ["voltage_range", "baseline_stability", "spike_shape",
                   "spike_count_agreement", "depolarisation_block",
                   "discontinuity", "finite", "feature_recomputation"],
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
    gate_rows = [
        gate_record(
            "WAVEFORM", r["key"], failures=list(r.get("problems", [])),
            measurements={"n_protocols": r["n_protocols"],
                          "n_features_recomputed": r["n_features_recomputed"],
                          "n_features_mismatched": r["n_features_mismatched"],
                          "campaign": args.experiment,
                          "status": r["status"]},
            inputs={"model_id": r["model_id"], "regime": r["regime"],
                    "stratum": r["stratum"], "shard": r.get("shard"),
                    "trace_file": r.get("trace_file"),
                    "trace_metadata_file": r.get("trace_metadata_file")},
            detail=f"{args.experiment}: {', '.join(record['checks'])}")
        for r in results
    ]
    # Coverage is itself a gate subject: an imbalanced retained set is a finding,
    # not a footnote (2026-08-09 audit, 3.3).
    gate_rows.append(gate_record(
        "WAVEFORM", f"{args.experiment}/_coverage",
        failures=([] if cov["satisfied"] else
                  [f"models with no retained trace: {cov['missing_models']}",
                   f"model/regime cells with no retained trace: "
                   f"{cov['missing_model_regime_cells'][:12]}"]),
        measurements=cov, detail="declared stratified coverage policy"))
    # A finding class with no prespecified disposition is itself a gate failure:
    # otherwise a new check could quietly appear with no statement of what it
    # does to the affected rows, which is the state P0-6 found the battery in.
    gate_rows.append(gate_record(
        "WAVEFORM", f"{args.experiment}/_disposition",
        failures=([] if not disp["unassigned_classes"] else
                  [f"finding classes with no declared disposition: "
                   f"{disp['unassigned_classes']}"]),
        measurements=disp,
        detail="every finding class is assigned a disposition before the run"))
    write_gate_records(gate_rows, OUT / f"waveform_gate_{args.experiment}.json")

    print(f"\n{len(ok)}/{len(res)} clean; {len(bad)}/{len(res)} flagged")
    print(f"findings by class (NOT a sample count): {disp['findings_by_class']}")
    print(f"findings by disposition: {disp['findings_by_disposition']}")
    print(f"features recomputed {record['n_features_recomputed']}, "
          f"mismatched {record['n_features_mismatched']}")
    for r in record["worst"][:5]:
        print(f"  ! {r['key']} [{r['stratum']}]: {r['problems'][:2]}")
    print(f"-> {OUT / f'retained_waveforms_{args.experiment}.json'}")
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
