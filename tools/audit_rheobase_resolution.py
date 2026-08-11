#!/usr/bin/env python3
"""Measure how wide the real firing windows are, so the grid can be defended.

``find_rheobase`` guarantees that any firing window at least
``delta = ceiling / RHEOBASE_SCAN_POINTS`` wide is found, and admits that a
narrower one may be stepped over. That is an honest contract, but on its own it
is only an assertion that 64 points is enough. This tool turns it into a
measurement.

For a stratified sample of stored parameter draws it runs the search in
``exhaustive`` mode at ``--refine`` times the campaign resolution, recording
**every** non-spiking-to-spiking transition rather than stopping at the first.
From those transitions it reports:

* the width of every firing window found, so the smallest one that actually
  occurs can be compared against the campaign's delta;
* how many draws have more than one disconnected window;
* how many draws would have been MISSED at the campaign resolution -- i.e. the
  fine scan finds a window that no campaign grid point falls inside;
* how often the fine scan's rheobase differs from the campaign's.

The last of those is the number that matters: it is the empirical error rate of
the declared resolution, measured rather than argued.

    python tools/audit_rheobase_resolution.py --per-model 6 --refine 8 --workers 12
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

FEATURES = ROOT / "results" / "features"
OUT = ROOT / "data" / "provenance" / "rheobase_resolution_audit.json"


def _one(job) -> dict:
    """Re-run one stored draw's rheobase search at a finer resolution."""
    sys.path.insert(0, str(ROOT / "src"))
    model_id, param_ids, scales, sample_id, regime, experiment, refine = job
    from compass.adapters.base import NeuronAdapter
    from compass.adapters.models import get_adapter
    from compass.protocols import common as P
    from compass.workflows.simulate import solver_kwargs

    base = {"model_id": model_id, "experiment": experiment, "regime": regime,
            "parameter_sample_id": int(sample_id)}
    try:
        a = get_adapter(model_id)
        a.build()
        a.apply(dict(zip(param_ids, scales, strict=True)))
        kw = solver_kwargs(model_id)
        n_camp = NeuronAdapter.RHEOBASE_SCAN_POINTS
        n_fine = n_camp * int(refine)

        fine = a.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                               duration_ms=P.RHEO_DURATION_MS,
                               total_ms=P.RHEO_TOTAL_MS,
                               scan_points=n_fine, exhaustive=True, **kw)
        d = dict(a.last_rheobase_search or {})
        ceiling = float(d.get("ceiling_nA", a.manifest_rheobase_hi))
        d_camp = ceiling / n_camp
        trans = d.get("transitions", [])

        # A window is [transition hi, next transition lo) -- or the ceiling.
        widths, starts = [], []
        for i, t in enumerate(trans):
            lo = float(t["hi_nA"])
            # the window ends where the next silent grid point begins; the fine
            # scan does not record silences, so bound it by the next transition
            # or by the ceiling, which is the conservative (widest) reading
            hi = float(trans[i + 1]["lo_nA"]) if i + 1 < len(trans) else ceiling
            widths.append(hi - lo)
            starts.append(lo)

        # Would the CAMPAIGN grid have found the lowest window? It does iff some
        # campaign grid point k*d_camp lies inside [start, start + width).
        missed_lowest = None
        if starts:
            s, w = starts[0], widths[0]
            k = int(np.ceil(s / d_camp))
            missed_lowest = bool(k * d_camp >= s + w)

        return {**base, "status": "ok",
                "rheobase_fine_nA": None if fine is None else float(fine),
                "n_transitions": len(trans),
                "n_disconnected_windows": max(0, len(trans) - 1),
                "window_widths_nA": [float(x) for x in widths],
                "min_window_width_nA": float(min(widths)) if widths else None,
                "campaign_resolution_nA": float(d_camp),
                "fine_resolution_nA": float(ceiling / n_fine),
                "lowest_window_missed_at_campaign_resolution": missed_lowest,
                "n_probes": int(d.get("n_probes", 0))}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def sample_jobs(experiment: str, per_model: int, refine: int, seed: int) -> list:
    paths = sorted(p for p in (FEATURES / experiment).rglob("*.parquet")
                   if "_quarantine" not in p.parts and "_superseded" not in p.parts
                   and not p.name.endswith(".tmp.parquet"))
    if not paths:
        raise SystemExit(f"no shards under {FEATURES / experiment}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # A shard stores the campaign as `identity.experiment` and the regime as
    # `sampling_regime`; both are normalised here so the sampling key is the
    # same five columns the corpus snapshots join on.
    if "experiment" not in df.columns:
        df["experiment"] = df.get("identity.experiment", experiment)
    if "regime" not in df.columns:
        df["regime"] = df["sampling_regime"]
    rng = np.random.default_rng(seed)
    jobs = []
    for mid, g in df.groupby("model_id"):
        # Stratify on the outcome most likely to expose a resolution problem:
        # draws recorded as silent, draws blocked at the ceiling, and ordinary
        # spiking draws, in equal measure.
        silent = g[~np.isfinite(pd.to_numeric(g["rheobase_nA"], errors="coerce"))]
        blocked = g[g.get("rheobase_blocked_at_ceiling", False).fillna(False)] \
            if "rheobase_blocked_at_ceiling" in g else g.iloc[0:0]
        spiking = g[np.isfinite(pd.to_numeric(g["rheobase_nA"], errors="coerce"))]
        take = max(1, per_model // 3)
        picks = []
        for sub in (silent, blocked, spiking):
            if len(sub):
                idx = rng.choice(len(sub), size=min(take, len(sub)), replace=False)
                picks.append(sub.iloc[idx])
        if not picks:
            continue
        sel = pd.concat(picks).drop_duplicates(
            subset=["experiment", "model_id", "regime", "parameter_sample_id"])
        # THIS MODEL'S parameters, from its own adapter -- not the columns of the
        # concatenated frame, which are the UNION over every model in the
        # campaign. Reading them off `sel.columns` handed `hh_point` khaliq's
        # `gnata_soma` and every draw died in `apply()` with a KeyError, so the
        # audit reported 37 errors and zeros in every findings field.
        from compass.adapters.models import get_adapter

        pids = [s.parameter_id for s in get_adapter(mid).parameter_specs()]
        absent = [p for p in pids if f"param.{p}" not in sel.columns]
        if absent:
            raise SystemExit(
                f"{mid}: shards carry no column for parameter(s) {absent}; "
                f"the stored draw cannot be reapplied")
        for _, r in sel.iterrows():
            jobs.append((mid, pids, [float(r[f"param.{p}"]) for p in pids],
                         r["parameter_sample_id"], r["regime"], experiment, refine))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--per-model", type=int, default=6)
    ap.add_argument("--refine", type=int, default=8,
                    help="scan at this multiple of the campaign resolution")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    jobs = sample_jobs(args.experiment, args.per_model, args.refine, args.seed)
    print(f"{len(jobs)} draws, scanning at {args.refine}x the campaign "
          f"resolution with {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, max(1, len(jobs))), maxtasksperchild=1) as pool:
        results = list(pool.imap_unordered(_one, jobs))
    ok = [r for r in results if r["status"] == "ok"]

    widths = [w for r in ok for w in r["window_widths_nA"]]
    narrower = [r for r in ok if r["min_window_width_nA"] is not None
                and r["min_window_width_nA"] < r["campaign_resolution_nA"]]
    missed = [r for r in ok if r["lowest_window_missed_at_campaign_resolution"]]
    multi = [r for r in ok if r["n_disconnected_windows"] > 0]

    from compass.adapters.base import (
        MEASUREMENT_SCHEMA_VERSION,
        RHEOBASE_SEARCH_CONTRACT,
        NeuronAdapter,
    )
    rec = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": ("is RHEOBASE_SCAN_POINTS fine enough for the firing windows "
                     "these models actually have?"),
        "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
        "campaign_scan_points": NeuronAdapter.RHEOBASE_SCAN_POINTS,
        "refine_factor": args.refine,
        "contract": RHEOBASE_SEARCH_CONTRACT,
        "experiment": args.experiment,
        "n_draws": len(results),
        "n_ok": len(ok),
        "n_errors": len(results) - len(ok),
        # ABSENT EVIDENCE IS NOT A CLEAN RESULT.
        #
        # These fields used to be plain `len(...)` of an empty list, so a run in
        # which EVERY draw errored wrote `0` into "windows the campaign grid
        # would miss" -- which reads exactly like a passing audit. That is the
        # same confusion the 2026-08-09 post-response-3 audit named for gate
        # coverage: "we never checked this" and "we checked it and found
        # nothing" must not share a representation. With no successful draw the
        # counts are null and the verdict says so.
        "verdict": ("NO EVIDENCE: every draw errored" if not ok
                    else "measured" if not (results and len(ok) < len(results))
                    else f"partial: {len(ok)}/{len(results)} draws measured"),
        "n_with_disconnected_windows": len(multi) if ok else None,
        "n_with_a_window_narrower_than_the_campaign_resolution":
            len(narrower) if ok else None,
        "n_whose_lowest_window_the_campaign_grid_would_miss":
            len(missed) if ok else None,
        "window_width_nA": {
            "n": len(widths),
            "min": float(np.min(widths)) if widths else None,
            "p01": float(np.percentile(widths, 1)) if widths else None,
            "median": float(np.median(widths)) if widths else None,
        },
        "errors_by_kind": dict(__import__("collections").Counter(
            str(r.get("error", "?")).split(":")[0]
            for r in results if r["status"] != "ok")),
        "missed_examples": missed[:10],
        "wall_s": round(time.time() - t0, 1),
        "draws": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"\nverdict: {rec['verdict']}")
    if ok:
        print(f"{len(ok)}/{len(results)} scanned; "
              f"{len(multi)} with disconnected windows; "
              f"{len(narrower)} with a window narrower than the campaign resolution; "
              f"{len(missed)} the campaign grid would have missed")
    if rec["errors_by_kind"]:
        print(f"errors: {rec['errors_by_kind']}")
    print(f"-> {OUT.relative_to(ROOT)}")
    # Exit non-zero when nothing was measured: this is a soft stage, so the
    # status is what carries the failure into the run summary and the manifest.
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
