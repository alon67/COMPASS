#!/usr/bin/env python3
"""Regenerate the campaign rows that probe-3 changes (2026-08-09 audit, 4.1).

Two defects in the measurement procedure changed at the same time:

1. ``find_rheobase`` returned ``None`` whenever the search ceiling did not spike,
   which equates depolarisation block at high current with silence.
2. The rheobase-normalised protocol family was suppressed at ``rheobase == 0``
   only for models listed in ``SPONTANEOUSLY_ACTIVE``, so a draw that becomes
   spontaneous in an ordinarily silent model ran "1.5x rheobase" at 0 nA.

Why a targeted regeneration is defensible here
----------------------------------------------
probe-3 differs from probe-2 on a **strictly enumerable** set of rows, because
the new code takes a bit-identical path everywhere else:

* the ladder runs only when the ceiling probe does not spike;
* when the ladder finds nothing, the function still returns ``None``;
* the normalised-family change touches only rows with ``rheobase == 0``.

So the candidate set is exactly ``non_spiking`` rows (re-probed here, one by
one, with the real solver) plus ``rheobase == 0`` rows on models that are not
globally spontaneous (a deterministic NaN, no simulation needed).  Everything
else is unchanged *by construction*.

That argument is not taken on trust.  ``--verify`` re-runs whole shards from
scratch under probe-3 and compares them to the repaired shards cell by cell, so
the claim "this shard is what a probe-3 rerun would produce" is measured rather
than asserted.

Usage
-----
    tools/repair_rheobase.py --scan     [--workers 12] [--models ...]
    tools/repair_rheobase.py --apply
    tools/repair_rheobase.py --verify 6 [--workers 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
PROV = ROOT / "data" / "provenance"
SCAN_PATH = PROV / "rheobase_probe3_scan.json"
REPAIR_PATH = PROV / "rheobase_probe3_repair.json"
VERIFY_PATH = PROV / "rheobase_probe3_verification.json"

CAMPAIGNS = ["screening", "screening_expand", "screening_scramble2",
             "screening_v2_replication", "sobol", "pilot"]

# Columns a repaired row rewrites.  The absolute family and rest are NOT in this
# list: probe-3 cannot change them, and the scan asserts that they in fact do not
# change when a row is fully re-simulated.
NORMALISED_PREFIXES = ("step_rheo_1p5x.", "step_rheo_3p0x.")

# The same protocols as they appear in `qc_flags`, which is namespaced with a
# COLON rather than a dot.  Keeping one tuple for both is how the first attempt
# at this repair silently stripped nothing.
NORMALISED_FLAG_PREFIXES = ("step_rheo_1p5x:", "step_rheo_3p0x:")


def shard_frames(campaigns=CAMPAIGNS, models=None):
    """Yield (campaign, model, path) for every feature shard."""
    for camp in campaigns:
        base = RESULTS / "features" / camp
        if not base.exists():
            continue
        for mdir in sorted(base.iterdir()):
            if not mdir.is_dir() or (models and mdir.name not in models):
                continue
            for p in sorted(mdir.glob("*.parquet")):
                yield camp, mdir.name, p


# --------------------------------------------------------------------------- #
# phase 1: re-probe
# --------------------------------------------------------------------------- #
def _reprobe_chunk(args) -> list[dict]:
    """Re-probe a list of samples of ONE model in a fresh process."""
    model_id, jobs = args
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from compass.adapters.models import get_adapter
    from compass.protocols import common as P
    from compass.workflows.simulate import _feature_row, solver_kwargs

    kw = solver_kwargs(model_id)
    protocols = P.common_core(model_id)
    adapter = get_adapter(model_id).build()
    out = []
    for job in jobs:
        adapter.reset()
        rec = {"model_id": model_id, "shard": job["shard"],
               "run_id": job["run_id"], "stored_rheobase": job["stored_rheobase"]}
        try:
            adapter.apply(job["params"])
            rheo = adapter.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                                         duration_ms=P.RHEO_DURATION_MS,
                                         total_ms=P.RHEO_TOTAL_MS, **kw)
            search = dict(adapter.last_rheobase_search or {})
            rec["new_rheobase"] = None if rheo is None else float(rheo)
            rec["blocked_at_ceiling"] = bool(search.get("blocked_at_ceiling"))
            rec["n_probes"] = int(search.get("n_probes", 0))
            if rheo is not None or job.get("force_recompute"):
                # The row changes: recompute the WHOLE row, so the absolute
                # family can be compared against the stored values as a
                # determinism check rather than assumed to be unaffected.
                feats, flags = _feature_row(adapter, kw, rheo, protocols)
                rec["features"] = {k: (None if v is None or not np.isfinite(v) else float(v))
                                   for k, v in feats.items()}
                rec["qc_flags"] = ";".join(sorted(set(flags)))
        except Exception as exc:                                  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def _recovery_run_ids() -> set[str]:
    """run_ids that the run_id-keyed apply could have written across campaigns.

    The first ``--apply`` keyed the fresh rows on ``run_id`` alone, which is
    unique within a campaign but not between campaigns, so 546 rows in campaigns
    the scan never touched received another campaign's simulation.  Those rows
    are recovered by re-simulating them from their own stored parameters, which
    are untouched -- not by trying to undo the write.
    """
    if not SCAN_PATH.exists():
        return set()
    return {r["run_id"] for r in json.loads(SCAN_PATH.read_text())["rows"]
            if r.get("new_rheobase") is not None}


def scan(workers: int, models=None, chunk_target: int = 25,
         recover: bool = False) -> dict:
    """Re-probe every stored non-spiking row under probe-3.

    With ``recover``, also re-simulate every row whose ``run_id`` matches one the
    earlier run_id-keyed apply could have overwritten, whatever its current
    ``non_spiking`` value.  Re-simulation is the recovery: a row's parameters are
    never rewritten, so a fresh run from them is the true probe-3 answer.
    """
    from multiprocessing import get_context

    recovery = _recovery_run_ids() if recover else set()
    jobs_by_model: dict[str, list[dict]] = {}
    n_rows = 0
    for _camp, model, path in shard_frames(models=models):
        df = pd.read_parquet(path)
        if "non_spiking" not in df.columns:
            continue
        sel = df["non_spiking"].fillna(False).astype(bool) & (df["simulation_status"] == "ok")
        if recovery:
            sel = sel | (df["run_id"].isin(recovery)
                         & (df["simulation_status"] == "ok"))
        if not sel.any():
            continue
        pcols = [c for c in df.columns if c.startswith("param.")]
        rel = str(path.relative_to(RESULTS))
        for _, row in df[sel].iterrows():
            jobs_by_model.setdefault(model, []).append({
                "shard": rel, "run_id": row["run_id"],
                "force_recompute": bool(recovery) and row["run_id"] in recovery,
                "stored_rheobase": (None if not np.isfinite(row.get("rheobase_nA", np.nan))
                                    else float(row["rheobase_nA"])),
                "params": {c[len("param."):]: float(row[c]) for c in pcols},
            })
            n_rows += 1

    tasks = []
    for model, jobs in jobs_by_model.items():
        for i in range(0, len(jobs), chunk_target):
            tasks.append((model, jobs[i:i + chunk_target]))
    # Longest models first so the tail does not dominate the wall clock.
    tasks.sort(key=lambda t: -len(t[1]))
    print(f"re-probing {n_rows} non-spiking rows in {len(tasks)} chunks "
          f"on {workers} workers", flush=True)

    t0 = time.time()
    results: list[dict] = []
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, part in enumerate(pool.imap_unordered(_reprobe_chunk, tasks), 1):
            results.extend(part)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} chunks, {len(results)} rows, "
                      f"{time.time() - t0:.0f}s", flush=True)

    changed = [r for r in results if r.get("new_rheobase") is not None]
    recomputed = [r for r in results if r.get("features") is not None]
    errors = [r for r in results if "error" in r]
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "measurement_schema_from": "probe-2",
        "measurement_schema_to": "probe-3",
        "n_reprobed": len(results),
        "n_changed": len(changed),
        "n_recomputed": len(recomputed),
        "recovery_mode": bool(recovery),
        "n_probe_errors": len(errors),
        "wall_s": round(time.time() - t0, 1),
        "by_model": {
            m: {"reprobed": sum(1 for r in results if r["model_id"] == m),
                "changed": sum(1 for r in changed if r["model_id"] == m)}
            for m in sorted({r["model_id"] for r in results})},
        "rows": results,
    }
    SCAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCAN_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(f"-> {SCAN_PATH.relative_to(ROOT)}  changed={len(changed)} "
          f"errors={len(errors)}")
    return payload


# --------------------------------------------------------------------------- #
# phase 2: rewrite the shards
# --------------------------------------------------------------------------- #
def apply_repair() -> dict:
    """Rewrite every shard so it holds what a probe-3 rerun would hold."""
    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION
    from compass.protocols import common as P
    from compass.workflows.simulate import VALIDATOR_VERSION, classify_row, write_sidecar

    if not SCAN_PATH.exists():
        raise SystemExit(f"run --scan first ({SCAN_PATH} is absent)")
    scanned = json.loads(SCAN_PATH.read_text())
    if scanned["n_probe_errors"]:
        raise SystemExit(f"{scanned['n_probe_errors']} probe errors in the scan; "
                         "resolve them before rewriting shards")
    # KEYED ON (shard, run_id), NOT run_id ALONE.
    #
    # `run_id` is `model:regime:replicate:sample_id` and is unique *within* a
    # campaign -- gate D1 checks exactly that -- but `screening`, `sobol` and
    # `screening_scramble2` all contain `hh_point:E1:1:0`.  Keying on run_id
    # alone therefore wrote one campaign's fresh row over the same-named rows in
    # every other campaign: 2,392 rows rewritten for 1,846 real ones, and 20,109
    # absolute-family cells changed that probe-3 cannot touch.  The
    # absolute-family assertion below is what caught it.
    by_key = {(r["shard"], r["run_id"]): r for r in scanned["rows"]
              if r.get("features") is not None}

    log = {"generated_utc": datetime.now(UTC).isoformat(),
           "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
           "validator_version": VALIDATOR_VERSION,
           "shards": 0, "shards_rewritten": 0,
           "rows_rheobase_recovered": 0, "rows_normalised_voided": 0,
           "rows_qc_flags_stripped": 0,
           "absolute_family_mismatches": [], "by_model": {}}

    for _camp, model, path in shard_frames():
        log["shards"] += 1
        df = pd.read_parquet(path)
        spont = model in P.SPONTANEOUSLY_ACTIVE
        stat = log["by_model"].setdefault(
            model, {"recovered": 0, "voided": 0, "shards": 0})

        # ---- (a) rows the ladder rescued: written from the fresh simulation --
        rel = str(path.relative_to(RESULTS))
        rescued = [i for i, rid in enumerate(df["run_id"]) if (rel, rid) in by_key]
        for i in rescued:
            idx = df.index[i]
            rec = by_key[(rel, df.at[idx, "run_id"])]
            for col, val in rec["features"].items():
                if col not in df.columns:
                    df[col] = np.nan
                # The absolute family and rest CANNOT change under probe-3.
                # Verify rather than assume, then write the fresh value anyway.
                if not col.startswith(NORMALISED_PREFIXES):
                    old = df.at[idx, col]
                    new = np.nan if val is None else val
                    both_nan = (not np.isfinite(old)) and (not np.isfinite(new))
                    if not both_nan and not np.isclose(old, new, rtol=0, atol=0,
                                                       equal_nan=True):
                        log["absolute_family_mismatches"].append(
                            {"shard": rel, "run_id": rec["run_id"], "column": col,
                             "stored": float(old), "reprobed": float(new)})
                df.at[idx, col] = np.nan if val is None else val
            df.at[idx, "rheobase_nA"] = (np.nan if rec["new_rheobase"] is None
                                         else rec["new_rheobase"])
            df.at[idx, "non_spiking"] = rec["new_rheobase"] is None
            df.at[idx, "qc_flags"] = rec.get("qc_flags", "")
            df.at[idx, "rheobase_blocked_at_ceiling"] = bool(rec["blocked_at_ceiling"])
            stat["recovered"] += 1
            log["rows_rheobase_recovered"] += 1

        # ---- (b) rows whose normalised family was measured at 0 nA -----------
        rheo = pd.to_numeric(df["rheobase_nA"], errors="coerce").to_numpy(float)
        void = np.isfinite(rheo) & (rheo == 0.0)
        ncols = [c for c in df.columns if c.startswith(NORMALISED_PREFIXES)]
        n_void = 0
        if void.any() and ncols:
            block = df.loc[void, ncols]
            n_void = int((~block.isna()).any(axis=1).sum())
            df.loc[void, ncols] = np.nan
        stat["voided"] += n_void

        # ...and the QC FLAGS those protocols raised.
        #
        # Voiding the feature columns is not enough: `qc_flags` still carried
        # `step_rheo_1p5x:unstable_baseline` and friends for protocols that
        # probe-3 never runs, so a stored row was not what a re-run produces.
        # `--verify` caught this on two of six re-simulated shards -- which is
        # what it is for -- and it affects 15,143 rows.  No classification can
        # change: a flag KIND that appears on a normalised protocol also appears
        # on an absolute one in every affected row (measured, 0 exceptions), and
        # `classify_row` reads the SET of kinds, not the per-protocol strings.
        #
        # Two things this needs and the feature-voiding does not.  QC flags are
        # namespaced with a COLON (`step_rheo_1p5x:unstable_baseline`) while
        # feature columns use a dot, so NORMALISED_PREFIXES does not match them;
        # and the flags must be stripped wherever the normalised family is
        # undefined, which includes rheobase = NaN, not only rheobase = 0.
        n_flags = 0
        undefined = ~np.isfinite(rheo) | (np.isfinite(rheo) & (rheo == 0.0))
        if undefined.any() and "qc_flags" in df.columns:
            def _strip(v):
                kept = [f for f in str(v or "").split(";")
                        if f and not f.startswith(NORMALISED_FLAG_PREFIXES)]
                return ";".join(kept)

            before = df.loc[undefined, "qc_flags"].fillna("").astype(str)
            after = before.map(_strip)
            n_flags = int((before != after).sum())
            df.loc[undefined, "qc_flags"] = after
        stat["qc_flags_stripped"] = stat.get("qc_flags_stripped", 0) + n_flags
        log["rows_qc_flags_stripped"] = log.get("rows_qc_flags_stripped", 0) + n_flags
        log["rows_normalised_voided"] += n_void

        # ---- (c) every row: the probe-3 / row-validity-3 columns -------------
        if "rheobase_blocked_at_ceiling" not in df.columns:
            df["rheobase_blocked_at_ceiling"] = False
        df["rheobase_blocked_at_ceiling"] = (
            df["rheobase_blocked_at_ceiling"].fillna(False).astype(bool))
        cls_rows = []
        for _, row in df.iterrows():
            flags = [f for f in str(row.get("qc_flags", "") or "").split(";") if f]
            r = row.get("rheobase_nA")
            r = None if (r is None or not np.isfinite(r)) else float(r)
            err = row.get("error") if row.get("simulation_status") == "error" else None
            cls_rows.append(classify_row(row.to_dict(), flags, r, spont, err))
        for key in ("execution_valid", "protocol_valid", "normalised_protocol_valid",
                    "feature_valid", "physiology_class", "analysis_eligible",
                    "validator_version", "failure_reason"):
            df[key] = [c[key] for c in cls_rows]

        # ---- (d) restamp identity and rewrite atomically ---------------------
        df["identity.measurement_schema"] = MEASUREMENT_SCHEMA_VERSION
        identity = {c[len("identity."):]: df.iloc[0][c]
                    for c in df.columns if c.startswith("identity.")}
        identity.pop("n_rows", None)
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        write_sidecar(path, identity, n_rows=len(df))
        log["shards_rewritten"] += 1
        stat["shards"] += 1

    REPAIR_PATH.write_text(json.dumps(log, indent=1, sort_keys=False))
    print(f"-> {REPAIR_PATH.relative_to(ROOT)}")
    print(f"   shards rewritten      : {log['shards_rewritten']}")
    print(f"   rheobase recovered    : {log['rows_rheobase_recovered']}")
    print(f"   normalised voided     : {log['rows_normalised_voided']}")
    print(f"   qc flags stripped     : {log['rows_qc_flags_stripped']}")
    print(f"   absolute-family drift : {len(log['absolute_family_mismatches'])}")
    return log


# --------------------------------------------------------------------------- #
# phase 3: verify the equivalence claim by re-running whole shards
# --------------------------------------------------------------------------- #
def _rerun_shard(args) -> dict:
    """Re-simulate one whole shard from its stored parameters, from scratch."""
    rel, model_id = args
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from compass.workflows.simulate import run_block

    path = RESULTS / rel
    df = pd.read_parquet(path)
    pcols = [c for c in df.columns if c.startswith("param.")]
    scales = df[pcols].to_numpy(dtype=float)
    param_ids = [c[len("param."):] for c in pcols]
    out = Path(os.environ["COMPASS_VERIFY_DIR"]) / rel.replace("/", "__")
    res = run_block(model_id, scales, param_ids,
                    df["parameter_sample_id"].to_numpy(),
                    str(df["sampling_regime"].iloc[0]),
                    int(df["replicate"].iloc[0]), int(df["seed"].iloc[0]),
                    str(df["prior_id"].iloc[0]), out, identity=None)
    fresh = pd.read_parquet(out)

    # Compare every column the repaired shard and a from-scratch probe-3 run
    # both carry.  Identity/provenance columns are excluded: they are metadata
    # about the run, not measurements.
    skip = {"run_id", "seed", "error"}
    cols = [c for c in df.columns
            if c in fresh.columns and not c.startswith("identity.") and c not in skip]
    diffs = []
    for c in cols:
        a, b = df[c], fresh[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            bad = ~((np.isnan(av) & np.isnan(bv)) | (av == bv))
        else:
            bad = (a.astype(str).to_numpy() != b.astype(str).to_numpy())
        if bad.any():
            diffs.append({"column": c, "n_rows": int(bad.sum()),
                          "example_stored": str(a[bad].iloc[0]),
                          "example_fresh": str(b[bad].iloc[0])})
    out.unlink(missing_ok=True)
    return {"shard": rel, "model_id": model_id, "n_rows": len(df),
            "n_columns_compared": len(cols), "wall_s": res["wall_s"],
            "differences": diffs}


def verify(n_shards: int, workers: int, seed: int = 20260809) -> dict:
    """Re-run whole shards from scratch and diff them against the repaired ones.

    This is the measurement that turns "probe-3 changes only the enumerated
    rows" from an argument into evidence.
    """
    import tempfile
    from multiprocessing import get_context

    rng = np.random.default_rng(seed)
    pool_of_shards = [(str(sp.relative_to(RESULTS)), m) for _, m, sp in shard_frames()]
    # Stratify by model so the sample is not all cheap point neurons, and
    # prefer shards that the repair actually touched.
    repaired = set()
    if SCAN_PATH.exists():
        repaired = {r["shard"] for r in json.loads(SCAN_PATH.read_text())["rows"]
                    if r.get("new_rheobase") is not None}
    by_model: dict[str, list] = {}
    for rel, m in pool_of_shards:
        by_model.setdefault(m, []).append((rel, m))
    picked = []
    for m in sorted(by_model):
        cand = [s for s in by_model[m] if s[0] in repaired] or by_model[m]
        picked.append(cand[int(rng.integers(len(cand)))])
    rng.shuffle(picked)
    picked = picked[:n_shards]

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["COMPASS_VERIFY_DIR"] = tmp
        ctx = get_context("spawn")
        with ctx.Pool(min(workers, len(picked)), maxtasksperchild=1) as pool:
            results = list(pool.imap_unordered(_rerun_shard, picked))

    n_diff = sum(len(r["differences"]) for r in results)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": ("does a from-scratch probe-3 re-run of a repaired shard "
                     "reproduce the repaired shard exactly?"),
        "n_shards": len(results),
        "n_rows": sum(r["n_rows"] for r in results),
        "n_columns_with_differences": n_diff,
        "verdict": "equivalent" if n_diff == 0 else "NOT equivalent",
        "shards": sorted(results, key=lambda r: r["shard"]),
    }
    VERIFY_PATH.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(f"-> {VERIFY_PATH.relative_to(ROOT)}: {payload['verdict']} "
          f"({payload['n_rows']} rows over {payload['n_shards']} shards)")
    for r in results:
        if r["differences"]:
            print(f"   {r['shard']}: {r['differences'][:3]}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--recover", action="store_true",
                    help="also re-simulate every row a run_id-keyed apply could "
                         "have overwritten across campaigns")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", type=int, default=0, metavar="N_SHARDS")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    if not (a.scan or a.apply or a.verify):
        ap.error("pick at least one of --scan / --apply / --verify")
    if a.scan:
        scan(a.workers, a.models, recover=a.recover)
    if a.apply:
        apply_repair()
    if a.verify:
        verify(a.verify, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
