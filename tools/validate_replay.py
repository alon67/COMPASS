#!/usr/bin/env python3
"""Deterministic replay of stored simulations (2026-08-06 audit, section 6.4.3).

    Rerunning only one friendly row per model is inadequate, while manually
    inspecting 98,048 voltage traces is neither scalable nor statistically well
    defined.

The layered design the audit specifies:

1. metadata and invariants for 100% of rows -- ``tools/validate_campaign.py``;
2. **at least one deterministic row from every primary shard**, selected by a
   published rule *before* its outputs are read;
3. **all scientifically risky strata** -- canonical rows, prior corners,
   failed/QC-flagged classes, nonspiking, spontaneous, depolarisation block and
   feature extrema, plus a random sample of ordinary rows;
4. **escalation by shard** -- a replay mismatch invalidates the whole shard, and
   clustered failures expand invalidation to every shard sharing the implicated
   model, mechanism or identity;
5. retained audit traces for everything replayed.

Selection rule (rule id ``blake2b-shard-v1``, published here so it cannot be
chosen after seeing the data): for each shard, take the row whose
``parameter_sample_id`` equals ``int(blake2b(shard_name, 8)) % n_rows``.  It
depends only on the file name and the row count.

Detection power.  For a simple random audit the sample size needed for
probability ``1 - alpha`` of seeing at least one defect at prevalence ``p`` is
``ceil(log(alpha)/log(1-p))``: ~2,995 replays for 95% detection of a 0.1%
defect rate, 598 for 0.5%, 299 for 1%.  One row per shard over the screening
and Sobol campaigns exceeds the first threshold while also covering every
worker output, seed range and file boundary.  Because implementation failures
cluster rather than occur independently, that calculation does not replace the
stratification or the whole-shard escalation rule.

    python tools/validate_replay.py --experiment screening --workers 12
    python tools/validate_replay.py --experiment screening --strata-only
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import sys
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
TRACES = ROOT / "results" / "traces" / "replay"

SELECTION_RULE = "blake2b-shard-v1"

# Columns that are DERIVED from the other stored columns rather than produced by
# the simulation. A validator bump changes them by design, and gate D5 validates
# them by exact recomputation. Counting them here would report "the simulation
# did not reproduce" when the simulation reproduced perfectly.
DERIVED_LABELS = frozenset({
    "physiology_class", "validator_version", "execution_valid", "protocol_valid",
    "feature_valid", "analysis_eligible", "failure_reason",
})

# Feature names whose DEFINITION changed under a documented schema drift
# (compass.workflows.simulate.DOCUMENTED_DRIFT).
#
# A difference here is NOT waived. It is *predicted*: features-2 changed
# `np.std(isi)` to `np.std(isi, ddof=1)` and required >=3 spikes, so a stored
# features-1 value must differ from its replay by exactly sqrt(n/(n-1)) for n
# interspike intervals, and must become undefined below 3 spikes. Anything else
# is a genuine reproduction failure. Verifying the prediction is what keeps this
# from being a hole in the gate.
SCHEMA_DRIFT_FIELDS = frozenset({"cv_isi"})


def explained_by_schema_drift(field: str, old: float, new: float,
                              spike_count: float | None) -> bool:
    """Is this difference exactly the declared features-1 -> features-2 change?"""
    if not any(field.endswith("." + f) for f in SCHEMA_DRIFT_FIELDS):
        return False
    if spike_count is None or not np.isfinite(spike_count) or spike_count < 3:
        # features-2 makes cv_isi undefined below 3 spikes; that IS the change.
        return np.isnan(new) if np.isfinite(old) else True
    n = float(spike_count) - 1.0          # number of ISIs
    if n < 2 or old == 0:
        return False
    predicted = old * np.sqrt(n / (n - 1.0))
    return bool(abs(new - predicted) <= max(1e-9, 1e-6 * abs(predicted)))

# Feature-level replay tolerances.  In an identical container a fixed-step rerun
# should be bit-identical; these exist for the cross-platform case the audit
# also asks for, and because a stored parquet round-trips through float64.
REPLAY_TOL = {
    "rheobase_nA": 1e-9,
    "_default_abs": 1e-6,
    "_default_rel": 1e-9,
}


def select_row(shard_name: str, n_rows: int) -> int:
    """Published deterministic rule; depends only on the name and the count."""
    h = hashlib.blake2b(shard_name.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % max(1, n_rows)


def risky_strata(df: pd.DataFrame, per_stratum: int = 3,
                 seed: int = 20260806) -> pd.DataFrame:
    """Rows the audit names as scientifically risky, plus ordinary controls."""
    rng = np.random.default_rng(seed)
    picks = []

    def take(sub, label):
        if not len(sub):
            return
        idx = rng.choice(len(sub), size=min(per_stratum, len(sub)), replace=False)
        chunk = sub.iloc[idx].copy()
        chunk["stratum"] = label
        picks.append(chunk)

    for _key, g in df.groupby(["model_id", "sampling_regime"]):
        take(g[g["simulation_status"] == "error"], "error")
        flags = g["qc_flags"].astype("string").fillna("")
        take(g[flags != ""], "qc_flagged")
        take(g[g["non_spiking"].astype(bool)], "nonspiking")
        if "physiology_class" in g:
            for cls in ("depolarisation_block", "spontaneous", "unstable"):
                take(g[g["physiology_class"] == cls], f"class_{cls}")
        # prior corners: the most extreme parameter draws in log space
        pcols = [c for c in g.columns if c.startswith("param.")]
        if pcols:
            x = np.log10(g[pcols].to_numpy(dtype=float))
            r = np.linalg.norm(x - np.nanmedian(x, axis=0), axis=1)
            order = np.argsort(r)
            take(g.iloc[order[-per_stratum:]], "prior_corner")
        # feature extrema on the primary readout
        col = "step_abs_hi.mean_firing_rate_Hz"
        if col in g:
            v = pd.to_numeric(g[col], errors="coerce")
            fin = g[np.isfinite(v)]
            if len(fin):
                take(fin.nsmallest(per_stratum, col), "rate_min")
                take(fin.nlargest(per_stratum, col), "rate_max")
        take(g.sample(min(per_stratum, len(g)), random_state=seed), "ordinary_random")

    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True)
    return out.drop_duplicates(subset=["run_id", "stratum"])


def _replay(job) -> dict:
    """Re-simulate one stored row in a fresh process and compare features."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    model_id, param_ids, scales, sample_id, regime, replicate, seed, prior_id, \
        shard, stored, keep_trace = job
    import numpy as _np

    from compass.workflows.simulate import run_block

    tmp = Path(ROOT) / "results" / "validation" / "_replay_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out_path = tmp / f"{model_id}_{regime}_{sample_id}_{os.getpid()}.parquet"
    try:
        run_block(model_id, _np.asarray([scales]), list(param_ids),
                  _np.asarray([sample_id]), regime, replicate, seed, prior_id,
                  out_path, identity=None)
        got = pd.read_parquet(out_path).iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"shard": shard, "run_id": f"{model_id}:{regime}:{replicate}:{sample_id}",
                "model_id": model_id, "status": "replay_error",
                "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        if out_path.exists():
            out_path.unlink()

    def undefined(v) -> bool:
        """None and NaN are the same thing: 'this feature has no value'.

        A stored parquet round-trips a missing float as NaN while a freshly
        built dict carries None. Treating that as a mismatch reported 69 fake
        failures per model and buried the real question.
        """
        if v is None:
            return True
        try:
            return bool(_np.isnan(float(v)))
        except (TypeError, ValueError):
            return False

    diffs, derived_drift, schema_drift = [], [], []
    compared = 0
    for k, want in stored.items():
        if k.startswith(("identity.", "param.", "_")) or k in ("run_id", "error"):
            continue
        have = got.get(k)
        if undefined(want) and undefined(have):
            continue
        # Derived labels are a pure function of the other columns and are
        # validated by recomputation in gate D5, not here. A validator bump
        # changes them by design; counting that as a replay failure would say
        # the SIMULATION did not reproduce, which is a different and false
        # claim.
        if k in DERIVED_LABELS:
            if str(want) != str(have):
                derived_drift.append(f"{k}: {want!r} -> {have!r}")
            continue
        if isinstance(want, str) or isinstance(have, str):
            compared += 1
            if str(want) != str(have):
                diffs.append(f"{k}: {want!r} -> {have!r}")
            continue
        try:
            w, h = float(want), float(have)
        except (TypeError, ValueError):
            continue
        compared += 1
        # spike count of the SAME protocol, needed to predict the drift
        proto = k.rsplit(".", 1)[0]
        sc = stored.get(f"{proto}.spike_count")
        try:
            sc = float(sc) if sc is not None else None
        except (TypeError, ValueError):
            sc = None

        if undefined(w) != undefined(h):
            if explained_by_schema_drift(k, w, h, sc):
                schema_drift.append(f"{k}: {want} -> {have} (predicted by features-2)")
            else:
                diffs.append(f"{k}: {want} -> {have} (definedness)")
            continue
        tol = REPLAY_TOL.get(k, REPLAY_TOL["_default_abs"])
        if abs(w - h) > max(tol, REPLAY_TOL["_default_rel"] * abs(w)):
            if explained_by_schema_drift(k, w, h, sc):
                schema_drift.append(
                    f"{k}: {w:.9g} -> {h:.9g} (= x sqrt(n/(n-1)), predicted by features-2)")
            else:
                diffs.append(f"{k}: {w:.9g} -> {h:.9g}")
    return {"shard": shard, "run_id": got.get("run_id"), "model_id": model_id,
            "status": "match" if not diffs else "mismatch",
            "n_compared": compared, "n_diff": len(diffs),
            "diffs": diffs[:8],
            "n_derived_label_drift": len(derived_drift),
            "derived_label_drift": derived_drift[:4],
            "n_schema_drift": len(schema_drift),
            "schema_drift": schema_drift[:4],
            "keep_trace": keep_trace}


def build_jobs(df: pd.DataFrame, shards, strata: pd.DataFrame,
               experiment: str) -> list:
    from compass.adapters.models import get_adapter

    wanted: dict[tuple, str] = {}
    for p in shards:
        sub = df[df["_shard_path"] == str(p.relative_to(ROOT))]
        if sub.empty:
            continue
        i = select_row(p.name, len(sub))
        row = sub.iloc[i]
        wanted[(row["model_id"], row["sampling_regime"], int(row["replicate"]),
                int(row["parameter_sample_id"]))] = "per_shard"
    for _, row in strata.iterrows():
        key = (row["model_id"], row["sampling_regime"], int(row["replicate"]),
               int(row["parameter_sample_id"]))
        wanted.setdefault(key, row.get("stratum", "risky"))

    pid_cache: dict[str, list[str]] = {}
    jobs = []
    for (mid, regime, rep, sid), tag in sorted(wanted.items(), key=lambda kv: str(kv[0])):
        m = ((df["model_id"] == mid) & (df["sampling_regime"] == regime)
             & (df["replicate"] == rep) & (df["parameter_sample_id"] == sid))
        sub = df[m]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if mid not in pid_cache:
            pid_cache[mid] = [s.parameter_id for s in get_adapter(mid).parameter_specs()]
        pids = pid_cache[mid]
        scales = [float(row[f"param.{p}"]) for p in pids]
        stored = {k: v for k, v in row.items()
                  if not k.startswith(("_", "identity."))}
        jobs.append((mid, pids, scales, sid, regime, int(rep), int(row["seed"]),
                     str(row["prior_id"]), row["_shard"], stored, tag))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of replays (a partial run is reported "
                         "as INDETERMINATE, never as a pass)")
    ap.add_argument("--strata-only", action="store_true")
    args = ap.parse_args()

    from validate_campaign import load_experiment  # noqa: PLC0415

    df, shards = load_experiment(args.experiment)
    if df.empty:
        print(f"no rows for {args.experiment}")
        return 1
    strata = risky_strata(df, per_stratum=args.per_stratum)
    jobs = build_jobs(df, [] if args.strata_only else shards, strata, args.experiment)
    total_planned = len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]

    p_detect = {p: math.ceil(math.log(0.05) / math.log(1 - p))
                for p in (0.001, 0.005, 0.01)}
    print(f"{args.experiment}: {len(df)} rows, {len(shards)} shards; "
          f"replaying {len(jobs)}/{total_planned} rows on {args.workers} workers")
    print(f"  95% detection needs {p_detect[0.001]} replays at 0.1% prevalence, "
          f"{p_detect[0.005]} at 0.5%, {p_detect[0.01]} at 1%")

    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_replay, jobs), 1):
            results.append(r)
            if r["status"] != "match":
                print(f"  [{r['status']}] {r['run_id']} ({r['shard']}): "
                      f"{'; '.join(r.get('diffs', [])[:3]) or r.get('detail', '')}",
                      flush=True)
            if i % 100 == 0:
                n_bad = sum(x["status"] != "match" for x in results)
                print(f"  {i}/{len(jobs)} replayed, {n_bad} mismatched", flush=True)

    res = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / f"replay_{args.experiment}.parquet", index=False)

    bad = res[res["status"] != "match"]
    # Escalation: a mismatch invalidates the whole shard, and a model with any
    # mismatch has every one of its shards invalidated.
    invalid_shards = sorted(bad["shard"].unique())
    clustered_models = sorted(bad["model_id"].unique())
    escalated = sorted({p.name for p in shards
                        if any(m in str(p) for m in clustered_models)})

    records = [gate_record(
        "REPLAY", args.experiment,
        [] if bad.empty else
        [f"{len(bad)} mismatches across {len(invalid_shards)} shards"],
        measurements={
            "n_derived_label_drift_rows": int((res["n_derived_label_drift"] > 0).sum())
            if "n_derived_label_drift" in res else 0,
            "n_schema_drift_rows": int((res["n_schema_drift"] > 0).sum())
            if "n_schema_drift" in res else 0,
            "drift_note": ("derived labels and documented schema drift are "
                           "reported separately: they are not reproduction "
                           "failures, and gate D5 validates them by exact "
                           "recomputation"),
            "n_rows_total": int(len(df)),
            "n_shards": len(shards),
            "n_replayed": int(len(res)),
            "n_planned": total_planned,
            "n_match": int((res["status"] == "match").sum()),
            "n_mismatch": int((res["status"] == "mismatch").sum()),
            "n_replay_error": int((res["status"] == "replay_error").sum()),
            "invalid_shards": invalid_shards[:50],
            "escalated_shards_if_clustered": len(escalated),
            "detection_power": p_detect,
        },
        tolerances=REPLAY_TOL,
        inputs={"selection_rule": SELECTION_RULE,
                "replayed_run_ids": sorted(res["run_id"].dropna().astype(str))[:500]},
        detail="one deterministic row per shard plus every risky stratum")]
    if args.limit and len(jobs) < total_planned:
        records[0].status = "INDETERMINATE"
        records[0].detail += (f" -- PARTIAL RUN ({len(jobs)}/{total_planned}); "
                              "a capped replay is not a pass")

    write_gate_records(records, OUT / f"replay_gate_{args.experiment}.json")
    print(f"\n{int((res['status'] == 'match').sum())}/{len(res)} rows reproduced "
          f"exactly -> {OUT / f'replay_gate_{args.experiment}.json'}")
    if not bad.empty:
        print(f"INVALIDATED SHARDS ({len(invalid_shards)}): {invalid_shards[:10]}")
        print(f"escalation would invalidate {len(escalated)} shards of "
              f"{clustered_models}")
    return 0 if records[0].passed else 1


if __name__ == "__main__":
    sys.exit(main())
