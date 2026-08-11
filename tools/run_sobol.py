"""Variance-based sensitivity campaign (blueprint Task 12, WP7).

Sobol indices must be computed on an intact Saltelli design drawn from the
declared prior -- never on a phenotype-filtered ensemble (blueprint 16.4).  This
tool therefore runs its own simulations rather than reusing the screening
ensembles.

    python tools/run_sobol.py --models hh_point pospischil_rs
    python tools/run_sobol.py --analyse-only
"""

from __future__ import annotations

import argparse
import glob
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
from compass.analysis.sensitivity import (  # noqa: E402
    CONF_WIDTH_CEILING,  # noqa: E402
    VALID_FRACTION_FLOOR,  # noqa: E402
    saltelli_design,
    sobol_convergence,
    sobol_indices,
)
from compass.sampling.priors import scales_from_unit  # noqa: E402
from compass.workflows.simulate import _worker, shard_is_reusable  # noqa: E402

# Only code that can change a simulated feature value belongs in a shard's
# identity.  `run_campaign.code_hash()` hashes all of `src/`, which is right for
# a campaign manifest but wrong here: editing a plotting or statistics module
# would invalidate every stored simulation and force an unnecessary re-run of
# the whole design.  These are the packages a feature row actually depends on.
SIM_CODE_PATHS = (
    "compass/adapters",
    "compass/protocols",
    "compass/features",
    "compass/sampling",
    "compass/workflows/simulate.py",
)


def sim_code_hash() -> str:
    """Hash of the simulation-relevant source only (see SIM_CODE_PATHS)."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    files: list[Path] = []
    for rel in SIM_CODE_PATHS:
        p = ROOT / "src" / rel
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    for f in sorted(files):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()

FEATURES = [
    "rheobase_nA",
    "hyperpol_abs.input_resistance_Mohm",
    "rest.resting_potential_mV",
    "step_abs_hi.mean_firing_rate_Hz",
    "step_abs_hi.ap_amplitude_mV",
    "step_abs_hi.ap_halfwidth_ms",
    "step_abs_hi.adaptation_index",
]

# Base sample size per model, chosen so the Saltelli design N*(2D+2) stays
# affordable at each model's measured per-sample cost.
N_BASE = {
    "hh_point": 256, "hh_cable": 256, "pospischil_rs": 256, "pospischil_fs": 256,
    "khaliq_purkinje": 128, "mainen_l5pc": 32, "bbp_l5pc": 64, "migliore_ca1": 32,
}
BLOCK = 64
SEED = 20260804
FEAT_DIR = ROOT / "results" / "features" / "sobol"
OUT = ROOT / "results" / "analyses"


def shard_identity(model_id: str) -> dict:
    """Everything that must be identical for a stored shard to be reusable.

    A shard is a cached simulation result.  Reusing one whose inputs have
    changed silently mixes two experiments in one table -- which is exactly what
    happened to `hh_point`, `khaliq_purkinje` and `bbp_l5pc`: their Sobol shards
    were produced under the pre-audit solver settings, those models were later
    re-simulated at corrected time steps for the screening campaign, and the
    Sobol resume check (`if not out.exists()`) happily kept the stale shards.
    Nothing failed; the numbers were just from a different model.
    Caught by the 2026-08-06 audit (C3).
    """
    from compass.workflows.simulate import CAMPAIGN_SOLVER

    return {
        "sim_code_hash": sim_code_hash(),
        "solver": dict(CAMPAIGN_SOLVER[model_id]),
        "design": {"kind": "saltelli", "n_base": N_BASE[model_id], "seed": SEED,
                   "calc_second_order": True, "block": BLOCK},
        "features": list(FEATURES),
    }


def _identity_path(shard: Path) -> Path:
    return shard.with_suffix(".identity.json")


# `_shard_is_current` used to live here and it could never return True.
#
# It compared the sidecar on disk against `shard_identity()` -- the RICH dict
# (`sim_code_hash`, `solver`, `design`, `features`) -- while the sidecar is
# written from `flat_identity()`, which has different keys and different value
# types. Two dictionaries that share no schema are never equal, so every Sobol
# shard was quarantined and re-simulated on every invocation: eight hours of
# NEURON discarded and redone each time, with the previous copies piling up
# under `_superseded/`. Nothing failed and nothing warned, because re-running a
# deterministic simulation produces the same bytes -- verified when this was
# found, 44 of 44 re-simulated shards were byte-identical to the originals they
# had displaced.
#
# It is the same failure as the receipt checks in this revision: a comparison
# whose outcome is fixed by its own construction rather than by the thing it
# compares. The reuse decision now goes through `shard_is_reusable`, which is
# the ONE rule the screening campaign uses and the same rule that decides,
# twenty lines below, what actually gets simulated -- so the quarantine and the
# work list can no longer disagree.


def flat_identity(model_id: str, identity: dict, start: int, n_rows: int) -> dict:
    """The Sobol shard identity in the flat form shard_is_reusable() compares.

    Sobol keeps its own richer `shard_identity()` (design, features, solver);
    this flattens it into the same string-valued dictionary the screening
    campaign uses, so there is ONE reuse rule in the project rather than two
    that can drift apart.
    """
    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION
    from compass.features.extract import FEATURE_SCHEMA_VERSION

    # `sim_code_hash` is RECORDED, not compared -- the same scoping decision the
    # screening campaign makes (see tools/run_campaign.py job_identity). A byte
    # hash of the simulation source invalidates every stored shard when a
    # comment changes, and a rule that discards correct data gets switched off.
    # The compared fields are the declared semantic versions; an undeclared
    # change to the extractor is caught by tests/unit/test_solver_contract.py
    # instead, which is earlier and cheaper than throwing away simulations.
    return {
        "experiment": "sobol",
        "model_id": model_id,
        "regime": "SOBOL",
        "block_start": int(start),
        "n_rows": int(n_rows),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
        "solver": json.dumps(identity["solver"], sort_keys=True),
        "design": json.dumps(identity["design"], sort_keys=True),
        "features": json.dumps(identity["features"], sort_keys=True),
        "recorded_only.sim_code_hash": identity["sim_code_hash"],
    }


def simulate(models: list[str], workers: int) -> None:
    jobs, quarantined = [], []
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for mid in models:
        specs = get_adapter(mid).parameter_specs()
        pids = [s.parameter_id for s in specs]
        n_base = N_BASE[mid]
        identity = shard_identity(mid)
        u = saltelli_design(len(specs), n_base, seed=SEED, calc_second_order=True)
        scales = scales_from_unit(u, specs)
        (FEAT_DIR / mid).mkdir(parents=True, exist_ok=True)
        np.save(FEAT_DIR / f"{mid}_design.npy", u)
        for start in range(0, len(scales), BLOCK):
            stop = min(start + BLOCK, len(scales))
            out = FEAT_DIR / mid / f"S_{start:06d}.parquet"
            ident = flat_identity(mid, identity, start, stop - start)
            if out.exists():
                ok, why = shard_is_reusable(out, ident, stop - start)
                if not ok:
                    # Move aside, never delete.  Overwriting an artefact is how
                    # the Mainen convergence evidence was lost (audit C1).
                    sup = FEAT_DIR / "_superseded" / stamp / mid
                    sup.mkdir(parents=True, exist_ok=True)
                    out.rename(sup / out.name)
                    if _identity_path(out).exists():
                        _identity_path(out).rename(sup / _identity_path(out).name)
                    quarantined.append(f"{mid}/{out.name} ({why})")
            jobs.append((mid, scales[start:stop], pids, np.arange(start, stop),
                         "SOBOL", 1, SEED, f"saltelli_N{n_base}", str(out), ident))

    if quarantined:
        print(f"quarantined {len(quarantined)} shard(s) whose identity no longer "
              f"matches -> results/features/sobol/_superseded/{stamp}/", flush=True)
        for q in quarantined[:10]:
            print(f"    {q}", flush=True)
        if len(quarantined) > 10:
            print(f"    ... and {len(quarantined) - 10} more", flush=True)

    # Identity-validated reuse, the same fail-closed rule the screening
    # campaign uses (audit 5.2). `_worker` quarantines anything that does not
    # match rather than silently reusing it.
    todo = [j for j in jobs
            if not shard_is_reusable(Path(j[-2]), j[-1], len(j[3]))[0]]
    print(f"sobol: {len(jobs)} blocks, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for i, _st in enumerate(pool.imap_unordered(_worker, todo), 1):
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} blocks  {(time.time()-t0)/60:.1f} min", flush=True)

    # `run_block` writes each shard's sidecar itself, atomically with the
    # parquet, so there is no window in which a shard exists without its
    # identity and no post-hoc stamping pass that could stamp a stale file.


def analyse(models: list[str]) -> None:
    rows, conv_rows, s2_rows = [], [], []
    for mid in models:
        files = sorted(glob.glob(str(FEAT_DIR / mid / "*.parquet")))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values("parameter_sample_id").reset_index(drop=True)
        specs = get_adapter(mid).parameter_specs()
        names = [s.parameter_id for s in specs]
        n_base = N_BASE[mid]
        expected = n_base * (2 * len(names) + 2)
        if len(df) != expected:
            print(f"  {mid}: have {len(df)}/{expected} rows, skipping "
                  f"(Sobol needs the intact design)", flush=True)
            continue
        for feat in FEATURES:
            if feat not in df.columns:
                continue
            y = df[feat].to_numpy(dtype=float)
            try:
                res = sobol_indices(y, names, n_base, output_name=feat, seed=SEED)
            except ValueError as exc:
                rows.append({"model_id": mid, "feature": feat, "error": str(exc)[:120],
                             "quotable": False, "valid_fraction": float("nan")})
                continue
            # Quotability now depends on PRECISION and PHYSICAL PLAUSIBILITY,
            # not only on completeness (2026-08-06 audit, M8 and 7.8).
            #
            # The previous rule was `valid_fraction >= 0.9` alone, which called
            # an index quotable while its confidence width exceeded the estimate
            # itself, and while some interaction fractions were negative and
            # some mean(ST-S1) values exceeded 1 -- both signatures of estimator
            # noise rather than of interaction mass.  All four conditions must
            # hold, and each is reported separately so a failure says why.
            valid_fraction = res.n_valid / len(y)
            max_s1_conf = float(np.nanmax(res.S1_conf))
            max_st_conf = float(np.nanmax(res.ST_conf))
            sum_st = float(np.nansum(res.ST))
            complete = bool(valid_fraction >= VALID_FRACTION_FLOOR)
            # A confidence half-width wider than the largest index it describes
            # means the ranking is not resolved at this base N.
            max_s1 = float(np.nanmax(np.abs(res.S1))) if np.isfinite(res.S1).any() else 0.0
            precise = bool(np.isfinite(max_s1_conf) and max_s1 > 0
                           and max_s1_conf <= CONF_WIDTH_CEILING * max_s1)
            # An interaction fraction outside [0, 1] or a mean(ST-S1) above 1 is
            # not a physical quantity; it is the estimator disagreeing with
            # itself.
            frac = res.interaction_fraction
            physical = bool(np.isfinite(frac) and -1e-9 <= frac <= 1.0 + 1e-9
                            and np.isfinite(res.mean_total_minus_first)
                            and abs(res.mean_total_minus_first) <= 1.0)
            reasons = []
            if not complete:
                reasons.append(f"valid_fraction {valid_fraction:.3f} < {VALID_FRACTION_FLOOR}")
            if not precise:
                reasons.append(f"max S1 confidence {max_s1_conf:.3f} > "
                               f"{CONF_WIDTH_CEILING} x max|S1| {max_s1:.3f}")
            if not physical:
                reasons.append(f"interaction_fraction {frac:.3f} or "
                               f"mean(ST-S1) {res.mean_total_minus_first:.3f} "
                               f"outside the physical range")
            rows.append({
                "model_id": mid, "feature": feat, "n_base": n_base,
                "n_valid": res.n_valid, "n_total": len(y),
                "valid_fraction": valid_fraction,
                "complete": complete,
                "precise": precise,
                "physically_plausible": physical,
                "quotable": bool(complete and precise and physical),
                "not_quotable_because": "; ".join(reasons),
                "interaction_fraction": res.interaction_fraction,
                "mean_ST_minus_S1": res.mean_total_minus_first,
                "sum_S1": float(np.nansum(res.S1)),
                "sum_ST": sum_st,
                "max_S1": max_s1,
                "max_S1_conf": max_s1_conf,
                "max_ST_conf": max_st_conf,
            })
            for j, nm in enumerate(names):
                s2_rows.append({"model_id": mid, "feature": feat, "parameter": nm,
                                "S1": float(res.S1[j]), "S1_conf": float(res.S1_conf[j]),
                                "ST": float(res.ST[j]), "ST_conf": float(res.ST_conf[j]),
                                "ST_minus_S1": float(res.ST[j] - res.S1[j])})
            if res.S2 is not None:
                s2 = res.S2
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        if np.isfinite(s2[a, b]):
                            s2_rows.append({
                                "model_id": mid, "feature": feat,
                                "parameter": f"{names[a]}*{names[b]}",
                                "S1": np.nan, "S1_conf": np.nan,
                                "ST": np.nan, "ST_conf": np.nan,
                                "S2": float(s2[a, b]),
                                "S2_conf": float(res.S2_conf[a, b])})
            for c in sobol_convergence(y, names, n_base, output_name=feat, seed=SEED):
                c.update({"model_id": mid, "feature": feat})
                conv_rows.append(c)
        print(f"  {mid}: analysed {len(FEATURES)} features (N={n_base}, "
              f"{len(df)} runs)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "sobol_summary.parquet", index=False)
    pd.DataFrame(s2_rows).to_parquet(OUT / "sobol_indices.parquet", index=False)
    pd.DataFrame(conv_rows).to_parquet(OUT / "sobol_convergence.parquet", index=False)
    print(f"wrote sobol tables ({len(rows)} summaries) -> {OUT}")


def verify(models: list[str], workers: int) -> dict[str, bool]:
    """Re-simulate one stored block per model and compare it bit-for-bit.

    Existing shards predate identity stamping, so their provenance cannot be
    read off the artefact.  Writing a stamp onto them anyway would assert
    exactly the thing that cannot be checked -- the mistake that cost this
    project the Mainen convergence evidence (audit C1).

    Instead this reproduces one real block per model under the current code and
    solver and compares the feature values.  A match is positive evidence that
    the stored shards are what today's pipeline produces, and only then is a
    stamp written.  A mismatch means the shards are from a different experiment
    and the model's whole design is re-run.
    """
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sobol_verify_"))
    jobs, probes = [], {}
    for mid in models:
        shards = sorted((FEAT_DIR / mid).glob("S_*.parquet"))
        if not shards:
            continue
        probe = shards[0]
        start = int(probe.stem.split("_")[1])
        specs = get_adapter(mid).parameter_specs()
        pids = [s.parameter_id for s in specs]
        u = saltelli_design(len(specs), N_BASE[mid], seed=SEED, calc_second_order=True)
        scales = scales_from_unit(u, specs)
        stop = min(start + BLOCK, len(scales))
        out = tmp / f"{mid}_{probe.name}"
        probes[mid] = (probe, out)
        jobs.append((mid, scales[start:stop], pids, np.arange(start, stop),
                     "SOBOL", 1, SEED, f"saltelli_N{N_BASE[mid]}", str(out)))

    print(f"verifying provenance: re-running 1 block for each of {len(jobs)} models",
          flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, max(1, len(jobs))), maxtasksperchild=1) as pool:
        list(pool.imap_unordered(_worker, jobs))

    ok: dict[str, bool] = {}
    for mid, (stored, fresh) in probes.items():
        if not fresh.exists():
            ok[mid] = False
            print(f"  {mid:18s} FAILED to reproduce (no output)", flush=True)
            continue
        a = pd.read_parquet(stored).sort_values("parameter_sample_id").reset_index(drop=True)
        b = pd.read_parquet(fresh).sort_values("parameter_sample_id").reset_index(drop=True)
        cols = [c for c in FEATURES if c in a.columns and c in b.columns]
        same = len(a) == len(b) and all(
            np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                        rtol=0, atol=0, equal_nan=True) for c in cols)
        ok[mid] = bool(same)
        if same:
            print(f"  {mid:18s} reproduces exactly ({len(cols)} features, {len(a)} rows)",
                  flush=True)
        else:
            worst = [c for c in cols
                     if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                                        rtol=0, atol=0, equal_nan=True)]
            print(f"  {mid:18s} DIFFERS -> stale, will re-run "
                  f"({len(worst)}/{len(cols)} features differ: {worst[:3]})", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(N_BASE))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--verify-stored", action="store_true",
                    help="reproduce one stored block per model; stamp the shards "
                         "that match and quarantine the models that do not")
    args = ap.parse_args()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_stored:
        ok = verify(args.models, args.workers)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for mid, good in ok.items():
            identity = shard_identity(mid)
            shards = sorted((FEAT_DIR / mid).glob("S_*.parquet"))
            if good:
                for s in shards:
                    _identity_path(s).write_text(json.dumps(identity, indent=1))
                print(f"  stamped {len(shards)} shards for {mid}", flush=True)
            else:
                sup = FEAT_DIR / "_superseded" / stamp / mid
                sup.mkdir(parents=True, exist_ok=True)
                for s in shards:
                    s.rename(sup / s.name)
                print(f"  quarantined {len(shards)} shards for {mid} -> {sup}", flush=True)

    if not args.analyse_only:
        simulate(args.models, args.workers)
    analyse(args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
