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
    VALID_FRACTION_FLOOR,  # noqa: E402
    saltelli_design,
    sobol_convergence,
    sobol_indices,
)
from compass.sampling.priors import scales_from_unit  # noqa: E402
from compass.workflows.simulate import _worker  # noqa: E402

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


def _shard_is_current(shard: Path, identity: dict) -> bool:
    if not shard.exists():
        return False
    ip = _identity_path(shard)
    if not ip.exists():
        return False            # unstamped shard: provenance unknown, re-run it
    try:
        return json.loads(ip.read_text()) == identity
    except json.JSONDecodeError:
        return False


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
            if out.exists() and not _shard_is_current(out, identity):
                # Move aside, never delete.  Overwriting an artefact is how the
                # Mainen convergence evidence was lost (audit C1).
                sup = FEAT_DIR / "_superseded" / stamp / mid
                sup.mkdir(parents=True, exist_ok=True)
                out.rename(sup / out.name)
                if _identity_path(out).exists():
                    _identity_path(out).rename(sup / _identity_path(out).name)
                quarantined.append(f"{mid}/{out.name}")
            jobs.append((mid, scales[start:stop], pids, np.arange(start, stop),
                         "SOBOL", 1, SEED, f"saltelli_N{n_base}", str(out)))

    if quarantined:
        print(f"quarantined {len(quarantined)} shard(s) whose identity no longer "
              f"matches -> results/features/sobol/_superseded/{stamp}/", flush=True)
        for q in quarantined[:10]:
            print(f"    {q}", flush=True)
        if len(quarantined) > 10:
            print(f"    ... and {len(quarantined) - 10} more", flush=True)

    todo = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"sobol: {len(jobs)} blocks, {len(todo)} to run, {workers} workers", flush=True)
    t0 = time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for i, _st in enumerate(pool.imap_unordered(_worker, todo), 1):
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} blocks  {(time.time()-t0)/60:.1f} min", flush=True)

    # Stamp after the fact, so a shard is only ever declared current once its
    # parquet is actually on disk.  Written per shard rather than per model:
    # a single index file would be a write race between workers.
    for mid in models:
        identity = shard_identity(mid)
        for shard in sorted((FEAT_DIR / mid).glob("S_*.parquet")):
            _identity_path(shard).write_text(json.dumps(identity, indent=1))


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
            rows.append({
                "model_id": mid, "feature": feat, "n_base": n_base,
                "n_valid": res.n_valid, "n_total": len(y),
                "valid_fraction": res.n_valid / len(y),
                "quotable": bool(res.n_valid / len(y) >= VALID_FRACTION_FLOOR),
                "interaction_fraction": res.interaction_fraction,
                "mean_ST_minus_S1": res.mean_total_minus_first,
                "sum_S1": float(np.nansum(res.S1)),
                "sum_ST": float(np.nansum(res.ST)),
                "max_S1_conf": float(np.nanmax(res.S1_conf)),
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
