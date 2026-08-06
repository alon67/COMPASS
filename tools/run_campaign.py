"""Execute a COMPASS simulation campaign (blueprint WP4/WP6).

    python tools/run_campaign.py --experiment pilot
    python tools/run_campaign.py --experiment screening --models bbp_l5pc

Ensembles E1 (independent prior), E2 (covariance-preserving prior) and E6
(clustered) require simulation.  E3/E4 (phenotype-conditioned), E7 (high-
covariance subset), E8 (shuffled) and E9 (copula control) are derived later from
the simulated tables without further simulation.
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

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.sampling.priors import (  # noqa: E402
    clustered_sample,
    correlated_sample,
    independent_sample,
)
from compass.workflows.simulate import _worker  # noqa: E402

RESULTS = ROOT / "results" / "features"


def code_hash() -> str:
    """Hash of the analysis-relevant source tree, stamped into every manifest."""
    h = hashlib.blake2b(digest_size=8)
    for f in sorted((ROOT / "src").rglob("*.py")):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()

# Sample sizes per model, sized from the canonical-audit cost measurements so
# that the whole campaign fits a single workstation.  Cheap point models get an
# order of magnitude more samples; every cross-model comparison is additionally
# repeated on a size-matched subsample (blueprint 26: "never compare unequal
# sample sizes without matched controls").
SAMPLE_SIZES = {
    "pilot": {m: 60 for m in MODEL_REGISTRY},
    # Sizes measured against the fixed-step cost per sample so that the whole
    # campaign fits ~3 h on 12 workers.  The 30x spread between the cheapest and
    # most expensive model is exactly why every cross-model endpoint is also
    # recomputed on a size-matched subsample.
    "screening": {
        "hh_point": 8192, "hh_cable": 8192, "pospischil_rs": 8192,
        "pospischil_fs": 8192, "khaliq_purkinje": 4096,
        "mainen_l5pc": 256, "bbp_l5pc": 1024, "migliore_ca1": 256,
    },
}
# E2 and E6 are run at half the E1 size (they support secondary endpoints).
SECONDARY_FRACTION = 0.5

BLOCK = 25
BASE_SEED = 20260804


SEED_LOCK = ROOT / "configs" / "seed_lock.json"


def model_seed(model_id: str, replicate: int, experiment: str = "screening") -> int:
    """Deterministic per-model seed.

    ``hash()`` on a str is salted per process (PYTHONHASHSEED), so the original
    ``abs(hash(model_id)) % 997`` produced a *different* seed on every
    invocation and the campaign could not be regenerated from its own
    configuration.  blake2b is stable across processes, machines and Python
    versions.

    ``configs/seed_lock.json`` pins the seeds that actually produced the stored
    shards (recovered from the ``seed`` column of the shards themselves).  A
    locked seed always wins, so the existing campaign stays *exactly*
    regenerable even though it predates this fix; anything not in the lock file
    gets the stable blake2b seed.
    """
    if SEED_LOCK.exists():
        locked = json.loads(SEED_LOCK.read_text())
        try:
            return int(locked[experiment][model_id][str(replicate)])
        except (KeyError, TypeError, ValueError):
            pass
    digest = hashlib.blake2b(model_id.encode(), digest_size=4).digest()
    return BASE_SEED + 1000 * replicate + int.from_bytes(digest, "big") % 997


def build_jobs(experiment: str, models: list[str], regimes: list[str],
               replicates: int) -> list[tuple]:
    jobs = []
    sizes = SAMPLE_SIZES[experiment]
    for model_id in models:
        adapter = get_adapter(model_id)
        specs = adapter.parameter_specs()
        pids = [s.parameter_id for s in specs]
        n1 = sizes[model_id]
        for rep in range(1, replicates + 1):
            seed = model_seed(model_id, rep, experiment)
            e1 = independent_sample(specs, n1, seed=seed, method="sobol")
            per_regime = {}
            if "E1" in regimes:
                per_regime["E1"] = (e1, "independent_sobol")
            if "E2" in regimes:
                n2 = max(BLOCK, int(n1 * SECONDARY_FRACTION))
                per_regime["E2"] = (
                    correlated_sample(specs, n2, seed=seed + 7, rho=0.6),
                    "correlated_rho0.6",
                )
            if "E6" in regimes:
                n6 = max(BLOCK, int(n1 * SECONDARY_FRACTION))
                # Basins are drawn from the *prior* here; the pilot has no
                # accepted set yet.  The screening run re-centres them on
                # accepted E1 solutions via --basins.
                centres = e1[:: max(1, len(e1) // 24)][:24]
                per_regime["E6"] = (
                    clustered_sample(centres, specs, n6, seed=seed + 13),
                    "clustered_prior_basins",
                )
            for regime, (scales, prior_id) in per_regime.items():
                n = len(scales)
                for start in range(0, n, BLOCK):
                    stop = min(start + BLOCK, n)
                    out = (RESULTS / experiment / model_id /
                           f"{regime}_r{rep}_{start:06d}.parquet")
                    jobs.append((
                        model_id, scales[start:stop], pids,
                        np.arange(start, stop), regime, rep, seed, prior_id, str(out),
                    ))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="pilot", choices=sorted(SAMPLE_SIZES))
    ap.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY))
    ap.add_argument("--regimes", nargs="*", default=["E1", "E2", "E6"])
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    jobs = build_jobs(args.experiment, args.models, args.regimes, args.replicates)
    todo = [j for j in jobs if not Path(j[-1]).exists()]
    print(f"experiment={args.experiment} models={len(args.models)} "
          f"blocks={len(jobs)} todo={len(todo)} workers={args.workers}", flush=True)

    t0 = time.time()
    stats = []
    # 'spawn' guarantees a clean interpreter (and therefore a clean NEURON) per
    # block, which is what makes one-model-per-process safe.
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, st in enumerate(pool.imap_unordered(_worker, todo), 1):
            stats.append(st)
            if i % 20 == 0 or i == len(todo):
                done = sum(s["n"] for s in stats)
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                print(f"  {i}/{len(todo)} blocks  {done} samples  "
                      f"{el/60:.1f} min  {rate:.1f} samples/s", flush=True)

    # Append-only manifest: a resumed or no-op invocation must not erase the
    # record of the runs that produced the stored shards.
    log = ROOT / "results" / "manifests" / f"campaign_{args.experiment}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "invocation_utc": datetime.now(UTC).isoformat(),
        "experiment": args.experiment, "models": args.models,
        "regimes": args.regimes, "replicates": args.replicates,
        "workers": args.workers,
        "seeds": {m: model_seed(m, r, args.experiment) for m in args.models
                  for r in range(1, args.replicates + 1)},
        "code_hash": code_hash(),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__},
        "wall_min": round((time.time() - t0) / 60, 2),
        "blocks": stats,
        "n_blocks_total": len(jobs),
        "n_blocks_run": len(todo),
        "n_samples": sum(s["n"] for s in stats),
        "n_error": sum(s["n_error"] for s in stats),
        "n_nonspiking": sum(s["n_nonspiking"] for s in stats),
    }
    history = json.loads(log.read_text()) if log.exists() else []
    if isinstance(history, dict):                       # migrate the old format
        history = [history]
    history.append(entry)
    log.write_text(json.dumps(history, indent=1))
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {log} ({len(history)} invocations recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
