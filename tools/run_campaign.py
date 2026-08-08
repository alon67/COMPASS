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
import functools
import hashlib
import json
import multiprocessing as mp
import os
import platform
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
from compass.workflows.simulate import _worker, shard_is_reusable  # noqa: E402

RESULTS = ROOT / "results" / "features"


CONFIG_PATH = ROOT / "configs" / "experiments" / "campaigns.json"
CONFIG = json.loads(CONFIG_PATH.read_text())

# Per-campaign sample sizes now live in configs/experiments/campaigns.json.
# They used to be hard-coded here while a *different*, unread file
# (configs/experiments/screening.json) looked authoritative (audit 5.5).
SAMPLE_SIZES = {name: c["sample_sizes"] for name, c in CONFIG["campaigns"].items()}
SECONDARY_FRACTION = float(CONFIG["secondary_fraction"])
BLOCK = int(CONFIG["block_size"])
BASE_SEED = int(CONFIG["base_seed"])
E2_RHO = float(CONFIG["e2_rho"])
E6_CENTRES = int(CONFIG["e6_n_centres"])


def code_hash() -> str:
    """Hash of the whole source tree, stamped into every campaign manifest."""
    h = hashlib.blake2b(digest_size=8)
    for f in sorted((ROOT / "src").rglob("*.py")):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# Only code that can change a simulated feature value belongs in a SHARD's
# identity.  `code_hash()` covers all of `src/`, which is right for a campaign
# manifest and wrong for shard reuse: editing a plotting or statistics module
# would invalidate every stored simulation.  Same rule `tools/run_sobol.py`
# already applies to its own shards.
SIM_CODE_PATHS = (
    "compass/adapters",
    "compass/protocols",
    "compass/features",
    "compass/sampling",
    "compass/workflows/simulate.py",
)


@functools.lru_cache(maxsize=1)
def sim_code_hash() -> str:
    """Hash of the simulation-relevant source only (see SIM_CODE_PATHS)."""
    h = hashlib.blake2b(digest_size=8)
    files: list[Path] = []
    for rel in SIM_CODE_PATHS:
        p = ROOT / "src" / rel
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    for f in sorted(files):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def config_hash() -> str:
    """Hash of the authoritative campaign configuration file, for the manifest."""
    return hashlib.blake2b(CONFIG_PATH.read_bytes(), digest_size=8).hexdigest()


@functools.cache
def job_config_hash(experiment: str, model_id: str) -> str:
    """Hash of ONLY the configuration that determines this model's design.

    Deliberately not the whole file.  A shard's identity must change when the
    thing that produced it changes and must not change otherwise: hashing the
    entire configuration would invalidate 3,057 correct shards because a note
    was corrected in a different campaign's entry, and a validation rule that
    cries wolf gets switched off.  The fields below are exactly those that enter
    ``build_jobs`` for this model.
    """
    cfg = CONFIG["campaigns"][experiment]
    payload = json.dumps({
        "n_samples": cfg["sample_sizes"][model_id],
        "regimes": cfg.get("regimes_by_model", {}).get(model_id, cfg["regimes"]),
        "seed_offset": cfg.get("seed_offset", 0),
        "block_size": BLOCK,
        "secondary_fraction": SECONDARY_FRACTION,
        "e2_rho": E2_RHO,
        "e6_n_centres": E6_CENTRES,
    }, sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def _hash_files(paths) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in sorted(paths):
        p = Path(p)
        if not p.is_file():
            continue
        h.update(p.name.encode())
        h.update(hashlib.blake2b(p.read_bytes(), digest_size=16).digest())
    return h.hexdigest()


@functools.cache
def asset_hash(model_id: str) -> str:
    """Hash of a model's external executable assets: HOC, MOD, morphology.

    ``code_hash`` covers only first-party Python.  A change to a downloaded
    ``.hoc`` morphology or a ``.mod`` mechanism source changes the model
    completely and used to leave no trace in any identity (audit 5.2).
    """
    cls = MODEL_REGISTRY[model_id]
    roots = []
    for attr in ("SRC",):
        p = getattr(cls, attr, None)
        if p is not None:
            roots.append(Path(p))
    md = cls.manifest.mechanism_dir
    if md:
        roots.append(ROOT / "data" / "models" / md / "mod")
    files = []
    for r in roots:
        if not r.exists():
            continue
        base = r if r.is_dir() else r.parent
        for pat in ("*.hoc", "*.mod", "*.asc", "*.swc", "*.nrn", "*.inc"):
            files.extend(base.rglob(pat))
    return _hash_files(files) if files else "builtin"


@functools.cache
def mechanism_hash(model_id: str) -> str:
    """Hash of the COMPILED mechanism library actually loaded at runtime."""
    builds = ROOT / "data" / "provenance" / "mechanism_builds.json"
    if not builds.exists():
        return "unknown"
    for r in json.loads(builds.read_text()):
        if r.get("model_id") == model_id:
            return str(r.get("sha256", "none"))
    md = MODEL_REGISTRY[model_id].manifest.mechanism_dir
    return "builtin" if md is None else "unknown"


@functools.lru_cache(maxsize=1)
def env_hash() -> str:
    """Hash of the runtime identity: Python, NEURON, eFEL, NumPy, platform."""
    import numpy as _np

    parts = [sys.version.split()[0], _np.__version__, platform.platform(),
             platform.machine()]
    for mod in ("neuron", "efel", "SALib", "scipy", "pandas", "sklearn"):
        try:
            m = __import__(mod)
            parts.append(f"{mod}={getattr(m, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            parts.append(f"{mod}=absent")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def job_identity(experiment: str, model_id: str, regime: str, replicate: int,
                 seed: int, prior_id: str, n_rows: int) -> dict:
    """Everything that must match before a stored shard may be reused.

    Computed BEFORE simulation, stored in the shard and in a sidecar, and
    checked by :func:`compass.workflows.simulate.shard_is_reusable`.  The
    previous rule was "the path exists" (audit 5.2), which reuses a shard
    written by different code, a different solver contract or a different
    mechanism build without any downstream artifact being able to tell.
    """
    from compass.adapters.base import MEASUREMENT_SCHEMA_VERSION
    from compass.features.extract import FEATURE_SCHEMA_VERSION
    from compass.workflows.simulate import solver_kwargs

    # What goes in the REUSE CRITERION vs what is merely recorded.
    #
    # A raw byte hash of the simulation source is the wrong reuse criterion: it
    # invalidates 98,048 stored simulations when a comment changes, and a
    # validation rule that discards correct data gets switched off. The fields
    # below are the SEMANTIC axes along which a feature row can differ --
    # declared feature schema, solver contract, protocol, model assets, compiled
    # mechanisms, design configuration and seed. `sim_code_hash` is still
    # recorded (see `recorded_only` below) so drift is visible, and
    # `tests/unit/test_solver_contract.py` fails if the extractor changes
    # without FEATURE_SCHEMA_VERSION changing -- which turns an undeclared code
    # change into a code-review failure instead of a data-discarding one.
    return {
        "experiment": experiment,
        "model_id": model_id,
        "regime": regime,
        "replicate": int(replicate),
        "seed": int(seed),
        "prior_id": prior_id,
        "n_rows": int(n_rows),
        "config_hash": job_config_hash(experiment, model_id),
        "asset_hash": asset_hash(model_id),
        "mechanism_hash": mechanism_hash(model_id),
        "env_hash": env_hash(),
        "solver": json.dumps(solver_kwargs(model_id), sort_keys=True),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "measurement_schema": MEASUREMENT_SCHEMA_VERSION,
        "protocol_hash": protocol_hash(model_id),
        # Recorded, not compared. Prefixed so shard_is_reusable ignores it.
        "recorded_only.sim_code_hash": sim_code_hash(),
    }


@functools.cache
def protocol_hash(model_id: str) -> str:
    from compass.protocols import common as P

    payload = json.dumps(
        {"amplitudes": P.ABSOLUTE_AMPLITUDES[model_id],
         "multiples": P.RHEOBASE_MULTIPLES,
         "timing": [P.DELAY_MS, P.DURATION_MS, P.TOTAL_MS, P.REST_TOTAL_MS,
                    P.RHEO_DELAY_MS, P.RHEO_DURATION_MS, P.RHEO_TOTAL_MS]},
        sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


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
    spec_cfg = CONFIG["campaigns"][experiment]
    sizes = SAMPLE_SIZES[experiment]
    per_model_regimes = spec_cfg.get("regimes_by_model", {})
    seed_offset = int(spec_cfg.get("seed_offset", 0))
    for model_id in models:
        adapter = get_adapter(model_id)
        specs = adapter.parameter_specs()
        pids = [s.parameter_id for s in specs]
        n1 = sizes[model_id]
        allowed = per_model_regimes.get(model_id, regimes)
        want = [r for r in regimes if r in allowed]
        for rep in range(1, replicates + 1):
            seed = model_seed(model_id, rep, experiment) + seed_offset
            e1 = independent_sample(specs, n1, seed=seed, method="sobol")
            per_regime = {}
            if "E1" in want:
                per_regime["E1"] = (e1, "independent_sobol")
            if "E2" in want:
                n2 = max(BLOCK, int(n1 * SECONDARY_FRACTION))
                per_regime["E2"] = (
                    correlated_sample(specs, n2, seed=seed + 7, rho=E2_RHO),
                    f"correlated_rho{E2_RHO}",
                )
            if "E6" in want:
                n6 = max(BLOCK, int(n1 * SECONDARY_FRACTION))
                # Basins are drawn from the *prior*.  The E6 docstring used to
                # claim the screening run re-centres them on accepted E1
                # solutions "via --basins"; no such option exists and no
                # campaign has ever done that (audit 5.5).
                centres = e1[:: max(1, len(e1) // E6_CENTRES)][:E6_CENTRES]
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
                    ident = job_identity(experiment, model_id, regime, rep, seed,
                                         prior_id, stop - start)
                    ident["block_start"] = int(start)
                    jobs.append((
                        model_id, scales[start:stop], pids,
                        np.arange(start, stop), regime, rep, seed, prior_id,
                        str(out), ident,
                    ))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="pilot", choices=sorted(SAMPLE_SIZES))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=["E1", "E2", "E6"])
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be simulated, reused or quarantined")
    args = ap.parse_args()

    cfg = CONFIG["campaigns"][args.experiment]
    models = args.models or sorted(cfg["sample_sizes"])
    unknown = [m for m in models if m not in cfg["sample_sizes"]]
    if unknown:
        raise SystemExit(
            f"{unknown} have no sample size in campaign {args.experiment!r}. "
            f"Add a NEW campaign to {CONFIG_PATH.name} rather than extending a "
            f"frozen one (audit 9.6 item 7).")

    jobs = build_jobs(args.experiment, models, args.regimes, args.replicates)
    todo, reused, stale = [], 0, []
    for j in jobs:
        shard, ident = Path(j[-2]), j[-1]
        ok, why = shard_is_reusable(shard, ident, len(j[3]))
        if ok:
            reused += 1
            continue
        if shard.exists():
            stale.append((shard.name, why))
        todo.append(j)
    print(f"experiment={args.experiment} models={len(models)} "
          f"blocks={len(jobs)} reusable={reused} todo={len(todo)} "
          f"stale={len(stale)} workers={args.workers}", flush=True)
    for name, why in stale[:10]:
        print(f"  STALE {name}: {why}", flush=True)
    if len(stale) > 10:
        print(f"  ... and {len(stale) - 10} more", flush=True)
    if args.dry_run:
        return 0

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
        "experiment": args.experiment, "models": models,
        "regimes": args.regimes, "replicates": args.replicates,
        "workers": args.workers,
        # Keep EVERY replicate's seed.  The old dict comprehension was keyed on
        # the model alone, so requesting several replicates silently retained
        # only the last one (audit 5.5).
        "seeds": {f"{m}:r{r}": model_seed(m, r, args.experiment)
                  + int(CONFIG["campaigns"][args.experiment].get("seed_offset", 0))
                  for m in models for r in range(1, args.replicates + 1)},
        "code_hash": code_hash(),
        "config_hash": config_hash(),
        "env_hash": env_hash(),
        "asset_hash": {m: asset_hash(m) for m in models},
        "mechanism_hash": {m: mechanism_hash(m) for m in models},
        "n_reused_shards": reused,
        "n_stale_quarantined": len(stale),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform()},
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
