"""Reconstruct identity sidecars for shards written before identity existed.

Shard reuse is now identity-validated and fails closed
(:func:`compass.workflows.simulate.shard_is_reusable`).  Applied naively that
would quarantine and recompute the entire stored campaign, because shards
written before the 2026-08-06 remediation carry no sidecar.

This tool writes a sidecar for each such shard from what can be *recovered*
(the model, regime, replicate, seed and prior recorded in the shard's own
columns) plus the identity fields that describe the current tree.  Every
backfilled sidecar is stamped ``identity_provenance: "backfilled"`` with the
reconstruction timestamp, so a release manifest can always distinguish a shard
whose identity was recorded when it was written from one whose identity was
inferred afterwards.

That distinction matters and is not a formality: a backfilled identity asserts
only "these fields describe the tree at reconstruction time", not "this shard
was produced by them".  The deterministic replay gate
(``tools/validate_replay.py``) is what actually tests the latter, and a
backfilled shard is exactly the kind that has to be replayed.

    python tools/backfill_shard_identity.py --experiment screening
    python tools/backfill_shard_identity.py --all --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEATURES = ROOT / "results" / "features"


def _campaign_module():
    spec = importlib.util.spec_from_file_location("_rc", ROOT / "tools" / "run_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _upgrade_sobol(shard: Path, side: Path, dry_run: bool) -> dict:
    """Bring a Sobol sidecar written in the pre-unification format up to date.

    The Sobol campaign already had identity-validated reuse (that was the
    previous audit's C2 repair), but in its own format.  Unifying the two rules
    must not force a re-run of 21,760 correctly-provenanced simulations, so the
    original payload is preserved verbatim inside the new one.
    """
    import importlib.util as iu

    from compass.workflows.simulate import write_sidecar

    spec = iu.spec_from_file_location("_rs", ROOT / "tools" / "run_sobol.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    old = json.loads(side.read_text())
    if "sha256" in old and "n_rows" in old:
        return {"shard": shard.name, "action": "kept"}
    df = pd.read_parquet(shard, columns=["model_id", "parameter_sample_id"])
    mid = str(df["model_id"].iloc[0])
    ident = mod.flat_identity(mid, old, int(df["parameter_sample_id"].min()), len(df))
    ident["identity_provenance"] = "upgraded_from_sobol_v1"
    ident["upgraded_utc"] = datetime.now(UTC).isoformat()
    ident["original_payload"] = json.dumps(old, sort_keys=True)
    if not dry_run:
        write_sidecar(shard, ident, n_rows=len(df))
    return {"shard": shard.name, "action": "upgraded", "model": mid, "n": len(df)}


def backfill(shard: Path, rc, experiment: str, dry_run: bool) -> dict:
    from compass.workflows.simulate import sidecar_path, write_sidecar

    side = sidecar_path(shard)
    if side.exists():
        if experiment == "sobol":
            return _upgrade_sobol(shard, side, dry_run)
        return {"shard": shard.name, "action": "kept"}
    df = pd.read_parquet(shard)
    if df.empty:
        return {"shard": shard.name, "action": "empty"}
    model_id = str(df["model_id"].iloc[0])
    regime = str(df["sampling_regime"].iloc[0])
    replicate = int(df["replicate"].iloc[0])
    seed = int(df["seed"].iloc[0])
    prior_id = str(df["prior_id"].iloc[0])
    ident = rc.job_identity(experiment, model_id, regime, replicate, seed,
                            prior_id, len(df))
    ident["block_start"] = int(df["parameter_sample_id"].min())
    ident["identity_provenance"] = "backfilled"
    ident["backfilled_utc"] = datetime.now(UTC).isoformat()
    if not dry_run:
        write_sidecar(shard, ident, n_rows=len(df))
    return {"shard": shard.name, "action": "backfilled", "model": model_id,
            "regime": regime, "n": len(df)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--all", action="store_true",
                    help="every experiment directory under results/features")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rc = _campaign_module()
    known = set(rc.CONFIG["campaigns"])
    experiments = ([p.name for p in sorted(FEATURES.iterdir())
                    if p.is_dir() and (p.name in known or p.name == "sobol")]
                   if args.all else [args.experiment])
    if args.all:
        skipped = [p.name for p in sorted(FEATURES.iterdir())
                   if p.is_dir() and p.name not in known and p.name != "sobol"]
        if skipped:
            # Superseded archives are evidence, not inputs: they keep whatever
            # identity they had and are never re-stamped against current code.
            print(f"not a live campaign, left untouched: {skipped}")

    summary = []
    for exp in experiments:
        base = FEATURES / exp
        if not base.exists():
            print(f"skip {exp}: no such directory")
            continue
        shards = sorted(p for p in base.rglob("*.parquet")
                        if "_quarantine" not in p.parts
                        and "_superseded" not in p.parts
                        and not p.name.endswith(".tmp.parquet"))
        counts: dict[str, int] = {}
        for s in shards:
            try:
                rec = backfill(s, rc, exp, args.dry_run)
            except Exception as exc:  # noqa: BLE001
                rec = {"shard": s.name, "action": f"error:{type(exc).__name__}"}
                print(f"  !! {s}: {exc}")
            counts[rec["action"]] = counts.get(rec["action"], 0) + 1
        print(f"{exp}: {len(shards)} shards -> {counts}", flush=True)
        summary.append({"experiment": exp, "n_shards": len(shards), **counts})

    out = ROOT / "data" / "provenance" / "identity_backfill.json"
    if not args.dry_run:
        out.write_text(json.dumps(
            {"written_utc": datetime.now(UTC).isoformat(), "summary": summary},
            indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
