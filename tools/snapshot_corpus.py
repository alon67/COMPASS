"""Freeze the identifying + outcome columns of the stored corpus into one file.

Why this exists.  The 2026-08-09 post-response-3 audit (P0-2) found the
probe-2 -> probe-3 repair impossible to reconcile: six different affected-row
counts appear across the response, the scan record, the repair record, the tool
comments and a docstring, and the *original* scan artifact had been overwritten
by a later recovery scan, so the number that the narrative rested on could no
longer be recomputed from anything in the tree.

The fix is not a better narrative.  It is to make the before-state a first-class
artifact.  This tool writes one Parquet file holding every stored row's identity
plus the columns a measurement change can move, at a size (~10 MB) that can be
kept indefinitely.  ``tools/reconcile_corpus.py`` then diffs a pair of snapshots
and emits the machine-generated count table the audit asked for, with each count
defined by an explicit inclusion criterion rather than by prose.

Snapshots are immutable by convention and by name: the filename carries the
measurement schema and the UTC minute, and the tool refuses to overwrite one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEATURES = ROOT / "results" / "features"
SNAPSHOTS = ROOT / "results" / "corpus_snapshots"

# Identity: enough to join two snapshots row-for-row.
#
# Two of these are not stored under these names -- the campaign is
# `identity.experiment` and the regime is `sampling_regime` -- so each key
# carries the aliases it may appear under, and a key that resolves to NOTHING is
# a hard error. The first version of this tool filtered the key list by
# `if c in df.columns`, which silently produced a snapshot with no campaign and
# no regime: 130,098 rows keyed on (model, replicate, sample), which collide
# across campaigns. That is the same class of defect as the probe-3 repair's
# `run_id`-only join, and it is exactly what this file exists to prevent.
KEY_COLUMNS = ("experiment", "model_id", "regime", "replicate",
               "parameter_sample_id")

KEY_ALIASES = {
    "experiment": ("experiment", "identity.experiment"),
    "regime": ("regime", "sampling_regime", "identity.regime"),
}

# Outcome: everything a measurement-schema change is allowed to move.  Feature
# columns are summarised by a digest rather than copied, so the snapshot stays
# small while still detecting any change to any of the ~180 feature values.
OUTCOME_COLUMNS = ("rheobase_nA", "non_spiking", "rheobase_blocked_at_ceiling",
                   "simulation_status", "physiology_class", "execution_valid",
                   "analysis_eligible", "protocol_valid",
                   "normalised_protocol_valid", "qc_flags",
                   "measurement_schema", "validator_version",
                   "rheobase_scan_points", "rheobase_scan_resolution_nA")

FEATURE_PREFIXES = ("rest.", "hyperpol_abs.", "step_abs_lo.", "step_abs_hi.",
                    "step_rheo_1p5x.", "step_rheo_3p0x.")


def shard_paths() -> list[Path]:
    """Every live shard: quarantined, superseded and in-flight files excluded."""
    out = []
    for p in sorted(FEATURES.rglob("*.parquet")):
        rel = p.relative_to(FEATURES).as_posix()
        if "_quarantine" in rel or "_superseded" in rel or p.name.endswith(".tmp.parquet"):
            continue
        out.append(p)
    return out


def _digest(values) -> str:
    """Order-independent-free digest of one row's feature vector.

    ``float64`` bytes, so it is exact: two rows agree here only if every feature
    value is bit-identical, NaN included.
    """
    h = hashlib.blake2b(digest_size=8)
    for name, v in values:
        h.update(name.encode())
        h.update(pd.array([v], dtype="float64").to_numpy().tobytes())
    return h.hexdigest()


def snapshot_shard(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = pd.DataFrame(index=df.index)
    for key in KEY_COLUMNS:
        src = next((c for c in KEY_ALIASES.get(key, (key,)) if c in df.columns), None)
        if src is None:
            raise SystemExit(
                f"{path.name} has no column for join key {key!r} (tried "
                f"{KEY_ALIASES.get(key, (key,))}). A snapshot missing a key "
                f"column joins rows that are not the same row.")
        out[key] = df[src].to_numpy()
    for col in OUTCOME_COLUMNS:
        src = next((c for c in (col, f"identity.{col}") if c in df.columns), None)
        if src is not None:
            out[col] = df[src].to_numpy()
    feat = sorted(c for c in df.columns if c.startswith(FEATURE_PREFIXES))
    if feat:
        sub = df[feat].astype("float64")
        out["feature_digest"] = [
            _digest(zip(feat, row, strict=True)) for row in sub.to_numpy()
        ]
        out["n_feature_columns"] = len(feat)
    else:
        out["feature_digest"] = ""
        out["n_feature_columns"] = 0
    out["shard"] = path.relative_to(FEATURES).as_posix()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="what this snapshot is of, e.g. 'probe-3-final'")
    ap.add_argument("--out", default=None, help="override the generated path")
    args = ap.parse_args()

    paths = shard_paths()
    if not paths:
        print(f"no shards under {FEATURES}", file=sys.stderr)
        return 1
    frames = []
    for i, p in enumerate(paths, 1):
        frames.append(snapshot_shard(p))
        if i % 500 == 0:
            print(f"  {i}/{len(paths)} shards", flush=True)
    df = pd.concat(frames, ignore_index=True)

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
    out = Path(args.out) if args.out else SNAPSHOTS / f"corpus_{args.label}_{stamp}.parquet"
    if out.exists():
        print(f"refusing to overwrite {out}: snapshots are immutable", file=sys.stderr)
        return 1
    df.to_parquet(out, index=False)

    meta = {
        "label": args.label,
        "written_utc": datetime.now(UTC).isoformat(),
        "n_rows": int(len(df)),
        "n_shards": len(paths),
        "key_columns": list(KEY_COLUMNS),
        "outcome_columns": [c for c in OUTCOME_COLUMNS if c in df.columns],
        "n_feature_columns": int(df["n_feature_columns"].max()),
        "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "purpose": ("immutable before/after state for the corpus reconciliation "
                    "table required by the 2026-08-09 post-response-3 audit, P0-2"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}  rows={len(df)}  shards={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
