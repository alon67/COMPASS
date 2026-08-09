#!/usr/bin/env python3
"""Attribute every release artifact to the stage that produced it (audit 4.4).

``run_all.sh`` was "an orchestration convenience, not yet proof that a release
can be reconstructed from a clean workspace": ``--force`` was parsed and never
used, nothing invalidated stored results, and a run could silently consume
artifacts left behind by an earlier one.

Clean-run semantics are defined here as three concrete, checkable properties:

1. **Declared start.**  A run records its git tree, the stage list, and an index
   of every release file that already existed, before anything executes.
2. **Per-stage attribution.**  After each stage, the index is re-taken and the
   difference is recorded, so the receipt says which stage created or rewrote
   which artifact.
3. **Checkable completeness.**  ``--check`` compares the release manifest
   against the receipt and reports every listed artifact that the recorded run
   did not produce.  A release built partly from leftovers is then visible
   rather than assumed away.

The index is keyed on (size, mtime_ns), which is cheap enough to take nineteen
times over 12,000 files.  It is used only to attribute outputs to stages;
integrity is the release manifest's job and is content-hashed there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results" / "validation" / "run_receipt.json"

# What counts as release output for attribution purposes.  Deliberately the
# generated tree only: source and configuration are inputs to a run, not outputs
# of it, and listing them here would make every stage look like it rewrote them.
WATCHED = ["results/**/*", "docs/*.md", "README.md", "notebooks/*.ipynb"]
IGNORE = ("results/logs/", "results/validation/run_receipt.json",
          "results/analyses/_perturbation_cache/")


def index() -> dict[str, list]:
    out: dict[str, list] = {}
    for pat in WATCHED:
        for p in ROOT.glob(pat):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ROOT))
            if rel.startswith(IGNORE):
                continue
            st = p.stat()
            out[rel] = [st.st_size, st.st_mtime_ns]
    return out


def git_tree() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:                                             # noqa: BLE001
        return "unversioned"


def begin(stages: list[str], force: bool) -> None:
    rec = {
        "started_utc": datetime.now(UTC).isoformat(),
        "git_tree_at_start": git_tree(),
        "stages_planned": stages,
        "force": bool(force),
        "pre_existing_files": len(index()),
        "_index": index(),
        "stages": [],
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(rec, indent=1))
    print(f"run receipt opened: {len(rec['_index'])} pre-existing release files")


def stage(name: str, rc: int, seconds: float) -> None:
    rec = json.loads(RECEIPT.read_text())
    before, after = rec["_index"], index()
    created = sorted(set(after) - set(before))
    rewritten = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    removed = sorted(set(before) - set(after))
    rec["stages"].append({
        "stage": name, "exit_code": int(rc), "seconds": round(float(seconds), 1),
        "created": created, "rewritten": rewritten, "removed": removed,
        "n_created": len(created), "n_rewritten": len(rewritten),
    })
    rec["_index"] = after
    RECEIPT.write_text(json.dumps(rec, indent=1))


def finish() -> None:
    rec = json.loads(RECEIPT.read_text())
    produced = sorted({f for s in rec["stages"]
                       for f in s["created"] + s["rewritten"]})
    rec["finished_utc"] = datetime.now(UTC).isoformat()
    rec["git_tree_at_finish"] = git_tree()
    rec["artifacts_produced_by_this_run"] = produced
    rec["n_produced"] = len(produced)
    rec["stages_run"] = [s["stage"] for s in rec["stages"]]
    rec["stages_failed"] = [s["stage"] for s in rec["stages"] if s["exit_code"]]
    rec.pop("_index", None)
    RECEIPT.write_text(json.dumps(rec, indent=1))
    print(f"run receipt closed: {len(produced)} artifacts produced by "
          f"{len(rec['stages'])} stages")


def check() -> int:
    """Does the release manifest reference only artifacts this run produced?"""
    man_path = ROOT / "results" / "validation" / "release_manifest.json"
    if not RECEIPT.exists():
        print("no run receipt; a clean-build claim cannot be checked")
        return 1
    if not man_path.exists():
        print("no release manifest to check against")
        return 1
    rec = json.loads(RECEIPT.read_text())
    man = json.loads(man_path.read_text())
    if "artifacts_produced_by_this_run" not in rec:
        print("the run receipt was never closed; the run did not finish")
        return 1
    produced = set(rec["artifacts_produced_by_this_run"])
    # Source, configuration and model assets are declared INPUTS: a run does not
    # produce them and must not be expected to.
    inputs = {"source", "configuration", "model_assets", "provenance"}
    listed = [r for r in man.get("artifacts", [])
              if r.get("inventory_class") not in inputs]
    leftovers = sorted(r["path"] for r in listed if r["path"] not in produced)
    print(f"receipt: {rec.get('git_tree_at_start')} -> "
          f"{rec.get('git_tree_at_finish')}")
    print(f"stages run: {rec.get('stages_run')}")
    if rec.get("stages_failed"):
        print(f"stages that exited non-zero: {rec['stages_failed']}")
    print(f"{len(listed) - len(leftovers)}/{len(listed)} generated artifacts are "
          f"attributable to this run")
    for p in leftovers[:25]:
        print(f"  NOT PRODUCED BY THIS RUN  {p}")
    if len(leftovers) > 25:
        print(f"  ... and {len(leftovers) - 25} more")
    return 0 if not leftovers else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("begin")
    b.add_argument("--stages", nargs="*", default=[])
    b.add_argument("--force", action="store_true")
    s = sub.add_parser("stage")
    s.add_argument("name")
    s.add_argument("rc", type=int)
    s.add_argument("seconds", type=float)
    sub.add_parser("finish")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "begin":
        begin(a.stages, a.force)
    elif a.cmd == "stage":
        stage(a.name, a.rc, a.seconds)
    elif a.cmd == "finish":
        finish()
    elif a.cmd == "check":
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
