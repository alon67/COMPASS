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
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results" / "validation" / "run_receipt.json"

# What counts as release output for attribution purposes.  Deliberately the
# generated tree only: source and configuration are inputs to a run, not outputs
# of it, and listing them here would make every stage look like it rewrote them.
#
# `data/provenance` and `data/models` are watched because `check()` DEMANDS
# them.  The previous revision reclassified the provenance and model-asset
# classes as outputs rather than inputs -- correctly, they are written by the
# assets, mechanisms, calibrate, campaign and gate stages -- but left this list
# alone, so every one of those files was required to be attributable and none of
# them could be seen by the index.  A check that asks for evidence it has made
# unobtainable reports the same failure whatever the run did, which is the same
# as reporting nothing.  This list and the inventory's non-input classes have to
# name the same tree; test_receipt_watches_every_class_it_demands enforces it.
WATCHED = ["results/**/*", "data/provenance/**/*", "data/models/**/*",
           "docs/*.md", "README.md", "notebooks/*.ipynb"]
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


# ONE DEFINITION OF THE TREE IDENTITY, NOT TWO.
#
# This module had its own copy, and it disagreed with
# `compass.validation.gates.git_tree` the moment that one was scoped to source
# paths -- so a receipt would have recorded a different revision from the gate
# records it is checked against, and the mismatch would have read as staleness.
# Imported rather than reimplemented, for the same reason `unattributed()` has
# exactly one implementation.
sys.path.insert(0, str(ROOT / "src"))
from compass.validation.gates import git_tree  # noqa: E402


def begin(stages: list[str], force: bool, force_simulations: bool = False,
          restart: bool = False) -> None:
    # A RECEIPT IS EVIDENCE; OPENING A NEW ONE MUST NOT DESTROY THE OLD ONE.
    #
    # `begin` overwrote whatever was there.  Two consequences.  A pipeline run
    # in segments -- which is how every long rebuild actually happens -- kept
    # only the last segment's attribution and silently discarded the rest.  And
    # an UNCLOSED receipt, which is the only trace that a previous run died
    # partway, was overwritten by the next attempt, so a crashed run and a run
    # that never happened left the same evidence.  Closed receipts are archived
    # under _receipts/; an unclosed one has to be discarded on purpose.
    if RECEIPT.exists():
        prev = json.loads(RECEIPT.read_text())
        closed = "artifacts_produced_by_this_run" in prev
        if not closed and not restart:
            print(f"REFUSING to open a receipt over an unclosed one started "
                  f"{prev.get('started_utc')} (stages so far: "
                  f"{[s['stage'] for s in prev.get('stages', [])]}). That "
                  f"receipt says a run died partway; keep it, or pass "
                  f"--restart to discard it deliberately.", file=sys.stderr)
            raise SystemExit(1)
        stamp = (prev.get("finished_utc") or prev.get("started_utc")
                 or "unknown").replace(":", "").replace("-", "")[:15]
        arch = RECEIPT.parent / "_receipts" / f"run_receipt_{stamp}.json"
        arch.parent.mkdir(parents=True, exist_ok=True)
        arch.write_text(RECEIPT.read_text())
        print(f"previous receipt ({'closed' if closed else 'UNCLOSED, discarded'}) "
              f"archived to {arch.relative_to(ROOT)}")
    rec = {
        "started_utc": datetime.now(UTC).isoformat(),
        "git_tree_at_start": git_tree(),
        "stages_planned": stages,
        "force": bool(force),
        # WHICH force. `covers_simulation` in the release manifest turns on
        # whether the feature corpus was rebuilt by this run, and a `--force`
        # run reuses stored shards by design -- so `force` alone cannot answer
        # it and the manifest must not have to guess from the stage list.
        "force_simulations": bool(force_simulations),
        "pre_existing_files": len(index()),
        "_index": index(),
        "stages": [],
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(rec, indent=1))
    print(f"run receipt opened: {len(rec['_index'])} pre-existing release files")


def stage(name: str, rc: int, seconds: float) -> None:
    rec = json.loads(RECEIPT.read_text())
    # A CLOSED RECEIPT CANNOT ACCEPT ANOTHER STAGE, AND MUST SAY SO.
    #
    # `finish()` drops `_index`, so this read used to raise a bare
    # `KeyError: '_index'`. The 2026-08-11 rebuild hit it on its last step: the
    # driver closes the receipt before building the manifest (so the manifest
    # does not hash a file that later changes), and then recorded the manifest
    # stage into the receipt it had just closed. Receipt operations are hard
    # failures by design, so a twenty-hour run ended in a traceback from its
    # final call -- and the traceback named a dictionary key, not the mistake.
    if "_index" not in rec:
        raise SystemExit(
            f"cannot record stage '{name}': the receipt was closed at "
            f"{rec.get('finished_utc')}. A stage that runs after `finish` is "
            f"outside the receipt by construction and must not try to write to "
            f"it -- see run_stage_unreceipted in tools/run_all.sh.")
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
    # DID THE SOURCE MOVE WHILE THE EVIDENCE WAS BEING PRODUCED?
    #
    # Both revisions were already recorded and nothing compared them. On
    # 2026-08-11 a source file was edited part-way through a gate re-run, so
    # the records written before the edit carried a clean revision and those
    # after carried `-dirty`; the manifest matched the second set and rejected
    # 4,783 records from the first. The evidence was self-inconsistent and the
    # only symptom was a large stale count that looked like a configuration
    # problem. It is a fact about the run, so the run states it.
    rec["source_revision_changed_during_run"] = (
        rec.get("git_tree_at_start") != rec["git_tree_at_finish"])
    if rec["source_revision_changed_during_run"]:
        print(f"WARNING: the source revision moved during this run "
              f"({rec.get('git_tree_at_start')} -> {rec['git_tree_at_finish']}). "
              f"Gate records written before and after the change name different "
              f"revisions and cannot all count towards the same release.")
    rec.pop("_index", None)
    RECEIPT.write_text(json.dumps(rec, indent=1))
    print(f"run receipt closed: {len(produced)} artifacts produced by "
          f"{len(rec['stages'])} stages")


# Source and configuration are declared INPUTS: a run reads them and must not be
# expected to produce them.
#
# `provenance` and `model_assets` are NOT inputs and were wrongly classified as
# such (2026-08-09 post-response-3 audit, P1-1).  The `assets` stage downloads
# model assets and writes data/provenance/downloads.json; the `mechanisms` stage
# writes mechanism_builds.json; `calibrate` writes amplitude_calibration.json;
# the campaign and gate stages write parameter_identifiability.json and
# numerical_qc.json. Treating all of that as input meant the one class of file
# most likely to be silently carried over from a previous run was the one class
# the clean-build check never looked at.
#
# This comment used to name waveform_findings.json in that list, and NOTHING
# wrote it -- it was hand-authored and two corpus regenerations stale while
# docs/RESULTS.md cited it for measured prevalences. Naming a producer that does
# not exist is the same error one level up: the justification for demanding
# attribution was itself unverified. It is generated now, by
# `tools/retain_waveforms.py --summarise`.
DECLARED_INPUT_CLASSES = frozenset({"source", "configuration"})


def unattributed(artifacts: list[dict], produced: set[str]) -> list[str]:
    """Manifest artifacts the run did not produce.

    ONE definition with two callers: this module's `check` subcommand, and
    tools/build_release_manifest.py, which needs the same list at build time
    because `check` reads a manifest from disk and on a `--force` run there
    isn't one yet -- the quarantine took it.  The check could therefore never
    pass on the one kind of run it exists to validate.

    Files under `data/provenance/_history` are exempt: they are RETAINED
    history, deliberately not regenerated, and are declared as such.
    """
    return sorted(r["path"] for r in artifacts
                  if r.get("inventory_class") not in DECLARED_INPUT_CLASSES
                  and "/_history/" not in r["path"]
                  and r["path"] not in produced)


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
    listed = [r for r in man.get("artifacts", [])
              if r.get("inventory_class") not in DECLARED_INPUT_CLASSES
              and "/_history/" not in r["path"]]
    leftovers = unattributed(man.get("artifacts", []), produced)
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
    b.add_argument("--force-simulations", action="store_true",
                   help="the run also rebuilt the feature corpus; this is what "
                        "`covers_simulation` in the release manifest reports")
    b.add_argument("--restart", action="store_true",
                   help="discard an unclosed receipt from a run that died")
    s = sub.add_parser("stage")
    s.add_argument("name")
    s.add_argument("rc", type=int)
    s.add_argument("seconds", type=float)
    sub.add_parser("finish")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "begin":
        begin(a.stages, a.force, a.force_simulations, a.restart)
    elif a.cmd == "stage":
        stage(a.name, a.rc, a.seconds)
    elif a.cmd == "finish":
        finish()
    elif a.cmd == "check":
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
