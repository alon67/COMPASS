# The probe-3 targeted repair — superseded history, retained as a record

**Status: superseded. Nothing in this directory supports any current number.**
The corpus these files describe no longer exists. It was replaced by a full
clean regeneration under `probe-4`, from the stored parameter draws, with no
targeted rewriting of any kind.

## What happened

The 2026-08-09 post-response-2 audit (§4.1) found that `find_rheobase` equated
depolarisation block at the search ceiling with silence. The response fixed the
search (`probe-2` → `probe-3`) and regenerated the affected rows **in place**,
on the argument that `probe-3` differs from `probe-2` on a strictly enumerable
set of rows. `tools/repair_rheobase.py` implemented that argument.

The post-response-3 audit (P0-1, P0-2) showed the argument was false, and that
the audit trail supporting it could not be reconciled:

1. **The enumeration premise was invalid.** `probe-3` searched a *sparse*
   geometric ladder (`hi/2**k`, k = 8…1) and ran it only when the ceiling did
   not spike. A cell firing between two rungs was still called silent, and a
   cell with a low disconnected firing window plus an upper branch was bisected
   on the upper branch. Neither case is in the enumerated set, so "every other
   row is unchanged by construction" did not hold.
2. **The equivalence verdict was negative.** `rheobase_probe3_verification.json`
   asks whether a from-scratch `probe-3` run reproduces a repaired shard
   exactly. Its retained verdict is `NOT equivalent`: 2 of 6 sampled shards
   differed. One was re-run after a fix; the other was argued away by an
   exhaustive D3 invariant that establishes something narrower than whole-shard
   equality.
3. **The counts do not reconcile, and the artifact needed to reconcile them was
   destroyed.** Six different affected-row counts appear across the response
   text, the scan record, the repair record, tool comments and a docstring
   (1,867 / 2,373 / 2,392 / 1,846 / 2,327 / 154-vs-142). The first scan was run
   without `--recover` and its record was then **overwritten** by the recovery
   scan, so `rheobase_probe3_scan.json` below is the recovery scan
   (`recovery_mode: true`) and the artifact behind the 1,867 figure cannot be
   recomputed from anything in this tree. That number should be treated as
   unsupported.
4. **The repair itself introduced two defects**, both caught and fixed but both
   recorded here because they set the evidential standard: a `run_id`-only join
   key wrote one campaign's simulations into another's shards (546 rows), and
   the first ghost-flag strip matched the wrong namespace and missed NaN
   rheobases.

## What replaced it

- `find_rheobase` now scans a **uniform grid of declared resolution** on every
  sample and states what that grid can and cannot detect
  (`compass.adapters.base.RHEOBASE_SEARCH_CONTRACT`).
- The corpus is **regenerated in full**, never patched. `measurement_schema` is
  part of every shard's reuse identity, so bumping it to `probe-4` invalidates
  every stored shard and the campaign runners re-simulate from the immutable
  parameter draws. There is no enumeration premise left to be wrong about.
- `tools/repair_rheobase.py` **was deleted**. In-place mutation of feature
  shards is no longer an operation this repository offers.
- `tools/snapshot_corpus.py` freezes the identifying and outcome columns of a
  corpus into one immutable file before any regeneration, and
  `tools/reconcile_corpus.py` diffs a pair of snapshots into a machine-generated
  count table in which every count carries its inclusion criterion. The
  before-state is now an artifact rather than a memory.

## Files

| file | what it is | how to read it |
|---|---|---|
| `rheobase_probe3_scan.json` | the **recovery** scan (`recovery_mode: true`), 10,356 rows re-probed | not the original scan; the original was overwritten and is gone |
| `rheobase_probe3_repair.json` | what the in-place rewrite did: 4,656 shards, 2,392 rheobase values recovered, 15,143 ghost QC flags stripped | describes a corpus that no longer exists |
| `rheobase_probe3_verification.json` | the shard-equivalence check | verdict `NOT equivalent`; the `resolution` block records what was and was not established afterwards |

The one durable outcome of the episode is a gate, not a number: D3 fails any row
carrying a `step_rheo_*` QC flag while its rheobase-normalised family is
undefined, on every row of every campaign, every run. That check is in
`tools/validate_campaign.py` and survives `probe-4`.
