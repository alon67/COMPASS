"""Diff two corpus snapshots into a count table where every count is defined.

The 2026-08-09 post-response-3 audit (P0-2) listed six mutually irreconcilable
affected-row counts for one measurement change, each stated in prose in a
different file, none of them recomputable. The problem was not arithmetic. It
was that no count carried its inclusion criterion, so two numbers that measured
different things looked like a contradiction, and two that measured the same
thing could not be checked against each other.

Every row of the table this tool emits is a named criterion, a boolean
expression over the joined before/after snapshots, and the count of rows
satisfying it. The criteria are declared once, in ``CRITERIA`` below, and the
report is generated -- so a number in the response document either appears in
this table under a stated criterion or it does not get written.

    python3 tools/reconcile_corpus.py --before <snap>.parquet --after <snap>.parquet

Rows present in one snapshot and not the other are reported separately rather
than dropped by the join: a regeneration that silently loses samples is exactly
the failure this table exists to make visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "provenance" / "corpus_reconciliation.json"

KEYS = ["experiment", "model_id", "regime", "replicate", "parameter_sample_id"]


def _f(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _b(s: pd.Series) -> np.ndarray:
    return s.fillna(False).astype(bool).to_numpy()


def _s(s: pd.Series) -> np.ndarray:
    return s.fillna("").astype(str).to_numpy()


# name -> (one-line inclusion criterion, predicate over the joined frame).
# `_a` is the before column, `_b` the after column.
CRITERIA: dict[str, tuple[str, object]] = {
    "rows_total": (
        "every row present in BOTH snapshots, joined on "
        "(experiment, model_id, regime, replicate, parameter_sample_id)",
        lambda d: np.ones(len(d), dtype=bool)),
    "rows_unchanged": (
        "identical rheobase (both NaN counts as identical) AND identical "
        "feature digest AND identical physiology_class",
        lambda d: _same_rheo(d) & (_s(d["feature_digest_a"]) == _s(d["feature_digest_b"]))
        & (_s(d["physiology_class_a"]) == _s(d["physiology_class_b"]))),
    "rows_any_change": (
        "the complement of rows_unchanged: rheobase, feature digest or "
        "physiology_class moved",
        lambda d: ~(_same_rheo(d) & (_s(d["feature_digest_a"]) == _s(d["feature_digest_b"]))
                    & (_s(d["physiology_class_a"]) == _s(d["physiology_class_b"])))),
    "rheobase_changed": (
        "rheobase differs by more than 1e-12, or one is NaN and the other is not",
        lambda d: ~_same_rheo(d)),
    "rescued_from_silence": (
        "rheobase was NaN before and is finite after: a row that was recorded "
        "as non-spiking and in fact fires",
        lambda d: ~np.isfinite(_f(d["rheobase_nA_a"])) & np.isfinite(_f(d["rheobase_nA_b"]))),
    "lost_to_silence": (
        "rheobase was finite before and is NaN after",
        lambda d: np.isfinite(_f(d["rheobase_nA_a"])) & ~np.isfinite(_f(d["rheobase_nA_b"]))),
    "rheobase_lowered": (
        "both finite and the new value is lower by more than 1e-12: a firing "
        "window below the one the previous search found",
        lambda d: np.isfinite(_f(d["rheobase_nA_a"])) & np.isfinite(_f(d["rheobase_nA_b"]))
        & (_f(d["rheobase_nA_b"]) < _f(d["rheobase_nA_a"]) - 1e-12)),
    "rheobase_raised": (
        "both finite and the new value is higher by more than 1e-12",
        lambda d: np.isfinite(_f(d["rheobase_nA_a"])) & np.isfinite(_f(d["rheobase_nA_b"]))
        & (_f(d["rheobase_nA_b"]) > _f(d["rheobase_nA_a"]) + 1e-12)),
    "became_spontaneous": (
        "rheobase is exactly 0 after and was not exactly 0 before",
        lambda d: (_f(d["rheobase_nA_b"]) == 0.0) & ~(_f(d["rheobase_nA_a"]) == 0.0)),
    "ceased_to_be_spontaneous": (
        "rheobase was exactly 0 before and is not exactly 0 after",
        lambda d: (_f(d["rheobase_nA_a"]) == 0.0) & ~(_f(d["rheobase_nA_b"]) == 0.0)),
    "features_changed": (
        "the float64 digest over all feature columns differs",
        lambda d: _s(d["feature_digest_a"]) != _s(d["feature_digest_b"])),
    "physiology_class_changed": (
        "the stored physiology_class label differs",
        lambda d: _s(d["physiology_class_a"]) != _s(d["physiology_class_b"])),
    "analysis_eligibility_changed": (
        "analysis_eligible differs: the row entered or left the inferential set",
        lambda d: _b(d["analysis_eligible_a"]) != _b(d["analysis_eligible_b"])),
    "entered_the_analysis_set": (
        "analysis_eligible False before, True after",
        lambda d: ~_b(d["analysis_eligible_a"]) & _b(d["analysis_eligible_b"])),
    "left_the_analysis_set": (
        "analysis_eligible True before, False after",
        lambda d: _b(d["analysis_eligible_a"]) & ~_b(d["analysis_eligible_b"])),
    "normalised_validity_changed": (
        "normalised_protocol_valid differs: the rheobase-normalised family "
        "became defined or undefined for this row",
        lambda d: _b(d["normalised_protocol_valid_a"]) != _b(d["normalised_protocol_valid_b"])),
    "blocked_at_ceiling_after": (
        "the new search found a firing window but the ceiling itself is silent",
        lambda d: _b(d["rheobase_blocked_at_ceiling_b"])),
    "simulation_errors_before": (
        "simulation_status != ok in the BEFORE snapshot",
        lambda d: _s(d["simulation_status_a"]) != "ok"),
    "simulation_errors_after": (
        "simulation_status != ok in the AFTER snapshot",
        lambda d: _s(d["simulation_status_b"]) != "ok"),
    "qc_flags_changed": (
        "the stored qc_flags string differs",
        lambda d: _s(d["qc_flags_a"]) != _s(d["qc_flags_b"])),
}


def _same_rheo(d: pd.DataFrame) -> np.ndarray:
    a, b = _f(d["rheobase_nA_a"]), _f(d["rheobase_nA_b"])
    both_nan = ~np.isfinite(a) & ~np.isfinite(b)
    close = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= 1e-12)
    return both_nan | close


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} lacks key columns {missing}")
    return df


def write_keys(before: pd.DataFrame, after: pd.DataFrame, criterion: str,
               out: Path) -> int:
    """Dump the row KEYS satisfying one criterion, so an arm can act on them.

    A count says how many rows a change moved; a sensitivity arm needs to know
    *which*.  The voltage-band unification is the case this exists for: it
    excluded rows the previous band admitted, that exclusion is a judgement
    call rather than something the audit specified, and the project's own
    practice is to measure such a choice rather than assert it (as the N4
    projection is measured by `n4_retained`).
    """
    if criterion not in CRITERIA:
        raise SystemExit(f"unknown criterion {criterion!r}; "
                         f"choose one of {sorted(CRITERIA)}")
    a, b = before.set_index(KEYS), after.set_index(KEYS)
    common = a.index.intersection(b.index)
    joined = (a.loc[common].add_suffix("_a")
              .join(b.loc[common].add_suffix("_b")).reset_index())
    mask = np.asarray(CRITERIA[criterion][1](joined), dtype=bool)
    sel = joined.loc[mask, KEYS]
    payload = {
        "criterion": criterion,
        "definition": CRITERIA[criterion][0],
        "generated_utc": datetime.now(UTC).isoformat(),
        "key_columns": KEYS,
        "n_rows": int(len(sel)),
        "by_model": sel.groupby("model_id").size().to_dict(),
        "by_experiment": sel.groupby("experiment").size().to_dict(),
        "rows": sel.to_dict("records"),
    }
    out.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    return len(sel)


def reconcile(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    a = before.set_index(KEYS)
    b = after.set_index(KEYS)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    common = a.index.intersection(b.index)

    joined = a.loc[common].add_suffix("_a").join(b.loc[common].add_suffix("_b"))
    joined = joined.reset_index()

    table = []
    for name, (criterion, pred) in CRITERIA.items():
        mask = np.asarray(pred(joined), dtype=bool)
        table.append({"count": name, "criterion": criterion,
                      "n": int(mask.sum()),
                      "pct_of_total": round(100.0 * mask.sum() / max(len(joined), 1), 3)})

    per_model = {}
    changed = np.asarray(CRITERIA["rows_any_change"][1](joined), dtype=bool)
    resc = np.asarray(CRITERIA["rescued_from_silence"][1](joined), dtype=bool)
    lower = np.asarray(CRITERIA["rheobase_lowered"][1](joined), dtype=bool)
    for m, idx in joined.groupby("model_id").groups.items():
        sel = joined.index.isin(idx)
        per_model[str(m)] = {
            "n_rows": int(sel.sum()),
            "n_changed": int((changed & sel).sum()),
            "n_rescued_from_silence": int((resc & sel).sum()),
            "n_rheobase_lowered": int((lower & sel).sum()),
        }

    fin_a, fin_b = _f(joined["rheobase_nA_a"]), _f(joined["rheobase_nA_b"])
    both = np.isfinite(fin_a) & np.isfinite(fin_b)
    delta = fin_b[both] - fin_a[both]
    return {
        "table": table,
        "per_model": per_model,
        "rows_only_in_before": int(len(only_a)),
        "rows_only_in_after": int(len(only_b)),
        "rheobase_delta_nA": {
            "n": int(both.sum()),
            "median": float(np.median(delta)) if delta.size else None,
            "p01": float(np.percentile(delta, 1)) if delta.size else None,
            "p99": float(np.percentile(delta, 99)) if delta.size else None,
            "min": float(delta.min()) if delta.size else None,
            "max": float(delta.max()) if delta.size else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--write-keys", metavar="CRITERION",
                    help="also dump the row keys satisfying this criterion, "
                         "for a sensitivity arm to act on")
    ap.add_argument("--keys-out",
                    default=str(ROOT / "data" / "provenance" / "affected_rows.json"))
    args = ap.parse_args()

    bp, ap_ = Path(args.before), Path(args.after)
    before, after = load(bp), load(ap_)
    if args.write_keys:
        n = write_keys(before, after, args.write_keys, Path(args.keys_out))
        print(f"wrote {n} row keys for {args.write_keys!r} -> {args.keys_out}")
    result = reconcile(before, after)
    result |= {
        "generated_utc": datetime.now(UTC).isoformat(),
        "before": {"file": bp.name, "n_rows": int(len(before)),
                   "sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
                   "measurement_schema": sorted(
                       set(before.get("measurement_schema", pd.Series(dtype=str)).dropna()))},
        "after": {"file": ap_.name, "n_rows": int(len(after)),
                  "sha256": hashlib.sha256(ap_.read_bytes()).hexdigest(),
                  "measurement_schema": sorted(
                      set(after.get("measurement_schema", pd.Series(dtype=str)).dropna()))},
        "_doc": ("Every count below is defined by its `criterion` field and "
                 "recomputable from the two snapshot files named above. No "
                 "affected-row count may be quoted anywhere in this repository "
                 "unless it appears here (2026-08-09 post-response-3 audit, P0-2)."),
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")

    w = max(len(r["count"]) for r in result["table"])
    print(f"{'count':{w}}  {'n':>8}  {'%':>7}")
    for r in result["table"]:
        print(f"{r['count']:{w}}  {r['n']:8d}  {r['pct_of_total']:7.3f}")
    if result["rows_only_in_before"] or result["rows_only_in_after"]:
        print(f"\nUNJOINED  before-only={result['rows_only_in_before']}  "
              f"after-only={result['rows_only_in_after']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
