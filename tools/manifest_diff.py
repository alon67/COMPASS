#!/usr/bin/env python3
"""Quantify what changed when the canonical manifests were rebuilt.

The manifests define the conditioning target that every ensemble is accepted
against.  Rebuilding them under a different solver contract (audit C2 / D24)
therefore moves the target, and "the provenance is now correct" is not the same
claim as "the numbers did not change".  This reports the second claim.

Shifts are expressed in units of each model's own unconditioned (E1) spread for
that feature, because that is the scale the conditioning tolerance is measured
in: a shift of 0.1 z moves the accept/reject boundary by a tenth of a tolerance
step, while a shift of 2 z relocates the target to a different part of the
ensemble.

    python tools/manifest_diff.py results/manifests/_superseded/<stamp>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

CUR = ROOT / "results" / "manifests"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    old_dir = Path(sys.argv[1])
    from run_analysis import load_model_table

    from compass.conditioning.ensembles import CONDITIONING_FEATURES

    rows = []
    for cur_p in sorted(CUR.glob("canonical_*.json")):
        mid = json.loads(cur_p.read_text())["model_id"]
        old_p = old_dir / cur_p.name
        if not old_p.exists():
            print(f"{mid}: no prior manifest to compare")
            continue
        new = json.loads(cur_p.read_text())
        old = json.loads(old_p.read_text())

        df = load_model_table("screening", mid)
        e1 = (df[(df.simulation_status == "ok") & (df.sampling_regime == "E1")]
              if df is not None else None)

        for f in CONDITIONING_FEATURES:
            a = old.get("canonical_features", {}).get(f)
            b = new.get("canonical_features", {}).get(f)
            if a is None or b is None:
                continue
            sd = np.nan
            if e1 is not None and f in e1:
                v = pd.to_numeric(e1[f], errors="coerce").to_numpy(float)
                sd = float(np.nanstd(v))
            rows.append({
                "model_id": mid, "feature": f, "old": float(a), "new": float(b),
                "abs_delta": abs(float(b) - float(a)),
                "delta_in_e1_sd": (abs(float(b) - float(a)) / sd
                                   if np.isfinite(sd) and sd > 0 else np.nan),
            })

        if old.get("solver") != new.get("solver"):
            print(f"{mid}: solver {old.get('solver')} -> {new.get('solver')}")
        if old.get("n_compartments") != new.get("n_compartments"):
            print(f"{mid}: n_compartments {old.get('n_compartments')} "
                  f"-> {new.get('n_compartments')}")
        if old.get("rheobase_nA") != new.get("rheobase_nA"):
            print(f"{mid}: rheobase {old.get('rheobase_nA')} -> {new.get('rheobase_nA')}")

    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing to compare")
        return 0
    d.to_parquet(ROOT / "results" / "analyses" / "manifest_diff.parquet", index=False)

    print("\nper-model target shift, in units of that model's E1 spread:")
    g = d.groupby("model_id")["delta_in_e1_sd"].agg(["max", "median", "count"])
    print(g.round(3).to_string())

    print("\nlargest individual shifts:")
    top = d.dropna(subset=["delta_in_e1_sd"]).nlargest(12, "delta_in_e1_sd")
    for r in top.itertuples():
        print(f"  {r.model_id:18s} {r.feature:38s} {r.old:12.4f} -> {r.new:12.4f}  "
              f"{r.delta_in_e1_sd:6.3f} E1-SD")

    n_big = int((d["delta_in_e1_sd"] > 0.5).sum())
    print(f"\nfeatures shifted by more than 0.5 E1-SD: {n_big} / {len(d)}")
    print("(the conditioning tolerances are 3.0 / 2.0 / 1.0 / 0.5 z, so a shift "
          "of this size is what would move ensemble membership materially)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
