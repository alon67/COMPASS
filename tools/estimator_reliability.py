#!/usr/bin/env python3
"""How many samples does `net_o_information` need before its SIGN is stable?

Why this exists
---------------
The conditioning analysis has no minimum-sample policy.  Sobol has one
(`VALID_FRACTION_FLOOR`); the entropy path does not, so an ensemble of 66
samples in 7 dimensions casts the same vote in a panel mean as one of 291.

That became load-bearing when the corrected conditioning targets (audit C2)
admitted a 66-sample very-strict ensemble for `pospischil_fs` whose
`net_o_information` is **+0.015**, opposite in sign to the other two models at
that tolerance and large enough to flip the panel mean from negative to
positive.

The question "is +0.015 a real reversal or sampling noise?" must not be settled
by choosing a cutoff that excludes it -- that is fitting the criterion to the
answer.  This measures the estimator instead: take ensembles that ARE large
enough, subsample them to matched n, and ask how often the statistic changes
sign.  The procedure never sees the value in question.

Writes ``results/analyses/estimator_reliability.json``.  No simulation required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "results" / "analyses"
N_GRID = (50, 66, 100, 150, 250)
N_REPLICATES = 60


def main() -> int:
    from run_analysis import (
        MAX_TUPLES,
        ORDERS,
        canonical_target,
        load_model_table,
    )

    from compass.adapters.models import get_adapter
    from compass.analysis.tuples import order_curve
    from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles

    def net_oi(x: np.ndarray, seed: int = 0) -> float:
        oc = order_curve(x, orders=ORDERS, max_tuples=MAX_TUPLES, seed=seed,
                         copula=False, bias_correct=True, n_surrogate=0)
        syn, red = oc.normalised_areas()
        return red - syn

    rows = []
    # Only ensembles comfortably larger than the grid can serve as references:
    # the reference value must itself be trustworthy for a sign flip to mean
    # anything.
    for mid in ("hh_cable", "hh_point", "pospischil_rs", "pospischil_fs"):
        df = load_model_table("screening", mid)
        if df is None:
            continue
        pids = [s.parameter_id for s in get_adapter(mid).parameter_specs()]
        ens = build_ensembles(df, mid, pids, CONDITIONING_FEATURES,
                              target=canonical_target(mid, CONDITIONING_FEATURES),
                              seed=1, protocol_family="absolute")
        for e in ens:
            if e.regime not in ("E4", "E4b") or "max_z" not in e.label:
                continue
            x = e.params
            if len(x) < 250:
                continue
            full = net_oi(x)
            rng = np.random.default_rng(0)
            for n in N_GRID:
                if n > len(x):
                    continue
                v = np.array([net_oi(x[rng.choice(len(x), n, replace=False)], seed=i)
                              for i in range(N_REPLICATES)])
                rows.append({
                    "model_id": mid, "label": e.label, "d": int(x.shape[1]),
                    "n_full": int(len(x)), "full_value": float(full),
                    "n": int(n), "mean": float(v.mean()), "sd": float(v.std()),
                    "p_sign_flip": float(np.mean(np.sign(v) != np.sign(full))),
                    "min": float(v.min()), "max": float(v.max()),
                })
                print(f"  {mid:16s} {e.label:34s} d={x.shape[1]} n={n:4d} "
                      f"sd={v.std():.5f} P(flip)={np.mean(np.sign(v) != np.sign(full)):.2f} "
                      f"max={v.max():+.5f}", flush=True)

    # Per-model resolvability at each tolerance.
    #
    # The subsampling curve above answers "how many samples does the SIGN need";
    # this answers the question that actually matters for Result 1: is each
    # model's conditioning effect larger than that model's own sampling noise?
    # The floor is obtained by bootstrapping the ensemble at its OWN size, so it
    # is specific to the ensemble rather than borrowed from a different one.
    resolv = []
    for mid in ("hh_point", "hh_cable", "migliore_ca1", "pospischil_rs",
                "pospischil_fs", "khaliq_purkinje", "mainen_l5pc", "bbp_l5pc"):
        df = load_model_table("screening", mid)
        if df is None:
            continue
        pids = [s.parameter_id for s in get_adapter(mid).parameter_specs()]
        ens = build_ensembles(df, mid, pids, CONDITIONING_FEATURES,
                              target=canonical_target(mid, CONDITIONING_FEATURES),
                              seed=1, protocol_family="absolute")
        e1 = next((x for x in ens if x.regime == "E1"), None)
        if e1 is None:
            continue
        base = net_oi(e1.params)
        for regime, tol in (("E3", "permissive"), ("E3b", "moderate"),
                            ("E4", "strict"), ("E4b", "very_strict")):
            e = next((x for x in ens if x.regime == regime and "max_z" in x.label), None)
            if e is None:
                continue
            x = e.params
            delta = net_oi(x) - base
            rng = np.random.default_rng(0)
            boot = np.array([net_oi(x[rng.integers(0, len(x), len(x))], seed=i)
                             for i in range(40)])
            sd = float(boot.std())
            resolv.append({
                "model_id": mid, "tolerance": tol, "regime": regime,
                "n": int(len(x)), "d": int(x.shape[1]), "delta": float(delta),
                "bootstrap_sd": sd,
                "sigma": float(abs(delta) / sd) if sd > 0 else float("inf"),
                "resolvable": bool(sd > 0 and abs(delta) / sd > 2.0),
            })
            print(f"  RESOLV {mid:16s} {tol:12s} n={len(x):5d} "
                  f"delta={delta:+.6f} sd={sd:.6f} "
                  f"{abs(delta)/sd if sd>0 else float('inf'):5.1f}sigma", flush=True)

    if not rows:
        print("no reference ensembles large enough")
        return 1

    by_n: dict[int, dict] = {}
    for n in N_GRID:
        sub = [r for r in rows if r["n"] == n]
        if sub:
            by_n[n] = {
                "p_sign_flip_max": max(r["p_sign_flip"] for r in sub),
                "p_sign_flip_mean": float(np.mean([r["p_sign_flip"] for r in sub])),
                "largest_wrong_sign_value": max(r["max"] for r in sub),
                "n_reference_ensembles": len(sub),
            }
    payload = {
        "statistic": "net_o_information (redundancy_area - synergy_area)",
        "note": ("Sign reliability of the conditioning endpoint as a function of "
                 "ensemble size, measured by subsampling ensembles that are large "
                 "enough to trust down to each n. Outcome-independent: no "
                 "ensemble under dispute is used as a reference."),
        "n_replicates": N_REPLICATES,
        "by_n": by_n,
        "rows": rows,
        "per_model_resolvability": resolv,
        "n_resolvable_permissive": sum(
            1 for r in resolv if r["tolerance"] == "permissive" and r["resolvable"]),
        "n_models_permissive": sum(
            1 for r in resolv if r["tolerance"] == "permissive"),
    }
    (OUT / "estimator_reliability.json").write_text(json.dumps(payload, indent=1))
    print("\nsign-flip probability by sample size:")
    for n, v in sorted(by_n.items()):
        print(f"  n={n:4d}  max P(flip)={v['p_sign_flip_max']:.2f}  "
              f"largest value of the wrong sign={v['largest_wrong_sign_value']:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
