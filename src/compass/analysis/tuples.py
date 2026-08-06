"""Tuple enumeration, surrogate nulls and module stability (blueprint 16.2, 16.6).

Three problems this module exists to solve.

1. **Search-space growth.**  The number of tuples explodes with order, so the
   *extreme* O-information found at order 7 is not comparable to the extreme at
   order 3 unless the size of the search is controlled.  Every summary here is
   therefore reported alongside a per-order surrogate distribution computed with
   the *same number of tuples*.

2. **Determinism.**  Tuple sampling is seeded and sorted, so an analysis rerun
   with the same configuration examines exactly the same tuples.

3. **Stability.**  A top-5% membership list from one run is not evidence.
   Selection probability and Jaccard similarity across replicates/estimators are
   computed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .oinfo import GaussianInfoEngine

__all__ = [
    "OrderCurve",
    "TupleSet",
    "generate_tuples",
    "module_stability",
    "order_curve",
    "surrogate_threshold",
]

MAX_EXHAUSTIVE = 20000  # enumerate exhaustively below this many tuples


@dataclass
class TupleSet:
    order: int
    tuples: list[tuple[int, ...]]
    exhaustive: bool
    n_possible: int

    def __len__(self) -> int:
        return len(self.tuples)


def generate_tuples(n_vars: int, order: int, max_tuples: int = 2000,
                    seed: int = 0) -> TupleSet:
    """Deterministically enumerate or sample index tuples of a given order."""
    from math import comb

    n_possible = comb(n_vars, order)
    if n_possible <= min(max_tuples, MAX_EXHAUSTIVE):
        return TupleSet(order, list(combinations(range(n_vars), order)), True, n_possible)
    rng = np.random.default_rng(seed * 1000 + order)
    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_tuples:
        t = tuple(sorted(rng.choice(n_vars, size=order, replace=False).tolist()))
        seen.add(t)
    return TupleSet(order, sorted(seen), False, n_possible)


@dataclass
class OrderCurve:
    """O-information summaries as a function of tuple order."""

    orders: list[int]
    n_tuples: list[int]
    n_possible: list[int]
    median: list[float]
    q05: list[float]
    q95: list[float]
    frac_synergy: list[float]
    frac_redundancy: list[float]
    frac_synergy_sig: list[float]
    frac_redundancy_sig: list[float]
    mean_negative: list[float]
    mean_positive: list[float]
    area_negative: list[float]
    area_positive: list[float]
    surrogate_lo: list[float]
    surrogate_hi: list[float]

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def normalised_areas(self) -> tuple[float, float]:
        """Normalised synergy / redundancy area under the order curve.

        Blueprint 18.1 primary endpoint 1.  At each order the *area* is
        ``mean_t min(Omega_t, 0)`` and ``mean_t max(Omega_t, 0)`` over all tuples
        examined -- not the mean over the negative tuples alone, which a single
        extreme tuple could dominate.  Each is divided by the tuple order so that
        orders contribute comparably (a tuple of order k contains k marginal
        terms), then averaged across orders.
        """
        neg = float(np.nanmean([a / o for a, o in zip(self.area_negative, self.orders, strict=True)]))
        pos = float(np.nanmean([a / o for a, o in zip(self.area_positive, self.orders, strict=True)]))
        return -neg, pos  # (synergy area >= 0, redundancy area >= 0)


def surrogate_threshold(x: np.ndarray, order: int, n_tuples: int,
                        n_surrogate: int = 40, alpha: float = 0.05,
                        seed: int = 0, copula: bool = False,
                        bias_correct: bool = True) -> tuple[float, float]:
    """Family-wise surrogate bounds for O-information at one tuple order.

    Columns are shuffled independently, destroying all dependence while
    preserving marginals exactly (blueprint 17.1).  For each surrogate ensemble
    the *same number* of tuples is evaluated and the extreme values retained, so
    the returned bounds are family-wise over a search of matched size.
    """
    rng = np.random.default_rng(seed)
    n_vars = x.shape[1]
    los, his = [], []
    for s in range(n_surrogate):
        xs = np.array(x, copy=True)
        for j in range(n_vars):
            rng.shuffle(xs[:, j])
        eng = GaussianInfoEngine(xs, copula=copula, bias_correct=bias_correct)
        ts = generate_tuples(n_vars, order, max_tuples=n_tuples, seed=seed + s).tuples
        vals = eng.o_information_batch(ts)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            los.append(vals.min())
            his.append(vals.max())
    if not los:
        return float("nan"), float("nan")
    return (float(np.quantile(los, alpha)), float(np.quantile(his, 1 - alpha)))


def order_curve(x: np.ndarray, orders=(3, 4, 5), max_tuples: int = 1500,
                seed: int = 0, copula: bool = False, bias_correct: bool = True,
                n_surrogate: int = 25) -> OrderCurve:
    """Compute the full O-information order curve with surrogate controls."""
    x = np.asarray(x, dtype=float)
    n_vars = x.shape[1]
    eng = GaussianInfoEngine(x, copula=copula, bias_correct=bias_correct)
    out = {k: [] for k in OrderCurve.__annotations__}
    for order in orders:
        if order > n_vars:
            continue
        ts = generate_tuples(n_vars, order, max_tuples=max_tuples, seed=seed)
        vals = eng.o_information_batch(ts.tuples)
        vals = vals[np.isfinite(vals)]
        lo, hi = surrogate_threshold(x, order, len(ts), n_surrogate=n_surrogate,
                                     seed=seed, copula=copula,
                                     bias_correct=bias_correct)
        out["orders"].append(order)
        out["n_tuples"].append(len(ts))
        out["n_possible"].append(ts.n_possible)
        out["median"].append(float(np.median(vals)) if vals.size else np.nan)
        out["q05"].append(float(np.quantile(vals, 0.05)) if vals.size else np.nan)
        out["q95"].append(float(np.quantile(vals, 0.95)) if vals.size else np.nan)
        out["frac_synergy"].append(float(np.mean(vals < 0)) if vals.size else np.nan)
        out["frac_redundancy"].append(float(np.mean(vals > 0)) if vals.size else np.nan)
        out["frac_synergy_sig"].append(
            float(np.mean(vals < lo)) if vals.size and np.isfinite(lo) else np.nan)
        out["frac_redundancy_sig"].append(
            float(np.mean(vals > hi)) if vals.size and np.isfinite(hi) else np.nan)
        neg, pos = vals[vals < 0], vals[vals > 0]
        out["mean_negative"].append(float(neg.mean()) if neg.size else 0.0)
        out["mean_positive"].append(float(pos.mean()) if pos.size else 0.0)
        out["area_negative"].append(float(np.minimum(vals, 0.0).mean()) if vals.size else 0.0)
        out["area_positive"].append(float(np.maximum(vals, 0.0).mean()) if vals.size else 0.0)
        out["surrogate_lo"].append(lo)
        out["surrogate_hi"].append(hi)
    return OrderCurve(**out)


def module_stability(rankings: list[list[tuple[int, ...]]], top_k: int = 20) -> dict:
    """Selection probability and Jaccard similarity of top tuples across runs.

    ``rankings`` is a list of per-run tuple lists already ordered from most to
    least extreme.  Returns the per-tuple selection probability and the mean
    pairwise Jaccard index of the top-``top_k`` sets.
    """
    tops = [set(r[:top_k]) for r in rankings if r]
    if len(tops) < 2:
        return {"mean_jaccard": float("nan"), "selection_probability": {},
                "n_runs": len(tops), "n_stable": 0}
    jac = []
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            u = tops[i] | tops[j]
            jac.append(len(tops[i] & tops[j]) / len(u) if u else np.nan)
    counts: dict[tuple[int, ...], int] = {}
    for s in tops:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    prob = {t: c / len(tops) for t, c in counts.items()}
    return {
        "mean_jaccard": float(np.nanmean(jac)),
        "selection_probability": prob,
        "n_runs": len(tops),
        # blueprint 19.4 candidate criterion: selection probability >= 0.8
        "n_stable": int(sum(p >= 0.8 for p in prob.values())),
    }
