"""Descriptive geometry of a parameter ensemble (blueprint 16.1).

These quantities are the *statistical-geometry* covariates of the hierarchical
model in blueprint 18.2.  The central claim under test is that they, rather than
neuron identity, explain most of the variance in the information endpoints, so
they must be measured with the same care as the information quantities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

__all__ = [
    "ID_ESTIMATOR_VERSION",
    "EnsembleGeometry",
    "describe_geometry",
    "intrinsic_dimension_mle",
    "intrinsic_dimension_twonn",
]

# Stamped into every endpoint row.  Bump whenever the estimator changes so that
# a stored geometry covariate can never be silently compared across algorithm
# versions (2026-08-06 audit, section 3.4 point 4).
#
# twonn-2 fixed the biased empirical CDF of twonn-1: it renormalised F by the
# *retained* count after trimming and assigned F=1 to the last retained point,
# so -log(1-F) hit the 1e-12 clip (~27.6) and dragged the through-origin slope
# up by 45-50%.
ID_ESTIMATOR_VERSION = "twonn-2+mle-1"


@dataclass
class EnsembleGeometry:
    n_samples: int
    n_params: int
    mean_abs_pearson: float
    max_abs_pearson: float
    mean_abs_spearman: float
    effective_rank: float
    participation_ratio: float
    pc1_variance_fraction: float
    intrinsic_dimension: float
    intrinsic_dimension_mle: float
    id_estimator_version: str
    log_generalised_variance: float
    condition_number: float

    def as_dict(self) -> dict:
        return asdict(self)


def _neighbour_ratios(x: np.ndarray) -> np.ndarray | None:
    """Sorted ``mu = r2/r1`` over points with two distinct nearest neighbours."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 20:
        return None
    tree = cKDTree(x)
    # Duplicate rows give r1 = 0.  Query enough neighbours that a point with a
    # few exact duplicates still finds two *distinct* radii instead of being
    # dropped, which is what a coarsely quantised ensemble produces.
    k = int(min(len(x), 8))
    dist, _ = tree.query(x, k=k, workers=-1)
    mu = []
    for row in dist[:, 1:]:
        pos = row[row > 0]
        if pos.size >= 2:
            mu.append(pos[1] / pos[0])
    if len(mu) < 20:
        return None
    return np.sort(np.asarray(mu, dtype=float))


def intrinsic_dimension_twonn(x: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Two-NN intrinsic-dimension estimator (Facco et al. 2017, Sci Rep 7:12140).

    ``mu = r2/r1`` is Pareto(d) distributed, so ``d`` is the through-origin slope
    of ``-log(1 - F(mu))`` against ``log(mu)``.

    The empirical CDF follows the reference implementation
    (``skdim.id.TwoNN``): ``F_emp = arange(n_keep) / N`` with
    ``n_keep = int(N * (1 - discard_fraction))`` and ``N`` the *untrimmed* count.
    Two properties matter and both were violated before the 2026-08-06 audit:

    1. the denominator stays ``N`` after trimming, so discarding the heavy tail
       does not stretch the retained CDF back over ``[0, 1]``; and
    2. no observation is assigned ``F = 1``, so ``-log(1-F)`` never has to be
       clipped -- the clip value dominated the least-squares slope.

    Synthetic calibration for both forms is in
    ``tests/numerical/test_intrinsic_dimension.py``.
    """
    mu = _neighbour_ratios(x)
    if mu is None:
        return float("nan")
    n_total = len(mu)
    n_keep = int(n_total * (1.0 - discard_fraction))
    if n_keep < 10:
        return float("nan")
    femp = np.arange(n_keep, dtype=float) / n_total
    xs = np.log(mu[:n_keep])
    ys = -np.log1p(-femp)
    good = np.isfinite(xs) & np.isfinite(ys)
    denom = float(np.sum(xs[good] ** 2))
    if good.sum() < 10 or denom <= 0:
        return float("nan")
    return float(np.sum(xs[good] * ys[good]) / denom)


def intrinsic_dimension_mle(x: np.ndarray, k: int = 10) -> float:
    """Levina & Bickel (2005) maximum-likelihood intrinsic dimension.

    An estimator with completely different bias behaviour from TwoNN, carried
    alongside it so that intrinsic dimension enters the hierarchical model as a
    *sensitivity covariate* rather than a uniquely measured truth (2026-08-06
    audit, section 3.4 point 5).  Uses the MacKay-Ghahramani averaging of the
    inverse estimates, which is unbiased for ``1/d`` rather than for ``d``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < max(20, k + 2):
        return float("nan")
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=k + 1, workers=-1)
    inv = []
    for row in dist[:, 1:]:
        pos = row[row > 0]
        if pos.size < 3:
            continue
        tk = pos[-1]
        s = np.sum(np.log(tk / pos[:-1]))
        if s > 0:
            inv.append(s / (pos.size - 1))
    if len(inv) < 20:
        return float("nan")
    mean_inv = float(np.mean(inv))
    return float(1.0 / mean_inv) if mean_inv > 0 else float("nan")


def describe_geometry(x: np.ndarray) -> EnsembleGeometry:
    """Full geometric description of an ensemble in sampling space."""
    x = np.asarray(x, dtype=float)
    n, d = x.shape
    sd = x.std(axis=0, ddof=1)
    sd = np.where(sd <= 0, 1.0, sd)
    z = (x - x.mean(axis=0)) / sd

    corr = np.atleast_2d(np.corrcoef(z, rowvar=False))
    off = ~np.eye(d, dtype=bool)
    with np.errstate(invalid="ignore"):
        rho = spearmanr(x).statistic if d > 2 else np.array([[1.0, spearmanr(x).statistic],
                                                             [spearmanr(x).statistic, 1.0]])
    rho = np.atleast_2d(rho)

    cov = np.atleast_2d(np.cov(z, rowvar=False))
    evals = np.linalg.eigvalsh(cov)[::-1]
    evals = np.clip(evals, 1e-12, None)
    p = evals / evals.sum()
    # effective rank = exp(spectral entropy) (Roy & Vetterli 2007)
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    part_ratio = float(evals.sum() ** 2 / np.sum(evals**2))

    return EnsembleGeometry(
        n_samples=int(n),
        n_params=int(d),
        mean_abs_pearson=float(np.nanmean(np.abs(corr[off]))),
        max_abs_pearson=float(np.nanmax(np.abs(corr[off]))),
        mean_abs_spearman=float(np.nanmean(np.abs(rho[off]))),
        effective_rank=eff_rank,
        participation_ratio=part_ratio,
        pc1_variance_fraction=float(p[0]),
        intrinsic_dimension=intrinsic_dimension_twonn(z),
        intrinsic_dimension_mle=intrinsic_dimension_mle(z),
        id_estimator_version=ID_ESTIMATOR_VERSION,
        log_generalised_variance=float(np.sum(np.log(evals))),
        condition_number=float(evals[0] / evals[-1]),
    )
