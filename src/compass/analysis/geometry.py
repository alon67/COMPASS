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

__all__ = ["EnsembleGeometry", "describe_geometry", "intrinsic_dimension_twonn"]


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
    log_generalised_variance: float
    condition_number: float

    def as_dict(self) -> dict:
        return asdict(self)


def intrinsic_dimension_twonn(x: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Two-NN intrinsic-dimension estimator (Facco et al. 2017, Sci Rep 7:12140).

    Uses the ratio ``mu = r2/r1`` of the distances to the first two neighbours;
    ``mu`` is Pareto(d) distributed, so ``d`` follows from a linear fit of
    ``-log(1-F(mu))`` against ``log(mu)``.  The largest ``discard_fraction`` of
    ratios is dropped, as recommended, to limit the influence of outliers.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return float("nan")
    # de-duplicate: identical rows give r1 = 0
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=3, workers=-1)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = (r1 > 0) & (r2 > 0)
    if ok.sum() < 20:
        return float("nan")
    mu = np.sort(r2[ok] / r1[ok])
    keep = int(len(mu) * (1.0 - discard_fraction))
    mu = mu[:keep]
    f = np.arange(1, len(mu) + 1) / len(mu)
    xs, ys = np.log(mu), -np.log(np.clip(1.0 - f, 1e-12, None))
    good = np.isfinite(xs) & np.isfinite(ys)
    if good.sum() < 10:
        return float("nan")
    # fit through the origin, as the estimator prescribes
    return float(np.sum(xs[good] * ys[good]) / np.sum(xs[good] ** 2))


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
        log_generalised_variance=float(np.sum(np.log(evals))),
        condition_number=float(evals[0] / evals[-1]),
    )
