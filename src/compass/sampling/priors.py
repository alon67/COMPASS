"""Parameter transforms and prior/ensemble generators (blueprint 9.1.3, 13.2).

All sampling happens in a common **standardised sampling space**: every
parameter is mapped to ``u in [0, 1]`` by its declared transform, so that a
Sobol sequence, a Gaussian copula and a jitter kernel can all be defined once
and reused for every model regardless of units.

Ensemble regimes implemented here (blueprint 13.2):

===== ============================================================= ==========
Code  Regime                                                        Simulated?
===== ============================================================= ==========
E1    unconditioned independent prior (Sobol / LHS)                 yes
E2    unconditioned covariance-preserving prior (Gaussian copula)   yes
E3    broad phenotype-conditioned (permissive tolerance)            derived
E4    tight phenotype-conditioned (strict tolerance)                derived
E6    clustered / oversampled around selected basins                yes
E7    high-covariance matched subset                                derived
E8    marginal-preserving shuffled control                          derived
E9    covariance-matched synthetic control                          derived
===== ============================================================= ==========

"derived" ensembles are produced from an already-simulated ensemble without
further simulation, which is what makes the design affordable.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from ..registry.schema import ParameterSpec

__all__ = [
    "clustered_sample",
    "correlated_sample",
    "gaussian_copula_control",
    "independent_sample",
    "scales_from_unit",
    "shuffle_control",
    "unit_from_scales",
]


# --------------------------------------------------------------------------- #
# Transform between unit hypercube and parameter scale factors
# --------------------------------------------------------------------------- #
def scales_from_unit(u: np.ndarray, specs: list[ParameterSpec],
                     width: float = 1.0) -> np.ndarray:
    """Map ``u`` in [0,1]^d to multiplicative scale factors.

    ``width`` rescales every prior around the canonical value 1.0 (in log space
    for log-transformed parameters).  ``width=0.5`` halves the prior width in
    decades; ``width=2`` doubles it.  This is the knob used for the prior-width
    sensitivity analysis (blueprint 16.6).
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    out = np.empty_like(u)
    for j, spec in enumerate(specs):
        lo, hi = spec.lower_bound, spec.upper_bound
        if spec.transform.value == "log10":
            l10, h10 = np.log10(lo), np.log10(hi)
            l10, h10 = l10 * width, h10 * width
            out[:, j] = 10.0 ** (l10 + u[:, j] * (h10 - l10))
        else:
            c = 1.0
            lo_w, hi_w = c + (lo - c) * width, c + (hi - c) * width
            out[:, j] = lo_w + u[:, j] * (hi_w - lo_w)
    return out


def unit_from_scales(x: np.ndarray, specs: list[ParameterSpec],
                     width: float = 1.0) -> np.ndarray:
    """Inverse of :func:`scales_from_unit` (used by the perturbation engine)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for j, spec in enumerate(specs):
        lo, hi = spec.lower_bound, spec.upper_bound
        if spec.transform.value == "log10":
            l10, h10 = np.log10(lo) * width, np.log10(hi) * width
            out[:, j] = (np.log10(x[:, j]) - l10) / (h10 - l10)
        else:
            lo_w, hi_w = 1.0 + (lo - 1.0) * width, 1.0 + (hi - 1.0) * width
            out[:, j] = (x[:, j] - lo_w) / (hi_w - lo_w)
    return out


# --------------------------------------------------------------------------- #
# E1: independent prior
# --------------------------------------------------------------------------- #
def independent_sample(specs: list[ParameterSpec], n: int, seed: int,
                       method: str = "sobol", width: float = 1.0) -> np.ndarray:
    """Quasi-random independent prior sample (blueprint 14.3).

    ``sobol`` and ``lhs`` are both provided because 14.3 requires at least two
    sampling methods where feasible; agreement between them is one of the
    robustness checks.
    """
    d = len(specs)
    if method == "sobol":
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = eng.random(n)
    elif method == "lhs":
        eng = qmc.LatinHypercube(d=d, seed=seed)
        u = eng.random(n)
    elif method == "random":
        u = np.random.default_rng(seed).random((n, d))
    else:
        raise ValueError(f"unknown sampling method {method!r}")
    return scales_from_unit(u, specs, width)


# --------------------------------------------------------------------------- #
# E2: covariance-preserving prior
# --------------------------------------------------------------------------- #
def _ar1_correlation(d: int, rho: float) -> np.ndarray:
    """Banded correlation matrix; a neutral, reproducible covariance structure.

    Used where no experimental joint distribution exists.  Declared as evidence
    tier D and never described as measured covariance (blueprint 7.2).
    """
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def correlated_sample(specs: list[ParameterSpec], n: int, seed: int,
                      rho: float = 0.6, corr: np.ndarray | None = None,
                      width: float = 1.0) -> np.ndarray:
    """Gaussian-copula sample with prescribed rank correlation.

    Marginals are identical to the independent prior; only the dependence
    structure differs.  That isolates the effect of parameter covariance from
    the effect of the marginal ranges (blueprint secondary question 6).
    """
    d = len(specs)
    corr = _ar1_correlation(d, rho) if corr is None else np.asarray(corr, float)
    rng = np.random.default_rng(seed)
    # nearest positive-definite repair, so an empirical matrix can be passed in
    w, v = np.linalg.eigh(corr)
    w = np.clip(w, 1e-8, None)
    corr = v @ np.diag(w) @ v.T
    dsq = np.sqrt(np.diag(corr))
    corr = corr / np.outer(dsq, dsq)
    z = rng.multivariate_normal(np.zeros(d), corr, size=n)
    from scipy.special import ndtr

    return scales_from_unit(ndtr(z), specs, width)


# --------------------------------------------------------------------------- #
# E6: clustered / oversampled ensemble
# --------------------------------------------------------------------------- #
def clustered_sample(centres: np.ndarray, specs: list[ParameterSpec], n: int,
                     seed: int, n_basins: int = 8, jitter: float = 0.06,
                     width: float = 1.0) -> np.ndarray:
    """Repeatedly resample a few local basins (blueprint E6).

    ``centres`` are accepted solutions in *scale* units.  ``n_basins`` of them
    are chosen deterministically and each is resampled with Gaussian jitter of
    ``jitter`` (in unit-cube units).  The result deliberately mimics an ensemble
    built by repeatedly re-fitting the same few solutions -- the situation that
    the generic-geometry hypothesis predicts will shift ensembles toward
    redundancy.
    """
    rng = np.random.default_rng(seed)
    u_centres = unit_from_scales(centres, specs, width)
    n_basins = min(n_basins, len(u_centres))
    pick = rng.choice(len(u_centres), size=n_basins, replace=False)
    base = u_centres[pick]
    reps = rng.integers(0, n_basins, size=n)
    u = base[reps] + rng.normal(0.0, jitter, size=(n, len(specs)))
    return scales_from_unit(np.clip(u, 0.0, 1.0), specs, width)


# --------------------------------------------------------------------------- #
# E8 / E9: statistical null controls (no simulation required)
# --------------------------------------------------------------------------- #
def shuffle_control(x: np.ndarray, seed: int) -> np.ndarray:
    """E8: shuffle each column independently, preserving marginals exactly."""
    rng = np.random.default_rng(seed)
    out = np.array(x, dtype=float, copy=True)
    for j in range(out.shape[1]):
        rng.shuffle(out[:, j])
    return out


def gaussian_copula_control(x: np.ndarray, seed: int) -> np.ndarray:
    """E9: synthetic ensemble with matched rank-correlation and marginals.

    Rank-transform to normal scores, fit a multivariate Gaussian, resample, then
    map back through the empirical marginal quantiles.  Any higher-order
    structure beyond the copula's second moments is destroyed, which is exactly
    what makes this the right control for a Gaussian O-information estimator.
    """
    from scipy.special import ndtr, ndtri
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    ranks = np.apply_along_axis(rankdata, 0, x)
    z = ndtri(ranks / (n + 1.0))
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-10, None)
    zs = rng.normal(size=(n, d)) @ (v * np.sqrt(w)).T
    us = ndtr(zs / np.sqrt(np.clip(np.diag(cov), 1e-12, None)))
    out = np.empty_like(zs)
    for j in range(d):
        out[:, j] = np.quantile(x[:, j], np.clip(us[:, j], 1e-9, 1 - 1e-9))
    return out
