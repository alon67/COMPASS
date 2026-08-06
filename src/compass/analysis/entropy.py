"""Differential-entropy estimators used by the COMPASS information engine.

Three estimator families are provided, matching blueprint section 16.2 which
requires the Gaussian estimator used in the original literature plus at least one
non-Gaussian / copula-based alternative:

``gaussian``
    Closed-form entropy of the maximum-likelihood Gaussian fit.  Optional
    analytic small-sample bias correction for ``log det`` (Ahmed & Gokhale 1989).
``copula``
    Gaussian-copula (rank-to-normal-score) estimator, i.e. the GCMI construction
    of Ince et al. (2017).  Invariant to any monotone reparameterisation of the
    marginals, which makes it the natural control for the log-scale parameter
    transforms used throughout this project.  It is a *lower bound* on the true
    entropy-based quantities for non-Gaussian dependence.
``knn``
    Kozachenko-Leonenko k-nearest-neighbour entropy.  Makes no parametric
    assumption at all but is variance-heavy above ~6 dimensions.

All estimators return entropies in **nats**.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

__all__ = [
    "ENTROPY_ESTIMATORS",
    "copula_transform",
    "entropy",
    "gaussian_entropy",
    "knn_entropy",
    "logdet_bias_correction",
]

_LOG_2PI_E = float(np.log(2.0 * np.pi * np.e))


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected 1d or 2d array, got shape {x.shape}")
    return x


def logdet_bias_correction(n_samples: int, dim: int) -> float:
    """Expected bias of ``log det`` of the sample covariance matrix.

    For ``X`` Gaussian with covariance ``S`` and ``N`` samples, the ML estimate
    satisfies

        E[log det S_hat] = log det S + sum_{i=1..d} psi((N - i)/2) - d*log((N-1)/2)

    Subtracting the returned value from ``log det S_hat`` removes this bias
    exactly.  This matters a great deal for O-information: the raw estimator
    combines determinants of different dimensionality, so the biases do **not**
    cancel and produce a spurious drift with tuple order.
    """
    if n_samples <= dim + 1:
        return np.nan
    i = np.arange(1, dim + 1)
    return float(np.sum(digamma((n_samples - i) / 2.0)) - dim * np.log((n_samples - 1) / 2.0))


def gaussian_entropy(x: np.ndarray, bias_correct: bool = True, ridge: float = 1e-12) -> float:
    """Differential entropy (nats) of the Gaussian fit to ``x`` (n_samples, n_dim)."""
    x = _as2d(x)
    n, d = x.shape
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    if ridge:
        cov = cov + ridge * np.eye(d) * max(1.0, float(np.trace(cov)) / d)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.nan
    if bias_correct:
        corr = logdet_bias_correction(n, d)
        if not np.isnan(corr):
            logdet = logdet - corr
    return 0.5 * (d * _LOG_2PI_E + logdet)


def copula_transform(x: np.ndarray) -> np.ndarray:
    """Map each column to standard-normal scores through its empirical rank CDF.

    Ties receive averaged ranks.  The result has exactly Gaussian marginals, so
    any dependence measured afterwards is a property of the copula alone.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    x = _as2d(x)
    n = x.shape[0]
    ranks = np.apply_along_axis(rankdata, 0, x)
    return ndtri(ranks / (n + 1.0))


def copula_entropy(x: np.ndarray, bias_correct: bool = True) -> float:
    """Gaussian entropy of the copula-transformed data."""
    return gaussian_entropy(copula_transform(x), bias_correct=bias_correct)


def knn_entropy(x: np.ndarray, k: int = 4, base_jitter: float = 1e-10,
                rng: np.random.Generator | None = None) -> float:
    """Kozachenko-Leonenko k-NN differential entropy (nats), Chebyshev metric.

    H_hat = -psi(k) + psi(N) + d*log(2) + (d/N) * sum_i log(eps_i)

    where ``eps_i`` is the distance to the i-th point's k-th neighbour.  A tiny
    jitter breaks exact ties (duplicated samples give ``eps = 0`` otherwise).
    """
    x = _as2d(x)
    n, d = x.shape
    if n <= k + 1:
        return np.nan
    rng = rng or np.random.default_rng(0)
    scale = np.std(x, axis=0)
    scale[scale == 0] = 1.0
    xs = x / scale
    xs = xs + rng.normal(0.0, base_jitter, size=xs.shape)
    tree = cKDTree(xs)
    dist, _ = tree.query(xs, k=k + 1, p=np.inf, workers=-1)
    eps = dist[:, k]
    eps = np.where(eps <= 0, np.finfo(float).tiny, eps)
    h = -digamma(k) + digamma(n) + d * np.log(2.0) + d * float(np.mean(np.log(eps)))
    # undo the per-column standardisation (entropy is not scale invariant)
    return h + float(np.sum(np.log(scale)))


def knn_entropy_euclidean(x: np.ndarray, k: int = 4) -> float:
    """Kozachenko-Leonenko entropy with the Euclidean metric (used for checks)."""
    x = _as2d(x)
    n, d = x.shape
    if n <= k + 1:
        return np.nan
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=k + 1, workers=-1)
    eps = np.where(dist[:, k] <= 0, np.finfo(float).tiny, dist[:, k])
    log_vol_unit_ball = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    return (
        -digamma(k)
        + digamma(n)
        + log_vol_unit_ball
        + d * float(np.mean(np.log(eps)))
    )


ENTROPY_ESTIMATORS = {
    "gaussian": gaussian_entropy,
    "copula": copula_entropy,
    "knn": knn_entropy,
}


def entropy(x: np.ndarray, estimator: str = "gaussian", **kwargs) -> float:
    """Dispatch to a named entropy estimator."""
    try:
        fn = ENTROPY_ESTIMATORS[estimator]
    except KeyError as exc:
        raise ValueError(
            f"unknown estimator {estimator!r}; choose from {sorted(ENTROPY_ESTIMATORS)}"
        ) from exc
    return fn(x, **kwargs)
