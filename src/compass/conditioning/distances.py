"""Target-distance metrics and phenotype conditioning (blueprint 15).

Conditioning is treated as an *object of study*, not preprocessing: the whole
point of the project is that conditioning an independent prior on a shared
output can manufacture higher-order dependence.  Five distance metrics are
implemented because blueprint 15.2 explicitly forbids relying on the
maximum-z-score criterion alone.

The conditioning target is the **canonical model's own feature vector**, with
dispersion taken from the unconditioned ensemble.  This gives a well-defined,
model-intrinsic phenotype for every model in the panel without claiming that
any of these dispersions are measured biological variability (they are not:
they are evidence tier C/D by construction, and are reported as such).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConditioningResult",
    "TOLERANCES",
    "condition",
    "max_abs_z",
    "mahalanobis",
    "robust_z",
    "soft_weights",
    "weighted_euclidean",
]

# Nested tolerance series (blueprint 15.3), frozen after the pilot.
# Values are in units of the distance metric being used (z-score units for the
# z-based metrics).
TOLERANCES = {
    "permissive": 3.0,
    "moderate": 2.0,
    "strict": 1.0,
    # A fourth, tighter level.  Solution-manifold contraction is the mechanism
    # the generic-geometry hypothesis proposes, so the tolerance series has to
    # reach a regime where the accepted set is a genuinely thin sheet, not a
    # mildly trimmed prior (blueprint 15.3, E4).
    "very_strict": 0.5,
}


def _standardise(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
    return (x - centre) / scale


def max_abs_z(x, centre, scale) -> np.ndarray:
    """Blueprint 15.2 metric 1: the criterion used by many fitting workflows."""
    z = np.abs(_standardise(x, centre, scale))
    return np.nanmax(z, axis=1)


def weighted_euclidean(x, centre, scale, weights=None) -> np.ndarray:
    """Blueprint 15.2 metric 2: RMS z-score across features."""
    z = _standardise(x, centre, scale)
    w = np.ones(x.shape[1]) if weights is None else np.asarray(weights, float)
    w = w / np.nansum(w)
    return np.sqrt(np.nansum(w * z**2, axis=1))


def mahalanobis(x, centre, cov, ridge: float = 1e-6) -> np.ndarray:
    """Blueprint 15.2 metric 3: accounts for correlations *between features*."""
    d = x - centre
    cov = np.atleast_2d(cov)
    cov = cov + ridge * np.eye(cov.shape[0]) * max(1.0, np.trace(cov) / cov.shape[0])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    d = np.nan_to_num(d, nan=0.0)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0) / x.shape[1])


def robust_z(x, centre_med, mad) -> np.ndarray:
    """Blueprint 15.2 metric 4: median / MAD, insensitive to outlier features."""
    scale = np.where(mad > 0, mad * 1.4826, np.nan)
    return np.nanmax(np.abs(_standardise(x, centre_med, scale)), axis=1)


def soft_weights(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    """Blueprint 15.2 metric 5: Gaussian kernel likelihood weights.

    Soft conditioning avoids the hard-threshold discontinuity; the weights are
    carried through to the information estimators as sample weights so that a
    "conditioned ensemble" need not be a subset.
    """
    w = np.exp(-0.5 * (distance / bandwidth) ** 2)
    s = w.sum()
    return w / s if s > 0 else w


@dataclass
class ConditioningResult:
    """Auditable record of one conditioning operation (blueprint WP3)."""

    metric: str
    tolerance_name: str
    threshold: float
    distance: np.ndarray
    accepted: np.ndarray          # boolean mask
    weights: np.ndarray | None
    n_input: int
    n_accepted: int
    acceptance_rate: float

    def summary(self) -> dict:
        return {
            "metric": self.metric,
            "tolerance": self.tolerance_name,
            "threshold": self.threshold,
            "n_input": self.n_input,
            "n_accepted": self.n_accepted,
            "acceptance_rate": self.acceptance_rate,
            "median_distance": float(np.nanmedian(self.distance)),
        }


def condition(
    features: np.ndarray,
    target: np.ndarray,
    dispersion: np.ndarray,
    metric: str = "max_z",
    tolerance_name: str = "moderate",
    cov: np.ndarray | None = None,
    threshold: float | None = None,
) -> ConditioningResult:
    """Apply one distance metric and tolerance to a feature matrix.

    ``features`` may contain NaN (a feature that is undefined for a given
    sample, e.g. spike shape when the sample does not spike).  Samples with any
    NaN in a conditioning feature are *rejected*, not imputed: a model that
    fails to produce the phenotype has failed the phenotype.
    """
    features = np.asarray(features, dtype=float)
    thr = TOLERANCES[tolerance_name] if threshold is None else threshold

    if metric == "max_z":
        dist = max_abs_z(features, target, dispersion)
    elif metric == "weighted_euclidean":
        dist = weighted_euclidean(features, target, dispersion)
    elif metric == "mahalanobis":
        if cov is None:
            raise ValueError("mahalanobis requires a feature covariance matrix")
        dist = mahalanobis(features, target, cov)
    elif metric == "robust_z":
        dist = robust_z(features, target, dispersion)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    finite_rows = np.isfinite(features).all(axis=1)
    accepted = finite_rows & np.isfinite(dist) & (dist <= thr)
    weights = soft_weights(np.where(np.isfinite(dist), dist, np.inf), bandwidth=thr)
    return ConditioningResult(
        metric=metric, tolerance_name=tolerance_name, threshold=float(thr),
        distance=dist, accepted=accepted, weights=weights,
        n_input=len(features), n_accepted=int(accepted.sum()),
        acceptance_rate=float(accepted.mean()),
    )
