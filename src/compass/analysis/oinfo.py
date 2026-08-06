"""O-information, total/dual total correlation, and target-specific decompositions.

Conventions (frozen; verified by ``tests/numerical/test_information_benchmarks.py``)
-----------------------------------------------------------------------------------
O-information (Rosas et al. 2019, Phys Rev E 100:032305)::

    Omega(X) = (n - 2) H(X) + sum_j [ H(X_j) - H(X_-j) ]
             = TC(X) - DTC(X)

    Omega > 0  ->  REDUNDANCY-dominated
    Omega < 0  ->  SYNERGY-dominated

Redundancy-synergy index (Chechik et al. 2001, NIPS 14)::

    RSI(X -> Y) = I(X_1..X_n ; Y) - sum_j I(X_j ; Y)

    RSI > 0  ->  SYNERGY   (the tuple tells us more jointly than separately)
    RSI < 0  ->  REDUNDANCY

Note the *opposite* sign polarity of the two quantities.  This is the single most
common source of sign errors in the literature, so both are asserted against
analytic discrete systems (XOR, duplicated predictors, independent predictors) in
the numerical test-suite rather than being taken on trust.

All quantities are returned in **nats** unless ``base=2`` is passed to the
convenience wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy import copula_transform, entropy, logdet_bias_correction

__all__ = [
    "GaussianInfoEngine",
    "InfoResult",
    "dual_total_correlation",
    "mutual_information",
    "o_information",
    "o_information_gradient",
    "rsi",
    "s_information",
    "total_correlation",
]

_LOG_2PI_E = float(np.log(2.0 * np.pi * np.e))


# --------------------------------------------------------------------------- #
# Generic (estimator-agnostic) implementations
# --------------------------------------------------------------------------- #
def _subset_entropies(x: np.ndarray, estimator: str, **kw) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``H(X)``, ``[H(X_j)]``, ``[H(X_-j)]`` for the given estimator."""
    x = np.asarray(x, dtype=float)
    n_var = x.shape[1]
    h_all = entropy(x, estimator, **kw)
    h_single = np.array([entropy(x[:, [j]], estimator, **kw) for j in range(n_var)])
    h_minus = np.array(
        [entropy(x[:, [k for k in range(n_var) if k != j]], estimator, **kw) for j in range(n_var)]
    )
    return h_all, h_single, h_minus


def o_information(x: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """O-information of an (n_samples, n_variables) array. ``>0`` redundancy."""
    x = np.asarray(x, dtype=float)
    n_var = x.shape[1]
    if n_var < 3:
        raise ValueError("O-information requires at least 3 variables")
    h_all, h_single, h_minus = _subset_entropies(x, estimator, **kw)
    return float((n_var - 2) * h_all + np.sum(h_single - h_minus))


def total_correlation(x: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """``TC(X) = sum_j H(X_j) - H(X)``; the multi-information / integration."""
    x = np.asarray(x, dtype=float)
    h_all = entropy(x, estimator, **kw)
    h_single = np.array([entropy(x[:, [j]], estimator, **kw) for j in range(x.shape[1])])
    return float(np.sum(h_single) - h_all)


def dual_total_correlation(x: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """``DTC(X) = (1-n) H(X) + sum_j H(X_-j)``; the binding information."""
    x = np.asarray(x, dtype=float)
    n_var = x.shape[1]
    h_all = entropy(x, estimator, **kw)
    h_minus = np.array(
        [entropy(x[:, [k for k in range(n_var) if k != j]], estimator, **kw) for j in range(n_var)]
    )
    return float((1 - n_var) * h_all + np.sum(h_minus))


def s_information(x: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """``S(X) = TC(X) + DTC(X)``; total shared randomness, always non-negative."""
    return total_correlation(x, estimator, **kw) + dual_total_correlation(x, estimator, **kw)


def mutual_information(x: np.ndarray, y: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """``I(X;Y) = H(X) + H(Y) - H(X,Y)`` with the requested entropy estimator."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return float(
        entropy(x, estimator, **kw)
        + entropy(y, estimator, **kw)
        - entropy(np.hstack([x, y]), estimator, **kw)
    )


def rsi(x: np.ndarray, y: np.ndarray, estimator: str = "gaussian", **kw) -> InfoResult:
    """Redundancy-synergy index of tuple ``x`` about target ``y``.

    Returns joint information and the summed individual informations separately,
    as required by blueprint section 16.3, so that a near-zero RSI arising from
    two large cancelling terms can be distinguished from a genuinely
    uninformative tuple.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    joint = mutual_information(x, y, estimator, **kw)
    singles = np.array([mutual_information(x[:, [j]], y, estimator, **kw) for j in range(x.shape[1])])
    return InfoResult(
        value=float(joint - np.sum(singles)),
        joint=float(joint),
        sum_individual=float(np.sum(singles)),
        individual=singles,
    )


def o_information_gradient(x: np.ndarray, y: np.ndarray, estimator: str = "gaussian", **kw) -> float:
    """Target-resolved O-information gradient ``dOmega = Omega(X, Y) - Omega(X)``.

    Rosas et al. (2019) section IV: the change in O-information caused by adding
    the target to the tuple.  Negative values mean the target adds synergy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(len(x), -1)
    return o_information(np.hstack([x, y]), estimator, **kw) - o_information(x, estimator, **kw)


@dataclass
class InfoResult:
    """Container that keeps the decomposed terms alongside the summary value."""

    value: float
    joint: float
    sum_individual: float
    individual: np.ndarray

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "joint": self.joint,
            "sum_individual": self.sum_individual,
        }


# --------------------------------------------------------------------------- #
# Fast covariance-based engine for large tuple sweeps
# --------------------------------------------------------------------------- #
@dataclass
class GaussianInfoEngine:
    """Vectorised O-information / RSI over many tuples of a fixed variable set.

    A single correlation matrix is factorised once; each tuple then costs only a
    handful of small ``slogdet`` calls.  This is what makes exhaustive order-3/4/5
    enumeration tractable.

    Parameters
    ----------
    data:
        ``(n_samples, n_variables)`` design matrix.
    copula:
        If ``True``, rank-transform columns to normal scores first (GCMI).  This
        yields the non-Gaussian estimator required by blueprint section 16.2
        while retaining closed-form speed.
    bias_correct:
        Apply the analytic ``log det`` small-sample correction.  Strongly
        recommended: without it O-information drifts systematically with tuple
        order purely because of estimation bias.
    """

    data: np.ndarray
    copula: bool = False
    bias_correct: bool = True
    ridge: float = 1e-10

    def __post_init__(self) -> None:
        x = np.asarray(self.data, dtype=float)
        if x.ndim != 2:
            raise ValueError("data must be 2d (n_samples, n_variables)")
        self.n_samples, self.n_vars = x.shape
        if self.copula:
            x = copula_transform(x)
        sd = x.std(axis=0, ddof=1)
        sd = np.where(sd <= 0, 1.0, sd)
        z = (x - x.mean(axis=0)) / sd
        corr = np.cov(z, rowvar=False)
        corr = np.atleast_2d(corr)
        corr.flat[:: self.n_vars + 1] += self.ridge
        self._corr = corr
        self._bias = {
            d: (logdet_bias_correction(self.n_samples, d) if self.bias_correct else 0.0)
            for d in range(1, self.n_vars + 2)
        }

    # -- primitives -------------------------------------------------------- #
    def _logdet(self, idx: np.ndarray | list[int]) -> float:
        sub = self._corr[np.ix_(idx, idx)]
        sign, ld = np.linalg.slogdet(sub)
        if sign <= 0:
            return np.nan
        return ld - self._bias[len(idx)]

    def entropy(self, idx) -> float:
        idx = list(idx)
        return 0.5 * (len(idx) * _LOG_2PI_E + self._logdet(idx))

    # -- quantities -------------------------------------------------------- #
    def o_information(self, idx) -> float:
        idx = list(idx)
        n = len(idx)
        if n < 3:
            raise ValueError("O-information requires at least 3 variables")
        # The (2*pi*e) constants cancel identically; only log-determinants remain.
        ld_all = self._logdet(idx)
        ld_single = sum(self._logdet([j]) for j in idx)
        ld_minus = sum(self._logdet([k for k in idx if k != j]) for j in idx)
        return 0.5 * ((n - 2) * ld_all + ld_single - ld_minus)

    def total_correlation(self, idx) -> float:
        idx = list(idx)
        return 0.5 * (sum(self._logdet([j]) for j in idx) - self._logdet(idx))

    def dual_total_correlation(self, idx) -> float:
        idx = list(idx)
        n = len(idx)
        return 0.5 * (
            (1 - n) * self._logdet(idx)
            + sum(self._logdet([k for k in idx if k != j]) for j in idx)
        )

    def mutual_information(self, idx_a, idx_b) -> float:
        a, b = list(idx_a), list(idx_b)
        return 0.5 * (self._logdet(a) + self._logdet(b) - self._logdet(a + b))

    def rsi(self, idx, target: int) -> InfoResult:
        idx = list(idx)
        joint = self.mutual_information(idx, [target])
        singles = np.array([self.mutual_information([j], [target]) for j in idx])
        return InfoResult(
            value=float(joint - singles.sum()),
            joint=float(joint),
            sum_individual=float(singles.sum()),
            individual=singles,
        )

    def o_information_gradient(self, idx, target: int) -> float:
        idx = list(idx)
        return self.o_information(idx + [target]) - self.o_information(idx)

    # -- batch ------------------------------------------------------------- #
    def o_information_batch(self, tuples) -> np.ndarray:
        """O-information for a sequence of index tuples."""
        return np.array([self.o_information(t) for t in tuples], dtype=float)

    def rsi_batch(self, tuples, target: int) -> np.ndarray:
        """RSI for a sequence of index tuples against a single target column."""
        return np.array([self.rsi(t, target).value for t in tuples], dtype=float)


# --------------------------------------------------------------------------- #
# Discrete reference implementations (used only to validate sign conventions)
# --------------------------------------------------------------------------- #
def discrete_entropy(samples: np.ndarray, base: float = 2.0) -> float:
    """Plug-in (maximum-likelihood) joint entropy of the *rows* of an integer array.

    Rows are treated as joint symbols, so ``discrete_entropy(X)`` for an
    ``(n_samples, n_variables)`` array is ``H(X_1, ..., X_n)``.
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    _, counts = np.unique(samples, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(base))


def discrete_o_information(x: np.ndarray, base: float = 2.0) -> float:
    """Exact plug-in O-information for a discrete sample matrix."""
    x = np.asarray(x)
    n = x.shape[1]
    h_all = discrete_entropy(x, base)
    h_single = sum(discrete_entropy(x[:, [j]], base) for j in range(n))
    h_minus = sum(
        discrete_entropy(x[:, [k for k in range(n) if k != j]], base) for j in range(n)
    )
    return (n - 2) * h_all + h_single - h_minus


def discrete_mutual_information(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    # NB: promote a 1-D input to a COLUMN, not a row.  `np.atleast_2d` turns
    # shape (n,) into (1, n), which silently made every 1-D call collapse to a
    # single sample and return 0.0 -- e.g. MI(x, x) for identical vectors.
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y).reshape(len(x), -1)
    return (
        discrete_entropy(x, base)
        + discrete_entropy(y, base)
        - discrete_entropy(np.hstack([x, y]), base)
    )


def discrete_rsi(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    x = np.asarray(x)
    joint = discrete_mutual_information(x, y, base)
    singles = sum(discrete_mutual_information(x[:, [j]], y, base) for j in range(x.shape[1]))
    return joint - singles
