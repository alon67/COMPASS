"""Synthetic benchmarks and the non-neuronal comparator (blueprint 17.2, 17.3).

Two jobs.

**Estimator validation.**  Systems with analytically known dependence structure
(independent, duplicated, XOR-like, multiplicative, linear-correlated) confirm
that the estimators assign the right sign and that sample-size bias is under
control before any neuronal number is believed.

**The generic-geometry test.**  The nonlinear many-to-one map and the
repressilator let us ask the project's central question in a system with *no
biology in it at all*: if conditioning an independent prior on a shared output
manufactures higher-order dependence here, then finding the same thing in a
neuron model is not evidence about neurons.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SYNTHETIC_SYSTEMS",
    "lotka_volterra_features",
    "repressilator_features",
    "sample_synthetic",
]


# --------------------------------------------------------------------------- #
# Synthetic systems with known structure
# --------------------------------------------------------------------------- #
def _independent(n, d, rng):
    """No dependence: O-information must be ~0 at every order."""
    return rng.normal(size=(n, d))


def _duplicated(n, d, rng, noise=0.2):
    """Predictors 0-2 are noisy copies of one latent: strongly redundant."""
    x = rng.normal(size=(n, d))
    z = rng.normal(size=n)
    for j in range(3):
        x[:, j] = z + noise * rng.normal(size=n)
    return x


def _xor_like(n, d, rng, noise=0.1):
    """x_last = product of two independent inputs: the continuous XOR analogue.

    Every pairwise correlation is exactly zero, yet the last variable is fully
    determined by the other two jointly.  This is the cleanest test of whether an
    estimator can see synergy that is invisible to second-order statistics.
    """
    x = rng.normal(size=(n, d))
    x[:, -1] = x[:, 0] * x[:, 1] + noise * rng.normal(size=n)
    return x


def _multiplicative(n, d, rng, noise=0.1):
    """x_last = product of the *three* preceding inputs: order-4 synergy."""
    x = rng.normal(size=(n, d))
    x[:, -1] = x[:, 0] * x[:, 1] * x[:, 2] + noise * rng.normal(size=n)
    return x


def _linear_correlated(n, d, rng, rho=0.7):
    """Banded Gaussian correlation: redundancy without any nonlinearity."""
    idx = np.arange(d)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    return rng.multivariate_normal(np.zeros(d), corr, size=n)


def _sum_constrained(n, d, rng, noise=0.15):
    """A sum constraint: synergy that IS visible to a Gaussian estimator.

    Contrast with :func:`_xor_like`: both are synergistic, but only this one has
    the synergy expressed in second-order statistics.  The pair separates
    "the estimator is wrong" from "the estimator cannot see this kind of
    structure".
    """
    x = rng.normal(size=(n, d))
    x[:, -1] = x[:, :-1].sum(axis=1) + noise * rng.normal(size=n)
    return x


def _mixture(n, d, rng, sep=2.5):
    """Two well-separated clusters: pooling induces spurious correlation.

    Within each cluster the variables are independent; the correlation exists
    only because two populations were pooled.  This is the statistical shape of
    an ensemble built by pooling separately-fitted cell models, and blueprint
    17.2 requires it as an explicit control.
    """
    x = rng.normal(size=(n, d))
    lab = rng.integers(0, 2, size=n)
    x += sep * (lab[:, None] - 0.5) * np.ones(d)
    return x


SYNTHETIC_SYSTEMS = {
    "independent": (_independent, "none", 0.0),
    "duplicated": (_duplicated, "redundancy", +1.0),
    "xor_like": (_xor_like, "synergy", -1.0),
    "multiplicative": (_multiplicative, "synergy", -1.0),
    "linear_correlated": (_linear_correlated, "redundancy", +1.0),
    "sum_constrained": (_sum_constrained, "synergy", -1.0),
    "mixture_pooled": (_mixture, "redundancy", +1.0),
}


def sample_synthetic(name: str, n: int, d: int, seed: int) -> np.ndarray:
    fn, _, _ = SYNTHETIC_SYSTEMS[name]
    return fn(n, d, np.random.default_rng(seed))


# --------------------------------------------------------------------------- #
# Non-neuronal comparator 1: the repressilator (Elowitz & Leibler 2000)
# --------------------------------------------------------------------------- #
def repressilator_features(params: np.ndarray, t_end: float = 160.0,
                           n_t: int = 640) -> np.ndarray:
    """Oscillation phenotype of the three-gene repressilator.

    Parameters (6): ``alpha0, alpha, beta, n_hill, k_m, k_p`` as multiplicative
    scale factors on the canonical values.  This is a nonlinear, many-to-one
    parameter-to-phenotype map with a comparable parameter count to the point
    neuron models -- and no neurons in it.

    Returns ``[period, amplitude, mean_level, n_peaks, cv_peak_interval,
    log_protein_mean]``; NaN where the system does not oscillate.
    """
    from scipy.integrate import solve_ivp
    from scipy.signal import find_peaks

    a0, a, b, nh, km, kp = params
    ALPHA0, ALPHA, BETA, NH = 0.2 * a0, 250.0 * a, 5.0 * b, 2.0 * nh
    KM, KP = 1.0 * km, 1.0 * kp

    def rhs(_t, y):
        m, p = y[:3], y[3:]
        prev = p[[2, 0, 1]]
        dm = -KM * m + ALPHA / (1.0 + prev**NH) + ALPHA0
        # Protein dynamics: translation from mRNA minus degradation.  An earlier
        # version read `-KP * (p - m) * 0 + BETA * (m - p)`; the `* 0` made k_p
        # an inert coordinate, so the "six-parameter" comparator really had five
        # active parameters and one nuisance axis that diluted every tuple
        # statistic involving it.  Caught by the 2026-08-06 audit and by the
        # parameter-influence test now in tests/numerical/.
        dp = -KP * p + BETA * m
        return np.concatenate([dm, dp])

    y0 = np.array([0.2, 0.1, 0.3, 1.0, 2.0, 3.0])
    out = np.full(6, np.nan)
    try:
        sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=np.linspace(0, t_end, n_t),
                        method="LSODA", rtol=1e-5, atol=1e-7)
    except Exception:  # noqa: BLE001
        return out
    if not sol.success:
        return out
    p1 = sol.y[3]
    half = n_t // 2
    seg, tseg = p1[half:], sol.t[half:]
    if not np.all(np.isfinite(seg)):
        return out
    span = float(np.ptp(seg))
    out[1] = span
    out[2] = float(np.mean(seg))
    out[5] = float(np.log10(max(np.mean(seg), 1e-12)))
    peaks, _ = find_peaks(seg, prominence=max(0.05 * span, 1e-9))
    out[3] = float(len(peaks))
    if len(peaks) >= 2:
        iv = np.diff(tseg[peaks])
        out[0] = float(np.mean(iv))
        out[4] = float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# Non-neuronal comparator 2: three-species Lotka-Volterra
# --------------------------------------------------------------------------- #
def lotka_volterra_features(params: np.ndarray, t_end: float = 200.0,
                            n_t: int = 1500) -> np.ndarray:
    """Coexistence phenotype of a 3-species competitive Lotka-Volterra system.

    Parameters (6): growth rates ``r1, r2, r3`` and competition strengths
    ``a12, a23, a31`` as scale factors.  Returns
    ``[mean_x1, mean_x2, mean_x3, total_biomass, oscillation_amplitude,
    n_surviving]``.
    """
    from scipy.integrate import solve_ivp

    r1, r2, r3, a12, a23, a31 = params
    r = np.array([1.0 * r1, 0.72 * r2, 1.53 * r3])
    A = np.array([
        [1.0, 1.09 * a12, 1.52],
        [0.0, 1.0, 0.44 * a23],
        [2.33 * a31, 0.0, 1.0],
    ])

    def rhs(_t, x):
        x = np.clip(x, 0.0, 1e6)
        return r * x * (1.0 - A @ x)

    out = np.full(6, np.nan)
    try:
        sol = solve_ivp(rhs, (0.0, t_end), np.array([0.3, 0.3, 0.3]),
                        t_eval=np.linspace(0, t_end, n_t), method="LSODA",
                        rtol=1e-7, atol=1e-9)
    except Exception:  # noqa: BLE001
        return out
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return out
    seg = sol.y[:, n_t // 2:]
    out[:3] = seg.mean(axis=1)
    out[3] = float(seg.sum(axis=0).mean())
    out[4] = float(np.mean(np.ptp(seg, axis=1)))
    out[5] = float(np.sum(seg.mean(axis=1) > 1e-3))
    return out
