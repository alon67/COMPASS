"""Analytical and synthetic validation of the COMPASS information engine.

Blueprint WP5 / Task 10 acceptance criteria:
  * known synergistic, redundant and independent systems are correctly classified;
  * sign conventions hold for both O-information and RSI;
  * results are deterministic for a fixed seed.

These are *numerical* tests: they assert on estimator output, not just on code
paths, and are the gate that licenses every downstream neuronal result.
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.analysis.entropy import (
    gaussian_entropy,
    knn_entropy,
    logdet_bias_correction,
)
from compass.analysis.oinfo import (
    GaussianInfoEngine,
    discrete_o_information,
    discrete_rsi,
    mutual_information,
    o_information,
    rsi,
    s_information,
)

SEED = 20260804
LOG2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# 1. Entropy estimators against closed-form Gaussian entropy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [1, 2, 4])
def test_gaussian_entropy_matches_analytic(dim):
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(dim, dim))
    cov = a @ a.T + dim * np.eye(dim)
    x = rng.multivariate_normal(np.zeros(dim), cov, size=200_000)
    analytic = 0.5 * (dim * np.log(2 * np.pi * np.e) + np.linalg.slogdet(cov)[1])
    assert gaussian_entropy(x) == pytest.approx(analytic, abs=0.02)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_knn_entropy_matches_analytic(dim):
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(40_000, dim))
    analytic = 0.5 * dim * np.log(2 * np.pi * np.e)
    assert knn_entropy(x, k=4) == pytest.approx(analytic, abs=0.06)


def test_logdet_bias_correction_removes_small_sample_bias():
    """The raw sample log-det is biased low; the correction must remove it."""
    rng = np.random.default_rng(SEED)
    dim, n = 6, 60
    raw, corrected = [], []
    for _ in range(400):
        x = rng.normal(size=(n, dim))          # true log det Sigma = 0
        ld = np.linalg.slogdet(np.cov(x, rowvar=False))[1]
        raw.append(ld)
        corrected.append(ld - logdet_bias_correction(n, dim))
    # analytic expectation of the bias for (N=60, d=6) is about -0.369 nats
    assert np.mean(raw) < -0.25                 # substantial negative bias
    assert abs(np.mean(corrected)) < 0.05       # corrected estimate is unbiased


# --------------------------------------------------------------------------- #
# 2. Discrete analytic systems - the sign-convention ground truth
# --------------------------------------------------------------------------- #
def test_xor_is_synergistic_discrete():
    """3-variable XOR: Omega = -1 bit exactly, RSI(X1,X2 -> Y) = +1 bit exactly."""
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    y = x1 ^ x2
    triple = np.column_stack([x1, x2, y])

    omega = discrete_o_information(triple)
    assert omega == pytest.approx(-1.0, abs=1e-9)   # negative -> SYNERGY
    assert discrete_rsi(np.column_stack([x1, x2]), y) == pytest.approx(1.0, abs=1e-9)


def test_duplicated_predictors_are_redundant_discrete():
    """Three exact copies of a fair bit: Omega = +1 bit, RSI = -1 bit."""
    rng = np.random.default_rng(SEED)
    b = rng.integers(0, 2, size=4000)
    triple = np.column_stack([b, b, b])
    assert discrete_o_information(triple) == pytest.approx(1.0, abs=0.02)
    assert discrete_rsi(np.column_stack([b, b]), b) == pytest.approx(-1.0, abs=0.02)


def test_independent_variables_have_zero_o_information_discrete():
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 2, size=(200_000, 3))
    assert abs(discrete_o_information(x)) < 0.01


# --------------------------------------------------------------------------- #
# 3. Continuous systems - the estimators actually used on neuronal ensembles
# --------------------------------------------------------------------------- #
def _independent(n, d, rng):
    return rng.normal(size=(n, d))


def _redundant(n, rng, noise=0.15):
    z = rng.normal(size=n)
    return np.column_stack([z + noise * rng.normal(size=n) for _ in range(3)])


def _synergistic_gaussian(n, rng, noise=0.15):
    """X1, X2 independent; X3 = X1 + X2 + noise.

    A sum-constrained triple is the canonical Gaussian synergy: neither X1 nor X2
    alone says much about X3, but jointly they determine it.
    """
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x3 = x1 + x2 + noise * rng.normal(size=n)
    return np.column_stack([x1, x2, x3])


@pytest.mark.parametrize("estimator", ["gaussian", "copula", "knn"])
def test_continuous_independent_is_near_zero(estimator):
    rng = np.random.default_rng(SEED)
    x = _independent(20_000, 3, rng)
    assert abs(o_information(x, estimator)) < 0.03


@pytest.mark.parametrize("estimator", ["gaussian", "copula", "knn"])
def test_continuous_redundant_is_positive(estimator):
    rng = np.random.default_rng(SEED)
    x = _redundant(20_000, rng)
    assert o_information(x, estimator) > 0.5


@pytest.mark.parametrize("estimator", ["gaussian", "copula", "knn"])
def test_continuous_synergistic_is_negative(estimator):
    rng = np.random.default_rng(SEED)
    x = _synergistic_gaussian(20_000, rng)
    assert o_information(x, estimator) < -0.3


def test_s_information_is_non_negative():
    rng = np.random.default_rng(SEED)
    for build in (_independent, _redundant, _synergistic_gaussian):
        x = build(10_000, rng) if build is not _independent else _independent(10_000, 3, rng)
        assert s_information(x) > -1e-2  # non-negative up to estimator noise


def test_o_information_is_invariant_to_monotone_rescaling_under_copula():
    """Log-transforming a conductance must not change a copula-based result."""
    rng = np.random.default_rng(SEED)
    x = np.exp(_redundant(20_000, rng))          # strictly positive, skewed
    plain = o_information(x, "copula")
    logged = o_information(np.log(x), "copula")
    assert plain == pytest.approx(logged, abs=1e-9)


def test_gaussian_estimator_is_not_invariant_to_monotone_rescaling():
    """Sanity check that the copula invariance above is a real property."""
    rng = np.random.default_rng(SEED)
    x = np.exp(_redundant(20_000, rng))
    assert abs(o_information(x, "gaussian") - o_information(np.log(x), "gaussian")) > 0.05


# --------------------------------------------------------------------------- #
# 4. RSI on continuous targets
# --------------------------------------------------------------------------- #
def test_rsi_positive_for_sum_target():
    """Y = X1 + X2 with independent inputs is synergistic: RSI > 0."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(20_000, 2))
    y = x[:, 0] + x[:, 1] + 0.1 * rng.normal(size=20_000)
    res = rsi(x, y)
    assert res.value > 0.2
    assert res.joint > res.sum_individual


def test_rsi_negative_for_duplicated_predictors():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=20_000)
    x = np.column_stack([z + 0.1 * rng.normal(size=20_000) for _ in range(2)])
    y = z + 0.1 * rng.normal(size=20_000)
    assert rsi(x, y).value < -0.3


def test_rsi_near_zero_for_independent_additive_target():
    """Y = f(X1) + g(X2) with independent X is neither synergistic nor redundant."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(60_000, 2))
    y = x[:, 0] * 2.0 + 0.0 * x[:, 1] + rng.normal(size=60_000)
    assert abs(rsi(x, y).value) < 0.02


def test_mutual_information_matches_analytic_gaussian():
    rng = np.random.default_rng(SEED)
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    xy = rng.multivariate_normal([0, 0], cov, size=100_000)
    analytic = -0.5 * np.log(1 - rho**2)
    assert mutual_information(xy[:, [0]], xy[:, 1]) == pytest.approx(analytic, abs=0.01)


# --------------------------------------------------------------------------- #
# 5. Fast engine agrees with the reference implementation, and is deterministic
# --------------------------------------------------------------------------- #
def test_engine_matches_reference_implementation():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(5_000, 6)) @ rng.normal(size=(6, 6))
    eng = GaussianInfoEngine(x, bias_correct=True)
    for idx in [(0, 1, 2), (1, 3, 5), (0, 2, 3, 4), (0, 1, 2, 3, 4)]:
        ref = o_information(x[:, list(idx)], "gaussian", bias_correct=True)
        assert eng.o_information(idx) == pytest.approx(ref, abs=1e-8)


def test_engine_copula_mode_matches_reference():
    rng = np.random.default_rng(SEED)
    x = np.exp(rng.normal(size=(5_000, 5)) @ rng.normal(size=(5, 5)))
    eng = GaussianInfoEngine(x, copula=True, bias_correct=True)
    ref = o_information(x[:, [0, 1, 2]], "copula", bias_correct=True)
    assert eng.o_information((0, 1, 2)) == pytest.approx(ref, abs=1e-8)


def test_engine_is_deterministic():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(3_000, 5))
    a = GaussianInfoEngine(x).o_information_batch([(0, 1, 2), (2, 3, 4)])
    b = GaussianInfoEngine(x).o_information_batch([(0, 1, 2), (2, 3, 4)])
    assert np.array_equal(a, b)


def test_gaussian_o_information_bias_cancels_across_orders():
    """The *mean* small-sample bias of Gaussian O-information cancels analytically.

    Writing ``c_i = psi((N-i)/2) - log((N-1)/2)`` and noting ``c_i ~ -i/(N-1)``,
    the bias combination entering O-information is

        -2 * sum_{i<n} c_i + (n-2) c_n + n c_1
          ~ [ n(n-1) - n^2 + 2n - n ] / (N-1) = 0.

    So O-information is, to leading order, an *unbiased* function of tuple order
    even at modest N -- unlike total correlation.  This is a load-bearing fact
    for the project: it means order-curves of Omega can be compared directly,
    while TC order-curves cannot.  Estimator *variance* still grows with order,
    which is why per-order surrogate controls remain mandatory.
    """
    rng = np.random.default_rng(SEED)
    orders = [3, 5, 7, 9, 12]
    mean_omega, mean_tc_raw, mean_tc_fix = [], [], []
    for n in orders:
        om, tc_raw, tc_fix = [], [], []
        for _ in range(120):
            x = rng.normal(size=(300, n))          # independent -> truth is 0
            raw = GaussianInfoEngine(x, bias_correct=False)
            fix = GaussianInfoEngine(x, bias_correct=True)
            om.append(raw.o_information(range(n)))
            tc_raw.append(raw.total_correlation(range(n)))
            tc_fix.append(fix.total_correlation(range(n)))
        mean_omega.append(np.mean(om))
        mean_tc_raw.append(np.mean(tc_raw))
        mean_tc_fix.append(np.mean(tc_fix))

    # (a) O-information: no appreciable drift with order, even uncorrected
    assert max(abs(v) for v in mean_omega) < 0.01
    # (b) total correlation: large positive bias that grows with order ...
    assert mean_tc_raw[-1] > 0.05
    assert mean_tc_raw[-1] > mean_tc_raw[0] * 5
    # ... which the analytic correction removes
    assert abs(mean_tc_fix[-1]) < 0.02


def test_estimator_variance_grows_with_order():
    """Motivates per-order surrogate controls (blueprint 16.2 'search-space control')."""
    rng = np.random.default_rng(SEED)
    sds = []
    for n in [3, 6, 10]:
        vals = [
            GaussianInfoEngine(rng.normal(size=(300, n))).o_information(range(n))
            for _ in range(200)
        ]
        sds.append(np.std(vals))
    assert sds[0] < sds[1] < sds[2]


# --------------------------------------------------------------------------- #
# 6. Comparator systems must have no inert parameters
# --------------------------------------------------------------------------- #
def test_no_comparator_parameter_is_inert():
    """Every declared parameter must actually influence the phenotype.

    The repressilator's protein equation once read ``-KP*(p-m)*0 + BETA*(m-p)``.
    The ``* 0`` made ``k_p`` an inert coordinate, so a "six-parameter"
    comparator really had five active parameters plus one nuisance axis that
    diluted every tuple statistic involving it -- while every test still passed,
    because nothing checked that the parameters did anything.
    """
    from compass.analysis.benchmarks import (
        lotka_volterra_features,
        repressilator_features,
    )

    for fn, n_par in ((repressilator_features, 6), (lotka_volterra_features, 6)):
        base = np.ones(n_par)
        f0 = np.asarray(fn(base), dtype=float)
        for j in range(n_par):
            p = base.copy()
            p[j] = 2.0
            delta = np.nanmax(np.abs(np.asarray(fn(p), dtype=float) - f0))
            assert np.isfinite(delta) and delta > 0, (
                f"{fn.__name__} parameter {j} is inert: doubling it changes "
                f"every feature by exactly {delta}"
            )
