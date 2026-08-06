"""Sobol estimator validation on functions with analytically known indices.

Blueprint 21.3: "Sobol convergence on known functions"; Task 12 acceptance:
"benchmark functions recover known first- and total-order effects".
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.analysis.perturbation import factorial_design, summarise_nonadditivity
from compass.analysis.sensitivity import saltelli_design, sobol_convergence, sobol_indices

SEED = 20260804


def _ishigami(u: np.ndarray, a=7.0, b=0.1) -> np.ndarray:
    """Ishigami function on [-pi, pi]^3 with closed-form Sobol indices."""
    x = u * 2 * np.pi - np.pi
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def _ishigami_analytic(a=7.0, b=0.1):
    v = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
    v1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
    v2 = a**2 / 8
    v3 = 0.0
    vt3 = b**2 * np.pi**8 * 8 / 225
    return (np.array([v1, v2, v3]) / v,
            np.array([v1 + vt3, v2, vt3]) / v)


def test_sobol_recovers_ishigami_indices():
    n_base = 8192
    u = saltelli_design(3, n_base, seed=SEED)
    y = _ishigami(u)
    res = sobol_indices(y, ["x1", "x2", "x3"], n_base, output_name="ishigami")
    s1, st = _ishigami_analytic()
    assert np.allclose(res.S1, s1, atol=0.03)
    assert np.allclose(res.ST, st, atol=0.03)
    # x3 has zero first-order but non-zero total effect: a pure interaction
    assert res.S1[2] == pytest.approx(0.0, abs=0.03)
    assert res.ST[2] > 0.2


def test_interaction_fraction_is_zero_for_a_purely_additive_function():
    n_base = 4096
    d = 4
    u = saltelli_design(d, n_base, seed=SEED)
    y = u @ np.array([1.0, 2.0, 3.0, 4.0])
    res = sobol_indices(y, [f"x{i}" for i in range(d)], n_base, output_name="additive")
    assert res.interaction_fraction == pytest.approx(0.0, abs=0.02)
    assert np.allclose(res.ST - res.S1, 0.0, atol=0.02)


def test_interaction_fraction_is_large_for_a_pure_product():
    n_base = 4096
    d = 3
    u = saltelli_design(d, n_base, seed=SEED)
    x = u * 2 - 1
    y = x[:, 0] * x[:, 1] * x[:, 2]
    res = sobol_indices(y, [f"x{i}" for i in range(d)], n_base, output_name="product")
    # a product of zero-mean factors has no first-order variance at all
    assert np.nansum(res.S1) < 0.1
    assert res.interaction_fraction > 0.9


def test_sobol_refuses_a_filtered_design():
    n_base = 512
    u = saltelli_design(3, n_base, seed=SEED)
    y = _ishigami(u)
    with pytest.raises(ValueError, match="intact Saltelli design"):
        sobol_indices(y[: len(y) // 2], ["x1", "x2", "x3"], n_base)


def test_sobol_convergence_reports_shrinking_confidence():
    n_base = 4096
    u = saltelli_design(3, n_base, seed=SEED)
    y = _ishigami(u)
    conv = sobol_convergence(y, ["x1", "x2", "x3"], n_base, output_name="ishigami")
    conf = [c["max_S1_conf"] for c in conv if "max_S1_conf" in c]
    assert len(conf) >= 3
    assert conf[-1] < conf[0]


# --------------------------------------------------------------------------- #
# Perturbation engine
# --------------------------------------------------------------------------- #
def test_factorial_design_shape():
    d = factorial_design((0, 2, 5), delta=0.1)
    assert [x["label"] for x in d] == ["baseline", "single_0", "single_2", "single_5", "joint"]
    assert d[-1]["shift"] == {0: 0.1, 2: 0.1, 5: 0.1}


def test_additive_model_has_zero_nonadditivity():
    """f(x) = sum a_i x_i: joint effect equals the sum of single effects."""
    rng = np.random.default_rng(SEED)
    n_bg, k = 8, 3
    a = np.array([1.0, -2.0, 0.5])
    delta = 0.15
    base = rng.normal(size=(n_bg, k))
    f0 = base @ a
    singles = np.column_stack([f0 + a[i] * delta for i in range(k)])
    joint = f0 + a.sum() * delta
    r = summarise_nonadditivity("m", (0, 1, 2), ("p0", "p1", "p2"), "y",
                                f0, singles, joint, feature_sd=1.0)
    assert abs(r.mean_nonadditivity_sd) < 1e-9
    assert not r.functionally_supported()


def test_multiplicative_model_shows_expected_nonadditivity():
    """f(x) = x0*x1: the blueprint's Task 13 verification case."""
    rng = np.random.default_rng(SEED)
    n_bg = 10
    delta = 0.2
    x = rng.uniform(1.0, 2.0, size=(n_bg, 2))
    f = lambda a, b: a * b  # noqa: E731
    f0 = f(x[:, 0], x[:, 1])
    s0 = f(x[:, 0] + delta, x[:, 1])
    s1 = f(x[:, 0], x[:, 1] + delta)
    joint = f(x[:, 0] + delta, x[:, 1] + delta)
    r = summarise_nonadditivity("m", (0, 1), ("p0", "p1"), "y",
                                f0, np.column_stack([s0, s1]), joint,
                                feature_sd=float(np.std(f0)))
    # exact non-additivity of a product is delta^2, independent of background
    assert np.allclose(r.nonadditivity, delta**2)
    assert r.sign_consistency == pytest.approx(1.0)
    # eta / SD here is delta^2 / std(f0) ~ 0.045, so the module is detected at a
    # threshold below that and correctly rejected at one above it
    assert r.functionally_supported(effect_threshold=0.04)
    assert not r.functionally_supported(effect_threshold=0.10)


def test_sign_consistency_detects_direction_flips():
    f0 = np.zeros(8)
    singles = np.zeros((8, 2))
    joint = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    r = summarise_nonadditivity("m", (0, 1), ("a", "b"), "y", f0, singles, joint, 1.0)
    assert r.sign_consistency == pytest.approx(0.5)
    assert not r.functionally_supported()
