"""Synthetic calibration of the intrinsic-dimension estimators.

The 2026-08-06 audit found that the previous TwoNN implementation inflated the
estimate by 45-50% because it renormalised the empirical CDF after trimming and
assigned ``F = 1`` to the last retained observation.  Intrinsic dimension is a
load-bearing covariate of the cross-model variance partition, so the estimator
now has to demonstrate on data of *known* dimension that it is unbiased, and it
has to keep demonstrating it on every release.

These tests are the regression guard for that defect.  They deliberately use
manifolds whose dimension is known by construction rather than any project
artifact, so they cannot drift with the campaign.
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.analysis.geometry import (
    ID_ESTIMATOR_VERSION,
    describe_geometry,
    intrinsic_dimension_mle,
    intrinsic_dimension_twonn,
)

SEED = 20260806


def _gaussian(d: int, n: int = 2000, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# Tolerances are one-sided-aware: TwoNN is known to be mildly *downward* biased
# at high dimension for finite nonuniform samples, so the accepted window widens
# with d.  The defect this file guards against was a 45-50% UPWARD bias, which
# fails every one of these bands.
@pytest.mark.parametrize(("true_d", "lo", "hi"), [
    (2, 1.80, 2.20),
    (3, 2.70, 3.30),
    (5, 4.50, 5.50),
    (8, 6.90, 8.60),
    (10, 8.50, 10.80),
    (15, 11.80, 15.80),
])
def test_twonn_recovers_gaussian_dimension(true_d, lo, hi):
    est = intrinsic_dimension_twonn(_gaussian(true_d))
    assert lo <= est <= hi, f"d={true_d}: estimate {est:.3f} outside [{lo}, {hi}]"


@pytest.mark.parametrize("true_d", [2, 3, 5, 8, 10])
def test_twonn_is_not_systematically_inflated(true_d):
    """The specific 2026-08-06 defect: a large positive bias at every dimension."""
    est = intrinsic_dimension_twonn(_gaussian(true_d))
    assert est < true_d * 1.20, (
        f"d={true_d}: estimate {est:.3f} exceeds +20% -- the renormalised-CDF "
        "bug (or an equivalent) is back"
    )


def test_twonn_matches_the_reference_empirical_cdf():
    """Independent transcription of skdim.id.TwoNN's estimator, same answer."""
    from scipy.spatial import cKDTree

    x = _gaussian(5)
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=3, workers=-1)
    r1, r2 = dist[:, 1], dist[:, 2]
    mu = np.sort(r2[(r1 > 0) & (r2 > 0)] / r1[(r1 > 0) & (r2 > 0)])
    n = len(mu)
    n_keep = int(n * 0.9)
    femp = np.arange(n_keep) / n
    xs, ys = np.log(mu[:n_keep]), -np.log(1.0 - femp)
    reference = float(np.sum(xs * ys) / np.sum(xs**2))
    assert intrinsic_dimension_twonn(x) == pytest.approx(reference, rel=1e-9)


def test_twonn_recovers_an_embedded_low_dimensional_manifold():
    """A 3-d manifold embedded in 12 ambient dimensions is still 3-d."""
    rng = np.random.default_rng(SEED)
    latent = rng.standard_normal((3000, 3))
    basis = rng.standard_normal((3, 12))
    est = intrinsic_dimension_twonn(latent @ basis)
    assert 2.6 <= est <= 3.5, est


@pytest.mark.parametrize("n", [200, 500, 1000, 4000])
def test_twonn_is_stable_across_sample_sizes(n):
    est = intrinsic_dimension_twonn(_gaussian(5, n=n))
    assert 4.0 <= est <= 6.0, f"n={n}: {est:.3f}"


@pytest.mark.parametrize("frac", [0.0, 0.05, 0.1, 0.2, 0.3])
def test_twonn_is_stable_across_trimming_fractions(frac):
    """The estimate must not be a strong function of the discard fraction.

    Under the old renormalising CDF it was: trimming moved the artificial F=1
    point to a smaller mu, which changed the slope substantially.
    """
    est = intrinsic_dimension_twonn(_gaussian(5), discard_fraction=frac)
    assert 4.3 <= est <= 5.8, f"discard={frac}: {est:.3f}"


def test_twonn_spread_over_trimming_fractions_is_small():
    vals = [intrinsic_dimension_twonn(_gaussian(5), discard_fraction=f)
            for f in (0.0, 0.05, 0.1, 0.2, 0.3)]
    assert (max(vals) - min(vals)) < 0.6, vals


def test_twonn_tolerates_duplicate_rows():
    """Exactly duplicated parameter draws must not silently destroy the estimate.

    A duplicate gives r1 = 0.  The estimator looks past zero radii instead of
    discarding the point, so a quantised or partially duplicated ensemble is
    still measurable.
    """
    x = _gaussian(5, n=1500)
    dup = np.vstack([x, x[:150]])
    est = intrinsic_dimension_twonn(dup)
    assert np.isfinite(est)
    assert 4.0 <= est <= 6.2, est


def test_twonn_returns_nan_for_degenerate_input():
    assert np.isnan(intrinsic_dimension_twonn(np.zeros((5, 3))))
    assert np.isnan(intrinsic_dimension_twonn(np.zeros((400, 3))))


@pytest.mark.parametrize(("true_d", "lo", "hi"), [
    (2, 1.7, 2.4), (5, 4.2, 5.9), (10, 8.0, 11.5),
])
def test_mle_estimator_recovers_gaussian_dimension(true_d, lo, hi):
    est = intrinsic_dimension_mle(_gaussian(true_d))
    assert lo <= est <= hi, f"d={true_d}: {est:.3f}"


def test_two_estimators_agree_on_a_clean_manifold():
    """Disagreement between the two would mean the covariate is not identified."""
    x = _gaussian(5, n=3000)
    a, b = intrinsic_dimension_twonn(x), intrinsic_dimension_mle(x)
    assert abs(a - b) < 1.2, (a, b)


def test_describe_geometry_reports_both_estimators_and_a_version():
    geo = describe_geometry(_gaussian(5, n=800)).as_dict()
    assert np.isfinite(geo["intrinsic_dimension"])
    assert np.isfinite(geo["intrinsic_dimension_mle"])
    assert geo["id_estimator_version"] == ID_ESTIMATOR_VERSION
    assert geo["id_estimator_version"].startswith("twonn-2")
