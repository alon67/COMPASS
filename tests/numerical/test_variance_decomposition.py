"""The mixed-model variance decomposition must not double-count group variance.

Regression guard for audit finding M7.  The original implementation took the
fixed-effect variance as ``np.var(m.fittedvalues)``.  statsmodels' MixedLM
``fittedvalues`` already include the random-effect predictions, so the
between-group variance appeared once inside that term and again as its own term.
The inflated denominator turned the reported "ICC" into a quantity that was
neither the adjusted ICC nor the variance-partition coefficient, and it was
wrong by a factor of six on the real data (0.048 reported, 0.289 correct).

These tests use synthetic data with a known group structure, so a recurrence is
caught by an identity that must hold rather than by a remembered number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.analysis.hierarchical import mixed_effects_components


def _panel(n_groups: int = 8, n_per: int = 60, group_sd: float = 0.8,
           resid_sd: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """A panel with a genuine between-family effect and known covariates."""
    rng = np.random.default_rng(seed)
    rows = []
    offsets = rng.normal(0.0, group_sd, size=n_groups)
    regimes = ["E1", "E3", "E4"]
    for g in range(n_groups):
        for _ in range(n_per):
            reg = regimes[rng.integers(len(regimes))]
            mp = float(rng.normal())
            er = float(rng.normal())
            idim = float(rng.normal())
            lns = float(rng.normal())
            y = (0.9 * mp + 0.4 * er + offsets[g]
                 + (0.6 if reg == "E3" else 0.0)
                 + rng.normal(0.0, resid_sd))
            rows.append({"model_id": f"m{g}", "regime": reg,
                         "mean_abs_pearson": mp, "effective_rank_norm": er,
                         "intrinsic_dimension_norm": idim,
                         "log_n_samples": lns, "synergy_area": y,
                         "n_samples": 1024, "estimator": "gaussian"})
    return pd.DataFrame(rows)


def _fit(**kw) -> dict:
    res = mixed_effects_components(_panel(**kw), "synergy_area")
    if "error" in res:
        pytest.skip(f"mixed model did not converge: {res['error']}")
    return res


def test_variance_components_are_a_partition() -> None:
    """The three components must define the ratios that are reported."""
    r = _fit()
    total = (r["var_fixed_effects"] + r["var_model_identity_random"]
             + r["var_residual"])
    assert total > 0
    assert r["vpc_model_identity"] == pytest.approx(
        r["var_model_identity_random"] / total, rel=1e-9)
    assert r["r2_marginal"] == pytest.approx(r["var_fixed_effects"] / total, rel=1e-9)
    assert r["r2_conditional"] == pytest.approx(
        (r["var_fixed_effects"] + r["var_model_identity_random"]) / total, rel=1e-9)
    assert r["icc_adjusted"] == pytest.approx(
        r["var_model_identity_random"]
        / (r["var_model_identity_random"] + r["var_residual"]), rel=1e-9)


def test_adjusted_icc_exceeds_the_variance_partition_coefficient() -> None:
    """The identity that the double-counting bug used to break.

    The adjusted ICC divides by (group + residual); the VPC divides by
    (fixed + group + residual).  With any non-zero fixed-effect variance the
    former MUST be the larger of the two.  The old code produced a single
    number labelled "ICC" that satisfied neither definition.
    """
    r = _fit()
    assert r["var_fixed_effects"] > 0
    assert r["icc_adjusted"] > r["vpc_model_identity"]
    assert 0.0 <= r["vpc_model_identity"] <= 1.0
    assert 0.0 <= r["icc_adjusted"] <= 1.0


def test_fixed_effect_variance_is_the_fixed_part_alone() -> None:
    """The direct signature of the bug, checked against the fitted model.

    ``var_fixed_effects`` must equal Var(X b) computed from the fixed design and
    coefficients only.  It must NOT equal Var(fittedvalues), which statsmodels
    forms as fixed + random and which the original code used.  With a real
    group effect the two differ, so this distinguishes the correct quantity
    from the buggy one rather than merely bounding it.
    """
    import statsmodels.formula.api as smf

    from compass.analysis.hierarchical import prepare

    d = prepare(_panel(group_sd=1.5)).dropna(subset=["synergy_area"]).copy()
    d["y"] = ((d["synergy_area"] - d["synergy_area"].mean())
              / (d["synergy_area"].std() or 1.0))
    fixed = ("C(regime) + mean_abs_pearson + effective_rank_norm "
             "+ intrinsic_dimension_norm + log_n_samples")
    m = smf.mixedlm(f"y ~ {fixed}", d, groups=d["model_id"]).fit(reml=True)

    fixed_only = float(np.var(np.asarray(m.model.exog) @ np.asarray(m.fe_params)))
    with_random = float(np.var(m.fittedvalues))

    r = mixed_effects_components(_panel(group_sd=1.5), "synergy_area")
    if "error" in r:
        pytest.skip("mixed model did not converge")

    assert r["var_fixed_effects"] == pytest.approx(fixed_only, rel=1e-6)
    # the buggy quantity is strictly larger here, so this is a real distinction
    assert with_random > fixed_only * 1.05
    assert r["var_fixed_effects"] != pytest.approx(with_random, rel=1e-3)


def test_icc_tracks_the_true_group_structure() -> None:
    """A panel with little between-family variance must report a smaller ICC.

    Stated as a comparison rather than an absolute bound: at group_sd near zero
    the REML fit sits on a variance boundary and the optimiser is unreliable, so
    an absolute threshold would test statsmodels rather than this code.
    """
    lo = _fit(group_sd=0.05, n_per=120)
    hi = _fit(group_sd=2.00, n_per=120)
    assert hi["icc_adjusted"] > lo["icc_adjusted"] + 0.2
    assert hi["vpc_model_identity"] > lo["vpc_model_identity"]
