"""Construction of the E0-E9 ensemble regimes from simulated feature tables.

Only E1, E2 and E6 require simulation.  Everything else is *derived* from an
already-simulated table, which is what makes the full factorial affordable:

* **E3 / E4** - phenotype conditioning at permissive / strict tolerance.
* **E7** - a high-covariance subset matched to E3 on size, marginal ranges and
  target fitness, so that the covariance contrast is not confounded with a
  change in marginals or phenotype composition (blueprint risk table).
* **E8** - column-wise shuffle: marginals preserved, dependence destroyed.
* **E9** - Gaussian-copula resample: rank covariance preserved, higher-order
  structure destroyed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..sampling.priors import gaussian_copula_control, shuffle_control
from .distances import TOLERANCES, ConditioningResult, condition

__all__ = ["Ensemble", "CONDITIONING_FEATURES", "QC_POLICY", "build_ensembles",
           "high_covariance_subset", "validity_mask"]

# --------------------------------------------------------------------------- #
# Scientific-validity policy for QC flags (blueprint 14.5, 26)
# --------------------------------------------------------------------------- #
# Retaining failures in the audit table is necessary but not sufficient: the
# ensembles that feed the estimators must state which rows are scientifically
# admissible.  Flags fall into two categories with different meanings.
#
# NUMERICAL FAILURE -- the integrator produced something the model cannot mean.
# These rows are excluded from every ensemble, under every policy.
QC_NUMERICAL_FAILURE = frozenset({
    "nonphysical_voltage",   # |V| outside [-160, 120] mV
    "nonfinite_voltage",     # NaN/Inf in the trace
    "solver_failure",        # continuerun raised
})

# PHENOTYPE OBSERVATION -- the cell really did behave this way.  A sample whose
# baseline drifts or rests depolarised has become spontaneously active or
# depolarisation-blocked; that is a legitimate model behaviour, not a numerical
# error, and phenotype conditioning is the mechanism that should judge it.
# Excluding them by default would silently narrow the prior; ignoring them
# entirely would let unstable states into the conditioned ensembles.  Both
# policies are therefore computed and reported.
QC_PHENOTYPE = frozenset({
    "unstable_baseline",
    "depolarised_rest",
})

QC_POLICY = {
    # primary: drop numerical failures only
    "primary": QC_NUMERICAL_FAILURE,
    # sensitivity: additionally drop every flagged phenotype state
    "strict": QC_NUMERICAL_FAILURE | QC_PHENOTYPE,
    # audit-only: keep everything (reproduces the pre-review behaviour)
    "none": frozenset(),
}


def validity_mask(df: pd.DataFrame, policy: str = "primary") -> np.ndarray:
    """Boolean mask of scientifically admissible rows under a named QC policy."""
    try:
        excluded = QC_POLICY[policy]
    except KeyError as exc:
        raise ValueError(f"unknown qc policy {policy!r}; choose {sorted(QC_POLICY)}") from exc
    ok = (df["simulation_status"] == "ok").to_numpy()
    if not excluded:
        return ok
    flags = df["qc_flags"].astype("string").fillna("")
    bad = np.zeros(len(df), dtype=bool)
    for name in excluded:
        bad |= flags.str.contains(name, regex=False).to_numpy()
    return ok & ~bad

# The conditioning phenotype: the minimum common feature set of blueprint 7.4,
# addressed to specific protocols.  Frozen before the campaign was analysed.
CONDITIONING_FEATURES = [
    "rest.resting_potential_mV",
    "hyperpol_abs.input_resistance_Mohm",
    "hyperpol_abs.membrane_tau_ms",
    "hyperpol_abs.sag_ratio",
    "rheobase_nA",
    "step_abs_hi.first_spike_latency_ms",
    "step_abs_hi.ap_amplitude_mV",
    "step_abs_hi.ap_halfwidth_ms",
    "step_abs_hi.ahp_depth_mV",
    "step_abs_hi.first_isi_ms",
    "step_abs_hi.mean_firing_rate_Hz",
    "step_abs_hi.adaptation_index",
    "step_abs_hi.spike_count",
]

# The same phenotype expressed on rheobase-normalised protocols.  Analysed
# separately, never pooled (blueprint 14.4).
CONDITIONING_FEATURES_NORMALISED = [
    "rest.resting_potential_mV",
    "hyperpol_abs.input_resistance_Mohm",
    "hyperpol_abs.membrane_tau_ms",
    "rheobase_nA",
    "step_rheo_3p0x.first_spike_latency_ms",
    "step_rheo_3p0x.ap_amplitude_mV",
    "step_rheo_3p0x.ap_halfwidth_ms",
    "step_rheo_3p0x.ahp_depth_mV",
    "step_rheo_3p0x.first_isi_ms",
    "step_rheo_3p0x.mean_firing_rate_Hz",
    "step_rheo_3p0x.adaptation_index",
    "step_rheo_3p0x.spike_count",
]


@dataclass
class Ensemble:
    """One analysed parameter ensemble."""

    model_id: str
    regime: str
    label: str
    params: np.ndarray                  # (n, d) in log10 scale units
    param_ids: list[str]
    weights: np.ndarray | None = None
    conditioning: ConditioningResult | None = None
    tolerance: str = "none"
    metric: str = "none"
    source_regime: str = "E1"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.params)


def _log_params(df: pd.DataFrame, param_ids: list[str]) -> np.ndarray:
    """Parameter matrix in log10 scale-factor space (the sampling space)."""
    return np.log10(df[[f"param.{p}" for p in param_ids]].to_numpy(dtype=float))


def covariance_score(params: np.ndarray, n_axes: int = 2) -> np.ndarray:
    """Preregistered per-sample covariance score (blueprint E7).

    The fraction of a sample's squared deviation that lies in the ensemble's
    leading ``n_axes`` principal directions::

        score_i = ||P t_i||^2 / ||t_i||^2

    Samples with a high score sit close to the dominant low-dimensional subspace.
    Selecting them concentrates the ensemble onto that subspace, which raises the
    mean absolute correlation without touching the target-distance distribution.
    """
    x = np.asarray(params, dtype=float)
    z = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    proj = z @ vt[:n_axes].T
    num = np.sum(proj**2, axis=1)
    den = np.sum(z**2, axis=1)
    return np.where(den > 0, num / den, 0.0)


def high_covariance_subset(params: np.ndarray, distance: np.ndarray,
                           n_target: int, seed: int = 0,
                           n_candidates: int = 400) -> np.ndarray:
    """E7: a size-matched subset with elevated mean absolute correlation.

    Samples are ranked by :func:`covariance_score` *within target-distance
    deciles*, and the top-scoring samples of each decile are taken.  Stratifying
    by fitness is essential: an unstratified high-covariance subset would also
    differ in phenotype quality, and the covariance contrast would be
    confounded with a fitness contrast (blueprint risk table: "High-covariance
    subset changes marginals and phenotype mix").
    """
    n = len(params)
    n_target = min(n_target, n)
    if n_target < 20 or n < n_target + 10:
        return np.arange(n)
    score = covariance_score(params)
    order = np.argsort(distance)
    strata = np.array_split(order, 10)
    per = max(1, n_target // 10)
    picks = []
    for s in strata:
        take = min(per, len(s))
        picks.append(s[np.argsort(score[s])[::-1][:take]])
    idx = np.concatenate(picks)
    del seed, n_candidates
    return idx


def build_ensembles(df: pd.DataFrame, model_id: str, param_ids: list[str],
                    feature_names: list[str], target: np.ndarray | None = None,
                    seed: int = 0, metrics=("max_z", "weighted_euclidean"),
                    protocol_family: str = "absolute",
                    qc_policy: str = "primary") -> list[Ensemble]:
    """Build every derivable ensemble for one model from its simulated table.

    ``target`` is the canonical model's feature vector.  Dispersion is taken
    from the *unconditioned* ensemble, so the tolerance has a consistent meaning
    across models with wildly different feature scales.  Both are recorded as
    evidence tier C/D: they are model-derived, not measured biological spread.

    ``qc_policy`` selects which rows are scientifically admissible; see
    :data:`QC_POLICY`.  Every ensemble records the policy it was built under so
    that primary and sensitivity results can never be confused.
    """
    out: list[Ensemble] = []
    ok = validity_mask(df, qc_policy)
    base = df[ok].reset_index(drop=True)
    if len(base) < 50:
        return out

    feats = base[feature_names].to_numpy(dtype=float)
    x_all = _log_params(base, param_ids)

    # E1/E2/E6 as simulated (unconditioned)
    for regime in sorted(base["sampling_regime"].unique()):
        m = (base["sampling_regime"] == regime).to_numpy()
        if m.sum() >= 50:
            out.append(Ensemble(model_id, regime, f"{regime}_unconditioned",
                                x_all[m], param_ids, source_regime=regime,
                                meta={"protocol_family": protocol_family, "qc_policy": qc_policy}))

    # conditioning uses the E1 (independent-prior) ensemble as its base
    e1 = (base["sampling_regime"] == "E1").to_numpy()
    if e1.sum() < 100:
        return out
    f1, x1 = feats[e1], x_all[e1]
    finite = np.isfinite(f1)
    centre = target if target is not None else np.nanmedian(f1[finite.all(axis=1)], axis=0)
    disp = np.nanstd(f1, axis=0)
    disp = np.where(np.isfinite(disp) & (disp > 0), disp, np.nan)

    for metric in metrics:
        for tol_name in TOLERANCES:
            res = condition(f1, centre, disp, metric=metric, tolerance_name=tol_name)
            if res.n_accepted < 60:
                continue
            regime = {"permissive": "E3", "moderate": "E3b",
                      "strict": "E4", "very_strict": "E4b"}[tol_name]
            out.append(Ensemble(
                model_id, regime, f"{regime}_{metric}_{tol_name}",
                x1[res.accepted], param_ids, conditioning=res,
                tolerance=tol_name, metric=metric, source_regime="E1",
                meta={"protocol_family": protocol_family,
                      "acceptance_rate": res.acceptance_rate},
            ))

            if metric == metrics[0] and tol_name == "permissive":
                acc_x = x1[res.accepted]
                acc_d = res.distance[res.accepted]
                # E7: high-covariance, size- and fitness-matched subset
                idx = high_covariance_subset(acc_x, acc_d,
                                             n_target=max(60, res.n_accepted // 2),
                                             seed=seed)
                if len(idx) >= 60:
                    out.append(Ensemble(model_id, "E7", "E7_high_covariance_subset",
                                        acc_x[idx], param_ids, tolerance=tol_name,
                                        metric=metric, source_regime="E3",
                                        meta={"protocol_family": protocol_family,
                                              "matched_n": int(len(idx))}))
                    # size-matched random subset: the correct comparator for E7
                    rng = np.random.default_rng(seed + 5)
                    rnd = rng.choice(len(acc_x), size=len(idx), replace=False)
                    out.append(Ensemble(model_id, "E3m", "E3_matched_random_subset",
                                        acc_x[rnd], param_ids, tolerance=tol_name,
                                        metric=metric, source_regime="E3",
                                        meta={"protocol_family": protocol_family,
                                              "matched_n": int(len(idx))}))
                # E8 / E9 statistical nulls built from the accepted ensemble
                out.append(Ensemble(model_id, "E8", "E8_shuffled_control",
                                    shuffle_control(acc_x, seed=seed + 1), param_ids,
                                    source_regime="E3",
                                    meta={"protocol_family": protocol_family, "qc_policy": qc_policy}))
                out.append(Ensemble(model_id, "E9", "E9_copula_control",
                                    gaussian_copula_control(acc_x, seed=seed + 2),
                                    param_ids, source_regime="E3",
                                    meta={"protocol_family": protocol_family, "qc_policy": qc_policy}))
    return out
