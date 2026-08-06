"""Direct factorial perturbation experiments (blueprint 16.5, Task 13).

This is the module that decides whether a *statistical* synergy module is
**functionally supported**.  O-information says nothing about the model
equations; a tuple can be strongly synergistic in an accepted ensemble purely
because conditioning carved a curved manifold out of an independent prior.  The
only way to find out whether the parameters actually interact in the model is to
change them and look.

Protocol, for a candidate tuple ``T = (p1..pk)`` at a background point ``x0``:

    baseline          f(x0)
    single effects    d_i = f(x0 + delta_i) - f(x0)          for each i in T
    joint effect      D   = f(x0 + sum_i delta_i) - f(x0)
    nonadditivity     eta = D - sum_i d_i

``eta`` is reported in units of the ensemble standard deviation of the feature,
so it is comparable across features and models.  A module counts as functionally
supported only if ``|eta|`` exceeds a preregistered threshold **and** its sign is
stable across backgrounds (blueprint 16.5 point 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PerturbationResult", "factorial_design", "summarise_nonadditivity"]


def factorial_design(tuple_idx: tuple[int, ...], delta: float = 0.15) -> list[dict]:
    """Perturbation set for one tuple: baseline, each single, and the joint.

    ``delta`` is a displacement in log10 units (0.15 ~ a 41% conductance change),
    small enough to probe local curvature but large enough to exceed numerical
    noise in spike-count-derived features.
    """
    design = [{"label": "baseline", "shift": {}}]
    for i in tuple_idx:
        design.append({"label": f"single_{i}", "shift": {i: delta}})
    design.append({"label": "joint", "shift": {i: delta for i in tuple_idx}})
    return design


@dataclass
class PerturbationResult:
    model_id: str
    tuple_idx: tuple[int, ...]
    parameter_ids: tuple[str, ...]
    feature: str
    n_backgrounds: int
    single_effects: np.ndarray        # (n_backgrounds, k)
    joint_effect: np.ndarray          # (n_backgrounds,)
    nonadditivity: np.ndarray         # (n_backgrounds,)
    feature_sd: float

    @property
    def mean_nonadditivity_sd(self) -> float:
        """Mean non-additivity in units of the ensemble feature SD."""
        if not np.isfinite(self.feature_sd) or self.feature_sd <= 0:
            return float("nan")
        return float(np.nanmean(self.nonadditivity) / self.feature_sd)

    @property
    def sign_consistency(self) -> float:
        """Fraction of backgrounds agreeing with the modal sign of ``eta``."""
        s = np.sign(self.nonadditivity[np.isfinite(self.nonadditivity)])
        s = s[s != 0]
        if s.size == 0:
            return float("nan")
        return float(max((s > 0).mean(), (s < 0).mean()))

    def functionally_supported(self, effect_threshold: float = 0.25,
                               consistency_threshold: float = 0.75) -> bool:
        """Preregistered decision rule (blueprint 19.4).

        The tuple is called functionally supported when the mean non-additivity
        exceeds ``effect_threshold`` ensemble SDs *and* its direction repeats in
        at least ``consistency_threshold`` of backgrounds.
        """
        eta = self.mean_nonadditivity_sd
        return bool(
            np.isfinite(eta)
            and abs(eta) >= effect_threshold
            and self.sign_consistency >= consistency_threshold
        )

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tuple": "|".join(self.parameter_ids),
            "order": len(self.tuple_idx),
            "feature": self.feature,
            "n_backgrounds": self.n_backgrounds,
            "mean_nonadditivity_sd": self.mean_nonadditivity_sd,
            "sign_consistency": self.sign_consistency,
            "mean_joint_effect": float(np.nanmean(self.joint_effect)),
            "mean_sum_singles": float(np.nanmean(np.sum(self.single_effects, axis=1))),
            "functionally_supported": self.functionally_supported(),
        }


def summarise_nonadditivity(model_id: str, tuple_idx, parameter_ids, feature: str,
                            baseline: np.ndarray, singles: np.ndarray,
                            joint: np.ndarray, feature_sd: float) -> PerturbationResult:
    """Assemble a :class:`PerturbationResult` from raw feature measurements.

    ``baseline`` is ``(n_backgrounds,)``, ``singles`` is ``(n_backgrounds, k)``
    and ``joint`` is ``(n_backgrounds,)``, all in raw feature units.
    """
    baseline = np.asarray(baseline, float)
    singles = np.asarray(singles, float)
    joint = np.asarray(joint, float)
    d_single = singles - baseline[:, None]
    d_joint = joint - baseline
    # Use `sum`, not `nansum`: a missing individual effect must propagate to NaN
    # rather than be silently treated as zero, which would manufacture apparent
    # non-additivity out of a failed simulation.
    eta = d_joint - np.sum(d_single, axis=1)
    return PerturbationResult(
        model_id=model_id,
        tuple_idx=tuple(tuple_idx),
        parameter_ids=tuple(parameter_ids),
        feature=feature,
        n_backgrounds=len(baseline),
        single_effects=d_single,
        joint_effect=d_joint,
        nonadditivity=eta,
        feature_sd=float(feature_sd),
    )
