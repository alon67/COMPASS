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

Raw outcomes are preserved (2026-08-06 audit, section 4.3)
----------------------------------------------------------
The previous version computed the per-background arrays, summarised them, and
threw the arrays away.  ``perturbations.parquet`` then held only means, which
made four things impossible after the fact: a discrete-outcome reanalysis of
spike counts, auditing outliers, showing the distribution over backgrounds, and
telling a stable small mean apart from large effects that cancel.
:meth:`PerturbationResult.as_rows` now emits one row per background so the raw
factorial outcome is archived alongside the summary.

On the withdrawn "measurement floor" (2026-08-06 audit, section 4)
------------------------------------------------------------------
An individual firing-rate observation over a 400 ms window is a spike count
times 2.5 Hz, and a per-background ``eta`` inherits that 2.5 Hz grid.  The
reported statistic, however, is the **mean over eight backgrounds**, whose grid
is 2.5/8 = 0.3125 Hz, and a per-model median over six tuple means can be finer
still.  A deterministic quantisation step is also not a stochastic detection
limit: a mean below one quantum is exactly observable and exactly repeatable.
Comparing a median of averaged effects against one unaveraged quantum therefore
tested nothing, and the conclusions drawn from it have been withdrawn.  What
replaces it lives in ``tools/perturbation_inference.py``: background-level
uncertainty for inferential precision, and re-running points at a finer time
step for numerical repeatability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PerturbationResult",
    "background_grid_note",
    "factorial_design",
    "summarise_nonadditivity",
]


def background_grid_note(n_backgrounds: int, window_ms: float = 400.0) -> dict:
    """Quantisation grid of a count-derived rate and of its background mean.

    Provided so that documents quote the arithmetic rather than restate the
    withdrawn claim.  Returns the single-observation quantum and the grid of the
    mean over ``n_backgrounds``; the latter is what the battery actually reports.
    """
    q = 1000.0 / window_ms
    return {
        "single_observation_quantum_Hz": q,
        "n_backgrounds": int(n_backgrounds),
        "mean_over_backgrounds_grid_Hz": q / max(1, int(n_backgrounds)),
        "note": ("a deterministic grid is not a stochastic detection limit; "
                 "use background-level uncertainty for inferential precision "
                 "and a finer-dt rerun for numerical repeatability"),
    }


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
    # Raw factorial outcome, retained so the summary can be re-derived and
    # audited (audit 4.3).  All in native feature units.
    baseline_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    singles_raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    joint_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-background, per-parameter signed displacement actually applied
    # (log10 units).  The battery flips a step inward when +delta would leave
    # the prior, so the applied design is not recoverable from `delta` alone.
    applied_steps: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    status: tuple[str, ...] = ()

    @property
    def mean_nonadditivity_sd(self) -> float:
        """Mean non-additivity in units of the ensemble feature SD."""
        if not np.isfinite(self.feature_sd) or self.feature_sd <= 0:
            return float("nan")
        return float(np.nanmean(self.nonadditivity) / self.feature_sd)

    @property
    def nonadditivity_sd_units(self) -> np.ndarray:
        """Per-background ``eta`` in feature-SD units."""
        if not np.isfinite(self.feature_sd) or self.feature_sd <= 0:
            return np.full_like(self.nonadditivity, np.nan, dtype=float)
        return self.nonadditivity / self.feature_sd

    @property
    def sem_nonadditivity_sd(self) -> float:
        """Standard error of the background mean, in feature-SD units.

        This -- not a quantisation step -- is the quantity that says whether the
        reported mean is distinguishable from zero (audit 4.4 point 4).
        """
        e = self.nonadditivity_sd_units
        e = e[np.isfinite(e)]
        if e.size < 2:
            return float("nan")
        return float(np.std(e, ddof=1) / np.sqrt(e.size))

    def as_rows(self) -> list[dict]:
        """One row per background: the raw factorial outcome, never summarised.

        Written to ``perturbations_raw.parquet``.  Keeping baseline, each
        single, the joint and the applied displacement means a later analysis
        can model spike counts directly instead of inheriting this module's
        choice of summary statistic.
        """
        rows = []
        n = int(self.n_backgrounds)
        for b in range(n):
            base = float(self.baseline_raw[b]) if b < len(self.baseline_raw) else float("nan")
            joint = float(self.joint_raw[b]) if b < len(self.joint_raw) else float("nan")
            row = {
                "model_id": self.model_id,
                "tuple": "|".join(self.parameter_ids),
                "tuple_idx": "-".join(map(str, self.tuple_idx)),
                "order": len(self.tuple_idx),
                "feature": self.feature,
                "background": b,
                "baseline_raw": base,
                "joint_raw": joint,
                "joint_effect": float(self.joint_effect[b]),
                "nonadditivity_raw": float(self.nonadditivity[b]),
                "nonadditivity_sd": float(self.nonadditivity_sd_units[b]),
                "feature_sd": float(self.feature_sd),
                "status": self.status[b] if b < len(self.status) else "ok",
            }
            for k, pid in enumerate(self.parameter_ids):
                if k < self.singles_raw.shape[1] if self.singles_raw.size else False:
                    row[f"single_raw.{pid}"] = float(self.singles_raw[b, k])
                row[f"single_effect.{pid}"] = float(self.single_effects[b, k])
                if self.applied_steps.size:
                    row[f"applied_step.{pid}"] = float(self.applied_steps[b, k])
            rows.append(row)
        return rows

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
            "median_nonadditivity_sd": float(np.nanmedian(self.nonadditivity_sd_units))
            if self.nonadditivity.size else float("nan"),
            "sem_nonadditivity_sd": self.sem_nonadditivity_sd,
            "sign_consistency": self.sign_consistency,
            "mean_joint_effect": float(np.nanmean(self.joint_effect)),
            "mean_sum_singles": float(np.nanmean(np.sum(self.single_effects, axis=1))),
            "n_finite_backgrounds": int(np.isfinite(self.nonadditivity).sum()),
            "functionally_supported": self.functionally_supported(),
        }


def summarise_nonadditivity(model_id: str, tuple_idx, parameter_ids, feature: str,
                            baseline: np.ndarray, singles: np.ndarray,
                            joint: np.ndarray, feature_sd: float,
                            applied_steps: np.ndarray | None = None,
                            status: tuple[str, ...] | None = None,
                            ) -> PerturbationResult:
    """Assemble a :class:`PerturbationResult` from raw feature measurements.

    ``baseline`` is ``(n_backgrounds,)``, ``singles`` is ``(n_backgrounds, k)``
    and ``joint`` is ``(n_backgrounds,)``, all in raw feature units.
    ``applied_steps`` is ``(n_backgrounds, k)`` signed log10 displacements, kept
    because the battery flips a step inward at the prior boundary and the design
    is therefore not recoverable from the nominal ``delta``.
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
    steps = (np.zeros((len(baseline), len(tuple(parameter_ids))))
             if applied_steps is None else np.asarray(applied_steps, float))
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
        baseline_raw=baseline,
        singles_raw=singles,
        joint_raw=joint,
        applied_steps=steps,
        status=tuple(status) if status is not None else tuple(
            "ok" if np.isfinite(v) else "nonfinite" for v in eta),
    )
