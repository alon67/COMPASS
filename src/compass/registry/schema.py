"""Typed schemas for model, parameter, protocol and experiment manifests.

Blueprint Task 1: "invalid units, missing provenance, and inconsistent identifiers
fail validation".  Validation is eager -- every dataclass checks itself in
``__post_init__`` -- so a malformed manifest cannot reach the simulation engine.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

__all__ = [
    "EvidenceTier",
    "ExperimentSpec",
    "ModelManifest",
    "ParameterSpec",
    "ProtocolSpec",
    "Transform",
    "ValidationError",
]

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

# Units accepted at the registry boundary.  Anything else must be converted by
# the adapter, not silently accepted (blueprint 14.2).
_KNOWN_UNITS = {
    "S/cm2", "mho/cm2", "cm/s", "uF/cm2", "ohm-cm", "mV", "ms", "um",
    "mM", "1", "dimensionless", "nA", "degC", "S/cm2_scale",
}


class ValidationError(ValueError):
    """Raised when a manifest violates the schema contract."""


class EvidenceTier(str, Enum):
    """Blueprint 7.2 evidence tiers.  Governs what a result may be *called*."""

    A = "A"  # direct experimental joint/marginal distribution
    B = "B"  # experimental mean/variance/bootstrap
    C = "C"  # published optimisation range or posterior  -> NOT biological variability
    D = "D"  # expert-defined plausible range             -> sensitivity range only

    @property
    def permits_biological_claim(self) -> bool:
        return self in (EvidenceTier.A, EvidenceTier.B)

    @property
    def description(self) -> str:
        return {
            "A": "Experimentally constrained population prior",
            "B": "Experimentally grounded uncertainty prior",
            "C": "Model-fitting prior, not a biological distribution",
            "D": "Sensitivity range only",
        }[self.value]


class Transform(str, Enum):
    """Sampling-space transform.  Recorded per parameter (blueprint 14.2)."""

    LINEAR = "linear"
    LOG10 = "log10"
    LOGIT = "logit"

    def to_sample_space(self, x):
        import numpy as np

        if self is Transform.LINEAR:
            return x
        if self is Transform.LOG10:
            return np.log10(x)
        return np.log(x / (1.0 - x))

    def from_sample_space(self, u):
        import numpy as np

        if self is Transform.LINEAR:
            return u
        if self is Transform.LOG10:
            return np.power(10.0, u)
        return 1.0 / (1.0 + np.exp(-u))


@dataclass(frozen=True)
class ParameterSpec:
    """One free parameter of one model.

    COMPASS samples *multiplicative scale factors* on the model's canonical
    spatial conductance profile rather than absolute densities.  This is what
    makes parameters semantically comparable across models with different
    morphologies and unit conventions (blueprint 14.2 forbids comparing raw
    magnitudes), and it preserves each model's published channel distribution,
    which becomes an explicit ablation condition instead of an accident.

    ``canonical_value`` is retained in absolute model units for the audit table.
    """

    model_id: str
    parameter_id: str
    mechanism_name: str
    attribute: str
    compartment: str
    channel_family: str
    biophysical_role: str
    units: str
    canonical_value: float
    lower_bound: float                       # in *scale* units (canonical = 1.0)
    upper_bound: float
    evidence_tier: EvidenceTier
    source_reference: str
    transform: Transform = Transform.LOG10
    distribution_family: str = "loguniform"
    distribution_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_id", "parameter_id"):
            val = getattr(self, name)
            if not _ID_RE.match(val):
                raise ValidationError(f"{name}={val!r} must match {_ID_RE.pattern}")
        if self.units not in _KNOWN_UNITS:
            raise ValidationError(
                f"{self.parameter_id}: unknown unit {self.units!r}; "
                f"convert at the adapter boundary or extend _KNOWN_UNITS"
            )
        if not (self.lower_bound < self.upper_bound):
            raise ValidationError(
                f"{self.parameter_id}: lower_bound {self.lower_bound} "
                f">= upper_bound {self.upper_bound}"
            )
        if self.transform is Transform.LOG10 and self.lower_bound <= 0:
            raise ValidationError(
                f"{self.parameter_id}: log10 transform requires a positive lower bound"
            )
        if not self.source_reference:
            raise ValidationError(f"{self.parameter_id}: source_reference is required")
        if not (self.lower_bound <= 1.0 <= self.upper_bound):
            raise ValidationError(
                f"{self.parameter_id}: canonical scale 1.0 must lie inside "
                f"[{self.lower_bound}, {self.upper_bound}]"
            )

    @property
    def log_width(self) -> float:
        """Prior width in decades -- the natural 'how broad is this prior' scalar."""
        import numpy as np

        return float(np.log10(self.upper_bound) - np.log10(self.lower_bound))

    def to_row(self) -> dict:
        d = asdict(self)
        d["evidence_tier"] = self.evidence_tier.value
        d["transform"] = self.transform.value
        d["distribution_parameters"] = json.dumps(self.distribution_parameters)
        d["log_width_decades"] = self.log_width
        return d


@dataclass(frozen=True)
class ProtocolSpec:
    """A stimulation protocol in the common core (blueprint 14.4)."""

    protocol_id: str
    kind: str                      # step | rest | iv | rheobase | short_pulse
    amplitude_nA: float | None = None
    amplitude_rheobase_mult: float | None = None
    delay_ms: float = 200.0
    duration_ms: float = 1000.0
    total_ms: float = 1500.0
    dt_ms: float = 0.025
    v_init_mV: float | None = None
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.protocol_id):
            raise ValidationError(f"protocol_id={self.protocol_id!r} is not a valid identifier")
        if self.kind not in {"step", "rest", "iv", "rheobase", "short_pulse"}:
            raise ValidationError(f"unknown protocol kind {self.kind!r}")
        if self.kind == "step" and self.amplitude_nA is None and self.amplitude_rheobase_mult is None:
            raise ValidationError(f"{self.protocol_id}: a step needs an absolute or relative amplitude")
        if self.delay_ms + self.duration_ms > self.total_ms:
            raise ValidationError(f"{self.protocol_id}: stimulus extends past total_ms")

    @property
    def is_normalised(self) -> bool:
        """Rheobase-normalised protocols must be analysed separately (blueprint 14.4)."""
        return self.amplitude_rheobase_mult is not None


@dataclass(frozen=True)
class ModelManifest:
    """Versioned description of one model in the panel (blueprint 8.1).

    Lineage fields (2026-08-06 audit, section 9.6 item 1)
    ----------------------------------------------------
    The previous panel had eight ``model_id`` values but only six independently
    developed sources, and every cross-model fold, bootstrap and variance
    component treated the eight as independent.  Three identifiers now make the
    dependence structure explicit and machine-readable:

    ``source_lineage_id``
        The independently developed body of model code and parameters.  Two
        entries share a lineage when one was derived from the other, when they
        come from the same publication or optimisation pipeline, or when the
        same person wrote both implementations.  **This is the unit of
        inference**: clustering, leave-one-out folds and bootstraps resample
        lineages, not model IDs.
    ``implementation_id``
        The specific realisation within a lineage (morphology variant, cell
        class, re-optimisation).  Nested inside the lineage.
    ``neuron_class_id``
        Biological cell class, used for class-specific claims.  A claim about a
        class needs several *lineages* in that class, not several
        implementations of one.
    """

    model_id: str
    neuron_class: str
    species: str
    brain_region: str
    simulator: str
    source_reference: str
    license: str
    temperature_c: float
    is_point_model: bool
    n_compartments: int
    mechanism_dir: str | None
    lab: str
    pipeline: str                 # "bbp" | "independent"
    source_lineage_id: str = ""
    implementation_id: str = ""
    neuron_class_id: str = ""
    source_accession: str = ""
    parent_lineage: str = ""      # non-empty when derived from another lineage
    cohort: str = "discovery"     # "discovery" | "replication"
    # "full"    -- the common-core protocol yields every conditioning feature
    # "partial" -- some features are structurally undefined for this cell (e.g. a
    #              slow pacemaker fires too few spikes in the 400 ms window for
    #              ISI-derived features).  Such a model is reported in a labelled
    #              stratum and excluded from primary cross-model inference rather
    #              than silently compared on a different scientific object
    #              (2026-08-06 audit, section 9.5 point 6).
    protocol_coverage: str = "full"
    v_init_mV: float = -70.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.model_id):
            raise ValidationError(f"model_id={self.model_id!r} is not a valid identifier")
        if not self.source_reference:
            raise ValidationError(f"{self.model_id}: source_reference (DOI/accession) is required")
        if not self.license:
            raise ValidationError(f"{self.model_id}: license is required for redistribution audit")
        if self.pipeline not in {"bbp", "independent"}:
            raise ValidationError(f"{self.model_id}: pipeline must be 'bbp' or 'independent'")
        if self.is_point_model and self.n_compartments != 1:
            raise ValidationError(f"{self.model_id}: point model must declare 1 compartment")
        if not (0.0 < self.temperature_c < 45.0):
            raise ValidationError(f"{self.model_id}: implausible temperature {self.temperature_c}")
        # Lineage bookkeeping is mandatory: an unset lineage silently defaults to
        # "this model is its own independent family", which is exactly the
        # overstatement the 2026-08-06 audit flagged (finding 4).
        for name in ("source_lineage_id", "implementation_id", "neuron_class_id"):
            val = getattr(self, name)
            if not val:
                raise ValidationError(f"{self.model_id}: {name} is required")
            if not _ID_RE.match(val):
                raise ValidationError(
                    f"{self.model_id}: {name}={val!r} must match {_ID_RE.pattern}")
        if self.cohort not in {"discovery", "replication"}:
            raise ValidationError(f"{self.model_id}: cohort must be discovery|replication")
        if self.protocol_coverage not in {"full", "partial"}:
            raise ValidationError(f"{self.model_id}: protocol_coverage must be full|partial")
        if self.parent_lineage and self.parent_lineage == self.source_lineage_id:
            raise ValidationError(f"{self.model_id}: parent_lineage cannot be its own lineage")

    def to_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    """A frozen, hashable experiment configuration (blueprint 12 and 20)."""

    experiment_id: str
    models: tuple[str, ...]
    regimes: tuple[str, ...]
    n_samples: int
    replicates: int
    base_seed: int
    tolerances: tuple[str, ...] = ("permissive", "moderate", "strict")
    prior_widths: tuple[str, ...] = ("default",)
    notes: str = ""

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.experiment_id):
            raise ValidationError(f"experiment_id={self.experiment_id!r} is invalid")
        if self.n_samples < 1 or self.replicates < 1:
            raise ValidationError("n_samples and replicates must be >= 1")
        if len(set(self.models)) != len(self.models):
            raise ValidationError("duplicate model ids in experiment")

    @property
    def config_hash(self) -> str:
        """Stable hash used to stamp every result row with its configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["config_hash"] = self.config_hash
        path.write_text(json.dumps(payload, indent=1, default=str))
        return path
