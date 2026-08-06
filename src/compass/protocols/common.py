"""Common-core stimulation protocols (blueprint 14.4).

Two protocol families are executed for every model and analysed **separately**,
because rheobase normalisation can hide exactly the differences the study is
about (blueprint 14.4, risk table 23):

``absolute``
    Fixed currents in nA, chosen per model before the campaign from the
    canonical model's rheobase so that the same *physical* stimulus is applied
    regardless of what a given parameter sample does to excitability.
``normalised``
    Multiples of each sample's own rheobase, so every sample is compared at a
    matched distance from its own threshold.

The per-model absolute amplitudes are frozen in ``ABSOLUTE_AMPLITUDES`` before
any ensemble is generated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..registry.schema import ProtocolSpec

# Rheobase of each canonical model (nA), measured by tools/canonical_audit.py.
# Absolute-current protocols are pinned to these so they are model-appropriate
# yet fixed across the whole ensemble.
CANONICAL_RHEOBASE = {
    "hh_point": 0.0640, "hh_cable": 0.1006, "pospischil_rs": 0.5449,
    "pospischil_fs": 0.3857, "khaliq_purkinje": 0.0, "mainen_l5pc": 0.1582,
    "migliore_ca1": 0.1777, "bbp_l5pc": 0.3384,
}

# Models that fire without injected current.  Rheobase is 0 for these, so the
# rheobase-normalised protocol family is not defined; they are analysed with the
# absolute family only and this is reported as a coverage limitation.
SPONTANEOUSLY_ACTIVE = frozenset({"khaliq_purkinje"})

# Absolute step amplitudes (nA) per model: a sub-threshold hyperpolarising step
# plus two depolarising steps bracketing the canonical repetitive-firing range.
# Chosen from the measured canonical f-I curves in results/manifests/ so that
# both depolarising steps land on the rising part of each model's f-I curve and
# clear of depolarisation block.
ABSOLUTE_AMPLITUDES = {
    "hh_point": (-0.050, 0.200, 0.400),      # ~58 / ~76 Hz
    "hh_cable": (-0.080, 0.320, 0.640),      # ~62 / ~82 Hz
    "pospischil_rs": (-0.200, 0.900, 1.600),  # ~30 / ~82 Hz
    "pospischil_fs": (-0.150, 0.700, 1.300),  # ~90 / ~160 Hz
    "khaliq_purkinje": (-0.050, 0.030, 0.100),  # ~88 / ~132 Hz
    "mainen_l5pc": (-0.150, 0.350, 0.750),   # ~20 / ~28 Hz
    # Migliore CA1 has a narrow dynamic range: with its published leak-reversal
    # compensation it fires from 0.18 nA and enters depolarisation block above
    # ~0.40 nA, so both steps sit inside the 0.18-0.36 nA firing window.
    "migliore_ca1": (-0.100, 0.220, 0.300),   # ~42 / ~78 Hz
    "bbp_l5pc": (-0.250, 0.600, 1.100),      # ~14 / ~18 Hz
}

RHEOBASE_MULTIPLES = (1.5, 3.0)

# Timing.  Long enough for adaptation to develop (>=3 spikes at the slowest
# model's ~18 Hz plateau), short enough that a multicompartment model can be run
# thousands of times under a broad prior.
DELAY_MS = 120.0
DURATION_MS = 400.0
TOTAL_MS = 550.0
REST_TOTAL_MS = 300.0

# Rheobase bisection uses a deliberately short probe: threshold crossing is a
# fast event, so a 140 ms step resolves it at ~1/3 the cost of a full protocol.
# The bisection runs ~9 probes per sample, so this dominates the sample budget.
RHEO_DELAY_MS = 40.0
RHEO_DURATION_MS = 140.0
RHEO_TOTAL_MS = 200.0


@dataclass(frozen=True)
class ProtocolResultKey:
    """Identifies one (protocol, family) pair in the feature table."""

    protocol_id: str
    family: str  # "absolute" | "normalised" | "rest"


def common_core(model_id: str) -> list[ProtocolSpec]:
    """The frozen common-core protocol list for one model."""
    lo, mid, hi = ABSOLUTE_AMPLITUDES[model_id]
    protocols = [
        ProtocolSpec("rest", "rest", amplitude_nA=0.0, delay_ms=0.0,
                     duration_ms=0.0, total_ms=REST_TOTAL_MS),
        ProtocolSpec("hyperpol_abs", "step", amplitude_nA=lo, delay_ms=DELAY_MS,
                     duration_ms=DURATION_MS, total_ms=TOTAL_MS),
        ProtocolSpec("step_abs_lo", "step", amplitude_nA=mid, delay_ms=DELAY_MS,
                     duration_ms=DURATION_MS, total_ms=TOTAL_MS),
        ProtocolSpec("step_abs_hi", "step", amplitude_nA=hi, delay_ms=DELAY_MS,
                     duration_ms=DURATION_MS, total_ms=TOTAL_MS),
    ]
    for mult in RHEOBASE_MULTIPLES:
        tag = str(mult).replace(".", "p")
        protocols.append(
            ProtocolSpec(f"step_rheo_{tag}x", "step", amplitude_rheobase_mult=mult,
                         delay_ms=DELAY_MS, duration_ms=DURATION_MS, total_ms=TOTAL_MS)
        )
    return protocols


def protocol_family(spec: ProtocolSpec) -> str:
    if spec.kind == "rest":
        return "rest"
    return "normalised" if spec.is_normalised else "absolute"
