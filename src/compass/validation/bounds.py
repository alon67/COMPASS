"""Model-specific canonical behaviour bounds (audit gate N1/N2).

Design rule, taken verbatim from the audit:

    Canonical biological bounds must be model-specific.  A Purkinje model,
    fast-spiking interneuron, reconstructed pyramidal cell and HH point model
    should not be forced through one generic "reasonable neuron" interval.  The
    bounds should be frozen before the release campaign and derived from the
    model's source publication, upstream validation scripts and protocol
    definitions.  They should be broad enough to accommodate legitimate
    implementation and platform variation, but narrow enough to detect unit
    errors, wrong temperatures, bad initialisation, inactive channels,
    incorrect current polarity and solver instability.

So each interval below carries the reason it has the width it has.  They are
deliberately *loose* -- they are unit-error and dead-channel detectors, not
goodness-of-fit tests.  Tight quantitative agreement with a source figure is a
different gate (N2), which needs the matching published stimulus, temperature
and recording site, and is recorded per model in ``docs/model_audits``.

A feature absent from a model's dict is **not applicable** to that model and is
not checked; a feature present must be finite and inside its interval.  Nothing
here is derived from a COMPASS result, so these cannot be tuned to a wanted
outcome.
"""

from __future__ import annotations

__all__ = ["CANONICAL_BOUNDS", "bounds_for"]

# key -> (lo, hi, why)
CANONICAL_BOUNDS: dict[str, dict[str, tuple[float, float, str]]] = {
    "hh_point": {
        "resting_potential_mV": (-70.0, -58.0, "HH squid rest ~ -65 mV at 6.3 C"),
        "rheobase_nA": (0.01, 0.30, "30 um sphere, gl = 3e-4 S/cm2"),
        "ap_amplitude_mV": (60.0, 130.0, "full Na spike; a dead Na channel falls below"),
        "ap_halfwidth_ms": (0.3, 3.0, "6.3 C squid kinetics are slow"),
        "rate_at_hi_Hz": (20.0, 150.0, "0.4 nA sits on the repetitive-firing branch"),
        "input_resistance_Mohm": (5.0, 200.0, "area 2.8e-5 cm2, Rm ~ 3.3 kOhm cm2"),
    },
    "hh_cable": {
        "resting_potential_mV": (-70.0, -58.0, "same channels as hh_point"),
        "rheobase_nA": (0.02, 0.60, "added dendritic load raises threshold"),
        "ap_amplitude_mV": (55.0, 130.0, "somatic spike with dendritic loading"),
        "ap_halfwidth_ms": (0.3, 3.0, "as hh_point"),
        "rate_at_hi_Hz": (20.0, 150.0, "0.64 nA on the repetitive branch"),
    },
    "pospischil_rs": {
        "resting_potential_mV": (-75.0, -62.0, "e_pas = -70 mV"),
        "rheobase_nA": (0.1, 1.2, "Rin ~ 34 MOhm, threshold ~ 15 mV above rest"),
        "ap_amplitude_mV": (55.0, 120.0, "hh2 Na/K at 36 C"),
        "ap_halfwidth_ms": (0.2, 1.5, "36 C cortical spike"),
        "rate_at_hi_Hz": (10.0, 120.0, "RS class saturates well below FS"),
        "input_resistance_Mohm": (15.0, 70.0, "template comment: Rin = 34 MOhm"),
        "adaptation_index": (0.0, 0.9, "I_M gives real spike-frequency adaptation"),
    },
    "pospischil_fs": {
        "resting_potential_mV": (-75.0, -62.0, "e_pas = -70 mV"),
        "rheobase_nA": (0.1, 1.2, "Rin ~ 48 MOhm"),
        "ap_amplitude_mV": (55.0, 120.0, "hh2 Na/K at 36 C"),
        "ap_halfwidth_ms": (0.1, 1.0, "FS spikes are narrower than RS"),
        "rate_at_hi_Hz": (40.0, 400.0, "FS class: no I_M, high maximum rate"),
        "input_resistance_Mohm": (20.0, 90.0, "template comment: Rin = 48 MOhm"),
    },
    "khaliq_purkinje": {
        "resting_potential_mV": (-70.0, -35.0, "spontaneously active: 'rest' is the "
                                               "mean of an ongoing spike train"),
        "spontaneous_rate_Hz": (10.0, 90.0, "Khaliq et al. 2003: 30-50 Hz at 22 C; "
                                            "widened for implementation variation"),
        "ap_amplitude_mV": (40.0, 120.0, "resurgent-Na Purkinje spike"),
        "ap_halfwidth_ms": (0.1, 1.5, "fast Purkinje spike at 22 C"),
    },
    "mainen_l5pc": {
        "resting_potential_mV": (-80.0, -60.0, "e_pas = v_init = -70 mV"),
        "rheobase_nA": (0.02, 1.0, "reconstructed L5 cell with axonal initiation"),
        "ap_amplitude_mV": (50.0, 120.0, "somatic spike, axonal initiation site"),
        "ap_halfwidth_ms": (0.2, 2.0, "37 C with vshift_na = -5"),
        "rate_at_hi_Hz": (5.0, 80.0, "burst/regular firing at 0.75 nA"),
    },
    "bbp_l5pc": {
        "resting_potential_mV": (-90.0, -65.0, "e_pas = -90 mV with Ih; v_init -80"),
        "rheobase_nA": (0.05, 1.5, "Hay L5b PC perisomatic step threshold"),
        "ap_amplitude_mV": (50.0, 120.0, "NaTa_t/SKv3_1 spike at 34 C"),
        "ap_halfwidth_ms": (0.2, 2.0, "34 C BBP e-model"),
        "rate_at_hi_Hz": (3.0, 60.0, "Hay Fig. 5 steps saturate near 10-20 Hz"),
    },
    "migliore_ca1": {
        "resting_potential_mV": (-72.0, -58.0, "pinned to -65 mV by the published "
                                               "e_pas init procedure"),
        "rheobase_nA": (0.05, 0.60, "narrow firing window, 0.18-0.40 nA"),
        "ap_amplitude_mV": (50.0, 120.0, "na3/kdr spike at 34 C"),
        "ap_halfwidth_ms": (0.2, 2.0, "34 C CA1 spike"),
        "rate_at_hi_Hz": (5.0, 120.0, "0.30 nA inside the firing window"),
    },
    # --- replication cohort; intervals set from the calibration scan before
    # any information endpoint was computed for these models -------------- #
    "miyasho_purkinje": {
        "resting_potential_mV": (-80.0, -55.0, "el_Leak = -80 mV with the full "
                                               "dendritic channel set"),
        "rheobase_nA": (0.2, 3.0, "electrically large dendritic tree"),
        "ap_amplitude_mV": (50.0, 130.0, "NaF spike, ena = 45 mV"),
        "ap_halfwidth_ms": (0.1, 2.0, "37 C Purkinje spike"),
        "rate_at_hi_Hz": (30.0, 250.0, "Purkinje high-frequency firing"),
    },
    "tzilivaki_fsbc": {
        "resting_potential_mV": (-80.0, -60.0, "e_pas = v_init = -70 mV"),
        "rheobase_nA": (0.01, 0.60, "FS basket cell, low threshold"),
        "ap_amplitude_mV": (50.0, 140.0, "Nafx spike"),
        "ap_halfwidth_ms": (0.1, 1.0, "FS: narrow spike is the defining feature"),
        "rate_at_hi_Hz": (60.0, 400.0, "FS: >100 Hz at modest current"),
    },
    "moubarak_snc_da": {
        "resting_potential_mV": (-75.0, -45.0, "pacemaking: the 'rest' trace is an "
                                               "ongoing slow spike train"),
        "spontaneous_rate_Hz": (0.5, 25.0, "SNc DA pacemaking 1-10 Hz in vitro; "
                                           "widened for the reconstruction"),
        "ap_amplitude_mV": (40.0, 120.0, "broad DA action potential"),
        "ap_halfwidth_ms": (0.3, 4.0, "DA spikes are broad (>1 ms in vitro)"),
    },
    "dougalis_vlpag_da": {
        "resting_potential_mV": (-80.0, -50.0, "voltage-clamp-constrained midbrain "
                                               "DA neuron"),
        "rheobase_nA": (0.0005, 0.10, "15 um sphere: very small area, tiny currents"),
        "ap_amplitude_mV": (40.0, 120.0, "NaT spike, ena = 50 mV"),
        "ap_halfwidth_ms": (0.2, 4.0, "broad DA spike"),
        "rate_at_hi_Hz": (5.0, 120.0, "measured f-I peaks near 60 Hz"),
    },
    "ortner_snc_da": {
        "resting_potential_mV": (-80.0, -45.0, "SNc single-compartment model"),
        "rheobase_nA": (0.02, 0.50, "30 um sphere"),
        "ap_amplitude_mV": (40.0, 120.0, "HHHmod spike"),
        "ap_halfwidth_ms": (0.2, 5.0, "broad, slow SNc spike"),
        "rate_at_hi_Hz": (1.0, 30.0, "slow pacemaker: 1-8 Hz is the phenotype, and "
                                     "this is why its protocol coverage is partial"),
    },
    "tomko_ca1": {
        "resting_potential_mV": (-80.0, -60.0, "v_init = -65 mV"),
        "rheobase_nA": (0.1, 1.5, "reduced-morphology CA1, measured 0.56 nA"),
        "ap_amplitude_mV": (50.0, 130.0, "nax/kdr spike at 35 C"),
        "ap_halfwidth_ms": (0.2, 2.0, "35 C CA1 spike"),
        "rate_at_hi_Hz": (5.0, 100.0, "block above ~1.2 nA, so hi sits below it"),
    },
}


def bounds_for(model_id: str) -> dict[str, tuple[float, float, str]]:
    """Bounds for one model.  Raises rather than silently waiving the gate."""
    try:
        return CANONICAL_BOUNDS[model_id]
    except KeyError as exc:
        raise KeyError(
            f"{model_id!r} has no canonical behaviour bounds. Gate N1 fails "
            f"closed: a model with no frozen bounds cannot pass by default. "
            f"Add an entry to compass.validation.bounds.CANONICAL_BOUNDS with a "
            f"source justification for every interval."
        ) from exc
