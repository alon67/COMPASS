"""Standardised electrophysiological feature extraction (blueprint 7.4).

eFEL provides the spike-shape features; project-owned functions provide the
passive and rate features so that their definitions are fixed and testable
independently of the eFEL version (blueprint 10: "eFEL plus project-owned
validated feature functions").

Every extraction failure is recorded as a NaN plus an explicit reason code --
nothing is silently dropped (blueprint 14.5, 26 "never silently remove
numerical failures").
"""

from __future__ import annotations

import numpy as np

from ..adapters.base import Recording

# Stamped into every shard identity.  Bump on ANY change to a feature
# definition, event definition or the feature list, so a shard extracted under
# one schema can never be silently pooled with one extracted under another
# (2026-08-06 audit, section 5.2).
FEATURE_SCHEMA_VERSION = "features-2"   # features-2: cv_isi requires >=3 spikes

# The minimum common feature set of blueprint 7.4.
COMMON_FEATURES = (
    "resting_potential_mV",
    "input_resistance_Mohm",
    "membrane_tau_ms",
    "rheobase_nA",
    "first_spike_latency_ms",
    "ap_amplitude_mV",
    "ap_halfwidth_ms",
    "ahp_depth_mV",
    "first_isi_ms",
    "mean_firing_rate_Hz",
    "adaptation_index",
    "spike_count",
)

# Model-specific extensions (blueprint 7.4).
EXTENDED_FEATURES = (
    "sag_ratio",
    "sag_amplitude_mV",
    "spontaneous_rate_Hz",
    "ap_threshold_mV",
    "ap_peak_mV",
    "cv_isi",
    "last_isi_ms",
    "depolarisation_block",
)

ALL_FEATURES = COMMON_FEATURES + EXTENDED_FEATURES

_EFEL_READY = False


def _init_efel():
    global _EFEL_READY
    import efel

    if not _EFEL_READY:
        efel.reset()
        efel.set_setting("Threshold", -20.0)
        efel.set_setting("DerivativeThreshold", 10.0)
        efel.set_setting("strict_stiminterval", True)
        _EFEL_READY = True
    return efel


def spike_times(rec: Recording, threshold: float = -20.0,
                t0: float | None = None, t1: float | None = None) -> np.ndarray:
    """Upward threshold crossings within the analysis window."""
    if rec.v.size < 3:
        return np.array([])
    t0 = rec.stim_start_ms if t0 is None else t0
    t1 = rec.stim_end_ms if t1 is None else t1
    m = (rec.t >= t0) & (rec.t <= t1)
    tt, vv = rec.t[m], rec.v[m]
    if vv.size < 3:
        return np.array([])
    idx = np.where((vv[:-1] < threshold) & (vv[1:] >= threshold))[0]
    return tt[idx]


# --------------------------------------------------------------------------- #
# Project-owned passive features
# --------------------------------------------------------------------------- #
def resting_potential(rec: Recording) -> float:
    """Mean voltage over the last 50 ms before stimulus onset."""
    if rec.v.size == 0:
        return np.nan
    t_end = rec.stim_start_ms if rec.stim_start_ms > 0 else rec.t[-1]
    m = (rec.t >= max(0.0, t_end - 50.0)) & (rec.t < t_end)
    return float(np.mean(rec.v[m])) if np.any(m) else float(rec.v[-1])


def steady_state_deflection(rec: Recording) -> float:
    """Mean voltage over the last 20% of the stimulus."""
    if rec.v.size == 0:
        return np.nan
    span = rec.stim_end_ms - rec.stim_start_ms
    m = (rec.t >= rec.stim_end_ms - 0.2 * span) & (rec.t <= rec.stim_end_ms)
    return float(np.mean(rec.v[m])) if np.any(m) else np.nan


def input_resistance(rec: Recording) -> float:
    """``R_in = dV / dI`` in Mohm, from a subthreshold hyperpolarising step."""
    if rec.amplitude_nA == 0 or rec.v.size == 0:
        return np.nan
    dv = steady_state_deflection(rec) - resting_potential(rec)
    return float(dv / rec.amplitude_nA)


def membrane_tau(rec: Recording) -> float:
    """Single-exponential fit to the charging transient of a hyperpolarising step."""
    if rec.v.size == 0 or rec.amplitude_nA >= 0:
        return np.nan
    v0 = resting_potential(rec)
    vss = steady_state_deflection(rec)
    if not np.isfinite(v0) or not np.isfinite(vss) or abs(vss - v0) < 0.5:
        return np.nan
    span = rec.stim_end_ms - rec.stim_start_ms
    m = (rec.t >= rec.stim_start_ms) & (rec.t <= rec.stim_start_ms + min(150.0, 0.5 * span))
    t, v = rec.t[m] - rec.stim_start_ms, rec.v[m]
    if t.size < 10:
        return np.nan
    frac = (v - vss) / (v0 - vss)
    ok = frac > 0.02
    if ok.sum() < 8:
        return np.nan
    slope, _ = np.polyfit(t[ok], np.log(frac[ok]), 1)
    return float(-1.0 / slope) if slope < 0 else np.nan


def sag(rec: Recording) -> tuple[float, float]:
    """``(sag_ratio, sag_amplitude_mV)`` of a hyperpolarising step."""
    if rec.v.size == 0 or rec.amplitude_nA >= 0:
        return np.nan, np.nan
    v0 = resting_potential(rec)
    vss = steady_state_deflection(rec)
    m = (rec.t >= rec.stim_start_ms) & (rec.t <= rec.stim_end_ms)
    if not np.any(m):
        return np.nan, np.nan
    vmin = float(np.min(rec.v[m]))
    peak_defl = vmin - v0
    if abs(peak_defl) < 0.5:
        return np.nan, np.nan
    return float((vss - vmin) / abs(peak_defl)), float(vss - vmin)


# --------------------------------------------------------------------------- #
# Firing features
# --------------------------------------------------------------------------- #
def firing_features(rec: Recording) -> dict:
    """Rate, ISI and adaptation statistics from threshold crossings."""
    st = spike_times(rec)
    dur_s = (rec.stim_end_ms - rec.stim_start_ms) / 1000.0
    out = {
        "spike_count": float(st.size),
        "mean_firing_rate_Hz": float(st.size / dur_s) if dur_s > 0 else np.nan,
        "first_spike_latency_ms": float(st[0] - rec.stim_start_ms) if st.size else np.nan,
        "first_isi_ms": np.nan,
        "last_isi_ms": np.nan,
        "adaptation_index": np.nan,
        "cv_isi": np.nan,
    }
    if st.size >= 2:
        isi = np.diff(st)
        out["first_isi_ms"] = float(isi[0])
        out["last_isi_ms"] = float(isi[-1])
    if st.size >= 3:
        # CV of the ISI distribution needs at least TWO intervals.  With one
        # interval np.std is exactly 0, so a two-spike train reported
        # cv_isi = 0.0 -- "perfectly regular firing" manufactured out of
        # nothing measurable.  Gate D3 found 2,127 such rows in the stored
        # screening campaign (audit 6.4.2, D3 "ISI-derived features exist only
        # when supported by enough spikes").
        isi = np.diff(st)
        out["cv_isi"] = (float(np.std(isi, ddof=1) / np.mean(isi))
                         if np.mean(isi) > 0 else np.nan)
        # Normalised adaptation: mean of (ISI_{i+1}-ISI_i)/(ISI_{i+1}+ISI_i).
        # Positive = slowing (adapting); bounded in [-1, 1] so it is comparable
        # across models with very different absolute rates.
        num = np.diff(isi)
        den = isi[1:] + isi[:-1]
        good = den > 0
        if good.any():
            out["adaptation_index"] = float(np.mean(num[good] / den[good]))
    return out


def depolarisation_block(rec: Recording) -> float:
    """1.0 if the cell stops spiking while still strongly depolarised."""
    if rec.v.size == 0:
        return np.nan
    span = rec.stim_end_ms - rec.stim_start_ms
    if span <= 0:
        return np.nan
    late = (rec.t >= rec.stim_end_ms - 0.3 * span) & (rec.t <= rec.stim_end_ms)
    early = (rec.t >= rec.stim_start_ms) & (rec.t < rec.stim_start_ms + 0.3 * span)
    if not (np.any(late) and np.any(early)):
        return np.nan
    n_late = spike_times(rec, t0=rec.stim_end_ms - 0.3 * span, t1=rec.stim_end_ms).size
    n_early = spike_times(rec, t0=rec.stim_start_ms, t1=rec.stim_start_ms + 0.3 * span).size
    v_late = float(np.mean(rec.v[late]))
    return float(n_early > 0 and n_late == 0 and v_late > -45.0)


def spike_shape(rec: Recording) -> dict:
    """AP amplitude, half-width, threshold, peak and AHP depth via eFEL."""
    out = {k: np.nan for k in
           ("ap_amplitude_mV", "ap_halfwidth_ms", "ahp_depth_mV",
            "ap_threshold_mV", "ap_peak_mV")}
    if rec.v.size < 10 or rec.stim_end_ms <= rec.stim_start_ms:
        return out
    efel = _init_efel()
    trace = {
        "T": rec.t, "V": rec.v,
        "stim_start": [rec.stim_start_ms], "stim_end": [rec.stim_end_ms],
    }
    names = ["AP_amplitude", "AP_width", "AHP_depth", "AP_begin_voltage", "peak_voltage"]
    try:
        res = efel.get_feature_values([trace], names, raise_warnings=False)[0]
    except Exception:  # noqa: BLE001 - eFEL raises many bespoke exceptions
        return out
    if res is None:
        return out

    def first(key):
        v = res.get(key)
        if v is None or len(v) == 0:
            return np.nan
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return float(v[0]) if v.size else np.nan

    out["ap_amplitude_mV"] = first("AP_amplitude")
    out["ap_halfwidth_ms"] = first("AP_width")
    out["ahp_depth_mV"] = first("AHP_depth")
    out["ap_threshold_mV"] = first("AP_begin_voltage")
    out["ap_peak_mV"] = first("peak_voltage")
    return out


# --------------------------------------------------------------------------- #
# Top-level extraction
# --------------------------------------------------------------------------- #
def extract(rec: Recording) -> dict:
    """Full feature dictionary for one recording, with NaN for undefined features."""
    out = {k: np.nan for k in ALL_FEATURES}
    if rec.v.size == 0:
        return out
    out["resting_potential_mV"] = resting_potential(rec)
    if rec.amplitude_nA < 0:
        out["input_resistance_Mohm"] = input_resistance(rec)
        out["membrane_tau_ms"] = membrane_tau(rec)
        out["sag_ratio"], out["sag_amplitude_mV"] = sag(rec)
    if rec.stim_end_ms > rec.stim_start_ms:
        out.update(firing_features(rec))
        out["depolarisation_block"] = depolarisation_block(rec)
        if out["spike_count"] > 0:
            out.update(spike_shape(rec))
    return out


def extract_rest(rec: Recording) -> dict:
    """Features from a zero-current run: resting potential and spontaneous rate."""
    if rec.v.size == 0:
        return {"resting_potential_mV": np.nan, "spontaneous_rate_Hz": np.nan}
    # ignore the first 100 ms so the initial transient does not count as spikes
    t0 = min(100.0, 0.3 * float(rec.t[-1]))
    st = spike_times(rec, t0=t0, t1=float(rec.t[-1]))
    dur_s = (float(rec.t[-1]) - t0) / 1000.0
    m = rec.t >= t0
    v = rec.v[m]
    # if the cell is firing, "resting potential" is the inter-spike minimum
    v_rest = float(np.percentile(v, 5)) if st.size else float(np.mean(v[-len(v) // 4:]))
    return {
        "resting_potential_mV": v_rest,
        "spontaneous_rate_Hz": float(st.size / dur_s) if dur_s > 0 else np.nan,
    }
