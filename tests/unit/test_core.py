"""Unit tests for schemas, transforms, sampling, conditioning and features.

Covers the blueprint 21.1 list: parameter transforms and inverses, unit
conversions, sampling bounds and covariance reproduction, feature calculations
on synthetic traces, tuple generation determinism, and target-distance functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.adapters.base import Recording
from compass.analysis.tuples import generate_tuples, module_stability
from compass.conditioning.distances import (
    TOLERANCES,
    condition,
    mahalanobis,
    max_abs_z,
    robust_z,
    soft_weights,
    weighted_euclidean,
)
from compass.features import extract as F
from compass.registry.schema import (
    EvidenceTier,
    ExperimentSpec,
    ModelManifest,
    ParameterSpec,
    ProtocolSpec,
    Transform,
    ValidationError,
)
from compass.sampling.priors import (
    correlated_sample,
    gaussian_copula_control,
    independent_sample,
    scales_from_unit,
    shuffle_control,
    unit_from_scales,
)

SEED = 7


def _spec(**kw):
    base = dict(
        model_id="m1", parameter_id="gna", mechanism_name="hh", attribute="gnabar",
        compartment="soma", channel_family="Nav", biophysical_role="fast_sodium",
        units="S/cm2", canonical_value=0.12, lower_bound=0.25, upper_bound=4.0,
        evidence_tier=EvidenceTier.D, source_reference="ref",
    )
    base.update(kw)
    return ParameterSpec(**base)


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def test_parameter_spec_accepts_valid():
    s = _spec()
    assert s.log_width == pytest.approx(np.log10(4.0) - np.log10(0.25))
    assert s.to_row()["evidence_tier"] == "D"


@pytest.mark.parametrize("kw,msg", [
    (dict(parameter_id="Bad-Id"), "match"),
    (dict(units="furlongs"), "unknown unit"),
    (dict(lower_bound=5.0, upper_bound=1.0), "lower_bound"),
    (dict(source_reference=""), "source_reference"),
    (dict(lower_bound=2.0, upper_bound=4.0), "canonical scale"),
])
def test_parameter_spec_rejects_invalid(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        _spec(**kw)


def test_log10_transform_requires_positive_lower_bound():
    with pytest.raises(ValidationError, match="positive"):
        _spec(lower_bound=-1.0, transform=Transform.LOG10)


def test_evidence_tier_gates_biological_claims():
    assert EvidenceTier.A.permits_biological_claim
    assert EvidenceTier.B.permits_biological_claim
    # C and D must never be described as measured biological variability
    assert not EvidenceTier.C.permits_biological_claim
    assert not EvidenceTier.D.permits_biological_claim


def test_model_manifest_rejects_inconsistent_point_model():
    kw = dict(model_id="m", neuron_class="c", species="s", brain_region="r",
              simulator="neuron", source_reference="doi", license="MIT",
              temperature_c=34.0, is_point_model=True, n_compartments=5,
              mechanism_dir=None, lab="l", pipeline="independent")
    with pytest.raises(ValidationError, match="1 compartment"):
        ModelManifest(**kw)


def test_protocol_spec_rejects_stimulus_past_end():
    with pytest.raises(ValidationError, match="past total_ms"):
        ProtocolSpec("p", "step", amplitude_nA=1.0, delay_ms=500,
                     duration_ms=600, total_ms=700)


def test_protocol_family_flag():
    a = ProtocolSpec("abs", "step", amplitude_nA=0.5)
    n = ProtocolSpec("norm", "step", amplitude_rheobase_mult=1.5)
    assert not a.is_normalised
    assert n.is_normalised


def test_experiment_hash_is_stable_and_sensitive():
    a = ExperimentSpec("e1", ("m1",), ("E1",), 100, 1, 42)
    b = ExperimentSpec("e1", ("m1",), ("E1",), 100, 1, 42)
    c = ExperimentSpec("e1", ("m1",), ("E1",), 200, 1, 42)
    assert a.config_hash == b.config_hash
    assert a.config_hash != c.config_hash


# --------------------------------------------------------------------------- #
# Transforms and priors
# --------------------------------------------------------------------------- #
def test_transform_roundtrip():
    for t in (Transform.LINEAR, Transform.LOG10, Transform.LOGIT):
        x = np.array([0.2, 0.5, 0.8])
        assert np.allclose(t.from_sample_space(t.to_sample_space(x)), x)


def test_scale_unit_roundtrip():
    specs = [_spec(parameter_id=f"p{i}") for i in range(4)]
    u = np.random.default_rng(SEED).random((200, 4))
    x = scales_from_unit(u, specs)
    assert np.allclose(unit_from_scales(x, specs), u, atol=1e-10)


def test_independent_sample_respects_bounds():
    specs = [_spec(parameter_id=f"p{i}") for i in range(5)]
    x = independent_sample(specs, 512, seed=SEED)
    for j, s in enumerate(specs):
        assert x[:, j].min() >= s.lower_bound - 1e-9
        assert x[:, j].max() <= s.upper_bound + 1e-9


def test_independent_sample_is_deterministic():
    specs = [_spec(parameter_id=f"p{i}") for i in range(4)]
    a = independent_sample(specs, 256, seed=3)
    b = independent_sample(specs, 256, seed=3)
    c = independent_sample(specs, 256, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_prior_width_scales_log_range():
    specs = [_spec(parameter_id="p0")]
    wide = independent_sample(specs, 4096, seed=SEED, width=1.0)
    narrow = independent_sample(specs, 4096, seed=SEED, width=0.5)
    assert np.ptp(np.log10(narrow)) < np.ptp(np.log10(wide)) * 0.6


def test_correlated_sample_reproduces_requested_correlation():
    specs = [_spec(parameter_id=f"p{i}") for i in range(4)]
    x = np.log10(correlated_sample(specs, 40000, seed=SEED, rho=0.6))
    c = np.corrcoef(x, rowvar=False)
    # neighbouring parameters correlate; the copula slightly attenuates rho
    assert 0.45 < c[0, 1] < 0.65
    assert abs(c[0, 3]) < 0.35


def test_shuffle_control_preserves_marginals_and_destroys_dependence():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=8000)
    x = np.column_stack([z + 0.1 * rng.normal(size=8000) for _ in range(3)])
    s = shuffle_control(x, seed=SEED)
    for j in range(3):
        assert np.allclose(np.sort(x[:, j]), np.sort(s[:, j]))
    assert abs(np.corrcoef(s, rowvar=False)[0, 1]) < 0.06


def test_copula_control_preserves_rank_correlation():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=8000)
    x = np.column_stack([z + 0.5 * rng.normal(size=8000) for _ in range(3)])
    c = gaussian_copula_control(x, seed=SEED)
    r_in = np.corrcoef(x, rowvar=False)[0, 1]
    r_out = np.corrcoef(c, rowvar=False)[0, 1]
    assert abs(r_in - r_out) < 0.08


# --------------------------------------------------------------------------- #
# Target distances and conditioning
# --------------------------------------------------------------------------- #
def test_distance_metrics_are_zero_at_the_target():
    t = np.array([1.0, 2.0, 3.0])
    s = np.array([0.5, 0.5, 0.5])
    x = t[None, :]
    assert max_abs_z(x, t, s)[0] == pytest.approx(0.0)
    assert weighted_euclidean(x, t, s)[0] == pytest.approx(0.0)
    assert robust_z(x, t, s / 1.4826)[0] == pytest.approx(0.0)
    assert mahalanobis(x, t, np.eye(3))[0] == pytest.approx(0.0)


def test_max_z_is_the_strictest_of_the_z_metrics():
    t = np.zeros(4)
    s = np.ones(4)
    x = np.array([[3.0, 0.0, 0.0, 0.0]])
    assert max_abs_z(x, t, s)[0] > weighted_euclidean(x, t, s)[0]


def test_soft_weights_sum_to_one_and_decay():
    d = np.array([0.0, 1.0, 5.0])
    w = soft_weights(d, bandwidth=1.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]


def test_conditioning_rejects_rows_with_undefined_features():
    rng = np.random.default_rng(SEED)
    f = rng.normal(size=(500, 4))
    f[:50, 0] = np.nan                      # e.g. no spike -> no spike width
    res = condition(f, np.zeros(4), np.ones(4), tolerance_name="permissive")
    assert not res.accepted[:50].any()
    assert res.n_accepted == int(res.accepted.sum())
    assert 0.0 <= res.acceptance_rate <= 1.0


def test_tighter_tolerance_accepts_fewer():
    rng = np.random.default_rng(SEED)
    f = rng.normal(size=(4000, 5))
    counts = [condition(f, np.zeros(5), np.ones(5), tolerance_name=t).n_accepted
              for t in ("permissive", "moderate", "strict", "very_strict")]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_tolerance_series_is_monotone():
    vals = [TOLERANCES[k] for k in ("permissive", "moderate", "strict", "very_strict")]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------------------------------------------- #
# Tuple generation
# --------------------------------------------------------------------------- #
def test_generate_tuples_exhaustive_when_small():
    ts = generate_tuples(6, 3, max_tuples=1000, seed=1)
    assert ts.exhaustive and len(ts) == 20 and ts.n_possible == 20


def test_generate_tuples_is_deterministic_when_sampled():
    a = generate_tuples(30, 5, max_tuples=200, seed=11)
    b = generate_tuples(30, 5, max_tuples=200, seed=11)
    c = generate_tuples(30, 5, max_tuples=200, seed=12)
    assert not a.exhaustive
    assert a.tuples == b.tuples
    assert a.tuples != c.tuples
    assert len(set(a.tuples)) == len(a.tuples)


def test_module_stability_detects_identical_and_disjoint_rankings():
    r = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    same = module_stability([r, r], top_k=2)
    assert same["mean_jaccard"] == pytest.approx(1.0)
    disjoint = module_stability([r[:2], r[2:]], top_k=2)
    assert disjoint["mean_jaccard"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Feature extraction on synthetic traces
# --------------------------------------------------------------------------- #
def _synthetic_spike_train(rate_hz: float, dur_ms=400.0, delay_ms=120.0,
                           total_ms=550.0, dt=0.025, amp_nA=0.5, v_rest=-70.0):
    t = np.arange(0.0, total_ms, dt)
    v = np.full_like(t, v_rest)
    isi = 1000.0 / rate_hz
    times = np.arange(delay_ms + isi / 2, delay_ms + dur_ms, isi)
    for ts in times:
        m = np.abs(t - ts) < 1.0
        v[m] += 110.0 * np.exp(-((t[m] - ts) ** 2) / 0.12)
    return Recording(t=t, v=v, stim_start_ms=delay_ms, stim_end_ms=delay_ms + dur_ms,
                     amplitude_nA=amp_nA, protocol_id="synthetic"), len(times)


def test_firing_rate_matches_synthetic_train():
    rec, n = _synthetic_spike_train(50.0)
    f = F.firing_features(rec)
    assert f["spike_count"] == n
    assert f["mean_firing_rate_Hz"] == pytest.approx(n / 0.4, rel=1e-6)
    assert f["first_isi_ms"] == pytest.approx(20.0, abs=0.1)


def test_adaptation_index_is_zero_for_a_regular_train():
    rec, _ = _synthetic_spike_train(40.0)
    assert abs(F.firing_features(rec)["adaptation_index"]) < 1e-6


def test_adaptation_index_is_positive_when_isis_lengthen():
    t = np.arange(0.0, 550.0, 0.025)
    v = np.full_like(t, -70.0)
    times = 120.0 + np.cumsum([10, 12, 15, 19, 24, 30, 38])
    for ts in times:
        v[np.abs(t - ts) < 1.0] += 110.0
    rec = Recording(t=t, v=v, stim_start_ms=120.0, stim_end_ms=520.0,
                    amplitude_nA=0.5, protocol_id="adapt")
    assert F.firing_features(rec)["adaptation_index"] > 0.05


def test_input_resistance_and_tau_from_synthetic_rc_step():
    dt, delay, dur, total = 0.025, 120.0, 400.0, 550.0
    t = np.arange(0.0, total, dt)
    v0, r_in, tau, amp = -70.0, 100.0, 20.0, -0.1   # -0.1 nA * 100 MOhm = -10 mV
    v = np.full_like(t, v0)
    m = (t >= delay) & (t <= delay + dur)
    v[m] = v0 + amp * r_in * (1.0 - np.exp(-(t[m] - delay) / tau))
    tail = t > delay + dur
    v[tail] = v0 + (v[m][-1] - v0) * np.exp(-(t[tail] - delay - dur) / tau)
    rec = Recording(t=t, v=v, stim_start_ms=delay, stim_end_ms=delay + dur,
                    amplitude_nA=amp, protocol_id="hyperpol")
    assert F.resting_potential(rec) == pytest.approx(v0, abs=1e-6)
    assert F.input_resistance(rec) == pytest.approx(r_in, rel=0.02)
    assert F.membrane_tau(rec) == pytest.approx(tau, rel=0.05)


def test_sag_is_zero_without_sag_and_positive_with_it():
    dt, delay, dur, total = 0.025, 120.0, 400.0, 550.0
    t = np.arange(0.0, total, dt)
    v0 = -70.0
    v = np.full_like(t, v0)
    m = (t >= delay) & (t <= delay + dur)
    v[m] = v0 - 10.0 * (1.0 - np.exp(-(t[m] - delay) / 20.0))
    rec = Recording(t=t, v=v, stim_start_ms=delay, stim_end_ms=delay + dur,
                    amplitude_nA=-0.1, protocol_id="hyp")
    ratio, _ = F.sag(rec)
    assert abs(ratio) < 0.02

    v2 = v.copy()
    tt = t[m] - delay
    v2[m] = v0 - 14.0 * np.exp(-tt / 60.0) - 8.0 * (1 - np.exp(-tt / 20.0)) + 4.0
    rec2 = Recording(t=t, v=v2, stim_start_ms=delay, stim_end_ms=delay + dur,
                     amplitude_nA=-0.1, protocol_id="hyp")
    ratio2, amp2 = F.sag(rec2)
    assert ratio2 > 0.05 and amp2 > 0.5


def test_extract_returns_all_declared_features_and_nan_on_empty():
    rec, _ = _synthetic_spike_train(30.0)
    out = F.extract(rec)
    assert set(F.ALL_FEATURES).issubset(out)
    empty = Recording(t=np.array([]), v=np.array([]), stim_start_ms=0.0,
                      stim_end_ms=0.0, amplitude_nA=0.0, protocol_id="x")
    assert all(np.isnan(v) for v in F.extract(empty).values())


def test_depolarisation_block_detected():
    dt, delay, dur, total = 0.025, 120.0, 400.0, 550.0
    t = np.arange(0.0, total, dt)
    v = np.full_like(t, -70.0)
    m = (t >= delay) & (t <= delay + dur)
    v[m] = -30.0                                  # depolarised plateau
    for ts in (140.0, 155.0, 172.0):              # a few early spikes only
        v[np.abs(t - ts) < 1.0] = 40.0
    rec = Recording(t=t, v=v, stim_start_ms=delay, stim_end_ms=delay + dur,
                    amplitude_nA=1.0, protocol_id="block")
    assert F.depolarisation_block(rec) == 1.0
