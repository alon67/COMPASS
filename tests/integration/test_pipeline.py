"""Integration tests: adapter -> protocol -> features -> conditioning -> analysis.

Blueprint 21.2. These exercise the real NEURON adapters, so they are slower than
the unit tests; the heavy multicompartment models are marked ``slow``.

NEURON's section list is process-global, so tests that build a model are
parameterised over a *single* model per process wherever possible and rely on
the adapter's one-model-per-process guard being explicit about violations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.adapters.models import MODEL_REGISTRY, get_adapter
from compass.conditioning.ensembles import CONDITIONING_FEATURES, build_ensembles
from compass.features.extract import extract, extract_rest
from compass.protocols import common as P
from compass.sampling.priors import independent_sample
from compass.workflows.simulate import run_block, solver_kwargs

CHEAP = "pospischil_rs"


@pytest.fixture(scope="module")
def adapter():
    # The Pospischil mechanisms are third-party sources this repository does not
    # redistribute (docs/ASSET_REDISTRIBUTION.md); a fresh checkout, including CI,
    # has none until tools/fetch_models.py and tools/build_mechanisms.py are run.
    try:
        return get_adapter(CHEAP).build()
    except ValueError as exc:
        if "density mechanism" not in str(exc):
            raise
        pytest.skip(f"{CHEAP} mechanisms are not compiled here ({exc}); run "
                    "tools/fetch_models.py and tools/build_mechanisms.py first")


def test_every_registered_model_declares_a_valid_parameter_table():
    """Schema validation runs at import; this asserts panel-wide invariants."""
    for mid, cls in MODEL_REGISTRY.items():
        specs = cls().parameter_specs()
        assert len(specs) >= 6, f"{mid} has too few free parameters"
        assert len({s.parameter_id for s in specs}) == len(specs), f"{mid} duplicate ids"
        assert all(s.model_id == mid for s in specs)
        assert all(s.lower_bound <= 1.0 <= s.upper_bound for s in specs)
        assert mid in P.ABSOLUTE_AMPLITUDES
        assert mid in P.CANONICAL_RHEOBASE


def test_common_core_protocols_are_wellformed():
    for mid in MODEL_REGISTRY:
        protos = P.common_core(mid)
        ids = [p.protocol_id for p in protos]
        assert len(set(ids)) == len(ids)
        assert "rest" in ids
        fams = {P.protocol_family(p) for p in protos}
        assert {"absolute", "normalised", "rest"} <= fams


def test_adapter_builds_and_reproduces_canonical_behaviour(adapter):
    kw = solver_kwargs(CHEAP)
    rec = adapter.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                           total_ms=P.REST_TOTAL_MS, protocol_id="rest", **kw)
    f = extract_rest(rec)
    assert f["resting_potential_mV"] == pytest.approx(-70.0, abs=1.5)
    assert f["spontaneous_rate_Hz"] == 0.0

    hyp = adapter.simulate(P.ABSOLUTE_AMPLITUDES[CHEAP][0], delay_ms=P.DELAY_MS,
                           duration_ms=P.DURATION_MS, total_ms=P.TOTAL_MS,
                           protocol_id="hyperpol", **kw)
    # published source comment: g_pas = 0.0001 -> Rin = 34 MOhm
    assert extract(hyp)["input_resistance_Mohm"] == pytest.approx(34.0, rel=0.15)


def test_parameter_injection_and_reset_are_exact(adapter):
    kw = solver_kwargs(CHEAP)
    pids = [s.parameter_id for s in adapter.parameter_specs()]
    base = adapter.simulate(P.ABSOLUTE_AMPLITUDES[CHEAP][2], delay_ms=P.DELAY_MS,
                            duration_ms=P.DURATION_MS, total_ms=P.TOTAL_MS, **kw)
    adapter.apply({pids[0]: 3.0})
    changed = adapter.simulate(P.ABSOLUTE_AMPLITUDES[CHEAP][2], delay_ms=P.DELAY_MS,
                               duration_ms=P.DURATION_MS, total_ms=P.TOTAL_MS, **kw)
    assert not np.allclose(base.v, changed.v), "injection had no effect"
    adapter.reset()
    restored = adapter.simulate(P.ABSOLUTE_AMPLITUDES[CHEAP][2], delay_ms=P.DELAY_MS,
                                duration_ms=P.DURATION_MS, total_ms=P.TOTAL_MS, **kw)
    assert np.array_equal(base.v, restored.v), "reset did not restore exactly"


def test_scale_factor_semantics(adapter):
    """A scale of s must set the segment value to canonical * s, exactly."""
    spec = next(s for s in adapter.parameter_specs() if s.parameter_id == "gna_soma")
    attr = f"{spec.attribute}_{spec.mechanism_name}"
    canonical = adapter.canonical_absolute("gna_soma")
    adapter.apply({"gna_soma": 2.5})
    seg = next(iter(adapter.soma))
    assert getattr(seg, attr) == pytest.approx(canonical * 2.5, rel=1e-12)
    adapter.reset()
    assert getattr(seg, attr) == pytest.approx(canonical, rel=1e-12)


def test_rheobase_bisection_brackets_the_threshold(adapter):
    kw = solver_kwargs(CHEAP)
    adapter.reset()
    rb = adapter.find_rheobase(delay_ms=P.RHEO_DELAY_MS, duration_ms=P.RHEO_DURATION_MS,
                               total_ms=P.RHEO_TOTAL_MS, **kw)
    assert rb is not None

    def n_spikes(amp):
        r = adapter.simulate(amp, delay_ms=P.RHEO_DELAY_MS, duration_ms=P.RHEO_DURATION_MS,
                             total_ms=P.RHEO_TOTAL_MS, **kw)
        m = (r.t >= P.RHEO_DELAY_MS)
        return int(np.sum((r.v[m][:-1] < 0) & (r.v[m][1:] >= 0)))

    tol = adapter.manifest_rheobase_tol
    assert n_spikes(rb + tol) >= 1
    assert n_spikes(max(0.0, rb - 3 * tol)) == 0


def test_qc_flags_catch_a_broken_trace(adapter):
    from compass.adapters.base import Recording

    empty = Recording(t=np.array([]), v=np.array([]), stim_start_ms=0.0,
                      stim_end_ms=0.0, amplitude_nA=0.0, protocol_id="x")
    assert adapter.qc_flags(empty) == ["solver_failure"]
    t = np.arange(0, 550, 0.025)
    bad = Recording(t=t, v=np.full_like(t, 500.0), stim_start_ms=120.0,
                    stim_end_ms=520.0, amplitude_nA=0.5, protocol_id="x")
    assert "nonphysical_voltage" in adapter.qc_flags(bad)


def test_full_block_produces_a_joinable_result_table(adapter, tmp_path):
    """One full smoke-test experiment per blueprint 21.2, end to end."""
    specs = adapter.parameter_specs()
    pids = [s.parameter_id for s in specs]
    scales = independent_sample(specs, 8, seed=1, method="sobol")
    out = tmp_path / "shard.parquet"
    stats = run_block(CHEAP, scales, pids, np.arange(8), "E1", 1, 1,
                      "independent_sobol", out)
    assert out.exists() and stats["n"] == 8
    df = pd.read_parquet(out)
    assert len(df) == 8
    assert (df["simulation_status"] == "ok").all()
    # parameter columns join back to the sample table exactly
    for j, pid in enumerate(pids):
        assert np.allclose(df[f"param.{pid}"].to_numpy(), scales[:, j])
    # every declared common feature has a column for every protocol
    for proto in P.common_core(CHEAP):
        if proto.kind == "rest":
            continue
        assert f"{proto.protocol_id}.mean_firing_rate_Hz" in df.columns
    assert {"run_id", "model_id", "sampling_regime", "prior_id", "replicate",
            "seed", "parameter_sample_id", "simulation_status", "qc_flags",
            "rheobase_nA"} <= set(df.columns)


def test_interrupted_block_is_not_recomputed(adapter, tmp_path):
    """Resume semantics: a shard with a MATCHING IDENTITY is skipped.

    Reuse used to be decided by "the path exists" (2026-08-06 audit, 5.2), so
    this test passed while a shard written by different code, a different
    solver contract or a different design was silently reused. The worker now
    takes an identity and reuses only on an exact match; the companion test
    below checks that a changed identity is quarantined instead.
    """
    from compass.workflows.simulate import _worker

    specs = adapter.parameter_specs()
    pids = [s.parameter_id for s in specs]
    scales = independent_sample(specs, 4, seed=2, method="sobol")
    out = tmp_path / "resume.parquet"
    identity = {"experiment": "test", "model_id": CHEAP, "regime": "E1",
                "seed": 2, "code_hash": "abc123"}
    args = (CHEAP, scales, pids, np.arange(4), "E1", 1, 2, "independent_sobol",
            str(out), identity)
    first = _worker(args)
    assert first["n"] == 4 and not first.get("skipped")
    second = _worker(args)
    assert second.get("skipped") is True and second["n"] == 0



def test_shard_with_a_changed_identity_is_quarantined_not_reused(adapter, tmp_path):
    """The defect the identity check exists for: silent reuse across a change."""
    from compass.workflows.simulate import _worker, shard_is_reusable, sidecar_path

    specs = adapter.parameter_specs()
    pids = [s.parameter_id for s in specs]
    scales = independent_sample(specs, 4, seed=2, method="sobol")
    out = tmp_path / "identity.parquet"
    ident = {"experiment": "test", "model_id": CHEAP, "regime": "E1",
             "seed": 2, "code_hash": "before"}
    _worker((CHEAP, scales, pids, np.arange(4), "E1", 1, 2, "independent_sobol",
             str(out), ident))
    assert out.exists() and sidecar_path(out).exists()
    assert shard_is_reusable(out, ident, 4)[0]

    changed = dict(ident, code_hash="after")
    reusable, why = shard_is_reusable(out, changed, 4)
    assert not reusable and why.startswith("identity_mismatch"), why

    st = _worker((CHEAP, scales, pids, np.arange(4), "E1", 1, 2,
                  "independent_sobol", str(out), changed))
    assert not st.get("skipped"), "a changed identity must force recomputation"
    quarantined = list((tmp_path / "_quarantine").glob("identity_*.parquet"))
    assert quarantined, "the superseded shard must be preserved, not overwritten"
    assert shard_is_reusable(out, changed, 4)[0]


def test_a_shard_without_a_sidecar_is_not_reusable(adapter, tmp_path):
    """Provenance-unknown is a recompute, not a pass."""
    import pandas as pd

    from compass.workflows.simulate import shard_is_reusable

    out = tmp_path / "orphan.parquet"
    pd.DataFrame({"x": [1, 2, 3, 4]}).to_parquet(out, index=False)
    reusable, why = shard_is_reusable(out, {"model_id": CHEAP}, 4)
    assert not reusable and why == "identity_unknown"

def test_ensembles_build_from_a_feature_table(adapter, tmp_path):
    """Conditioning + ensemble construction on a real (small) feature table."""
    specs = adapter.parameter_specs()
    pids = [s.parameter_id for s in specs]
    scales = independent_sample(specs, 260, seed=5, method="sobol")
    out = tmp_path / "ens.parquet"
    run_block(CHEAP, scales, pids, np.arange(260), "E1", 1, 5, "independent_sobol", out)
    df = pd.read_parquet(out)
    feats = [f for f in CONDITIONING_FEATURES if f in df.columns]
    ens = build_ensembles(df, CHEAP, pids, feats, seed=0)
    labels = {e.label for e in ens}
    assert any(lb.startswith("E1_") for lb in labels)
    assert any("E3" in lb for lb in labels), "no conditioned ensemble was produced"
    for e in ens:
        assert e.params.shape[1] == len(pids)
        assert np.isfinite(e.params).all()
        if e.conditioning is not None:
            assert 0.0 <= e.conditioning.acceptance_rate <= 1.0
            assert e.n == int(e.conditioning.accepted.sum())


@pytest.mark.slow
def test_information_pipeline_on_a_real_ensemble(adapter, tmp_path):
    """Conditioning a real neuronal prior must create measurable dependence."""
    from compass.analysis.geometry import describe_geometry
    from compass.analysis.tuples import order_curve

    specs = adapter.parameter_specs()
    pids = [s.parameter_id for s in specs]
    scales = independent_sample(specs, 512, seed=9, method="sobol")
    out = tmp_path / "info.parquet"
    run_block(CHEAP, scales, pids, np.arange(512), "E1", 1, 9, "independent_sobol", out)
    df = pd.read_parquet(out)
    feats = [f for f in CONDITIONING_FEATURES if f in df.columns]
    ens = {e.regime: e for e in build_ensembles(df, CHEAP, pids, feats, seed=0)}
    if "E1" not in ens or "E3" not in ens:
        pytest.skip("insufficient accepted samples at this smoke-test size")
    g1 = describe_geometry(ens["E1"].params)
    g3 = describe_geometry(ens["E3"].params)
    # an independent quasi-random prior is essentially uncorrelated ...
    assert g1.mean_abs_pearson < 0.05
    # ... and conditioning on a shared phenotype introduces dependence
    assert g3.mean_abs_pearson > g1.mean_abs_pearson
    oc = order_curve(ens["E1"].params, orders=(3,), max_tuples=60, n_surrogate=8)
    assert abs(oc.median[0]) < 0.02
