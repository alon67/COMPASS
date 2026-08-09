"""Gate N5: rheobase probes are tri-state and failures propagate.

Audit 5.4 and 6.4.1 gate N5:

    ``find_rheobase()`` treats an empty/failed trace as "does not spike" and
    detects a spike by any sample above 0 mV during the stimulus. A solver
    failure at high current can therefore become ``None``/non-spiking rather
    than an error, and an already depolarised trace can count without an upward
    crossing. Each probe should return ``{spiking, nonspiking, failed}``; any
    failed bracket or bisection probe should fail the parameter sample, and
    spike detection should use validated threshold crossings with baseline
    checks.

These run on a synthetic adapter, so they are fast, deterministic and belong on
every pull request.  The point is the *semantics*, and semantics do not need
NEURON to test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.base import NeuronAdapter, Recording  # noqa: E402
from compass.registry.schema import ModelManifest  # noqa: E402


class FakeAdapter(NeuronAdapter):
    """Scriptable stand-in: a caller decides what each probe amplitude returns."""

    manifest = ModelManifest(
        model_id="fake_probe_model", neuron_class="test", species="none",
        brain_region="none", simulator="neuron", source_reference="unit test",
        license="none", temperature_c=20.0, is_point_model=True,
        n_compartments=1, mechanism_dir=None, lab="test", pipeline="independent",
        source_lineage_id="test_lineage", implementation_id="test_impl",
        neuron_class_id="test_class",
    )
    manifest_rheobase_hi = 1.0
    manifest_rheobase_tol = 0.01

    def __init__(self, behaviour):
        super().__init__()
        self.behaviour = behaviour
        self.calls: list[float] = []

    def _build(self):
        pass

    def parameter_specs(self):
        return []

    def _targets(self, spec):
        return []

    @property
    def soma(self):
        return None

    def simulate(self, amplitude_nA, delay_ms=200.0, duration_ms=1000.0,
                 total_ms=1400.0, dt_ms=0.025, protocol_id="step", **kw):
        self.calls.append(amplitude_nA)
        t = np.arange(0.0, total_ms, dt_ms)
        kind = self.behaviour(amplitude_nA)
        if kind == "empty":
            v = np.array([])
            t = np.array([])
        elif kind == "nonfinite":
            v = np.full_like(t, -65.0)
            v[len(v) // 2] = np.nan
        elif kind == "spike":
            v = np.full_like(t, -65.0)
            step = (t >= delay_ms) & (t <= delay_ms + duration_ms)
            # three clean upward crossings of 0 mV
            for frac in (0.2, 0.5, 0.8):
                i = int(np.argmax(step) + frac * step.sum())
                v[i:i + 8] = 30.0
        elif kind == "silent":
            v = np.full_like(t, -65.0)
        elif kind == "depolarised_plateau":
            # sits above 0 mV for the whole step with no crossing at all
            v = np.full_like(t, -65.0)
            v[(t >= delay_ms) & (t <= delay_ms + duration_ms)] = 12.0
        else:
            raise AssertionError(kind)
        return Recording(t=t, v=v, stim_start_ms=delay_ms,
                         stim_end_ms=delay_ms + duration_ms,
                         amplitude_nA=amplitude_nA, protocol_id=protocol_id)


def probe(kinds):
    return FakeAdapter(kinds)


def test_probe_reports_three_distinct_states():
    a = probe(lambda amp: {0.0: "silent", 0.5: "spike", 1.0: "empty"}[amp])
    assert a.probe_spiking(0.0, 10, 100, 200, 0.025) == a.NONSPIKING
    assert a.probe_spiking(0.5, 10, 100, 200, 0.025) == a.SPIKING
    assert a.probe_spiking(1.0, 10, 100, 200, 0.025) == a.FAILED


def test_nonfinite_trace_is_failed_not_silent():
    a = probe(lambda amp: "nonfinite")
    assert a.probe_spiking(0.5, 10, 100, 200, 0.025) == a.FAILED


def test_depolarised_plateau_is_not_counted_as_spiking():
    """The old rule 'any sample above 0 mV' called this a spike."""
    a = probe(lambda amp: "depolarised_plateau")
    assert a.probe_spiking(0.5, 10, 100, 200, 0.025) == a.NONSPIKING


def test_failed_probe_at_the_bracket_raises_instead_of_returning_none():
    """The exact defect: a blow-up at hi became 'this cell does not spike'."""
    a = probe(lambda amp: "empty" if amp >= 1.0 else "silent")
    with pytest.raises(NeuronAdapter.RheobaseProbeFailure):
        a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)


def test_failed_probe_during_bisection_also_raises():
    def behaviour(amp):
        if amp >= 1.0:
            return "spike"
        if amp == 0.0:
            return "silent"
        return "empty"          # every bisection midpoint fails

    a = probe(behaviour)
    with pytest.raises(NeuronAdapter.RheobaseProbeFailure):
        a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)


def test_genuinely_silent_cell_still_returns_none():
    """A real non-spiking sample must NOT be turned into an error."""
    a = probe(lambda amp: "silent")
    assert a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200) is None


def test_bracket_is_demonstrated_by_a_silent_point_below_and_a_spike_above():
    threshold = 0.37

    def behaviour(amp):
        return "spike" if amp >= threshold else "silent"

    a = probe(behaviour)
    rheo = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)
    assert rheo is not None
    assert abs(rheo - threshold) <= a.manifest_rheobase_tol
    below = [c for c in a.calls if c < rheo - 1e-12]
    above = [c for c in a.calls if c >= rheo]
    assert below and above, "the search must probe on both sides of its answer"


def test_spontaneously_spiking_cell_reports_zero_rheobase():
    a = probe(lambda amp: "spike")
    assert a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200) == 0.0


def test_probe_failure_fails_the_parent_sample_in_the_campaign_worker():
    """The whole point of the tri-state: the row must be an error row."""
    from compass.workflows.simulate import classify_row

    row = classify_row({}, [], None, spontaneous=False,
                       error="rheobase_probe_failure: probe at 1 nA gave no trace")
    assert row["execution_valid"] is False
    assert row["analysis_eligible"] is False
    assert row["physiology_class"] == "simulation_error"
    assert "rheobase_probe_failure" in row["failure_reason"]


def test_a_genuinely_silent_sample_is_eligible_and_classified_nonspiking():
    """A broad prior is *supposed* to produce silent cells; they are data."""
    from compass.workflows.simulate import classify_row

    row = classify_row({"step_abs_hi.spike_count": 0.0}, [], None,
                       spontaneous=False, error=None)
    assert row["execution_valid"] is True
    assert row["physiology_class"] == "nonspiking"
    assert row["analysis_eligible"] is True


# --------------------------------------------------------------------------- #
# 2026-08-09 audit, 4.1: firing is not monotonic in current
# --------------------------------------------------------------------------- #

def test_depolarisation_block_at_the_ceiling_no_longer_reads_as_silent():
    """The exact defect: fires from 0.15 nA, blocked from 0.60 nA, ceiling 1.0.

    The old search probed 1.0 nA, saw no spike and returned ``None``, so the
    sample was stored as non-spiking with a NaN rheobase and no normalised
    protocols at all.
    """
    def behaviour(amp):
        return "spike" if 0.15 <= amp < 0.60 else "silent"

    a = probe(behaviour)
    rheo = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)
    assert rheo is not None, "a cell blocked at the ceiling is not a silent cell"
    assert abs(rheo - 0.15) <= a.manifest_rheobase_tol
    assert a.last_rheobase_search["blocked_at_ceiling"] is True
    assert a.last_rheobase_search["scan_hit_nA"] is not None


def test_the_ladder_brackets_a_narrow_low_current_firing_window():
    """khaliq_purkinje fires at 0.03 nA and is blocked long before its 1.0 ceiling.

    A linear ladder would step straight over a window this narrow; the geometric
    one reaches hi/256 = 0.0039 nA.
    """
    def behaviour(amp):
        return "spike" if 0.020 <= amp < 0.045 else "silent"

    a = probe(behaviour)
    rheo = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200, tol_nA=0.001)
    assert rheo is not None
    assert abs(rheo - 0.020) <= 0.001


def test_a_truly_silent_cell_still_returns_none_after_the_ladder():
    a = probe(lambda amp: "silent")
    assert a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200) is None
    assert a.last_rheobase_search["blocked_at_ceiling"] is False
    assert a.last_rheobase_search["scan_rungs"] == a.RHEOBASE_SCAN_RUNGS


def test_the_ladder_costs_nothing_when_the_ceiling_spikes():
    """The majority path must be bit-identical to probe-2, not merely similar."""
    a = probe(lambda amp: "spike" if amp >= 0.37 else "silent")
    a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)
    assert a.last_rheobase_search["scan_rungs"] == 0
    assert a.last_rheobase_search["blocked_at_ceiling"] is False


def test_a_failed_probe_inside_the_ladder_still_raises():
    """A blow-up must not be absorbed by the new search path either."""
    def behaviour(amp):
        if amp >= 1.0:
            return "silent"      # forces the ladder
        return "empty"

    a = probe(behaviour)
    with pytest.raises(NeuronAdapter.RheobaseProbeFailure):
        a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)


def test_a_plateau_at_the_ceiling_is_bracketed_not_reported_as_silent():
    """Depolarisation block expressed as a plateau above 0 mV, the realistic form."""
    def behaviour(amp):
        if amp >= 0.5:
            return "depolarised_plateau"
        return "spike" if amp >= 0.1 else "silent"

    a = probe(behaviour)
    rheo = a.find_rheobase(delay_ms=10, duration_ms=100, total_ms=200)
    assert rheo is not None
    assert abs(rheo - 0.1) <= a.manifest_rheobase_tol


# --------------------------------------------------------------------------- #
# draw-specific spontaneous activity
# --------------------------------------------------------------------------- #

def test_zero_rheobase_makes_the_normalised_family_undefined_for_every_model():
    """Not only for models listed in SPONTANEOUSLY_ACTIVE (audit 4.1)."""
    from compass.workflows.simulate import normalised_family_defined

    assert normalised_family_defined(0.25) is True
    assert normalised_family_defined(0.0) is False
    assert normalised_family_defined(None) is False


def test_a_draw_that_becomes_spontaneous_gets_nan_normalised_features():
    from compass.registry.schema import ProtocolSpec
    from compass.workflows.simulate import _feature_row

    a = probe(lambda amp: "spike")
    specs = [ProtocolSpec("step_rheo_1p5x", "step", amplitude_rheobase_mult=1.5,
                          delay_ms=10, duration_ms=100, total_ms=200)]
    row, _ = _feature_row(a, {"dt_ms": 0.025}, 0.0, specs)
    assert row, "the columns must exist"
    assert all(np.isnan(v) for v in row.values())
    assert not a.calls, "no protocol may be run at 0 nA and labelled 1.5x rheobase"


def test_classify_row_reports_normalised_validity_separately():
    """A silent draw stays in the primary (absolute) cohort but not the normalised one."""
    from compass.workflows.simulate import classify_row

    silent = classify_row({"step_abs_hi.spike_count": 0.0}, [], None,
                          spontaneous=False, error=None)
    assert silent["analysis_eligible"] is True
    assert silent["normalised_protocol_valid"] is False

    spont = classify_row({"step_abs_hi.spike_count": 12.0}, [], 0.0,
                         spontaneous=False, error=None)
    assert spont["analysis_eligible"] is True
    assert spont["normalised_protocol_valid"] is False

    ok = classify_row({"step_abs_hi.spike_count": 12.0}, [], 0.25,
                      spontaneous=False, error=None)
    assert ok["analysis_eligible"] is True
    assert ok["normalised_protocol_valid"] is True
