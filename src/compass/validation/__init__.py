"""Release-validation gates (2026-08-06 audit, section 6.4).

The audit's central complaint about the previous test suite was not that it was
small but that it measured the wrong thing: real NEURON physiology assertions
existed for one adapter, and every other model was build-checked and
contract-checked only.  Passing the suite therefore could not be evidence that
the stored simulations were numerically and scientifically valid.

This package holds what that evidence requires:

``bounds``  model-specific, source-justified canonical behaviour intervals,
            frozen before the release campaign.  A Purkinje cell, a fast-spiking
            interneuron, a reconstructed pyramidal cell and an HH point model
            are *not* forced through one generic "reasonable neuron" interval.
``gates``   the N1-N7 execution/behaviour gates and the D1-D6 whole-campaign
            data-integrity gates, each returning a machine-readable record.

Every gate fails closed: a missing, skipped or indeterminate result is a
failure, never a pass.
"""

from .bounds import CANONICAL_BOUNDS, bounds_for
from .gates import GateResult, gate_record

__all__ = ["CANONICAL_BOUNDS", "GateResult", "bounds_for", "gate_record"]
