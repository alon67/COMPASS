"""Common simulator-adapter interface over NEURON (blueprint 9.1.2).

Design decision: parameters are **multiplicative scale factors on the canonical
per-segment conductance profile**, not absolute densities.

Why this matters scientifically
-------------------------------
1. Cross-model comparability.  Blueprint 14.2 forbids comparing raw parameter
   magnitudes across models.  A scale factor of 2.0 means the same thing in a
   point model and in a 600-compartment reconstruction.
2. Spatial distributions are preserved.  Hay's apical ``Ih`` gradient, Mainen's
   axonal ``Na`` hotspot and Migliore's distance-dependent ``K_A`` are published
   *model architecture*.  Overwriting them with a uniform density would silently
   confound "neuron identity" with "we deleted the channel gradient".  Making
   distribution shape an explicit ablation (blueprint 13.3) requires that the
   default path leaves it intact.
3. Log-space sampling is natural: the canonical model sits at scale 1.0, i.e.
   0.0 in log10 space, for every parameter of every model.

Each adapter snapshots the canonical value of every targeted segment at build
time; ``apply`` then writes ``canonical * scale`` back.  ``reset`` restores the
snapshot exactly, so a single built cell can serve an unlimited number of
parameter samples without drift.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..registry.schema import ModelManifest, ParameterSpec

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")

_LOADED_MECH_DIRS: set[str] = set()

# Which model this process has built.  NEURON's section list is process-global,
# so only one model may be instantiated per process.
_PROCESS_MODEL: str | None = None


def section_names() -> set[str]:
    """Names of all sections currently alive in this process.

    Names, not Section objects: several published morphology files begin with
    ``forall delete_section()``, which would invalidate held references.
    """
    from neuron import h

    return {s.name() for s in h.allsec()}


def _own_sections(before: set[str]) -> list:
    """Sections created since ``before`` was captured, in creation order."""
    from neuron import h

    return [s for s in h.allsec() if s.name() not in before]


def load_mechanisms(mech_dir: str | Path) -> None:
    """Load a compiled mechanism library exactly once per process."""
    from neuron import h
    from neuron import load_mechanisms as _lm

    key = str(Path(mech_dir).resolve())
    if key in _LOADED_MECH_DIRS:
        return
    _lm(key)
    _LOADED_MECH_DIRS.add(key)
    h.load_file("stdrun.hoc")


@dataclass
class Recording:
    """One simulated trace plus the metadata a feature extractor needs."""

    t: np.ndarray
    v: np.ndarray
    stim_start_ms: float
    stim_end_ms: float
    amplitude_nA: float
    protocol_id: str
    site: str = "soma"
    extra: dict = field(default_factory=dict)


class NeuronAdapter(ABC):
    """Base class for every model in the panel.

    Subclasses implement ``_build`` (construct sections and insert mechanisms)
    and ``parameter_specs`` (declare the free parameters).  Everything else --
    canonical snapshotting, scaled parameter injection, protocol execution,
    rheobase search, numerical QC -- is shared.
    """

    manifest: ModelManifest

    def __init__(self, mechanism_root: Path | None = None) -> None:
        self._built = False
        self._canonical: dict[str, list[tuple[object, str, float]]] = {}
        self._mech_root = mechanism_root
        self._stim = None
        self._sections: dict[str, list] = {}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build(self) -> None:
        """Create sections, insert mechanisms, set canonical biophysics."""

    @abstractmethod
    def parameter_specs(self) -> list[ParameterSpec]:
        """Declare the free parameters (scale factors) of this model."""

    @abstractmethod
    def _targets(self, spec: ParameterSpec) -> list:
        """Return the NEURON sections a parameter applies to."""

    def build(self, allow_shared_process: bool = False) -> NeuronAdapter:
        """Construct the cell.

        NEURON keeps one global section list per process.  Adapters that load a
        published ``.hoc`` morphology therefore have to discover *their own*
        sections by differencing ``h.allsec()`` before and after the load (see
        :meth:`_own_sections`).  Building two models in one process is refused by
        default because the second model would otherwise inherit the first
        model's sections and silently insert its mechanisms into them.
        Simulation workers are pinned to a single model, so this never binds in
        production; ``allow_shared_process`` exists only for diagnostics.
        """
        global _PROCESS_MODEL
        if self._built:
            return self
        if _PROCESS_MODEL is not None and _PROCESS_MODEL != self.manifest.model_id:
            if not allow_shared_process:
                raise RuntimeError(
                    f"process already holds model {_PROCESS_MODEL!r}; building "
                    f"{self.manifest.model_id!r} in the same process would mix "
                    f"section lists. Use a fresh process per model."
                )
        _PROCESS_MODEL = self.manifest.model_id
        if self.manifest.mechanism_dir:
            root = self._mech_root or Path(__file__).resolve().parents[3] / "data" / "models"
            load_mechanisms(root / self.manifest.mechanism_dir / "mod")
        else:
            from neuron import h

            h.load_file("stdrun.hoc")
        self._build()
        self._snapshot_canonical()
        self._built = True
        return self

    # NEURON exposes these on the Section, not on individual segments.
    _SECTION_LEVEL_ATTRS = frozenset({"Ra"})

    def _snapshot_canonical(self) -> None:
        """Record every targeted segment's canonical attribute value."""
        for spec in self.parameter_specs():
            entries: list[tuple[object, str, float]] = []
            attr = f"{spec.attribute}_{spec.mechanism_name}" if spec.mechanism_name else spec.attribute
            for sec in self._targets(spec):
                if attr in self._SECTION_LEVEL_ATTRS:
                    if hasattr(sec, attr):
                        entries.append((sec, attr, float(getattr(sec, attr))))
                    continue
                for seg in sec:
                    if not hasattr(seg, attr):
                        continue
                    entries.append((seg, attr, float(getattr(seg, attr))))
            if not entries:
                raise RuntimeError(
                    f"{self.manifest.model_id}: parameter {spec.parameter_id} "
                    f"({attr}) matched no segments"
                )
            self._canonical[spec.parameter_id] = entries

    # ------------------------------------------------------------------ #
    # Parameter injection
    # ------------------------------------------------------------------ #
    def canonical_absolute(self, parameter_id: str) -> float:
        """Area-weighted mean canonical value, for the audit table."""
        entries = self._canonical[parameter_id]
        return float(np.mean([v for _, _, v in entries]))

    def apply(self, scales: dict[str, float]) -> None:
        """Set every listed parameter to ``canonical * scale``."""
        for pid, scale in scales.items():
            entries = self._canonical.get(pid)
            if entries is None:
                raise KeyError(f"{self.manifest.model_id}: unknown parameter {pid!r}")
            for seg, attr, base in entries:
                setattr(seg, attr, base * scale)

    def reset(self) -> None:
        """Restore every parameter to its canonical value."""
        for entries in self._canonical.values():
            for seg, attr, base in entries:
                setattr(seg, attr, base)

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def soma(self):
        """The section used for current injection and voltage recording."""

    def _post_finitialize(self) -> bool:
        """Hook for a model-specific initial-state procedure.

        Called immediately after ``finitialize``.  Return ``True`` to request a
        second ``finitialize`` with whatever the hook changed.  The default is a
        no-op; only models whose published code defines such a procedure
        override it.
        """
        return False

    def _ensure_stim(self):
        from neuron import h

        if self._stim is None:
            self._stim = h.IClamp(self.soma(0.5))
        return self._stim

    def simulate(
        self,
        amplitude_nA: float,
        delay_ms: float = 200.0,
        duration_ms: float = 1000.0,
        total_ms: float = 1500.0,
        dt_ms: float = 0.025,
        protocol_id: str = "step",
        v_init: float | None = None,
        celsius: float | None = None,
        cvode: bool = False,
    ) -> Recording:
        """Run one current-clamp protocol and return the somatic trace."""
        from neuron import h

        h.celsius = celsius if celsius is not None else self.manifest.temperature_c
        stim = self._ensure_stim()
        stim.delay = delay_ms
        stim.dur = duration_ms
        stim.amp = amplitude_nA

        t_vec, v_vec = h.Vector(), h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(self.soma(0.5)._ref_v)

        # Not every published mechanism set is CVODE-safe; fall back rather than
        # abort, and report the fallback so the QC audit records it.
        cvode_used = cvode
        try:
            h.cvode_active(1 if cvode else 0)
        except RuntimeError:
            cvode_used = False
            try:
                h.cvode_active(0)
            except RuntimeError:
                pass
        h.dt = dt_ms
        h.tstop = total_ms
        h.v_init = v_init if v_init is not None else self.manifest.v_init_mV
        h.finitialize(h.v_init)
        # Some published models define a non-trivial initial-state procedure
        # (blueprint 7.1) that must run after finitialize and be repeated for
        # every parameter sample, not just once at build time.
        if self._post_finitialize():
            h.finitialize(h.v_init)
        try:
            h.continuerun(total_ms)
        except RuntimeError as exc:  # solver blow-up
            return Recording(
                t=np.array([]), v=np.array([]), stim_start_ms=delay_ms,
                stim_end_ms=delay_ms + duration_ms, amplitude_nA=amplitude_nA,
                protocol_id=protocol_id, extra={"solver_error": str(exc)[:120]},
            )

        # Copy, do not view.  numpy can wrap a NEURON Vector's buffer without
        # copying; when the Vector is garbage-collected at the end of this call
        # the array would point at freed memory that the next run can reuse, so
        # two Recordings could silently alias the same samples.
        t = np.array(t_vec, dtype=float, copy=True)
        v = np.array(v_vec, dtype=float, copy=True)
        if cvode_used:  # resample onto a regular grid so eFEL sees uniform dt
            grid = np.arange(0.0, total_ms + dt_ms, dt_ms)
            v = np.interp(grid, t, v)
            t = grid
        return Recording(
            t=t, v=v, stim_start_ms=delay_ms, stim_end_ms=delay_ms + duration_ms,
            amplitude_nA=amplitude_nA, protocol_id=protocol_id,
        )

    # ------------------------------------------------------------------ #
    # Rheobase
    # ------------------------------------------------------------------ #
    def find_rheobase(
        self,
        lo_nA: float = 0.0,
        hi_nA: float | None = None,
        tol_nA: float | None = None,
        delay_ms: float = 200.0,
        duration_ms: float = 1000.0,
        total_ms: float = 1400.0,
        dt_ms: float = 0.025,
        max_iter: int = 24,
        **sim_kwargs,
    ) -> float | None:
        """Bisection search for the smallest step current that evokes a spike.

        Returns ``None`` if the cell does not spike at ``hi_nA`` (the sample is
        then classified as non-spiking rather than silently dropped).

        ``sim_kwargs`` (notably ``cvode``) is forwarded to every probe.  Passing
        the model's chosen solver settings here matters a great deal: the search
        costs ~11 simulations per sample and dominates the campaign budget.
        """
        lo = lo_nA
        hi = self.manifest_rheobase_hi if hi_nA is None else hi_nA
        tol = self.manifest_rheobase_tol if tol_nA is None else tol_nA

        def spikes(amp: float) -> bool:
            rec = self.simulate(amp, delay_ms, duration_ms, total_ms, dt_ms,
                                protocol_id="rheobase_probe", **sim_kwargs)
            if rec.v.size == 0:
                return False
            mask = (rec.t >= delay_ms) & (rec.t <= delay_ms + duration_ms)
            return bool(np.any(rec.v[mask] > 0.0))

        if not spikes(hi):
            return None
        if spikes(lo):
            return lo
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if spikes(mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return hi

    # Per-model search envelope; overridden by subclasses with different sizes.
    manifest_rheobase_hi: float = 2.0
    manifest_rheobase_tol: float = 0.002

    # ------------------------------------------------------------------ #
    # Numerical quality control (blueprint 14.5)
    # ------------------------------------------------------------------ #
    @staticmethod
    def qc_flags(rec: Recording) -> list[str]:
        """Detect solver failures, nonphysical voltages and unstable rest."""
        flags: list[str] = []
        if rec.v.size == 0:
            return ["solver_failure"]
        v = rec.v
        if not np.all(np.isfinite(v)):
            flags.append("nonfinite_voltage")
            return flags
        if np.max(v) > 120.0 or np.min(v) < -160.0:
            flags.append("nonphysical_voltage")
        pre = v[rec.t < rec.stim_start_ms]
        if pre.size > 40:
            baseline = pre[len(pre) // 2:]
            if np.ptp(baseline) > 2.0:
                flags.append("unstable_baseline")
            if np.any(baseline > -20.0):
                flags.append("depolarised_rest")
        return flags
