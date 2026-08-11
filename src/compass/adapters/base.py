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

# Declared version of the MEASUREMENT procedure: how a protocol is executed and
# how a rheobase probe decides "did this cell spike".  Stamped into every shard
# identity and compared on reuse.
#
# This exists because the identity previously had a hole. `feature_schema`
# covers extraction, `solver` covers the time step, `protocol_hash` covers the
# stimulus -- but nothing covered the probe semantics in this module, so the N5
# tri-state fix changed what a re-run produces on ~0.56% of rows (measured by
# the deterministic replay gate) without anything forcing regeneration. A
# declared-version scheme only works if it covers every input that can change a
# stored value.
#
# probe-2: `find_rheobase` requires a genuine upward threshold crossing and
#          raises on a failed probe. probe-1 accepted "any sample above 0 mV",
#          so an already-depolarised trace was recorded as spiking at 0 nA and
#          got rheobase = 0.0 plus rheobase-normalised features evaluated at
#          0 nA (2026-08-06 audit, section 5.4).
# probe-3: `find_rheobase` no longer assumes firing is monotonic in current. A
#          cell in depolarisation block at the search ceiling used to be
#          reported as non-spiking; it is now bracketed by a geometric downward
#          ladder and its true rheobase is measured (2026-08-09 audit, 4.1).
#          `_feature_row` also treats rheobase == 0 as undefined for the
#          rheobase-normalised family for EVERY model, not only for the ones
#          listed as spontaneously active, so a draw that becomes spontaneous in
#          an ordinarily silent model no longer runs "1.5x rheobase" at 0 nA.
# probe-4: probe-3's geometric ladder was a SPARSE search presented as an
#          exhaustive one.  It probed hi/2**k for k = 8..1, so a cell that fires
#          only between two rungs -- say 0.30-0.40 nA with a 1.0 nA ceiling,
#          which the ladder straddles at 0.25 and 0.50 -- was still returned as
#          silent (2026-08-09 post-response-3 audit, P0-1).  probe-3 also ran
#          the ladder ONLY when the ceiling did not spike, so a cell with a
#          disconnected LOWER firing window plus an upper firing branch was
#          bisected on the upper branch and its true rheobase overstated.
#          probe-4 replaces the ladder with a uniform grid of declared
#          resolution, scanned upward from the bottom on EVERY sample, and
#          states the detection guarantee that grid actually provides.
MEASUREMENT_SCHEMA_VERSION = "probe-4"

# What the probe-4 search does and does not guarantee.  This string is quoted in
# the release manifest and in docs/RESULTS.md so the contract travels with the
# number; the guarantee is a property of the GRID, not of the biology.
RHEOBASE_SEARCH_CONTRACT = (
    "uniform grid of RHEOBASE_SCAN_POINTS amplitudes spanning (lo, hi] with "
    "spacing delta = (hi - lo) / RHEOBASE_SCAN_POINTS, plus lo itself; scanned "
    "upward from lo and refined by bisection inside the first bracket that "
    "spikes. GUARANTEED: every firing window of width >= delta lying in "
    "[lo, hi] is detected, whether or not the ceiling spikes and whether or not "
    "firing is monotonic in current. NOT GUARANTEED: firing windows narrower "
    "than delta. None therefore means 'no spike at any probed amplitude at this "
    "resolution', never 'silent at every real-valued amplitude'."
)

# probe-4 is NOT a targeted change: the scan now runs on every sample, so no row
# is unchanged by construction and no enumeration of affected rows is possible.
# The corpus is regenerated in full from the stored parameter draws.
PROBE4_CHANGES_ROWS_WHERE = (
    "any row may change: the upward scan runs on every sample, including the "
    "ones that spike at the ceiling, so a lower disconnected firing window is "
    "now found where probe-3 bisected onto the upper branch"
)

# THE band outside which a membrane potential is a broken integration rather
# than an unusual phenotype.  One band, one place, two consumers: `qc_flags`
# raises `nonphysical_voltage` here on every one of the corpus's simulations --
# which sets `execution_valid = False` and removes the row from every downstream
# analysis -- and `tools/retain_waveforms.py` reports it as a `voltage_range`
# finding on its retained sample.
#
# They used to disagree: the corpus excluded outside [-160, 120] and the
# waveform battery flagged outside [-150, 80], so a trace reaching -155 mV was
# "physiologically impossible" in the waveform report and fully analysis-
# eligible in the corpus.  The 2026-08-09 post-response-3 audit (P0-6) required
# every invalid class to be excluded by a rule propagated to every downstream
# analysis; two bands cannot do that, so there is now one, set at the tighter
# pair.  Measured cost of the tightening on the previous retained sample: 4 of
# 914 protocol traces move from "flagged only" to "excluded", against 17 that
# were already excluded under the wider band.
PHYSICAL_V_MIN_MV = -150.0
PHYSICAL_V_MAX_MV = 80.0

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
        """Membrane-area-weighted mean canonical value, for the audit table.

        The docstring said "area-weighted" and the body was ``np.mean`` over
        segments (2026-08-09 audit, 5.3).  That is not a harmless simplification
        on a reconstructed morphology: a d_lambda mesh puts many short segments
        in thin distal dendrites and few long ones in thick proximal sections,
        so a plain segment mean over-weights the distal tree by an order of
        magnitude.  ``bbp_l5pc`` has 642 segments spanning about three decades of
        area.

        Weighting is by each segment's membrane area.  Where an area cannot be
        obtained -- a synthetic adapter in a unit test, or a mechanism attached
        to something that is not a segment -- the weight falls back to 1 and the
        result degrades to the old segment mean rather than raising.
        """
        entries = self._canonical[parameter_id]
        vals = np.array([v for _, _, v in entries], dtype=float)
        areas = []
        for seg, _attr, _v in entries:
            try:
                areas.append(float(seg.area()))
            except Exception:                                     # noqa: BLE001
                areas.append(1.0)
        w = np.asarray(areas, dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return float(np.mean(vals))
        return float(np.sum(w * vals) / np.sum(w))

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
            # The comment above promises the CVODE fallback will be "reported so
            # the QC audit records it", and until now only the FAILING path
            # carried anything: a successful Recording said nothing about which
            # solver actually integrated it, so a silent fallback from adaptive
            # to fixed step left no trace at all (2026-08-09 audit, 5.2).  The
            # effective mode now travels with every trace.
            extra={"cvode_requested": bool(cvode), "cvode_used": bool(cvode_used),
                   "cvode_fallback": bool(cvode and not cvode_used),
                   "dt_ms": float(dt_ms)},
        )

    # ------------------------------------------------------------------ #
    # Rheobase
    # ------------------------------------------------------------------ #
    class RheobaseProbeFailure(RuntimeError):
        """A rheobase probe did not produce a usable trace.

        Raised rather than returned, because the only correct response is to
        fail the whole parameter sample.  Silently mapping a failed probe to
        "does not spike" is how a solver blow-up at high current became a
        finite feature row (2026-08-04/2026-08-06 audits, section 5.4).
        """

    # Probe outcomes.  The tri-state is the point: a probe that could not be
    # evaluated is NOT evidence of silence (gate N5).
    SPIKING, NONSPIKING, FAILED = "spiking", "nonspiking", "failed"

    def probe_spiking(
        self,
        amplitude_nA: float,
        delay_ms: float,
        duration_ms: float,
        total_ms: float,
        dt_ms: float,
        threshold_mV: float = 0.0,
        **sim_kwargs,
    ) -> str:
        """Three-state spike probe: ``spiking`` | ``nonspiking`` | ``failed``.

        Two defects of the previous boolean version are fixed here.

        1. **A failed simulation returned "no spike".**  An empty trace, a
           non-finite sample or a solver exception all became ``False``, so a
           blow-up at high current was indistinguishable from a genuinely
           silent cell -- and, because the bracket check runs at ``hi_nA``
           first, it turned into ``rheobase = None`` and a "non-spiking" sample.
        2. **Detection was "any sample above 0 mV".**  A trace that is *already*
           depolarised above 0 mV counts as spiking without any upward
           crossing.  Detection now requires a genuine rising crossing of the
           threshold from below, and a baseline check rejects a trace that
           never sits below it.
        """
        rec = self.simulate(amplitude_nA, delay_ms, duration_ms, total_ms, dt_ms,
                            protocol_id="rheobase_probe", **sim_kwargs)
        if rec.v.size == 0 or rec.t.size != rec.v.size:
            return self.FAILED
        if not np.all(np.isfinite(rec.v)):
            return self.FAILED
        mask = (rec.t >= delay_ms) & (rec.t <= delay_ms + duration_ms)
        v = rec.v[mask]
        if v.size < 2:
            return self.FAILED
        above = v > threshold_mV
        if above.all():
            # Never crosses from below during the step.  This is a depolarised
            # or blocked state, not a spike train; report it as such instead of
            # counting the plateau as a spike.
            return self.NONSPIKING
        crossings = int(np.count_nonzero((~above[:-1]) & above[1:]))
        return self.SPIKING if crossings > 0 else self.NONSPIKING

    # The declared resolution of the rheobase search.  The interval (lo, hi] is
    # divided into this many equal steps and EVERY step is a candidate probe, so
    # the smallest firing window the search is guaranteed to find is
    # `(hi - lo) / RHEOBASE_SCAN_POINTS` wide.  See RHEOBASE_SEARCH_CONTRACT.
    #
    # 64 is a cost choice as much as a resolution choice and is recorded as one.
    # The scan stops at the first amplitude that spikes and the measured corpus
    # has a median rheobase at 0.148 of the ceiling, so the expected scan length
    # is ~15 probes rather than 64; a silent cell pays the full 64.  Raising this
    # to 128 would halve the smallest detectable window and roughly double the
    # campaign's simulation budget.  `tools/audit_rheobase_resolution.py` measures
    # the observed firing-window widths on a stratified sample at 8x this
    # resolution, which is the evidence for whether 64 is fine enough.
    RHEOBASE_SCAN_POINTS = 64

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
        scan_points: int | None = None,
        exhaustive: bool = False,
        **sim_kwargs,
    ) -> float | None:
        """Search for the smallest step current that evokes a spike.

        **The guarantee this search provides is a property of its grid, and it
        is stated rather than implied.**  ``(lo_nA, hi_nA]`` is divided into
        ``scan_points`` equal steps of width ``delta``; every one of them is a
        candidate probe, and ``lo_nA`` itself is probed too.  A contiguous
        firing window at least ``delta`` wide must contain a grid point, so it
        is found.  A window narrower than ``delta`` may be stepped over.
        ``None`` therefore means *no spike at any of the probed amplitudes*, and
        never *silent at every real-valued amplitude* -- no finite search can
        establish the latter.  ``RHEOBASE_SEARCH_CONTRACT`` is the same statement
        in one string, and it travels into the release manifest.

        **Firing is not assumed to be monotonic in current, and the search no
        longer assumes it in the one place it still did.**  probe-3 walked a
        geometric ladder ``hi/2**k`` and only when the ceiling did not spike.
        Both halves were wrong.  The ladder is sparse: a cell firing from 0.30
        to 0.40 nA under a 1.0 nA ceiling is straddled by the 0.25 and 0.50 nA
        rungs and was returned as silent.  And running it only for
        ceiling-silent cells meant that a cell with a low disconnected window
        *plus* an upper firing branch was bisected on the upper branch, so its
        rheobase was overstated rather than missed (2026-08-09
        post-response-3 audit, P0-1).  probe-4 scans the uniform grid upward
        from the bottom on **every** sample and brackets the first step that
        spikes, so the value returned is the lowest detectable firing
        amplitude whatever the shape of the f-I curve above it.

        With ``exhaustive=True`` the scan does not stop at the first hit; it
        probes the whole grid and records every non-spiking-to-spiking
        transition in ``last_rheobase_search["transitions"]``.  That mode costs
        ``scan_points`` simulations per sample and is used by
        ``tools/audit_rheobase_resolution.py`` on a stratified sample to measure
        how wide the real firing windows are, not by the campaign.

        ``sim_kwargs`` (notably ``cvode``) is forwarded to every probe.  Cost:
        the scan stops at the first spiking step, so a cell whose rheobase sits
        at the corpus median of 0.148 x ceiling pays ~10 scan probes, and only a
        genuinely silent cell pays all ``scan_points``.

        The diagnosis is recorded on ``self.last_rheobase_search`` so a caller
        can tell a genuinely silent cell from one that is blocked at the ceiling
        without re-running the probes.
        """
        lo = lo_nA
        ceiling = self.manifest_rheobase_hi if hi_nA is None else hi_nA
        tol = self.manifest_rheobase_tol if tol_nA is None else tol_nA
        n = int(self.RHEOBASE_SCAN_POINTS if scan_points is None else scan_points)
        if n < 1:
            raise ValueError(f"scan_points must be >= 1, got {n}")
        delta = (ceiling - lo) / n
        diag: dict = {
            "contract": "uniform-grid",
            "ceiling_nA": float(ceiling),
            "scan_points": n,
            "scan_resolution_nA": float(delta),
            "smallest_detectable_window_nA": float(delta),
            "scan_probes": 0,
            "scan_hit_index": None,
            "scan_hit_nA": None,
            "spikes_at_ceiling": None,
            "blocked_at_ceiling": False,
            "spontaneous": False,
            "exhaustive": bool(exhaustive),
            "transitions": [],
            "n_probes": 0,
        }
        self.last_rheobase_search = diag
        seen: dict[float, str] = {}

        def state(amp: float) -> str:
            amp = float(amp)
            if amp in seen:
                return seen[amp]
            diag["n_probes"] += 1
            s = self.probe_spiking(amp, delay_ms, duration_ms, total_ms, dt_ms,
                                   **sim_kwargs)
            if s == self.FAILED:
                raise self.RheobaseProbeFailure(
                    f"{self.manifest.model_id}: rheobase probe at {amp:.6g} nA "
                    f"produced no usable trace")
            seen[amp] = s
            return s

        def grid(k: int) -> float:
            # k == n reproduces the ceiling exactly rather than by accumulation.
            return ceiling if k == n else lo + k * delta

        # A cell that already fires with no injected current has rheobase lo by
        # definition; probing it first also costs nothing on the majority path,
        # because the bisection below needs the same value.
        if state(lo) == self.SPIKING:
            diag["spontaneous"] = True
            return lo

        # The ceiling is grid point n and is memoised, so probing it up front to
        # record `blocked_at_ceiling` does not add a simulation to the scan.
        diag["spikes_at_ceiling"] = state(ceiling) == self.SPIKING

        hit = None
        prev = self.NONSPIKING          # state(lo), established above
        for k in range(1, n + 1):
            amp = grid(k)
            diag["scan_probes"] += 1
            cur = state(amp)
            if cur == self.SPIKING and prev != self.SPIKING:
                diag["transitions"].append(
                    {"index": k, "lo_nA": float(grid(k - 1)), "hi_nA": float(amp)})
                if hit is None:
                    hit = k
                    if not exhaustive:
                        break
            prev = cur

        if hit is None:
            return None      # no spike at any probed amplitude, at resolution delta
        diag["scan_hit_index"] = hit
        diag["scan_hit_nA"] = float(grid(hit))
        # Spikes somewhere below the ceiling but not at it: depolarisation block
        # (or any other non-monotonicity) at the top of the range.
        diag["blocked_at_ceiling"] = not diag["spikes_at_ceiling"]

        # Refine inside the one bracket known to straddle a transition.  Both
        # ends are already measured, so bisection never re-derives them.
        lo, hi = grid(hit - 1), grid(hit)
        for _ in range(max_iter):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            if state(mid) == self.SPIKING:
                hi = mid
            else:
                lo = mid
        return hi

    # Diagnosis of the most recent `find_rheobase` call; see that method.
    last_rheobase_search: dict | None = None

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
        # A silent fall back from the requested adaptive solver to fixed step
        # changes what was integrated and must reach the audit table, not just
        # the Recording's metadata (2026-08-09 audit, 5.2).
        if rec.extra.get("cvode_fallback"):
            flags.append("cvode_fallback")
        if rec.v.size == 0:
            return [*flags, "solver_failure"]
        v = rec.v
        if not np.all(np.isfinite(v)):
            flags.append("nonfinite_voltage")
            return flags
        if np.max(v) > PHYSICAL_V_MAX_MV or np.min(v) < PHYSICAL_V_MIN_MV:
            flags.append("nonphysical_voltage")
        pre = v[rec.t < rec.stim_start_ms]
        if pre.size > 40:
            baseline = pre[len(pre) // 2:]
            if np.ptp(baseline) > 2.0:
                flags.append("unstable_baseline")
            if np.any(baseline > -20.0):
                flags.append("depolarised_rest")
        return flags
