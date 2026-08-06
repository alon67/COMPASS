"""Concrete adapters for the COMPASS model panel.

Every model is reconstructed from its published source (ModelDB accession or a
pinned Blue Brain Project commit) with the canonical parameter values written
out explicitly in this file, so that the canonical-reproduction audit compares
*our* construction against the published traces rather than against itself.

Panel composition follows blueprint 6.1.  Two independently developed cortical
pyramidal models (Mainen & Sejnowski 1996 vs. the Hay/Blue Brain lineage) and
two morphological variants of the minimal HH model provide the replication
structure required by 6.4 and the morphology ablation required by 13.3.
"""

from __future__ import annotations

from pathlib import Path

from ..registry.schema import EvidenceTier, ModelManifest, ParameterSpec, Transform
from .base import NeuronAdapter, _own_sections, section_names

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

# Default expert-defined prior half-width in decades (Tier D).  A scale range of
# [1/4, 4] is the conventional "broad but not absurd" conductance range in the
# degenerate-solution literature (Prinz et al. 2004 spans ~an order of magnitude
# per conductance).  Prior width is varied systematically in the experiments.
D_WIDE = 4.0
D_MED = 2.0


def _p(model_id, pid, mech, attr, comp, family, role, units, canonical, ref,
       lo=1.0 / D_WIDE, hi=D_WIDE, tier=EvidenceTier.D, transform=Transform.LOG10):
    return ParameterSpec(
        model_id=model_id, parameter_id=pid, mechanism_name=mech, attribute=attr,
        compartment=comp, channel_family=family, biophysical_role=role, units=units,
        canonical_value=canonical, lower_bound=lo, upper_bound=hi,
        evidence_tier=tier, source_reference=ref, transform=transform,
    )


# --------------------------------------------------------------------------- #
# 1. Minimal Hodgkin-Huxley point model
# --------------------------------------------------------------------------- #
class HHPoint(NeuronAdapter):
    """Single-compartment Hodgkin-Huxley cell using NEURON's built-in ``hh``."""

    REF = "Hodgkin & Huxley 1952 J Physiol 117:500; NEURON built-in hh mechanism"
    manifest = ModelManifest(
        model_id="hh_point", neuron_class="minimal_hh", species="generic",
        brain_region="none", simulator="neuron", source_reference=REF,
        license="NEURON built-in (BSD-like)", temperature_c=6.3, is_point_model=True,
        n_compartments=1, mechanism_dir=None, lab="hodgkin_huxley",
        pipeline="independent", v_init_mV=-65.0,
        notes="Reference minimal model; morphology partner of hh_cable.",
    )
    manifest_rheobase_hi = 1.0
    manifest_rheobase_tol = 0.002

    def _build(self) -> None:
        from neuron import h

        self._soma = h.Section(name="soma_hhpoint")
        self._soma.L = self._soma.diam = 30.0
        self._soma.cm = 1.0
        self._soma.Ra = 100.0
        self._soma.insert("hh")
        self._sections = {"soma": [self._soma]}

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "hh_point"
        return [
            _p(m, "gna_soma", "hh", "gnabar", "soma", "Nav", "fast_sodium", "S/cm2", 0.12, self.REF),
            _p(m, "gk_soma", "hh", "gkbar", "soma", "Kv", "delayed_rectifier", "S/cm2", 0.036, self.REF),
            _p(m, "gl_soma", "hh", "gl", "soma", "leak", "leak", "S/cm2", 0.0003, self.REF),
            _p(m, "el_soma", "hh", "el", "soma", "leak", "leak_reversal", "mV", -54.3, self.REF,
               lo=0.85, hi=1.15),
            _p(m, "cm_soma", None, "cm", "soma", "passive", "membrane_capacitance", "uF/cm2", 1.0,
               self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_soma", None, "diam", "soma", "morphology", "somatic_area", "um", 30.0,
               self.REF, lo=0.5, hi=2.0),
        ]


class HHCable(NeuronAdapter):
    """Two-compartment HH cable: identical channels, different morphology.

    Paired with :class:`HHPoint` this isolates the effect of adding a dendritic
    compartment while holding the channel repertoire fixed -- the cleanest
    possible instance of the morphology ablation in blueprint 13.3.
    """

    REF = "Hodgkin & Huxley 1952 J Physiol 117:500 (soma-dendrite cable variant)"
    manifest = ModelManifest(
        model_id="hh_cable", neuron_class="minimal_hh", species="generic",
        brain_region="none", simulator="neuron", source_reference=REF,
        license="NEURON built-in (BSD-like)", temperature_c=6.3, is_point_model=False,
        n_compartments=11, mechanism_dir=None, lab="hodgkin_huxley",
        pipeline="independent", v_init_mV=-65.0,
        notes="Morphology partner of hh_point: same channels, added dendrite.",
    )
    manifest_rheobase_hi = 2.0
    manifest_rheobase_tol = 0.004

    def _build(self) -> None:
        from neuron import h

        self._soma = h.Section(name="soma_hhcable")
        self._dend = h.Section(name="dend_hhcable")
        self._soma.L = self._soma.diam = 30.0
        self._dend.L, self._dend.diam, self._dend.nseg = 400.0, 2.0, 10
        for sec in (self._soma, self._dend):
            sec.cm, sec.Ra = 1.0, 100.0
            sec.insert("hh")
        # dendrite is excitable but far less so than the soma
        for seg in self._dend:
            seg.hh.gnabar, seg.hh.gkbar = 0.012, 0.0036
        self._dend.connect(self._soma(1), 0)
        self._sections = {"soma": [self._soma], "dend": [self._dend],
                          "all": [self._soma, self._dend]}

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "hh_cable"
        return [
            _p(m, "gna_soma", "hh", "gnabar", "soma", "Nav", "fast_sodium", "S/cm2", 0.12, self.REF),
            _p(m, "gk_soma", "hh", "gkbar", "soma", "Kv", "delayed_rectifier", "S/cm2", 0.036, self.REF),
            _p(m, "gl_soma", "hh", "gl", "soma", "leak", "leak", "S/cm2", 0.0003, self.REF),
            _p(m, "gna_dend", "hh", "gnabar", "dend", "Nav", "fast_sodium", "S/cm2", 0.012, self.REF),
            _p(m, "gk_dend", "hh", "gkbar", "dend", "Kv", "delayed_rectifier", "S/cm2", 0.0036, self.REF),
            _p(m, "gl_dend", "hh", "gl", "dend", "leak", "leak", "S/cm2", 0.0003, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance", "uF/cm2", 1.0,
               self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_dend", None, "diam", "dend", "morphology", "dendritic_diameter", "um", 2.0,
               self.REF, lo=0.5, hi=2.0),
        ]


# --------------------------------------------------------------------------- #
# 2. Pospischil et al. 2008 minimal cortical models (ModelDB 123623)
# --------------------------------------------------------------------------- #
class _PospischilBase(NeuronAdapter):
    REF = "Pospischil et al. 2008 Biol Cybern 99:427 (ModelDB 123623)"
    LICENSE = "ModelDB terms; free for academic use"
    _diam = 96.0
    _g_pas = 1e-4
    _gna = 0.05
    _gk = 0.005
    _gim: float | None = 7e-5

    def _build(self) -> None:
        from neuron import h

        self._soma = h.Section(name=f"soma_{self.manifest.model_id}")
        self._soma.L = self._soma.diam = self._diam
        self._soma.nseg, self._soma.Ra, self._soma.cm = 1, 100.0, 1.0
        self._soma.insert("pas")
        self._soma.e_pas, self._soma.g_pas = -70.0, self._g_pas
        self._soma.insert("hh2")
        self._soma.ek, self._soma.ena = -100.0, 50.0
        for seg in self._soma:
            seg.hh2.vtraub = -55.0
            seg.hh2.gnabar = self._gna
            seg.hh2.gkbar = self._gk
        if self._gim is not None:
            self._soma.insert("im")
            from neuron import h as _h

            _h.taumax_im = 1000.0
            for seg in self._soma:
                seg.im.gkbar = self._gim
        self._sections = {"soma": [self._soma]}

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def _common_specs(self, m):
        return [
            _p(m, "gna_soma", "hh2", "gnabar", "soma", "Nav", "fast_sodium", "S/cm2",
               self._gna, self.REF),
            _p(m, "gk_soma", "hh2", "gkbar", "soma", "Kv", "delayed_rectifier", "S/cm2",
               self._gk, self.REF),
            _p(m, "gpas_soma", "pas", "g", "soma", "leak", "leak", "S/cm2", self._g_pas, self.REF),
            _p(m, "epas_soma", "pas", "e", "soma", "leak", "leak_reversal", "mV", -70.0,
               self.REF, lo=0.85, hi=1.15),
            _p(m, "vtraub_soma", "hh2", "vtraub", "soma", "Nav", "activation_offset", "mV",
               -55.0, self.REF, lo=0.9, hi=1.1),
            _p(m, "cm_soma", None, "cm", "soma", "passive", "membrane_capacitance", "uF/cm2",
               1.0, self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_soma", None, "diam", "soma", "morphology", "somatic_area", "um",
               self._diam, self.REF, lo=0.6, hi=1.6),
        ]


class PospischilRS(_PospischilBase):
    """Regular-spiking cortical pyramidal cell (sPY template)."""

    _diam, _g_pas, _gna, _gk, _gim = 96.0, 1e-4, 0.05, 0.005, 7e-5
    manifest = ModelManifest(
        model_id="pospischil_rs", neuron_class="cortical_pyramidal_rs", species="rat",
        brain_region="neocortex", simulator="neuron",
        source_reference=_PospischilBase.REF, license=_PospischilBase.LICENSE,
        temperature_c=36.0, is_point_model=True, n_compartments=1,
        mechanism_dir="pospischil_rs", lab="destexhe", pipeline="independent",
        v_init_mV=-70.0,
        notes="Same framework and lab as pospischil_fs: isolates cell class from implementation.",
    )
    manifest_rheobase_hi = 1.5
    manifest_rheobase_tol = 0.004

    def parameter_specs(self):
        m = "pospischil_rs"
        return self._common_specs(m) + [
            _p(m, "gim_soma", "im", "gkbar", "soma", "Kv7_M", "adaptation", "S/cm2",
               7e-5, self.REF),
        ]


class PospischilFS(_PospischilBase):
    """Fast-spiking cortical interneuron (sIN template)."""

    _diam, _g_pas, _gna, _gk, _gim = 67.0, 1.5e-4, 0.05, 0.01, None
    manifest = ModelManifest(
        model_id="pospischil_fs", neuron_class="cortical_interneuron_fs", species="rat",
        brain_region="neocortex", simulator="neuron",
        source_reference=_PospischilBase.REF, license=_PospischilBase.LICENSE,
        temperature_c=36.0, is_point_model=True, n_compartments=1,
        mechanism_dir="pospischil_rs", lab="destexhe", pipeline="independent",
        v_init_mV=-70.0,
        notes="Shares mechanisms and lab with pospischil_rs; no M current.",
    )
    manifest_rheobase_hi = 2.0
    manifest_rheobase_tol = 0.004

    def parameter_specs(self):
        return self._common_specs("pospischil_fs")


# --------------------------------------------------------------------------- #
# 3. Khaliq & Raman 2003 cerebellar Purkinje cell (ModelDB 48332)
# --------------------------------------------------------------------------- #
class KhaliqPurkinje(NeuronAdapter):
    """Purkinje soma with resurgent Na and the Purkinje K/Ca repertoire."""

    REF = "Khaliq, Gouwens & Raman 2003 J Neurosci 23:4899 (ModelDB 48332)"
    manifest = ModelManifest(
        model_id="khaliq_purkinje", neuron_class="cerebellar_purkinje", species="mouse",
        brain_region="cerebellum", simulator="neuron", source_reference=REF,
        license="ModelDB terms; free for academic use", temperature_c=22.0,
        is_point_model=True, n_compartments=1, mechanism_dir="khaliq_purkinje",
        lab="raman", pipeline="independent", v_init_mV=-68.0,
        notes="Spontaneously active; resurgent sodium current.",
    )
    manifest_rheobase_hi = 1.0
    manifest_rheobase_tol = 0.004

    def _build(self) -> None:
        from neuron import h

        s = h.Section(name="soma_khaliq")
        s.L = s.diam = 20.0
        s.nseg, s.Ra, s.cm = 1, 100.0, 1.0
        for mech in ("naRsg", "kpkj", "kpkj2", "kpkjslow", "bkpkj",
                     "cadiff", "cap", "lkpkj", "hpkj"):
            s.insert(mech)
        s.ena, s.ek = 60.0, -88.0
        self._soma = s
        self._sections = {"soma": [s]}

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "khaliq_purkinje"
        return [
            _p(m, "gna_rsg", "naRsg", "gbar", "soma", "Nav_resurgent", "fast_sodium", "S/cm2",
               0.015, self.REF),
            _p(m, "gk_fast", "kpkj", "gkbar", "soma", "Kv3", "fast_rectifier", "S/cm2",
               0.004, self.REF),
            _p(m, "gk_mid", "kpkj2", "gkbar", "soma", "Kv3", "mid_rectifier", "S/cm2",
               0.002, self.REF),
            _p(m, "gk_slow", "kpkjslow", "gkbar", "soma", "Kv1", "slow_rectifier", "S/cm2",
               0.004, self.REF),
            _p(m, "gk_bk", "bkpkj", "gkbar", "soma", "BK", "calcium_activated_k", "S/cm2",
               0.007, self.REF),
            _p(m, "pca_p", "cap", "pcabar", "soma", "Cav_P", "p_type_calcium", "cm/s",
               5e-5, self.REF),
            _p(m, "gh", "hpkj", "ghbar", "soma", "HCN", "h_current", "S/cm2", 1e-4, self.REF),
            _p(m, "gleak", "lkpkj", "g", "soma", "leak", "leak", "S/cm2", 5e-5, self.REF),
            _p(m, "cm_soma", None, "cm", "soma", "passive", "membrane_capacitance", "uF/cm2",
               1.0, self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_soma", None, "diam", "soma", "morphology", "somatic_area", "um",
               20.0, self.REF, lo=0.6, hi=1.6),
        ]


# --------------------------------------------------------------------------- #
# 4. Mainen & Sejnowski 1996 layer-5 pyramidal cell (ModelDB 2488)
# --------------------------------------------------------------------------- #
class MainenL5PC(NeuronAdapter):
    """Reconstructed L5 pyramidal neuron with axonal spike-initiation zone.

    Reconstructed from ``demofig1.hoc``; the morphology is the published
    layer-5 cell ``j4a.hoc``.  Axon, spine correction and channel placement
    follow the original code exactly (values re-stated here for audit).
    """

    REF = "Mainen & Sejnowski 1996 Nature 382:363 (ModelDB 2488)"
    SRC = RAW / "modeldb_2488" / "cells"
    manifest = ModelManifest(
        model_id="mainen_l5pc", neuron_class="cortical_pyramidal", species="rat",
        brain_region="neocortex_layer5", simulator="neuron", source_reference=REF,
        license="ModelDB terms; free for academic use", temperature_c=37.0,
        is_point_model=False, n_compartments=0, mechanism_dir="mainen_l5pc",
        lab="mainen_sejnowski", pipeline="independent", v_init_mV=-70.0,
        notes="Independent (non-BBP) cortical pyramidal model; replication partner of bbp_l5pc.",
    )
    manifest_rheobase_hi = 3.0
    manifest_rheobase_tol = 0.002

    # canonical densities, pS/um2 as in the original hoc
    GNA_DEND, GNA_NODE = 20.0, 30000.0
    GKV_AXON, GKV_SOMA = 2000.0, 200.0
    GCA, GKM, GKCA = 0.3, 0.1, 3.0
    RA, RM, CM, CM_MYELIN, GPAS_NODE = 150.0, 30000.0, 0.75, 0.04, 0.02

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.celsius = 37.0
        before = section_names()
        morph = self.SRC / "cells" / "j4a.hoc"
        h.xopen(str(morph))
        own = _own_sections(before)

        soma = h.soma
        self._soma = soma

        dendritic = []
        for sec in own:
            name = sec.name()
            if "axon" in name or "myelin" in name or "node" in name or "hill" in name or "iseg" in name:
                continue
            dendritic.append(sec)
        # split long compartments (original: no compartment longer than 50 um)
        for sec in dendritic:
            sec.nseg = max(1, int(sec.L / 50.0) * 2 + 1)

        self._build_axon(h, soma)

        dend_only = [s for s in dendritic if s is not soma]
        self._add_spines(dend_only)

        axon_secs = [self._iseg, self._hill] + list(self._myelin) + list(self._node)
        all_secs = own + axon_secs
        for sec in all_secs:
            sec.insert("pas")
            sec.Ra, sec.cm = self.RA, self.CM
            for seg in sec:
                seg.pas.g = 1.0 / self.RM
                seg.pas.e = -70.0
        for sec in self._myelin:
            sec.cm = self.CM_MYELIN
        for sec in self._node:
            for seg in sec:
                seg.pas.g = self.GPAS_NODE

        # Spatial-convergence refinement.  The published segmentation (the
        # "no compartment longer than 50 um" rule, 754 segments) is NOT
        # converged: at every time step tested it fails to reproduce the
        # reference feature vector (mean agreement 0.897-0.987; mean firing rate
        # and AP amplitude disagree on 1-3 of 6 prior samples).  Twice the
        # d_lambda(100 Hz) mesh at dt=0.0125 ms reproduces all 13 conditioning
        # features on all 6 samples.  Placed after Ra/cm are assigned, because
        # the electrotonic length constant depends on both.
        # See data/provenance/convergence_audit_mainen_l5pc.json.
        for sec in all_secs:
            lam = 1e5 * (sec.diam / (4.0 * 3.141592653589793 * 100.0 * sec.Ra * sec.cm)) ** 0.5
            base = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1
            sec.nseg = max(1, (base * 2) // 2 * 2 + 1)

        for sec in all_secs:
            sec.insert("na")
        for sec in dendritic + list(self._myelin):
            for seg in sec:
                seg.na.gbar = self.GNA_DEND
        for sec in [self._hill, self._iseg] + list(self._node):
            for seg in sec:
                seg.na.gbar = self.GNA_NODE

        for sec in (self._iseg, self._hill):
            sec.insert("kv")
            for seg in sec:
                seg.kv.gbar = self.GKV_AXON
        soma.insert("kv")
        for seg in soma:
            seg.kv.gbar = self.GKV_SOMA

        for sec in dendritic:
            for mech in ("km", "kca", "ca", "cad"):
                sec.insert(mech)
            for seg in sec:
                seg.km.gbar, seg.kca.gbar, seg.ca.gbar = self.GKM, self.GKCA, self.GCA

        for sec in all_secs:
            if h.ismembrane("k_ion", sec=sec):
                sec.ek = -90.0
            if h.ismembrane("na_ion", sec=sec):
                sec.ena = 60.0
                h.vshift_na = -5.0
            if h.ismembrane("ca_ion", sec=sec):
                sec.eca = 140.0
                h.ion_style("ca_ion", 0, 1, 0, 0, 0, sec=sec)

        self._sections = {
            "soma": [soma],
            "dendritic": dendritic,
            "dend_only": dend_only,
            "axonal": axon_secs,
            "axon_na": [self._hill, self._iseg] + list(self._node),
            "axon_kv": [self._iseg, self._hill],
            "all": all_secs,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in all_secs))

    def _build_axon(self, h, soma) -> None:
        n_seg = 5
        area = sum(seg.area() for seg in soma)
        equiv_diam = (area / (4.0 * 3.141592653589793)) ** 0.5
        self._iseg = h.Section(name="iseg")
        self._hill = h.Section(name="hill")
        self._myelin = [h.Section(name=f"myelin_{i}") for i in range(n_seg)]
        self._node = [h.Section(name=f"node_{i}") for i in range(n_seg)]
        self._iseg.L, self._iseg.nseg, self._iseg.diam = 15.0, 5, equiv_diam / 10.0
        self._hill.L, self._hill.nseg = 10.0, 5
        self._hill.diam = 2.5 * self._iseg.diam
        for i in range(n_seg):
            self._myelin[i].nseg, self._myelin[i].L = 5, 100.0
            self._myelin[i].diam = self._iseg.diam
            self._node[i].nseg, self._node[i].L = 1, 1.0
            self._node[i].diam = self._iseg.diam * 0.75
        self._hill.connect(soma(0.5), 0)
        self._iseg.connect(self._hill(1), 0)
        self._myelin[0].connect(self._iseg(1), 0)
        for i in range(n_seg - 1):
            self._node[i].connect(self._myelin[i](1), 0)
            self._myelin[i + 1].connect(self._node[i](1), 0)
        self._node[n_seg - 1].connect(self._myelin[n_seg - 1](1), 0)

    @staticmethod
    def _add_spines(sections, spine_dens: float = 1.0, spine_area: float = 0.83) -> None:
        """Folding-factor spine correction (Jack et al. 1989, Major et al. 1994)."""
        for sec in sections:
            a = sum(seg.area() for seg in sec)
            if a <= 0:
                continue
            f = (sec.L * spine_area * spine_dens + a) / a
            sec.L = sec.L * f ** (2.0 / 3.0)
            for seg in sec:
                seg.diam = seg.diam * f ** (1.0 / 3.0)

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "mainen_l5pc"
        return [
            _p(m, "gna_dend", "na", "gbar", "dendritic", "Nav", "fast_sodium", "1",
               self.GNA_DEND, self.REF),
            _p(m, "gna_axon", "na", "gbar", "axon_na", "Nav", "axonal_sodium", "1",
               self.GNA_NODE, self.REF),
            _p(m, "gkv_soma", "kv", "gbar", "soma", "Kv", "delayed_rectifier", "1",
               self.GKV_SOMA, self.REF),
            _p(m, "gkv_axon", "kv", "gbar", "axon_kv", "Kv", "axonal_rectifier", "1",
               self.GKV_AXON, self.REF),
            _p(m, "gkm_dend", "km", "gbar", "dendritic", "Kv7_M", "adaptation", "1",
               self.GKM, self.REF),
            _p(m, "gkca_dend", "kca", "gbar", "dendritic", "SK", "calcium_activated_k", "1",
               self.GKCA, self.REF),
            _p(m, "gca_dend", "ca", "gbar", "dendritic", "Cav", "high_threshold_calcium", "1",
               self.GCA, self.REF),
            _p(m, "gpas_all", "pas", "g", "all", "leak", "leak", "S/cm2",
               1.0 / self.RM, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance", "uF/cm2",
               self.CM, self.REF, lo=0.5, hi=2.0),
            _p(m, "ra_all", None, "Ra", "all", "passive", "axial_resistance", "ohm-cm",
               self.RA, self.REF, lo=0.4, hi=2.5),
        ]


# --------------------------------------------------------------------------- #
# 5. Hay / Blue Brain L5b pyramidal cell (ModelDB 139653 + BBP mechanisms)
# --------------------------------------------------------------------------- #
class BBPL5PC(NeuronAdapter):
    """Blue Brain-lineage layer-5b thick-tufted pyramidal cell.

    The model is the Hay et al. (2011) L5b PC, which is the direct ancestor of
    the Blue Brain Project's cortical e-models: its mechanism set (``NaTa_t``,
    ``SKv3_1``, ``Ca_LVAst``, ``CaDynamics_E2`` ...) is exactly the mechanism
    library shipped with BluePyOpt, and the parameter bounds used here are taken
    from BBP's own published optimisation configuration
    (``BluePyOpt/examples/l5pc/config/params.json``), which makes them
    **evidence tier C -- a model-fitting prior, not biological variability**.
    """

    REF = ("Hay et al. 2011 PLoS Comput Biol 7:e1002107 (ModelDB 139653); "
           "bounds from BlueBrain/BluePyOpt examples/l5pc/config/params.json")
    SRC = RAW / "modeldb_139653" / "L5bPCmodelsEH"
    manifest = ModelManifest(
        model_id="bbp_l5pc", neuron_class="cortical_pyramidal", species="rat",
        brain_region="neocortex_layer5", simulator="neuron", source_reference=REF,
        license="ModelDB terms (Hay et al.); BluePyOpt LGPL-3.0 for BBP config",
        temperature_c=34.0, is_point_model=False, n_compartments=0,
        mechanism_dir="bbp_l5pc", lab="blue_brain_project", pipeline="bbp",
        v_init_mV=-80.0,
        notes="BBP-pipeline lineage; replication partner of mainen_l5pc.",
    )
    manifest_rheobase_hi = 3.0
    manifest_rheobase_tol = 0.002

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")
        h.celsius = 34.0
        h.load_file(str(self.SRC / "models" / "L5PCbiophys3.hoc"))
        h.load_file(str(self.SRC / "models" / "L5PCtemplate.hoc"))
        self._cell = h.L5PCtemplate(str(self.SRC / "morphologies" / "cell1.asc"))
        self._soma = self._cell.soma[0]
        self._sections = {
            "soma": list(self._cell.somatic),
            "apical": list(self._cell.apical),
            "basal": list(self._cell.basal),
            "axonal": list(self._cell.axonal),
            "all": list(self._cell.all),
        }
        object.__setattr__(self.manifest, "n_compartments",
                           sum(s.nseg for s in self._cell.all))

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "bbp_l5pc"
        C = EvidenceTier.C
        # Bounds are expressed as scale factors around the canonical value; the
        # ranges reproduce the span of BBP's published optimisation bounds
        # relative to the Hay canonical densities.
        return [
            _p(m, "gnata_soma", "NaTa_t", "gNaTa_tbar", "soma", "Nav", "fast_sodium",
               "S/cm2", 2.04, self.REF, lo=0.25, hi=2.0, tier=C),
            _p(m, "gskv3_soma", "SKv3_1", "gSKv3_1bar", "soma", "Kv3", "fast_rectifier",
               "S/cm2", 0.693, self.REF, lo=0.1, hi=4.0, tier=C),
            _p(m, "gkpst_soma", "K_Pst", "gK_Pstbar", "soma", "Kv2", "persistent_k",
               "S/cm2", 0.00223, self.REF, lo=0.05, hi=20.0, tier=C),
            _p(m, "gktst_soma", "K_Tst", "gK_Tstbar", "soma", "Kv4", "transient_k",
               "S/cm2", 0.0812, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gnap_soma", "Nap_Et2", "gNap_Et2bar", "soma", "Nav_persistent",
               "persistent_sodium", "S/cm2", 0.00172, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gsk_soma", "SK_E2", "gSK_E2bar", "soma", "SK", "calcium_activated_k",
               "S/cm2", 0.0441, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gcahva_soma", "Ca_HVA", "gCa_HVAbar", "soma", "Cav_HVA",
               "high_threshold_calcium", "S/cm2", 0.000992, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gcalva_soma", "Ca_LVAst", "gCa_LVAstbar", "soma", "Cav_LVA",
               "low_threshold_calcium", "S/cm2", 0.00343, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gih_apical", "Ih", "gIhbar", "apical", "HCN", "h_current",
               "S/cm2", 2e-4, self.REF, lo=0.1, hi=5.0, tier=C),
            _p(m, "gnata_apical", "NaTa_t", "gNaTa_tbar", "apical", "Nav",
               "dendritic_sodium", "S/cm2", 0.0213, self.REF, lo=0.05, hi=5.0, tier=C),
            _p(m, "gskv3_apical", "SKv3_1", "gSKv3_1bar", "apical", "Kv3",
               "dendritic_rectifier", "S/cm2", 0.000261, self.REF, lo=0.05, hi=20.0, tier=C),
            _p(m, "gim_apical", "Im", "gImbar", "apical", "Kv7_M", "adaptation",
               "S/cm2", 6.75e-5, self.REF, lo=0.05, hi=15.0, tier=C),
            _p(m, "gpas_all", "pas", "g", "all", "leak", "leak", "S/cm2",
               3.38e-5, self.REF, lo=0.3, hi=3.0, tier=C),
            # Apical Ca_LVAst carries the dendritic calcium "hot zone" (a
            # distance-dependent sigmoid centred at 685-885 um) that defines this
            # cell's BAC-firing behaviour; the scale factor preserves its shape.
            _p(m, "gcalva_apical", "Ca_LVAst", "gCa_LVAstbar", "apical", "Cav_LVA",
               "dendritic_calcium_hotzone", "S/cm2", 0.0187, self.REF,
               lo=0.05, hi=5.0, tier=C),
            _p(m, "gih_basal", "Ih", "gIhbar", "basal", "HCN", "basal_h_current",
               "S/cm2", 2e-4, self.REF, lo=0.1, hi=5.0, tier=C),
        ]


# --------------------------------------------------------------------------- #
# 6. Migliore et al. 1999 CA1 pyramidal cell (ModelDB 2796)
# --------------------------------------------------------------------------- #
class MiglioreCA1(NeuronAdapter):
    """Hippocampal CA1 pyramidal neuron with a distance-dependent A-current.

    The published model raises ``K_A`` density linearly with path distance from
    the soma and switches from the proximal (``kap``) to the distal (``kad``)
    kinetic variant beyond 100 um -- a spatial channel gradient that the
    scale-factor parameterisation preserves.
    """

    REF = "Migliore et al. 1999 J Comput Neurosci 7:5 (ModelDB 2796)"
    SRC = RAW / "modeldb_2796" / "ca1"
    manifest = ModelManifest(
        model_id="migliore_ca1", neuron_class="hippocampal_pyramidal", species="rat",
        brain_region="hippocampus_ca1", simulator="neuron", source_reference=REF,
        license="ModelDB terms; free for academic use", temperature_c=34.0,
        is_point_model=False, n_compartments=0, mechanism_dir="migliore_ca1",
        lab="migliore", pipeline="independent", v_init_mV=-65.0,
        notes="Distance-dependent K_A gradient; independent implementation.",
    )
    manifest_rheobase_hi = 2.0
    manifest_rheobase_tol = 0.002

    RM, RA_ALL, RA_AX, CM = 28000.0, 150.0, 50.0, 1.0
    GNA, GKDR, KMULT, KMULTP, AXONM = 0.032, 0.01, 0.048, 0.048, 2.0

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.celsius = 34.0
        before = section_names()
        h.xopen(str(self.SRC / "n160_mod.nrn"))
        own = _own_sections(before)

        soma_secs, axon_secs, dend_secs = [], [], []
        for sec in own:
            nm = sec.name()
            (soma_secs if "soma" in nm else
             axon_secs if "axon" in nm else dend_secs).append(sec)
        self._soma = soma_secs[len(soma_secs) // 2] if soma_secs else own[0]

        # Passive properties first -- the d_lambda rule below depends on Ra and cm.
        for sec in own:
            sec.Ra, sec.cm = self.RA_ALL, self.CM
        for sec in dend_secs:
            sec.cm = self.CM * 2.0
        for sec in axon_secs:
            sec.Ra = self.RA_AX

        # Spatial-convergence refinement.  The as-distributed segmentation
        # (559 segments) is NOT spatially converged: it yields 5 spikes to a
        # 0.5 nA step where both the d_lambda(100 Hz) mesh and a mesh twice that
        # fine yield 6.  We therefore run the model on the d_lambda mesh and
        # record the discrepancy in the model audit rather than silently
        # inheriting an under-resolved published mesh.
        # See data/provenance/numerical_qc.json (migliore_ca1 -> spatial).
        for sec in own:
            lam = 1e5 * (sec.diam / (4.0 * 3.141592653589793 * 100.0 * sec.Ra * sec.cm)) ** 0.5
            sec.nseg = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1

        for sec in own:
            sec.insert("pas")
            sec.insert("na3")
            sec.insert("kdr")
            for seg in sec:
                seg.pas.g = 1.0 / self.RM
                seg.pas.e = -65.0
                seg.na3.gbar = self.GNA
                seg.kdr.gkdrbar = self.GKDR
        for sec in dend_secs:
            for seg in sec:
                seg.pas.g = 2.0 / self.RM
        for sec in axon_secs:
            for seg in sec:
                seg.na3.gbar = self.GNA * self.AXONM

        # distance-dependent A-current: proximal kap, distal kad (Migliore 1999)
        h.distance(0, 0.5, sec=self._soma)
        prox, dist = [], []
        for sec in own:
            for seg in sec:
                d = h.distance(seg.x, sec=sec)
                (dist if d > 100.0 else prox).append((sec, seg, d))
        for sec_p in {s for s, _, _ in prox}:
            sec_p.insert("kap")
        for sec_d in {s for s, _, _ in dist}:
            sec_d.insert("kad")
        for _, seg, d in prox:
            seg.kap.gkabar = self.KMULTP * (1.0 + d / 100.0)
        for _, seg, d in dist:
            seg.kad.gkabar = self.KMULT * (1.0 + d / 100.0)

        for sec in own:
            if h.ismembrane("k_ion", sec=sec):
                sec.ek = -90.0
            if h.ismembrane("na_ion", sec=sec):
                sec.ena = 55.0

        self._sections = {
            "soma": soma_secs, "axonal": axon_secs, "dendritic": dend_secs,
            "kap_secs": sorted({s for s, _, _ in prox}, key=lambda s: s.name()),
            "kad_secs": sorted({s for s, _, _ in dist}, key=lambda s: s.name()),
            "all": own,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in own))

    @property
    def soma(self):
        return self._soma

    def _post_finitialize(self) -> bool:
        """Migliore's initial-state procedure (``proc init`` in fig_1a.hoc).

        The published model pins the resting potential at ``Vrest`` by solving
        for the leak reversal that exactly cancels the standing sodium and
        potassium currents in every segment::

            e_pas(x) = v(x) + (ina(x) + ik(x)) / g_pas(x)

        This has to be redone for every parameter sample, because the standing
        currents depend on the conductances being sampled.  Omitting it leaves
        the cell resting near -75 mV instead of the published -65 mV, and makes
        resting potential an artefact of the conductance draw rather than the
        model's own compensation.
        """
        from neuron import h

        vrest = self.manifest.v_init_mV
        for sec in self._sections["all"]:
            has_na = h.ismembrane("na_ion", sec=sec)
            has_k = h.ismembrane("k_ion", sec=sec)
            for seg in sec:
                if seg.pas.g <= 0:
                    continue
                if has_na or has_k:
                    i_ion = (seg.ina if has_na else 0.0) + (seg.ik if has_k else 0.0)
                    seg.pas.e = vrest + i_ion / seg.pas.g
                else:
                    seg.pas.e = vrest
        return True

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "migliore_ca1"
        return [
            _p(m, "gna_all", "na3", "gbar", "all", "Nav", "fast_sodium", "S/cm2",
               self.GNA, self.REF),
            _p(m, "gkdr_all", "kdr", "gkdrbar", "all", "Kv", "delayed_rectifier", "S/cm2",
               self.GKDR, self.REF),
            _p(m, "gka_prox", "kap", "gkabar", "kap_secs", "Kv4_A", "proximal_a_current",
               "S/cm2", self.KMULTP, self.REF),
            _p(m, "gka_dist", "kad", "gkabar", "kad_secs", "Kv4_A", "distal_a_current",
               "S/cm2", self.KMULT, self.REF),
            _p(m, "gna_axon", "na3", "gbar", "axonal", "Nav", "axonal_sodium", "S/cm2",
               self.GNA * self.AXONM, self.REF),
            _p(m, "gpas_all", "pas", "g", "all", "leak", "leak", "S/cm2",
               1.0 / self.RM, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance",
               "uF/cm2", self.CM, self.REF, lo=0.5, hi=2.0),
            _p(m, "ra_all", None, "Ra", "all", "passive", "axial_resistance", "ohm-cm",
               self.RA_ALL, self.REF, lo=0.4, hi=2.5),
        ]


# --------------------------------------------------------------------------- #
MODEL_REGISTRY: dict[str, type[NeuronAdapter]] = {
    "hh_point": HHPoint,
    "hh_cable": HHCable,
    "pospischil_rs": PospischilRS,
    "pospischil_fs": PospischilFS,
    "khaliq_purkinje": KhaliqPurkinje,
    "mainen_l5pc": MainenL5PC,
    "migliore_ca1": MiglioreCA1,
    "bbp_l5pc": BBPL5PC,
}


def get_adapter(model_id: str) -> NeuronAdapter:
    """Instantiate (but do not build) the adapter for ``model_id``."""
    try:
        cls = MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODEL_REGISTRY)}") from exc
    return cls()
