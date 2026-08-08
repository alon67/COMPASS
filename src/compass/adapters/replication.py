"""External replication cohort (2026-08-06 audit, section 9).

The discovery panel had eight ``model_id`` values but only **six independently
developed source lineages**, and several biological classes were represented by
exactly one lineage.  That is enough for a descriptive screening result and not
enough for any claim about generality or about class-specific modules
(audit section 9.1).

This module adds a second, separately versioned cohort.  It is deliberately
built for *replication structure* rather than breadth: every entry either turns
a singleton class into a replicated class or adds a nested implementation that
makes the lineage/implementation distinction measurable.

===================  ===================  ==========================  ===========
model_id             source_lineage_id    neuron_class_id             role
===================  ===================  ==========================  ===========
miyasho_purkinje     miyasho_2001         cerebellar_purkinje         2nd Purkinje lineage (with khaliq_2003)
tzilivaki_fsbc       tzilivaki_2019       cortical_interneuron_fs     2nd FS lineage (with pospischil_2008)
moubarak_snc_da      moubarak_2022        dopaminergic_snc            1st dopaminergic lineage
dougalis_vlpag_da    dougalis_2017        dopaminergic_midbrain       2nd dopaminergic lineage
ortner_snc_da        dougalis_2017        dopaminergic_snc            nested implementation of dougalis_2017
tomko_ca1            migliore_1999        hippocampal_pyramidal_ca1   nested implementation of migliore_1999
===================  ===================  ==========================  ===========

Four new lineages, two new nested implementations: ten lineages and fourteen
implementations in total.  Lineage attribution is deliberately **conservative**,
because the failure mode the audit identified is overstating independence:

* ``ortner_snc_da`` (ModelDB 2017402) and ``dougalis_vlpag_da`` (ModelDB 226010)
  model different cells in different regions and were published separately, but
  the NEURON code of both was written by the same author (the ``.hoc`` headers
  say so) and they share the ``A.mod``/``H.mod`` implementations.  They are one
  code lineage with two implementations, not two independent lineages.
* ``tomko_ca1`` (ModelDB 266901) re-optimised every conductance de novo with
  BluePyOpt against HippoUnit on a different morphology, but its channel
  kinetics are taken from the Migliore lineage, so it is filed as a nested
  implementation of ``migliore_1999`` rather than as an independent CA1 source.

Two further candidates were evaluated and **excluded before any information
endpoint was computed for them**, on the outcome-blind gate 2 of audit section
9.5 ("model executes in the supported simulator without project-specific
biological retuning"):

* ModelDB 20212 (Poirazi et al. 2003, CA1): the published biophysics is only
  expressible through the package's own ``ExperimentControl``/``defvar``
  framework, so adapting it would pull a large third-party experiment harness
  across the adapter boundary rather than restating the model.
* ModelDB 116835 (Diwakar et al. 2009, cerebellar granule): the canonical
  channel distribution is applied by ``Parametri.hoc``, a GUI panel builder; the
  cell template alone rests at -33 mV, which is not the published state.

Both exclusions are recorded in ``EXCLUDED_CANDIDATES`` so the panel's
composition stays auditable.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..registry.schema import EvidenceTier, ModelManifest, ParameterSpec, Transform
from .base import NeuronAdapter, _own_sections, section_names

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

D_WIDE = 4.0

# Candidates fetched, built and rejected on frozen technical gates, never on the
# sign or magnitude of an information result (audit section 9.5 point 8).
EXCLUDED_CANDIDATES = {
    "poirazi_ca1": {
        "accession": "modeldb:20212",
        "reference": "Poirazi, Brannon & Mel 2003 Neuron 37:989",
        "intended_slot": "independent CA1 pyramidal lineage",
        "gate_failed": "9.5.2 executes without importing third-party experiment scaffolding",
        "detail": "cell-setup.hoc expresses every conductance gradient through the "
                  "package's ExperimentControl.defvar() harness; NEURON raises "
                  "hocobj_call without that object graph.",
    },
    "diwakar_granule": {
        "accession": "modeldb:116835",
        "reference": "Diwakar et al. 2009 J Neurophysiol 101:519",
        "intended_slot": "cerebellar granule class",
        "gate_failed": "9.5.4 source canonical traces reproduce prespecified ranges",
        "detail": "channel distribution is applied by the GUI-only Parametri.hoc; the "
                  "bare Grc_Cell.hoc template rests at -32.7 mV instead of the "
                  "published -70 mV.",
    },
}


def _p(model_id, pid, mech, attr, comp, family, role, units, canonical, ref,
       lo=1.0 / D_WIDE, hi=D_WIDE, tier=EvidenceTier.D, transform=Transform.LOG10):
    return ParameterSpec(
        model_id=model_id, parameter_id=pid, mechanism_name=mech, attribute=attr,
        compartment=comp, channel_family=family, biophysical_role=role, units=units,
        canonical_value=canonical, lower_bound=lo, upper_bound=hi,
        evidence_tier=tier, source_reference=ref, transform=transform,
    )


class _HocCellAdapter(NeuronAdapter):
    """Shared machinery for adapters that load a published ``.hoc`` template.

    Holds a strong Python reference to the template object.  NEURON deletes a
    template instance's sections as soon as the last reference is dropped, and
    the sections then raise ``ReferenceError`` from any held Section handle --
    an easy way to lose an entire cell mid-campaign.
    """

    _cell = None

    def _chdir_source(self, path: Path):
        """Run a hoc load with the source tree as cwd, then restore.

        Several published packages ``xopen`` sibling files by relative path.
        Leaving the process cwd changed would then make every later relative
        path in the campaign resolve against a model's source tree.
        """
        prev = os.getcwd()
        os.chdir(str(path))
        return prev


# --------------------------------------------------------------------------- #
# 9. Miyasho et al. 2001 cerebellar Purkinje cell (ModelDB 17664)
# --------------------------------------------------------------------------- #
class MiyashoPurkinje(_HocCellAdapter):
    """Multicompartment Purkinje cell with the De Schutter-Bower channel set.

    The independent partner of :class:`~compass.adapters.models.KhaliqPurkinje`.
    Khaliq is a single-compartment soma built around resurgent sodium; Miyasho
    is a 1,088-section reconstruction whose dendrites carry the low-threshold
    K and Ca conductances the paper is about.  Same biological class, different
    laboratory, different channel implementations, different morphological
    commitment -- which is exactly what turns a singleton class into a
    replicated one.
    """

    REF = "Miyasho et al. 2001 Brain Res 891:106 (ModelDB 17664)"
    SRC = RAW / "modeldb_17664" / "prknj"
    manifest = ModelManifest(
        model_id="miyasho_purkinje", neuron_class="cerebellar_purkinje",
        species="guinea_pig", brain_region="cerebellum", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=37.0, is_point_model=False, n_compartments=0,
        mechanism_dir="miyasho_purkinje", lab="miyakawa", pipeline="independent",
        source_lineage_id="miyasho_2001", implementation_id="miyasho_shelton_morphology",
        neuron_class_id="cerebellar_purkinje", source_accession="modeldb:17664",
        cohort="replication", v_init_mV=-68.0,
        notes="Independent Purkinje lineage; replication partner of khaliq_purkinje.",
    )
    # Purkinje dendrites are electrically large: the canonical cell needs ~1.5 nA.
    manifest_rheobase_hi = 6.0
    manifest_rheobase_tol = 0.01

    # Canonical densities restated from purkinje.hoc for the audit table.
    SOMA_GNAF, SOMA_GCAP2, SOMA_GKH, SOMA_GL = 10.0, 0.0005, 0.0005, 0.00006
    DEND_GCAP2, DEND_GCAT, DEND_GCAE = 0.004, 0.0015, 0.008
    DEND_GKHH, DEND_GKM, DEND_GKA = 0.0006, 0.00001, 0.08
    DEND_GKD, DEND_GKC3, DEND_GL = 0.09, 0.06, 0.00021

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.celsius = 37.0
        before = section_names()
        prev = self._chdir_source(self.SRC)
        try:
            h.xopen(str(self.SRC / "purkinje.hoc"))
        finally:
            os.chdir(prev)
        own = _own_sections(before)

        self._soma = h.soma
        smooth, spiny = [], []
        for sec in own:
            nm = sec.name()
            if "SmoothDendrite" in nm:
                smooth.append(sec)
            elif "SpinyDendrite" in nm:
                spiny.append(sec)
        dendritic = smooth + spiny
        self._sections = {
            "soma": [self._soma],
            "smooth": smooth,
            "spiny": spiny,
            "dendritic": dendritic,
            "all": own,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in own))

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "miyasho_purkinje"
        return [
            _p(m, "gna_soma", "NaF", "gnabar", "soma", "Nav", "fast_sodium",
               "mho/cm2", self.SOMA_GNAF, self.REF),
            _p(m, "gcap_soma", "CaP2", "gcabar", "soma", "Cav_P", "p_type_calcium",
               "mho/cm2", self.SOMA_GCAP2, self.REF),
            _p(m, "gkh_soma", "Kh", "gkbar", "soma", "HCN", "h_current",
               "mho/cm2", self.SOMA_GKH, self.REF),
            _p(m, "gleak_all", "Leak", "gl", "all", "leak", "leak",
               "mho/cm2", self.DEND_GL, self.REF),
            # Dendritic conductances: the low-threshold K and Ca currents the
            # paper identifies as the regulators of dendritic Ca spikes.
            _p(m, "gcap_dend", "CaP2", "gcabar", "dendritic", "Cav_P",
               "dendritic_p_calcium", "mho/cm2", self.DEND_GCAP2, self.REF),
            _p(m, "gcat_dend", "CaT", "gcabar", "dendritic", "Cav_T",
               "low_threshold_calcium", "mho/cm2", self.DEND_GCAT, self.REF),
            _p(m, "gcae_dend", "CaE", "gcabar", "dendritic", "Cav_E",
               "e_type_calcium", "mho/cm2", self.DEND_GCAE, self.REF),
            _p(m, "gka_dend", "KA", "gkbar", "dendritic", "Kv4_A",
               "a_type_potassium", "mho/cm2", self.DEND_GKA, self.REF),
            _p(m, "gkd_dend", "KD", "gkbar", "dendritic", "Kv1",
               "low_threshold_potassium", "mho/cm2", self.DEND_GKD, self.REF),
            _p(m, "gkc3_dend", "KC3", "gkbar", "dendritic", "BK",
               "calcium_activated_k", "mho/cm2", self.DEND_GKC3, self.REF),
            _p(m, "gkm_dend", "KM", "gkbar", "dendritic", "Kv7_M",
               "adaptation", "mho/cm2", self.DEND_GKM, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance",
               "uF/cm2", 0.8, self.REF, lo=0.5, hi=2.0),
        ]


# --------------------------------------------------------------------------- #
# 10. Tzilivaki et al. 2019 fast-spiking basket cell (ModelDB 237595)
# --------------------------------------------------------------------------- #
class TzilivakiFSBC(_HocCellAdapter):
    """Reconstructed fast-spiking basket cell with a full axonal arbor.

    The independent partner of
    :class:`~compass.adapters.models.PospischilFS`.  Pospischil FS is a
    single-compartment reduced template; this is a morphologically detailed
    hippocampal basket cell with 879 axonal sections.  Section 9.4 of the audit
    forbids using the XPPAUT-only Golomb model as the FS replicate, so this is
    the executable NEURON FS lineage it asks for.
    """

    REF = "Tzilivaki, Kastellakis & Poirazi 2019 Nat Commun 10:3664 (ModelDB 237595)"
    SRC = (RAW / "modeldb_237595" / "TzilivakiEtal_FSBCs_model"
           / "Multicompartmental_Biophysical_models")
    manifest = ModelManifest(
        model_id="tzilivaki_fsbc", neuron_class="hippocampal_interneuron_fs",
        species="rat", brain_region="hippocampus_ca3", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=34.0, is_point_model=False, n_compartments=0,
        mechanism_dir="tzilivaki_fsbc", lab="poirazi", pipeline="independent",
        source_lineage_id="tzilivaki_2019", implementation_id="tzilivaki_somogyi_1",
        neuron_class_id="cortical_interneuron_fs", source_accession="modeldb:237595",
        cohort="replication", v_init_mV=-70.0,
        notes="Independent fast-spiking lineage; replication partner of pospischil_fs.",
    )
    manifest_rheobase_hi = 1.0
    manifest_rheobase_tol = 0.004

    # Canonical densities restated from experiment/tempSomogyi1.hoc.
    SOMA_NAF = 0.045 * 0.8 * 11
    SOMA_KDR = 0.018 * 2 * 1.2
    SOMA_KSLOW, SOMA_H, SOMA_KAP = 0.000725, 0.00001, 0.0032
    SOMA_KCT, SOMA_KCA = 0.0001, 0.02
    G_PAS_SOMA, RA, CM = 1.0 / 7600.0, 172.0, 1.2

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.celsius = 34.0
        h("v_init = -70")
        before = section_names()
        prev = self._chdir_source(self.SRC)
        try:
            h.xopen("experiment/tempSomogyi1.hoc")
            self._cell = h.FScell("experiment/Somogyi_1.hoc")
        finally:
            os.chdir(prev)
        own = _own_sections(before)

        self._soma = self._cell.soma[0]
        somatic = list(self._cell.somatic)
        axonal = list(self._cell.axonal)
        prox = list(self._cell.basal_prox)
        dist = list(self._cell.basal_dist)
        self._sections = {
            "soma": somatic,
            "axonal": axonal,
            "basal_prox": prox,
            "basal_dist": dist,
            "dendritic": prox + dist,
            "all": own,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in own))

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "tzilivaki_fsbc"
        return [
            _p(m, "gnaf_soma", "Nafx", "gnafbar", "soma", "Nav", "fast_sodium",
               "S/cm2", self.SOMA_NAF, self.REF),
            _p(m, "gkdr_soma", "kdrin", "gkdrbar", "soma", "Kv3",
               "fast_rectifier", "S/cm2", self.SOMA_KDR, self.REF),
            _p(m, "gkslow_soma", "IKsin", "gKsbar", "soma", "Kv7_M",
               "slow_potassium", "S/cm2", self.SOMA_KSLOW, self.REF),
            _p(m, "gka_soma", "kapin", "gkabar", "soma", "Kv4_A",
               "a_type_potassium", "S/cm2", self.SOMA_KAP, self.REF),
            _p(m, "gkct_soma", "kctin", "gkcbar", "soma", "BK",
               "calcium_activated_k", "S/cm2", self.SOMA_KCT, self.REF),
            _p(m, "gkca_soma", "kcain", "gbar", "soma", "SK",
               "slow_calcium_activated_k", "S/cm2", self.SOMA_KCA, self.REF),
            _p(m, "gh_soma", "hin", "gbar", "soma", "HCN", "h_current",
               "S/cm2", self.SOMA_H, self.REF),
            _p(m, "gnaf_axon", "Nafx", "gnafbar", "axonal", "Nav",
               "axonal_sodium", "S/cm2", 0.045 * 0.8 * 11, self.REF),
            _p(m, "gkdr_axon", "kdrin", "gkdrbar", "axonal", "Kv3",
               "axonal_rectifier", "S/cm2", self.SOMA_KDR, self.REF),
            _p(m, "gnaf_dend", "Nafx", "gnafbar", "dendritic", "Nav",
               "dendritic_sodium", "S/cm2", 0.045, self.REF),
            _p(m, "gka_dend", "kapin", "gkabar", "dendritic", "Kv4_A",
               "dendritic_a_current", "S/cm2", self.SOMA_KAP, self.REF),
            _p(m, "gpas_all", "pas", "g", "all", "leak", "leak",
               "S/cm2", self.G_PAS_SOMA, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance",
               "uF/cm2", self.CM, self.REF, lo=0.5, hi=2.0),
            _p(m, "ra_all", None, "Ra", "all", "passive", "axial_resistance",
               "ohm-cm", self.RA, self.REF, lo=0.4, hi=2.5),
        ]


# --------------------------------------------------------------------------- #
# 11. Moubarak et al. 2022 substantia nigra dopaminergic neuron (ModelDB 267306)
# --------------------------------------------------------------------------- #
class MoubarakSNcDA(_HocCellAdapter):
    """Reconstructed SNc dopaminergic neuron with an axon-bearing dendrite.

    The first dopaminergic lineage in the panel.  The published package
    contains 40+ reconstructed cells from one study; audit section 9.4 says not
    to count multiple morphologies from one paper as independent lineages, so
    exactly one representative cell is adapted and the rest are left as an
    available within-lineage morphology series.
    """

    REF = "Moubarak et al. 2022 J Neurosci 42:6178 (ModelDB 267306)"
    SRC = RAW / "modeldb_267306" / "MODELE_ DA_NEURON"
    CELL = "04_02_2014_C2"
    manifest = ModelManifest(
        model_id="moubarak_snc_da", neuron_class="dopaminergic_snc",
        species="mouse", brain_region="substantia_nigra", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=35.0, is_point_model=False, n_compartments=0,
        mechanism_dir="moubarak_snc_da", lab="goaillard", pipeline="independent",
        source_lineage_id="moubarak_2022", implementation_id="moubarak_cell_04022014c2",
        neuron_class_id="dopaminergic_snc", source_accession="modeldb:267306",
        cohort="replication", v_init_mV=-60.0,
        notes="Spontaneously active pacemaker; first dopaminergic lineage in the panel.",
    )
    manifest_rheobase_hi = 1.0
    manifest_rheobase_tol = 0.004

    G_PAS, E_PAS, RA, CM = 1e-5, -50.0, 150.0, 0.75
    GKCA, GNA12, GKDR, GKA, GIH, GCAV13 = 0.1, 50.0, 100.0, 10.0, 3.0, 1.25
    GNA12_AIS, GKDR_AIS = 4000.0, 4000.0
    GKA_SOMA, GKDR_SOMA = 50.0, 150.0

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.celsius = 35.0
        before = section_names()
        prev = self._chdir_source(self.SRC)
        try:
            h.xopen(str(self.SRC / f"{self.CELL}.hoc"))
        finally:
            os.chdir(prev)
        own = _own_sections(before)

        # Spatial-convergence refinement.  The published segmentation uses
        # `lambda_f(1000)` (254 segments); the feature-level convergence audit
        # rejects it at every time step tested -- input resistance never
        # reproduces the reference, because the hyperpolarising step is measured
        # on an ongoing pacemaker train and the standing current depends on the
        # mesh.  The d_lambda(100 Hz) mesh (174 segments) at dt = 0.003125 ms
        # reproduces all 13 conditioning features on all 7 audit points.
        # Applied after Ra/cm are set by the source hoc, because the electrotonic
        # length constant depends on both.
        # See data/provenance/convergence_audit_moubarak_snc_da.json.
        for sec in own:
            lam = 1e5 * (sec.diam / (4.0 * 3.141592653589793 * 100.0
                                     * sec.Ra * sec.cm)) ** 0.5
            sec.nseg = max(1, int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1)

        self._soma = h.soma
        ais = [s for s in own if s.name().startswith("AIS")]
        axon = [s for s in own if s.name().split("[")[0] == "axon"]
        # "all_biophys" in the source is everything that carries the full
        # dendritic channel set: not the soma, not the AIS, not the bare axon.
        special = {id(self._soma)} | {id(s) for s in ais} | {id(s) for s in axon}
        dendritic = [s for s in own if id(s) not in special]
        self._sections = {
            "soma": [self._soma],
            "ais": ais,
            "axon": axon,
            "dendritic": dendritic,
            "axonal": ais + axon,
            "all": own,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in own))

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "moubarak_snc_da"
        return [
            _p(m, "gna_soma", "Na12", "gbar", "soma", "Nav", "fast_sodium",
               "1", self.GNA12, self.REF),
            _p(m, "gkdr_soma", "kdrDA", "gbar", "soma", "Kv", "delayed_rectifier",
               "1", self.GKDR_SOMA, self.REF),
            _p(m, "gka_soma", "kaDasoma", "gbar", "soma", "Kv4_A",
               "a_type_potassium", "1", self.GKA_SOMA, self.REF),
            _p(m, "gkca_soma", "kca", "gbar", "soma", "SK",
               "calcium_activated_k", "1", self.GKCA, self.REF),
            _p(m, "gih_soma", "Ih", "gbar", "soma", "HCN", "h_current",
               "1", self.GIH, self.REF),
            _p(m, "gcav13_soma", "CAV13", "gbar", "soma", "Cav_L",
               "l_type_calcium_pacemaker", "1", self.GCAV13, self.REF),
            _p(m, "gna_dend", "Na12", "gbar", "dendritic", "Nav",
               "dendritic_sodium", "1", self.GNA12, self.REF),
            _p(m, "gkdr_dend", "kdrDA", "gbar", "dendritic", "Kv",
               "dendritic_rectifier", "1", self.GKDR, self.REF),
            _p(m, "gka_dend", "kaDa", "gbar", "dendritic", "Kv4_A",
               "dendritic_a_current", "1", self.GKA, self.REF),
            _p(m, "gcav13_dend", "CAV13", "gbar", "dendritic", "Cav_L",
               "dendritic_l_calcium", "1", self.GCAV13, self.REF),
            _p(m, "gna_ais", "Na12", "gbar", "ais", "Nav", "axonal_sodium",
               "1", self.GNA12_AIS, self.REF),
            _p(m, "gkdr_ais", "kdrDA", "gbar", "ais", "Kv", "axonal_rectifier",
               "1", self.GKDR_AIS, self.REF),
            _p(m, "gpas_all", "pasnts", "g", "all", "leak", "leak",
               "S/cm2", self.G_PAS, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive", "membrane_capacitance",
               "uF/cm2", self.CM, self.REF, lo=0.5, hi=2.0),
        ]


# --------------------------------------------------------------------------- #
# 12/13. Dougalis-lineage single-compartment dopamine neurons
# --------------------------------------------------------------------------- #
class _DougalisLineageBase(NeuronAdapter):
    """Single-compartment midbrain dopamine neurons from one code lineage.

    Both packages declare the same author in their ``.hoc`` headers and reuse
    the same ``A.mod``/``H.mod`` implementations, so they are recorded as two
    implementations of ``dougalis_2017`` rather than as two lineages.  They are
    still worth having as separate entries: they are different cells in
    different regions, which is exactly the nested
    class-within-lineage structure the audit's hierarchical design needs.
    """

    _mechs: tuple[str, ...] = ()
    _diam = 15.0
    _ra, _cm = 35.4, 1.0

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdrun.hoc")
        h.celsius = self.manifest.temperature_c
        s = h.Section(name=f"soma_{self.manifest.model_id}")
        s.L = s.diam = self._diam
        s.nseg, s.Ra, s.cm = 1, self._ra, self._cm
        for mech in self._mechs:
            s.insert(mech)
        self._soma = s
        self._sections = {"soma": [s]}

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]


class DougalisVlpagDA(_DougalisLineageBase):
    """vlPAG/DRN dopamine neuron (Dougalis et al. 2017, ModelDB 226010)."""

    REF = "Dougalis et al. 2017 J Comput Neurosci 42:275 (ModelDB 226010)"
    _mechs = ("NaT", "kDR", "leak", "kA", "H", "caHVA", "caLVA", "kM", "NaP")
    _diam = 15.0
    manifest = ModelManifest(
        model_id="dougalis_vlpag_da", neuron_class="dopaminergic_vlpag_drn",
        species="mouse", brain_region="periaqueductal_gray", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=32.0, is_point_model=True, n_compartments=1,
        mechanism_dir="dougalis_vlpag_da", lab="dougalis_ungless",
        pipeline="independent",
        source_lineage_id="dougalis_2017", implementation_id="dougalis_vlpag_drn",
        neuron_class_id="dopaminergic_midbrain", source_accession="modeldb:226010",
        cohort="replication", v_init_mV=-60.0,
        notes="Voltage-clamp-constrained midbrain DA neuron; narrow firing range.",
    )
    manifest_rheobase_hi = 0.5
    manifest_rheobase_tol = 0.002

    def parameter_specs(self):
        m = "dougalis_vlpag_da"
        return [
            _p(m, "gnat_soma", "NaT", "gnaTbar", "soma", "Nav", "fast_sodium",
               "S/cm2", 0.02, self.REF),
            _p(m, "gnap_soma", "NaP", "gnaPbar", "soma", "Nav_persistent",
               "persistent_sodium", "S/cm2", 2e-5, self.REF),
            _p(m, "gkdr_soma", "kDR", "gkDRbar", "soma", "Kv",
               "delayed_rectifier", "S/cm2", 0.003, self.REF),
            _p(m, "gka_soma", "kA", "gkAbar", "soma", "Kv4_A",
               "a_type_potassium", "S/cm2", 0.006, self.REF),
            _p(m, "gkm_soma", "kM", "gkMbar", "soma", "Kv7_M", "adaptation",
               "S/cm2", 0.001, self.REF),
            _p(m, "gh_soma", "H", "ghbar", "soma", "HCN", "h_current",
               "S/cm2", 8e-5, self.REF),
            _p(m, "gcahva_soma", "caHVA", "gcaHVAbar", "soma", "Cav_HVA",
               "high_threshold_calcium", "S/cm2", 4e-5, self.REF),
            _p(m, "gcalva_soma", "caLVA", "gcaLVAbar", "soma", "Cav_LVA",
               "low_threshold_calcium", "S/cm2", 4e-5, self.REF),
            _p(m, "gleak_soma", "leak", "gleakbar", "soma", "leak", "leak",
               "S/cm2", 2e-5, self.REF),
            _p(m, "cm_soma", None, "cm", "soma", "passive",
               "membrane_capacitance", "uF/cm2", 1.0, self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_soma", None, "diam", "soma", "morphology",
               "somatic_area", "um", 15.0, self.REF, lo=0.6, hi=1.6),
        ]


class OrtnerSNcDA(_DougalisLineageBase):
    """SNc dopamine neuron (Ortner et al. 2017, ModelDB 2017402).

    Nested implementation of the ``dougalis_2017`` code lineage.  The three
    L-type variants ``cal2``/``cal3s``/``cal3l`` ship with ``gbar = 0`` and are
    therefore **not** exposed as parameters: a multiplicative scale on a zero
    canonical value can never change the model, which is precisely the inactive
    parameter that gate N4 of the audit exists to catch.
    """

    REF = "Ortner et al. 2017 Front Cell Neurosci 11:410 (ModelDB 2017402)"
    _mechs = ("HHHmod", "A", "H", "gkca", "cabalanmod", "capumpmod",
              "calcha", "calchaT", "cal2", "cal3s", "cal3l")
    _diam = 30.0
    manifest = ModelManifest(
        model_id="ortner_snc_da", neuron_class="dopaminergic_snc",
        species="mouse", brain_region="substantia_nigra", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=32.0, is_point_model=True, n_compartments=1,
        mechanism_dir="ortner_snc_da", lab="dougalis_striessnig",
        pipeline="independent",
        source_lineage_id="dougalis_2017", implementation_id="ortner_snc_single",
        neuron_class_id="dopaminergic_snc", source_accession="modeldb:2017402",
        parent_lineage="", cohort="replication", protocol_coverage="partial",
        v_init_mV=-60.0,
        notes="Nested implementation of dougalis_2017: same author's NEURON code, "
              "different cell and region. NOT an independent lineage. Maximum "
              "firing rate ~7.5 Hz gives <=3 spikes in the frozen 400 ms window, "
              "so ISI-derived features sit at their definitional floor: partial "
              "protocol coverage, excluded from primary inference.",
    )
    manifest_rheobase_hi = 0.6
    manifest_rheobase_tol = 0.002

    def parameter_specs(self):
        m = "ortner_snc_da"
        return [
            _p(m, "gna_soma", "HHHmod", "gnabar", "soma", "Nav", "fast_sodium",
               "S/cm2", 0.050, self.REF),
            _p(m, "gk_soma", "HHHmod", "gkbar", "soma", "Kv",
               "delayed_rectifier", "S/cm2", 0.01, self.REF),
            _p(m, "gleak_soma", "HHHmod", "glbar", "soma", "leak", "leak",
               "S/cm2", 8e-5, self.REF),
            _p(m, "gka_soma", "A", "gkAbar", "soma", "Kv4_A",
               "a_type_potassium", "S/cm2", 0.01, self.REF),
            _p(m, "gh_soma", "H", "ghbar", "soma", "HCN", "h_current",
               "S/cm2", 2e-4, self.REF),
            _p(m, "gkca_soma", "gkca", "gkcabar", "soma", "SK",
               "calcium_activated_k", "S/cm2", 4e-4, self.REF),
            _p(m, "gcan_soma", "calcha", "gcanbar", "soma", "Cav_N",
               "n_type_calcium", "S/cm2", 5.0e-5, self.REF),
            _p(m, "gcar_soma", "calcha", "gcarbar", "soma", "Cav_R",
               "r_type_calcium", "S/cm2", 5.0e-5, self.REF),
            _p(m, "gcat_soma", "calchaT", "gcatbar", "soma", "Cav_T",
               "low_threshold_calcium", "S/cm2", 7.5e-5, self.REF),
            _p(m, "cm_soma", None, "cm", "soma", "passive",
               "membrane_capacitance", "uF/cm2", 1.0, self.REF, lo=0.5, hi=2.0),
            _p(m, "diam_soma", None, "diam", "soma", "morphology",
               "somatic_area", "um", 30.0, self.REF, lo=0.6, hi=1.6),
        ]


# --------------------------------------------------------------------------- #
# 14. Tomko et al. 2021 reduced CA1 pyramidal cell (ModelDB 266901)
# --------------------------------------------------------------------------- #
class TomkoCA1(_HocCellAdapter):
    """Reduced-morphology CA1 pyramidal cell, re-optimised against HippoUnit.

    Filed as a **nested implementation of the Migliore lineage**, not as an
    independent CA1 source: the morphology and optimisation are new (Cutsuridis
    geometry, BluePyOpt against the HippoUnit validation suite) but the channel
    kinetics come from Migliore et al. 2018.  Pairing it with
    :class:`~compass.adapters.models.MiglioreCA1` measures
    implementation-within-lineage variation, which is the term the nested
    cross-model model needs and which no other pair in the panel supplies
    cleanly.
    """

    REF = ("Tomko, Benuskova & Jedlicka 2021 Front Neuroinform 15:630607 "
           "(ModelDB 266901); kinetics from Migliore et al. 2018")
    SRC = RAW / "modeldb_266901" / "TomkoEtAl2021" / "TomkoEtAl2021"
    manifest = ModelManifest(
        model_id="tomko_ca1", neuron_class="hippocampal_pyramidal",
        species="rat", brain_region="hippocampus_ca1", simulator="neuron",
        source_reference=REF, license="ModelDB terms; free for academic use",
        temperature_c=35.0, is_point_model=False, n_compartments=0,
        mechanism_dir="tomko_ca1", lab="jedlicka", pipeline="independent",
        source_lineage_id="migliore_1999", implementation_id="tomko_reduced_weak_bap",
        neuron_class_id="hippocampal_pyramidal_ca1", source_accession="modeldb:266901",
        cohort="replication", v_init_mV=-65.0,
        notes="Nested implementation of migliore_1999: independent morphology and "
              "de novo optimisation, shared channel kinetics. NOT an independent lineage.",
    )
    manifest_rheobase_hi = 2.0
    manifest_rheobase_tol = 0.004

    GKA_SOMA, GKM_SOMA, GKDR_SOMA, GNA_SOMA = 0.0075, 0.001, 0.0015, 0.035
    GCAL_SOMA, GCAN_SOMA, GCAT_SOMA = 0.0005, 2.2618914062501833e-06, 0.00005
    GKCA_SOMA, GCAGK_SOMA = 0.0015, 4.4820097108998517e-05
    GPAS_SOMA, RA_SOMA = 9.031387191839301e-05, 115.3957607556371
    GNA_AXON, GKDR_AXON, GKM_AXON, GKA_AXON = (
        0.035, 0.011664045469379856, 0.026473888790212396, 0.1636942175250268)
    GNA_DEND, GKDR_DEND = 0.038280628170345957, 0.0043036502438625682
    GHD, GKAD = 1.9042409723832741e-05, 0.012921529390557651

    def _build(self) -> None:
        from neuron import h

        h.load_file("stdlib.hoc")
        h.load_file("stdrun.hoc")
        h.celsius = 35.0
        before = section_names()
        prev = self._chdir_source(self.SRC)
        try:
            h.load_file(str(self.SRC / "pyramidal_cell_weak_bAP_updated.hoc"))
            self._cell = h.CA1_PC_Tomko()
        finally:
            os.chdir(prev)
        own = _own_sections(before)

        self._soma = self._cell.soma[0]
        self._sections = {
            "soma": list(self._cell.somatic),
            "axonal": list(self._cell.axonal),
            "apical": list(self._cell.apical),
            "basal": list(self._cell.basal),
            "trunk": list(self._cell.trunk_sec_list),
            "dendritic": list(self._cell.apical) + list(self._cell.basal),
            "all": own,
        }
        object.__setattr__(self.manifest, "n_compartments", sum(s.nseg for s in own))

    @property
    def soma(self):
        return self._soma

    def _targets(self, spec):
        return self._sections[spec.compartment]

    def parameter_specs(self):
        m = "tomko_ca1"
        return [
            _p(m, "gna_soma", "nax", "gbar", "soma", "Nav", "fast_sodium",
               "S/cm2", self.GNA_SOMA, self.REF),
            _p(m, "gkdr_soma", "kdr", "gkdrbar", "soma", "Kv",
               "delayed_rectifier", "S/cm2", self.GKDR_SOMA, self.REF),
            _p(m, "gka_soma", "kap", "gkabar", "soma", "Kv4_A",
               "proximal_a_current", "S/cm2", self.GKA_SOMA, self.REF),
            _p(m, "gkm_soma", "kmb", "gbar", "soma", "Kv7_M", "adaptation",
               "S/cm2", self.GKM_SOMA, self.REF),
            _p(m, "gkca_soma", "kca", "gbar", "soma", "SK",
               "calcium_activated_k", "S/cm2", self.GKCA_SOMA, self.REF),
            _p(m, "gcal_soma", "cal", "gcalbar", "soma", "Cav_L",
               "l_type_calcium", "S/cm2", self.GCAL_SOMA, self.REF),
            _p(m, "gcat_soma", "cat", "gcatbar", "soma", "Cav_T",
               "low_threshold_calcium", "S/cm2", self.GCAT_SOMA, self.REF),
            _p(m, "gna_axon", "nax", "gbar", "axonal", "Nav", "axonal_sodium",
               "S/cm2", self.GNA_AXON, self.REF),
            _p(m, "gkm_axon", "kmb", "gbar", "axonal", "Kv7_M",
               "axonal_adaptation", "S/cm2", self.GKM_AXON, self.REF),
            _p(m, "gka_axon", "kap", "gkabar", "axonal", "Kv4_A",
               "axonal_a_current", "S/cm2", self.GKA_AXON, self.REF),
            _p(m, "gna_dend", "nax", "gbar", "dendritic", "Nav",
               "dendritic_sodium", "S/cm2", self.GNA_DEND, self.REF),
            _p(m, "gkdr_dend", "kdr", "gkdrbar", "dendritic", "Kv",
               "dendritic_rectifier", "S/cm2", self.GKDR_DEND, self.REF),
            # Distance-dependent A-current and h-current gradients: the scale
            # factor multiplies the fitted sigmoid, preserving its shape.
            _p(m, "gkad_dend", "kad", "gkabar", "dendritic", "Kv4_A",
               "distal_a_current_gradient", "S/cm2", self.GKAD, self.REF),
            _p(m, "ghd_all", "hd", "ghdbar", "all", "HCN",
               "h_current_gradient", "S/cm2", self.GHD, self.REF),
            _p(m, "gpas_all", "pas", "g", "all", "leak", "leak",
               "S/cm2", self.GPAS_SOMA, self.REF),
            _p(m, "cm_all", None, "cm", "all", "passive",
               "membrane_capacitance", "uF/cm2", 1.0, self.REF, lo=0.5, hi=2.0),
        ]


REPLICATION_REGISTRY: dict[str, type[NeuronAdapter]] = {
    "miyasho_purkinje": MiyashoPurkinje,
    "tzilivaki_fsbc": TzilivakiFSBC,
    "moubarak_snc_da": MoubarakSNcDA,
    "dougalis_vlpag_da": DougalisVlpagDA,
    "ortner_snc_da": OrtnerSNcDA,
    "tomko_ca1": TomkoCA1,
}
