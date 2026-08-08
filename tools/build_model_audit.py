"""Generate docs/model_audits/model_audit.md (blueprint WP1 deliverable).

Compares each model's reproduced canonical behaviour against values stated in
the original publication or source code, and records the numerical evidence and
any deviation.  Reference values are curated here by hand with their source, so
the audit compares our construction against the *literature*, not against
itself.

Corrections forced by the 2026-08-06 audit (section 5.7)
--------------------------------------------------------
1. **Numerical evidence comes from the current convergence audit.**  This tool
   used to read ``data/provenance/numerical_qc.json``, the superseded first-pass
   sweep that judged one canonical point on spike count alone.  The generated
   document therefore said ``mainen_l5pc`` PASSed while
   ``convergence_audit_mainen_l5pc.json`` recorded ``chosen: null``, and said
   ``migliore_ca1`` accepts only 0.0125 ms while the feature-level audit accepts
   0.025 ms at every point.  It now reads ``convergence_audit_<model>.json``
   only; ``numerical_qc.json`` is cited as history, never as evidence.
2. **Numerical acceptance is part of the verdict.**  A model whose exact
   production setting is not an accepted setting of its own convergence audit
   can no longer be reported as a canonical PASS.  Gate N3 of the audit's
   release decision fails closed, and so does this document.
3. **Sections, segments and recording compartments are distinguished.**  The
   summary column used to be headed "compartments" while carrying NEURON's
   ``nseg`` total, which overstates the biological compartment count and mixes
   morphology with numerical discretisation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAN = ROOT / "results" / "manifests"
PROV = ROOT / "data" / "provenance"
OUT = ROOT / "docs" / "model_audits" / "model_audit.md"

# Published / source-code reference values, with where each one comes from.
REFERENCE = {
    "hh_point": {
        "V_rest_mV": (-65.0, "HH 1952 squid axon resting potential; NEURON hh defaults at 6.3 C"),
        "behaviour": ("single spike near threshold, repetitive firing above ~3x rheobase",
                      "classic HH subcritical Hopf behaviour"),
    },
    "hh_cable": {
        "V_rest_mV": (-65.0, "as hh_point; dendrite added, channels unchanged"),
        "behaviour": ("as hh_point, higher rheobase from added load", "cable loading"),
    },
    "pospischil_rs": {
        "V_rest_mV": (-70.0, "sPY_template: e_pas = -70"),
        "R_in_Mohm": (34.0, "sPY_template comment: 'g_pas = 0.0001  // Rin = 34 Meg'"),
        "behaviour": ("regular spiking with spike-frequency adaptation via I_M",
                      "Pospischil et al. 2008, RS class"),
    },
    "pospischil_fs": {
        "V_rest_mV": (-70.0, "sIN_template: e_pas = -70"),
        "R_in_Mohm": (48.0, "sIN_template comment: 'g_pas = 0.00015 // Rin = 48Meg'"),
        "behaviour": ("fast spiking, no adaptation, higher maximum rate than RS",
                      "Pospischil et al. 2008, FS class"),
    },
    "khaliq_purkinje": {
        "spontaneous_Hz": (37.0, "Khaliq et al. 2003: spontaneous firing ~30-50 Hz at 22 C"),
        "behaviour": ("spontaneously active; depolarisation block at high current",
                      "Khaliq, Gouwens & Raman 2003"),
    },
    "mainen_l5pc": {
        "V_rest_mV": (-70.0, "demofig1.hoc: v_init = -70, e_pas = v_init"),
        "behaviour": ("burst firing driven by dendritic Ca, axonal spike initiation",
                      "Mainen & Sejnowski 1996"),
    },
    "migliore_ca1": {
        "V_rest_mV": (-65.0, "fig_1a.hoc: Vrest = -65, enforced by the e_pas init procedure"),
        "behaviour": ("narrow firing range; depolarisation block above ~0.4 nA",
                      "Migliore et al. 1999 (built for dendritic propagation, not f-I)"),
    },
    "bbp_l5pc": {
        "V_rest_mV": (-80.0, "L5PCbiophys3.hoc: e_pas = -90 with Ih; init.hoc v_init = -80"),
        "behaviour": ("perisomatic step firing saturating near 10-20 Hz",
                      "Hay et al. 2011, Fig. 5 step-current protocol"),
    },
    # --- external replication cohort -------------------------------------- #
    "miyasho_purkinje": {
        "V_rest_mV": (-68.0, "purkinje.hoc: el_Leak = -80 with the full channel set; "
                             "Miyasho et al. 2001 report rest near -68 mV"),
        "behaviour": ("high-frequency Na spiking on somatic current with dendritic "
                      "Ca spikes; rates above 100 Hz at 2-3 nA",
                      "Miyasho et al. 2001, Fig. 2"),
    },
    "tzilivaki_fsbc": {
        "V_rest_mV": (-70.0, "tempSomogyi1.hoc: e_pas = v_init = -70"),
        "behaviour": ("fast spiking without adaptation; rates above 100 Hz at "
                      "modest current, non-saturating over the tested range",
                      "Tzilivaki et al. 2019, FS basket cell validation"),
    },
    "moubarak_snc_da": {
        "V_rest_mV": (-60.0, "cell_init*.hoc: v_init = -70 with pacemaking to a "
                             "trough near -60 mV"),
        "spontaneous_Hz": (5.0, "SNc dopaminergic pacemaking, 1-10 Hz in vitro "
                                "(Moubarak et al. 2022)"),
        "behaviour": ("spontaneous regular pacemaking, broad action potentials, "
                      "axon-bearing dendrite spike initiation",
                      "Moubarak et al. 2022"),
    },
    "dougalis_vlpag_da": {
        "V_rest_mV": (-60.0, "Dougalis et al. 2017: modelled resting potential "
                             "around -55 to -65 mV"),
        "behaviour": ("slow regular firing driven by I_h and I_NaP; depolarisation "
                      "block at modest current",
                      "Dougalis et al. 2017, Fig. 8"),
    },
    "ortner_snc_da": {
        "V_rest_mV": (-60.0, "Ortner et al. 2017 SNc single-compartment model"),
        "behaviour": ("slow SNc pacemaking, 1-8 Hz; L-type-dependent oscillation",
                      "Ortner et al. 2017"),
    },
    "tomko_ca1": {
        "V_rest_mV": (-65.0, "pyramidal_cell_weak_bAP_updated.hoc: v_init = -65"),
        "behaviour": ("regular spiking with adaptation; depolarisation block "
                      "above ~1.2 nA; weak bAP by construction",
                      "Tomko et al. 2021, HippoUnit somatic-features test"),
    },
}

TOL = {"V_rest_mV": 3.0, "R_in_Mohm": 0.20, "spontaneous_Hz": 0.35}  # abs mV, rel, rel


def verdict(key, ref, got):
    if got is None:
        return "not measured", "-"
    if key == "V_rest_mV":
        ok = abs(got - ref) <= TOL[key]
        return ("PASS" if ok else "DEVIATION"), f"{got - ref:+.1f} mV"
    rel = abs(got - ref) / ref if ref else float("inf")
    return ("PASS" if rel <= TOL[key] else "DEVIATION"), f"{100 * (got / ref - 1):+.0f}%"


def convergence_status(model_id: str, solver: dict, prod_nseg: int | None = None) -> dict:
    """Is the EXACT production setting an accepted setting of the current audit?

    This is gate N3.  A missing audit, a null ``chosen``, or a production
    ``dt``/mesh that has no directly tested row all fail -- the point of the gate
    is that acceptance may not be inferred, interpolated or assumed.

    Matching is on ``(dt, total_nseg)``, not on the audit's mesh *label*.  Three
    adapters (`mainen_l5pc`, `migliore_ca1`, `moubarak_snc_da`) apply a d_lambda
    rule inside ``_build``, so their "as-built" mesh *is* a d_lambda multiple:
    matching on the label alone reported `migliore_ca1` as untested when its
    production 1,106-segment mesh had in fact been audited and accepted under
    the name ``mesh1.0x``. The segment count is what the solver actually sees.
    """
    path = PROV / f"convergence_audit_{model_id}.json"
    if not path.exists():
        return {"status": "MISSING", "detail": f"no {path.name}", "accepted": False}
    d = json.loads(path.read_text())
    prod_dt = float(solver.get("dt_ms", float("nan")))
    settings = d.get("settings", {})

    def matches(s) -> bool:
        if abs(float(s.get("dt_ms", -1)) - prod_dt) >= 1e-12:
            return False
        if prod_nseg is not None and s.get("total_nseg") is not None:
            return int(s["total_nseg"]) == int(prod_nseg)
        return s.get("mesh_factor") in (None, "as_built")

    direct = [(k, s) for k, s in settings.items() if matches(s)]
    accepted_keys = sorted(k for k, s in settings.items() if s.get("accepted"))
    if not direct:
        return {
            "status": "UNTESTED",
            "detail": (f"production dt={prod_dt} ms on the as-built mesh has no direct "
                       f"row in the audit; tested: {sorted(settings)}"),
            "accepted": False, "chosen": d.get("chosen"),
            "accepted_settings": accepted_keys,
        }
    key, s = direct[0]
    failing = sorted(f.split(".")[-1] for f, v in
                     s.get("per_feature_agreement", {}).items() if v < 1.0)
    return {
        "status": "ACCEPTED" if s.get("accepted") else "NOT ACCEPTED",
        "detail": (f"{key}: {s['total_nseg']} segments, mean feature agreement "
                   f"{s['fraction_features_reproduced']:.3f}"
                   + (f", failing {failing}" if failing else "")),
        "accepted": bool(s.get("accepted")),
        "chosen": d.get("chosen"),
        "accepted_settings": accepted_keys,
        "reference": (f"dt={d['reference']['dt_ms']} ms, mesh "
                      f"{d['reference'].get('mesh_factor')}x, "
                      f"{d['reference'].get('total_nseg')} segments"),
        "n_points": d.get("n_points"),
    }


def main() -> int:
    from compass.adapters.replication import EXCLUDED_CANDIDATES

    downloads = json.loads((PROV / "downloads.json").read_text())
    builds = {r["model_id"]: r for r in json.loads((PROV / "mechanism_builds.json").read_text())}

    lines = [
        "# COMPASS model audit",
        "",
        "Canonical-reproduction audit for every model in the panel "
        "(blueprint WP1, §6.2/6.3).",
        "",
        "A model is included only if it executes reproducibly under a pinned "
        "simulator, has named mechanisms and a documented morphology or explicit "
        "point-model designation, reproduces the qualitative and (where stated) "
        "quantitative behaviour of its source, **and** runs its production "
        "campaign at a time step and mesh that its own feature-level convergence "
        "audit accepts. Every deviation found is recorded here rather than "
        "corrected silently.",
        "",
        "Numerical evidence is read from "
        "`data/provenance/convergence_audit_<model>.json`, the feature-level "
        "audit over prior samples. The older `numerical_qc.json` judged a single "
        "canonical point on spike count alone and is retained only as history; "
        "it is not consulted here.",
        "",
        "Counting convention: **sections** are morphological compartments as the "
        "published source defines them; **segments** (`nseg`) are the numerical "
        "discretisation COMPASS runs, which is a solver choice and not a "
        "biological quantity; voltage is recorded at one somatic segment.",
        "",
        "Generated by `tools/build_model_audit.py` from "
        "`results/manifests/canonical_*.json` and `data/provenance/`.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| model | cohort | lineage | class | sections | segments | params | "
        "canonical | numerical (N3) | overall |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    details = []
    failures = []
    for p in sorted(MAN.glob("canonical_*.json")):
        r = json.loads(p.read_text())
        mid = r["model_id"]
        m = r["manifest"]
        ref = REFERENCE.get(mid, {})
        got = {
            "V_rest_mV": r["resting_potential_mV"],
            "R_in_Mohm": r["canonical_features"].get("hyperpol_abs.input_resistance_Mohm"),
            "spontaneous_Hz": r["spontaneous_rate_Hz"],
        }
        checks = []
        for key, (val, src) in ref.items():
            if key == "behaviour":
                continue
            v, delta = verdict(key, val, got.get(key))
            checks.append((key, val, got.get(key), delta, v, src))
        canonical = ("PASS" if all(c[4] == "PASS" for c in checks if c[4] != "not measured")
                     else "PASS with deviation")
        conv = convergence_status(mid, r.get("solver", {}),
                                  prod_nseg=r.get("n_compartments"))
        overall = "ACCEPTED" if (conv["accepted"] and canonical == "PASS") else (
            "ACCEPTED with deviation" if conv["accepted"] else "**NOT ACCEPTED**")
        if not conv["accepted"]:
            failures.append((mid, conv))

        nmod = builds.get(mid, {}).get("n_mod", 0)
        n_sec = r.get("n_sections", m.get("n_sections", "-"))
        lines.append(
            f"| `{mid}` | {m.get('cohort', 'discovery')} | "
            f"`{m.get('source_lineage_id', '?')}` | {m['neuron_class']} | "
            f"{n_sec} | {r['n_compartments']} | {r['n_parameters']} | "
            f"{canonical} | {conv['status']} | {overall} |"
        )

        d = [f"### `{mid}`", "",
             f"* **Class / species / region:** {m['neuron_class']} / {m['species']} / {m['brain_region']}",
             f"* **Source:** {m['source_reference']}",
             f"* **Accession:** {m.get('source_accession', '-')}  •  "
             f"**Cohort:** {m.get('cohort', 'discovery')}",
             f"* **Source lineage / implementation / class id:** "
             f"`{m.get('source_lineage_id', '?')}` / `{m.get('implementation_id', '?')}` / "
             f"`{m.get('neuron_class_id', '?')}`",
             f"* **Lab / pipeline:** {m['lab']} / {m['pipeline']}",
             f"* **License:** {m['license']}",
             f"* **Temperature:** {m['temperature_c']} °C  •  "
             f"**Protocol coverage:** {m.get('protocol_coverage', 'full')}",
             f"* **Discretisation:** {n_sec} sections, {r['n_compartments']} segments "
             f"(`nseg` total)  •  **Free parameters:** {r['n_parameters']}  •  "
             f"{nmod} mod files",
             f"* **Mechanism library:** {builds.get(mid, {}).get('lib', 'NEURON built-ins')}"
             + (f" (sha256 {builds[mid]['sha256']})" if builds.get(mid, {}).get("sha256") else ""),
             f"* **Production solver:** `{r.get('solver')}`",
             f"* **Measured rheobase:** {r['rheobase_nA']} nA"
             f"  •  **V_rest:** {r['resting_potential_mV']:.2f} mV"
             f"  •  **spontaneous rate:** {r['spontaneous_rate_Hz']:.1f} Hz",
             ""]
        if checks:
            d += ["| quantity | published | reproduced | difference | verdict | source of reference |",
                  "|---|---|---|---|---|---|"]
            for key, val, g, delta, v, src in checks:
                gs = "-" if g is None else f"{g:.2f}"
                d.append(f"| {key} | {val} | {gs} | {delta} | **{v}** | {src} |")
            d.append("")
        if "behaviour" in ref:
            d += [f"**Expected qualitative behaviour:** {ref['behaviour'][0]} "
                  f"({ref['behaviour'][1]}).", ""]

        d.append(f"**Numerical convergence (gate N3): {conv['status']}.** {conv['detail']}")
        if conv.get("reference"):
            d.append(f" Reference setting: {conv['reference']}, "
                     f"{conv.get('n_points')} parameter points.")
        d.append(f" Audit `chosen`: `{conv.get('chosen')}`; accepted settings: "
                 f"{conv.get('accepted_settings') or 'none'}.")
        if not conv["accepted"]:
            d.append("")
            d.append("> This model does **not** satisfy gate N3 at its production "
                     "setting. It is excluded from primary cross-model inference; "
                     "see the sensitivity cascade in `docs/RESULTS.md`.")
        d.append("")
        details.append("\n".join(d))

    # Registered models that never reached a canonical manifest, because a gate
    # rejected them first. They must appear here, or the document silently
    # describes a smaller panel than the registry does.
    from compass.adapters.models import MODEL_REGISTRY
    from compass.analysis.cohorts import PRIMARY_EXCLUSIONS

    audited = {json.loads(p.read_text())["model_id"] for p in MAN.glob("canonical_*.json")}
    unaudited = sorted(set(MODEL_REGISTRY) - audited)
    if unaudited:
        lines += [
            "",
            "## Registered but not simulated",
            "",
            "These adapters exist, their mechanisms are compiled and their "
            "convergence audits ran, but a gate rejected them before any "
            "campaign. They are listed so the registry and this document "
            "describe the same panel.",
            "",
            "| model | lineage | class | gate | reason |",
            "|---|---|---|---|---|",
        ]
        for mid in unaudited:
            m = MODEL_REGISTRY[mid].manifest
            exc = PRIMARY_EXCLUSIONS.get(mid, {})
            lines.append(
                f"| `{mid}` | `{m.source_lineage_id}` | {m.neuron_class_id} | "
                f"{exc.get('gate', '-')} | {exc.get('reason', 'not evaluated')} |")

    lines += [
        "",
        "## Numerical acceptance",
        "",
    ]
    if failures:
        lines += [
            f"**{len(failures)} model(s) fail gate N3 at their production setting** "
            "and are therefore excluded from primary cross-model inference:",
            "",
        ]
        for mid, conv in failures:
            lines.append(f"* `{mid}` — {conv['status']}: {conv['detail']}")
        lines.append("")
    else:
        lines += ["Every model runs its campaign at a setting its own "
                  "feature-level convergence audit accepts.", ""]

    lines += [
        "## Deviations and declared limitations",
        "",
        "1. **`migliore_ca1` spatial discretisation.** The as-distributed mesh "
        "(559 segments) is not spatially converged: a 0.5 nA step gives 5 spikes "
        "there but 6 on both the d_lambda(100 Hz) mesh and a mesh twice as fine. "
        "The model is run on the d_lambda(100 Hz) mesh (1106 segments).",
        "2. **`migliore_ca1` time step.** The feature-level convergence audit "
        "accepts dt = 0.025 ms at every audited point, and that is what the "
        "campaign runs. (The superseded `numerical_qc.json` rejected 0.025 ms on "
        "an exact spike-count comparison at a single canonical point sitting on a "
        "4-versus-5-spike knife edge; that record is history, not evidence.)",
        "3. **`migliore_ca1` initial state.** The published model pins V_rest by "
        "solving `e_pas(x) = v(x) + (ina(x)+ik(x))/g_pas(x)` after `finitialize`. "
        "This is reproduced (`MiglioreCA1._post_finitialize`) and re-run for every "
        "parameter sample; without it the cell rests near -75 mV instead of -65 mV.",
        "4. **Spontaneously active models** (`khaliq_purkinje`, `moubarak_snc_da`) "
        "have rheobase 0, so the rheobase-normalised protocol family is undefined "
        "for them. Reported as a coverage limitation, not imputed.",
        "5. **`bbp_l5pc` axon is passive** in `L5PCbiophys3.hoc` (leak only), so no "
        "axonal conductance is sampled for this model.",
        "6. **`ortner_snc_da` partial protocol coverage.** Its maximum firing rate "
        "over the tested range is ~7.5 Hz, i.e. three spikes in the frozen 400 ms "
        "analysis window, so ISI-derived features sit at their definitional floor. "
        "It is reported in a labelled partial-coverage stratum and excluded from "
        "primary cross-model inference.",
        "7. **`ortner_snc_da` and `dougalis_vlpag_da` share a code lineage.** Both "
        "NEURON implementations name the same author and reuse the same `A.mod` / "
        "`H.mod`, so they are recorded as two implementations of `dougalis_2017`, "
        "not as two independent lineages.",
        "8. **`tomko_ca1` shares the Migliore mechanism lineage.** Morphology and "
        "optimisation are independent (Cutsuridis geometry, BluePyOpt against "
        "HippoUnit) but the channel kinetics come from Migliore et al. 2018, so it "
        "is a nested implementation of `migliore_1999`.",
        "",
        "## Models downloaded but excluded from the panel",
        "",
        "Exclusions are decided on the frozen technical gates of the 2026-08-06 "
        "audit §9.5 and were made before any information endpoint was computed "
        "for the candidate.",
        "",
        "| accession | intended slot | gate failed | detail |",
        "|---|---|---|---|",
        "| ModelDB 97747 | fast-spiking interneuron (replication) | 9.5.2 executes "
        "in the supported simulator | distributed as an XPPAUT `.ode` file, not an "
        "executable NEURON model. Its slot is filled by `tzilivaki_fsbc`. |",
    ]
    for _mid, rec in sorted(EXCLUDED_CANDIDATES.items()):
        lines.append(f"| {rec['accession']} | {rec['intended_slot']} | "
                     f"{rec['gate_failed']} | {rec['detail']} |")

    used_accessions = set()
    for p in MAN.glob("canonical_*.json"):
        acc = json.loads(p.read_text())["manifest"].get("source_accession", "")
        if acc.startswith("modeldb:"):
            used_accessions.add(int(acc.split(":")[1]))
    excluded_ids = {int(r["accession"].split(":")[1])
                    for r in EXCLUDED_CANDIDATES.values()}
    for d in downloads:
        if d["source"] != "modeldb":
            continue
        if d["id"] in used_accessions or d["id"] in excluded_ids or d["id"] == 97747:
            continue
        lines.append(f"| ModelDB {d['id']} | {d['slot']} | not evaluated | "
                     f"archived with provenance; deferred on simulation cost |")

    lines += ["", "---", "", "## Per-model detail", ""] + details

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    if failures:
        print(f"gate N3 FAILS for: {', '.join(m for m, _ in failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
