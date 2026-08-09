"""Who is in the primary cohort, and at what level inference is clustered.

Two separate corrections from the 2026-08-06 audit live here, because both are
about *which rows may speak* and both were previously implicit.

1. Numerical exclusion (audit finding 3, release rule 7)
--------------------------------------------------------
``mainen_l5pc`` fails gate N3: at its exact production setting (dt = 0.0125 ms,
2,304 segments) it reproduces 12 of the 13 conditioning features on all seven
audit points, but fails ``adaptation_index`` on one.  The previous revision
characterised that accurately and then kept the model in every headline number.
The audit is explicit that this is not allowed:

    An exception requires exclusion of the affected model/data from primary
    inference and regeneration of dependent artifacts; it does not convert the
    gate to a pass.

So the exclusion is prespecified here, applied by every analysis that reads
:func:`primary_mask`, and paired with a **sensitivity cascade** (see
:data:`SENSITIVITY_ARMS`) that reports what changes if the model is retained,
and what changes if only the failing feature is dropped.

2. Lineage, not model id (audit finding 4)
------------------------------------------
The panel has more implementations than independent sources.  ``hh_point`` and
``hh_cable`` are one HH lineage; ``pospischil_rs`` and ``pospischil_fs`` are one
Destexhe-lab framework; ``ortner_snc_da`` and ``dougalis_vlpag_da`` share an
author's NEURON code; ``tomko_ca1`` inherits Migliore's channel kinetics.
Clustering on ``model_id`` therefore overstates the inferential sample.  Every
fold, bootstrap and random effect must group on
:func:`lineage_column` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "N4_FAILING_MODELS",
    "PRIMARY_EXCLUSIONS",
    "SENSITIVITY_ARMS",
    "UNIDENTIFIABLE_PARAMETERS",
    "annotate_lineage",
    "cohort_label",
    "drop_duplicate_scientific_rows",
    "duplicate_arms",
    "effective_sample_summary",
    "lineage_column",
    "primary_mask",
]

_IDENTIFIABILITY = (Path(__file__).resolve().parents[3] / "data" / "provenance"
                    / "parameter_identifiability.json")


def _load_unidentifiable() -> dict[str, list[str]]:
    """model_id -> parameters that gate N4 measured as unidentifiable.

    Read from the provenance record rather than restated here, so the cohort
    logic and the published measurement cannot drift apart.  A missing file is
    an error and not an empty dict: silently reporting "no unidentifiable
    parameters" is precisely the false all-clear the post-response audit found
    in the cohort labelling.
    """
    if not _IDENTIFIABILITY.exists():
        raise FileNotFoundError(
            f"{_IDENTIFIABILITY} is missing. Gate N4's measured parameter "
            f"identifiability is a precondition for defining the cohort; "
            f"refusing to assume every parameter is identifiable.")
    rec = json.loads(_IDENTIFIABILITY.read_text())
    return {m: sorted(v["unidentifiable"])
            for m, v in rec.get("models", {}).items() if v.get("unidentifiable")}


UNIDENTIFIABLE_PARAMETERS: dict[str, list[str]] = _load_unidentifiable()

# Models that FAIL gate N4.  They are not excluded from the primary cohort --
# see the `n4_strict` and `n4_projected` arms below for why, and for what is
# reported instead -- but nothing in this project may describe a cohort
# containing them as "gate-passing" or "passing all gates".
N4_FAILING_MODELS: frozenset[str] = frozenset(UNIDENTIFIABLE_PARAMETERS)

# model_id -> why it is out of the primary cohort.  Every entry is a gate
# failure, never a result the analyst disliked.
PRIMARY_EXCLUSIONS: dict[str, dict] = {
    "mainen_l5pc": {
        "gate": "N3",
        "reason": ("production dt=0.0125 ms / 2304 segments fails "
                   "step_abs_hi.adaptation_index at 1 of 7 audit points; the "
                   "other 12 conditioning features reproduce on all 7"),
        "remediation": ("finer dt or mesh was measured and is not affordable at "
                        "campaign scale; see docs/decisions"),
        "sensitivity_arm": "retain_mainen",
    },
    "miyasho_purkinje": {
        "gate": "N3",
        "reason": ("no tested step converges. dt = 0.05 and 0.025 ms fail "
                   "ap_amplitude_mV (agreement 0.40) on both the published "
                   "1,088-segment mesh and a 1,264-segment d_lambda mesh; "
                   "0.0125 ms still fails ap_amplitude_mV (0.60) and 0.00625 ms "
                   "fails membrane_tau_ms (0.80) against a 0.003125 ms "
                   "reference. The failing feature MOVES between steps, which is "
                   "the signature of threshold-adjacent audit points rather than "
                   "of an under-resolved integration"),
        "remediation": ("not simulated. Halving again would make the reference "
                        "the production setting; the honest reading is that this "
                        "model's prior samples include points where a feature "
                        "sits on a discontinuity, which a finer step cannot fix"),
        "sensitivity_arm": "retain_partial_coverage",
    },
    "tzilivaki_fsbc": {
        "gate": "N3",
        "reason": ("step_abs_hi.mean_firing_rate_Hz reproduces on only 25-50% of "
                   "audit points at dt = 0.05 and 0.025 ms, on both the published "
                   "9,875-segment mesh and the 4,096-segment d_lambda mesh. A "
                   "fast-spiking cell at ~145 Hz puts one or two spikes per "
                   "400 ms window inside the tolerance, and the step needed to "
                   "resolve them is finer than 0.0125 ms"),
        "remediation": ("a converged campaign would cost ~4 h per 256 samples at "
                        "0.00625 ms on this morphology; not affordable in this "
                        "release. Adapter, mechanisms, calibration and the "
                        "failing audit are retained so it can be resumed"),
        "sensitivity_arm": "retain_partial_coverage",
    },
    "ortner_snc_da": {
        "gate": "N3",
        "reason": ("no tested time step converges: step_abs_hi.ap_amplitude_mV "
                   "fails on 1-2 of 7 prior points at dt = 0.025, 0.0125 AND "
                   "0.00625 ms, non-monotonically. Separately, its maximum rate "
                   "of ~7.5 Hz gives <=3 spikes in the frozen 400 ms window, so "
                   "ISI-derived conditioning features sit at their definitional "
                   "floor (protocol_coverage='partial')"),
        "remediation": ("not simulated at all; the adapter and its failing "
                        "convergence audit are retained as the documented record"),
        "sensitivity_arm": "retain_partial_coverage",
    },
}

# Prespecified sensitivity cascade.  Each arm is a complete re-run of the
# cross-model analysis under a stated deviation from the primary rule, so the
# consequence of every exclusion is measured rather than asserted.
SENSITIVITY_ARMS: dict[str, dict] = {
    "primary": {
        # NOT "gate-passing".  The exclusion list above is built from N3
        # failures only, and `bbp_l5pc` -- which is in this arm and casts one of
        # its five lineage votes -- fails N4 on two apical conductances.  The
        # post-response audit is right that calling this cohort "gate-passing"
        # or "passing all gates" is false.  The N4 consequence is measured by
        # the two arms below rather than asserted away.
        "description": ("N3-passing models; the headline analysis. Contains "
                        "bbp_l5pc, which fails N4 -- see the n4_strict and "
                        "n4_projected arms"),
        "exclude": sorted(PRIMARY_EXCLUSIONS),
        "drop_features": [],
    },
    "n4_strict": {
        "description": ("every N4-failing implementation excluded as well, so "
                        "the headline can be read under full N1-N7 enforcement"),
        "exclude": sorted(set(PRIMARY_EXCLUSIONS) | N4_FAILING_MODELS),
        "drop_features": [],
    },
    "n4_projected": {
        "description": ("N4-failing implementations RETAINED but projected onto "
                        "their identifiable dimensions: the parameters gate N4 "
                        "measured as unobservable through the frozen protocol "
                        "are dropped before ensembles, geometry and "
                        "O-information are computed. Covers bbp_l5pc, the only "
                        "N4-failing model in the DISCOVERY cohort; tomko_ca1 is "
                        "a replication model and its projection lives in "
                        "endpoints_screening_v2_replication_n4proj.parquet"),
        "exclude": sorted(PRIMARY_EXCLUSIONS),
        "drop_features": [],
        # Dropping a parameter changes the sampled space the ensembles are built
        # on, so -- exactly like `drop_features` -- it cannot be applied by
        # filtering an endpoint table afterwards.  The arm needs its own pass:
        #   python tools/run_analysis.py --experiment screening \
        #          --drop-unidentifiable-parameters --suffix _n4proj
        #   python tools/run_analysis.py --experiment screening_v2_replication \
        #          --drop-unidentifiable-parameters --suffix _n4proj
        # (the second pass is what covers tomko_ca1: it is a replication-cohort
        # model and has no rows in the discovery table at all)
        # The dropped set is read from data/provenance/parameter_identifiability
        # .json, which was measured by gate N4 before any endpoint was inspected,
        # so the projection is outcome-blind.
        "source_experiment": "screening_n4proj",
    },
    "retain_mainen": {
        "description": ("mainen_l5pc retained despite its N3 failure, to show "
                        "how much the exclusion moves the result"),
        "exclude": [m for m in PRIMARY_EXCLUSIONS if m != "mainen_l5pc"],
        "drop_features": [],
    },
    "drop_failed_feature": {
        "description": ("mainen_l5pc retained but step_abs_hi.adaptation_index "
                        "-- the one feature that fails convergence -- removed "
                        "from the conditioning phenotype for EVERY model, so the "
                        "comparison stays like-for-like"),
        "exclude": [m for m in PRIMARY_EXCLUSIONS if m != "mainen_l5pc"],
        "drop_features": ["step_abs_hi.adaptation_index"],
        # `drop_features` changes the conditioning phenotype, which is fixed when
        # ensembles are built -- it cannot be applied by filtering an endpoint
        # table afterwards.  The arm therefore needs its own analysis pass:
        #   python tools/run_analysis.py --experiment screening \
        #          --drop-features step_abs_hi.adaptation_index \
        #          --suffix _dropfeat
        # Until that table exists the arm is skipped rather than silently
        # duplicating `retain_mainen`, which is what it did before: every
        # substantive artifact was byte-identical and only the metadata differed.
        "source_experiment": "screening_dropfeat",
    },
    "retain_partial_coverage": {
        "description": "ortner_snc_da retained despite partial protocol coverage",
        "exclude": [m for m in PRIMARY_EXCLUSIONS if m != "ortner_snc_da"],
        "drop_features": [],
    },
    "all_models": {
        "description": "every registered model, no gate exclusions at all",
        "exclude": [],
        "drop_features": [],
    },
    "discovery_only": {
        "description": ("the frozen discovery cohort alone, so the replication "
                        "cohort can be evaluated against it rather than pooled "
                        "with it before heterogeneity is assessed"),
        "exclude": sorted(PRIMARY_EXCLUSIONS),
        "cohort": "discovery",
        "drop_features": [],
    },
    "replication_only": {
        "description": "the external replication cohort alone",
        "exclude": sorted(PRIMARY_EXCLUSIONS),
        "cohort": "replication",
        "drop_features": [],
        # The replication models are simulated into their OWN campaign namespace,
        # so their endpoints are not rows of the discovery table.  Filtering the
        # discovery table by cohort=="replication" therefore matched nothing and
        # the arm reported "too few models" while
        # endpoints_screening_v2_replication.parquet sat unread on disk.
        "source_experiment": "screening_v2_replication",
    },
}


def lineage_column(df: pd.DataFrame) -> pd.Series:
    """``source_lineage_id`` for each row, falling back to the model id.

    The fallback is deliberately visible rather than silent: a model with no
    registered lineage becomes its own cluster, which is the *conservative*
    direction only for the model itself and the anticonservative direction for
    the panel, so :func:`effective_sample_summary` reports how often it happened.
    """
    from ..adapters.models import lineage_map

    lm = lineage_map()
    return df["model_id"].map(lambda m: lm.get(m, m))


def annotate_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lineage, implementation, class and cohort columns."""
    from ..adapters.models import MODEL_REGISTRY

    out = df.copy()

    def field(name, default=""):
        return out["model_id"].map(
            lambda m: getattr(MODEL_REGISTRY[m].manifest, name, default)
            if m in MODEL_REGISTRY else default)

    out["source_lineage_id"] = lineage_column(out)
    out["implementation_id"] = field("implementation_id")
    out["neuron_class_id"] = field("neuron_class_id")
    out["cohort"] = field("cohort", "discovery")
    out["protocol_coverage"] = field("protocol_coverage", "full")
    out["gate_excluded"] = out["model_id"].isin(PRIMARY_EXCLUSIONS)
    return out


def primary_mask(df: pd.DataFrame, arm: str = "primary") -> np.ndarray:
    """Boolean mask of rows admissible under a named sensitivity arm."""
    try:
        spec = SENSITIVITY_ARMS[arm]
    except KeyError as exc:
        raise ValueError(f"unknown sensitivity arm {arm!r}; "
                         f"choose {sorted(SENSITIVITY_ARMS)}") from exc
    mask = ~df["model_id"].isin(spec["exclude"]).to_numpy()
    want_cohort = spec.get("cohort")
    if want_cohort is not None:
        from ..adapters.models import MODEL_REGISTRY

        cohort = df["model_id"].map(
            lambda m: MODEL_REGISTRY[m].manifest.cohort if m in MODEL_REGISTRY
            else "discovery")
        mask &= (cohort == want_cohort).to_numpy()
    return mask


def cohort_label(arm: str = "primary") -> str:
    """The honest one-line name of a cohort, for prose and figure captions.

    Exists so that "gate-passing" cannot be reintroduced by hand.  A cohort is
    only called gate-passing when it actually excludes every model that fails a
    gate; otherwise the label names the gates it does enforce.
    """
    spec = SENSITIVITY_ARMS[arm]
    excluded = set(spec["exclude"])
    unresolved = sorted(N4_FAILING_MODELS - excluded)
    if not unresolved:
        return "gate-passing (N3 and N4 enforced)"
    if arm == "n4_projected":
        return ("N3-passing, with N4-unidentifiable dimensions projected out of "
                f"{', '.join(unresolved)}")
    return f"N3-passing; N4 unresolved for {', '.join(unresolved)}"


def duplicate_arms(df: pd.DataFrame) -> dict[str, str]:
    """Arms whose admissible rows are identical to an earlier arm's.

    An arm that selects exactly the rows another arm already selected is not an
    independent robustness demonstration, however differently it is described.
    Three of the seven arms were in that position -- ``retain_partial_coverage``
    because the model it retains has no screening data at all, and ``all_models``
    because the only models it adds back are ones no other arm removed -- and
    each was being counted and reported as though it had confirmed something
    (post-response audit, section 4).

    Detected from the data rather than hard-coded, because whether two arms
    coincide depends on which models have rows in the table being analysed.
    Arms drawing on a different ``source_experiment`` are never duplicates:
    their rows come from another table entirely.
    """
    seen: dict[frozenset, str] = {}
    dup: dict[str, str] = {}
    for arm in SENSITIVITY_ARMS:
        if SENSITIVITY_ARMS[arm].get("source_experiment"):
            continue
        rows = df[primary_mask(df, arm)]
        key = frozenset(map(tuple, rows[["model_id"]].drop_duplicates().to_numpy())) \
            if len(rows) else frozenset()
        if key in seen:
            dup[arm] = seen[key]
        else:
            seen[key] = arm
    return dup


def drop_duplicate_scientific_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse endpoint rows that are the same ensemble under two labels.

    Audit 6.2: "There are 40 exact duplicate scientific endpoint rows when
    ``protocol_family`` is ignored, primarily E1/E2/E6 parameter ensembles
    copied under absolute and normalised labels. These are valid for display but
    not independent observations for regression or classification."

    The unconditioned regimes do not depend on the protocol family at all -- the
    same parameter matrix is analysed twice and stored twice -- so one copy is
    kept, deterministically the ``absolute`` one.  Conditioned regimes DO differ
    by family (different phenotype, different accepted set) and are untouched.
    """
    if "protocol_family" not in df.columns:
        return df
    unconditioned = df["regime"].isin(["E1", "E2", "E6"])
    keep = (~unconditioned) | (df["protocol_family"] == "absolute")
    return df[keep].reset_index(drop=True)


def effective_sample_summary(df: pd.DataFrame, arm: str = "primary") -> dict:
    """State the inferential sample size honestly, in one place."""
    ann = annotate_lineage(df)
    prim = ann[primary_mask(ann, arm)]
    dedup = drop_duplicate_scientific_rows(prim)
    return {
        "n_rows_all": int(len(ann)),
        "n_rows_primary": int(len(prim)),
        "n_rows_primary_deduplicated": int(len(dedup)),
        "n_model_ids": int(ann["model_id"].nunique()),
        "n_model_ids_primary": int(prim["model_id"].nunique()),
        "n_source_lineages": int(ann["source_lineage_id"].nunique()),
        "n_source_lineages_primary": int(prim["source_lineage_id"].nunique()),
        "n_neuron_classes": int(ann["neuron_class_id"].nunique()),
        "n_classes_with_2plus_lineages": int(
            (ann.groupby("neuron_class_id")["source_lineage_id"].nunique() >= 2).sum()),
        "n_classes_with_3plus_lineages": int(
            (ann.groupby("neuron_class_id")["source_lineage_id"].nunique() >= 3).sum()),
        "lineage_members": {
            lin: sorted(g["model_id"].unique())
            for lin, g in ann.groupby("source_lineage_id")},
        "excluded_from_primary": {
            m: PRIMARY_EXCLUSIONS[m]["reason"]
            for m in sorted(set(ann["model_id"]) & set(PRIMARY_EXCLUSIONS))},
        # Stated on every artifact that carries an effective sample, so no
        # downstream document can describe this cohort as gate-passing without
        # contradicting the record it was generated from.
        "cohort_label": cohort_label(arm),
        "n4_failing_models_retained": sorted(
            set(prim["model_id"]) & N4_FAILING_MODELS),
        "n4_unidentifiable_parameters": {
            m: UNIDENTIFIABLE_PARAMETERS[m]
            for m in sorted(set(prim["model_id"]) & N4_FAILING_MODELS)},
        "note": ("clustering, folds and bootstraps use source_lineage_id, not "
                 "model_id; the two differ and the difference is the whole point "
                 "of the 2026-08-06 audit's finding 4"),
    }
