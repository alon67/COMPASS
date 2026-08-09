"""Block-based simulation workflow with atomic result shards (blueprint Task 7).

Execution model
---------------
* One **process per model** -- NEURON's section list is process-global, so a
  worker is pinned to a single model and builds the cell exactly once, then
  reuses it for every sample in its block (parameter injection + reset).
* Work is split into **blocks** of samples.  Each block writes one parquet shard
  atomically (temp file + rename), so an interrupted campaign resumes by simply
  skipping shards that already exist -- no duplicated samples, no partial files.
* Failures are recorded, never dropped: ``simulation_status`` and ``qc_flags``
  columns carry solver failures, non-physical voltages and extraction failures
  into the audit table (blueprint 26 "never silently remove numerical failures").
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"


# Campaign solver policy: DETERMINISTIC FIXED TIME STEP.
#
# The convergence audit (tools/numerical_qc.py) selects the cheapest setting that
# reproduces the finest-step result at *canonical* parameters, and for most
# models that is CVODE.  Under a broad prior, however, CVODE's cost is a
# function of how stiff a particular parameter sample happens to be, and the
# tail is severe: profiling mainen_l5pc over six prior samples gave a median of
# 11 s but individual samples of 233 s and 307 s (mean 96 s).  A campaign cannot
# be budgeted against that distribution, and a per-sample cost that depends on
# the sample is also a subtle sampling bias if any sample is ever abandoned.
#
# Each dt below is the largest step the audit accepted for that model
# (data/provenance/numerical_qc.json, "accepted" field), so cost is constant per
# simulation and every result is bit-reproducible.
CAMPAIGN_SOLVER = {
    # Settings accepted by tools/convergence_audit.py: every one of the 13
    # conditioning features reproduced, on 6 parameter points drawn from each
    # model's own prior, against a reference at half the step on twice the mesh.
    # This SUPERSEDES tools/numerical_qc.py, which judged a single canonical
    # point on spike count alone -- a knife-edge criterion that both passed
    # settings that do not converge (mainen, bbp, hh_point, khaliq) and rejected
    # one that does (migliore).
    "hh_point": {"cvode": False, "dt_ms": 0.025},        # 0.05 fails membrane tau
    "hh_cable": {"cvode": False, "dt_ms": 0.05},
    "pospischil_rs": {"cvode": False, "dt_ms": 0.05},
    "pospischil_fs": {"cvode": False, "dt_ms": 0.025},
    # Spontaneously active, resurgent Na + spatial Ca diffusion: the stiffest
    # model in the panel.  0.0125 still misses membrane tau and AP amplitude on
    # 1 of 6 samples; 0.00625 reproduces everything.
    "khaliq_purkinje": {"cvode": False, "dt_ms": 0.00625},
    # Needs BOTH a finer step and a finer mesh; the adapter builds 2x the
    # d_lambda(100 Hz) mesh (2304 segments) for this reason.
    "mainen_l5pc": {"cvode": False, "dt_ms": 0.0125},
    "bbp_l5pc": {"cvode": False, "dt_ms": 0.025},        # 0.05 fails AP amplitude
    # migliore_ca1: dt=0.025 on the d_lambda(100 Hz) mesh (1106 segments).
    # numerical_qc.py originally REJECTED this step on an exact spike-count
    # comparison at the canonical point, where a 0.5 nA step sits on a knife
    # edge (4 vs 5 spikes).  The feature-level audit accepts it: agreement is
    # 1.00 for every feature on every point.
    "migliore_ca1": {"cvode": False, "dt_ms": 0.025},
    # --- external replication cohort ------------------------------------- #
    # Each entry is the cheapest setting accepted by that model's own
    # feature-level convergence audit; see
    # data/provenance/convergence_audit_<model>.json.  They are placeholders
    # ONLY until that audit has run: tests/integration/test_convergence_contract.py
    # fails the release if the exact production setting is not a directly tested,
    # accepted row.
    # miyasho_purkinje: NO setting converges. ap_amplitude_mV fails at 0.05,
    # 0.025 and 0.0125 ms (on the published 1,088-segment mesh AND a
    # 1,264-segment d_lambda mesh) and membrane_tau_ms at 0.00625 ms, against a
    # 0.003125 ms reference. The failing feature MOVES between steps, which
    # points at threshold-adjacent prior samples rather than an under-resolved
    # integration -- a finer step does not fix that. NOT simulated; the entry
    # names the finest AUDITED setting so the failing audit stays reproducible
    # and gate N3 can find a directly tested row to reject.
    "miyasho_purkinje": {"cvode": False, "dt_ms": 0.00625},
    "tzilivaki_fsbc": {"cvode": False, "dt_ms": 0.025},
    # Spontaneous pacemaker on a reconstructed axon-bearing dendrite. Its
    # published lambda_f(1000) mesh never converges (input resistance measured
    # on an ongoing spike train); the adapter therefore builds the d_lambda(100
    # Hz) mesh (174 segments), where dt = 0.003125 ms reproduces all 13 features
    # on all 7 audit points.
    "moubarak_snc_da": {"cvode": False, "dt_ms": 0.003125},
    # 0.025 fails ap_amplitude_mV; 0.0125 reproduces all 13 features on all 7
    # points against a dt=0.0015625 reference.
    "dougalis_vlpag_da": {"cvode": False, "dt_ms": 0.0125},
    # ortner_snc_da: NO setting converges. ap_amplitude_mV fails on 1-2 of 7
    # prior points at 0.025, 0.0125 AND 0.00625, non-monotonically -- the
    # signature of a threshold-adjacent point rather than a step-size problem.
    # Combined with its partial protocol coverage (<=3 spikes in the 400 ms
    # window) it fails two independent outcome-blind gates, so it is NOT
    # simulated and NOT in the replication campaign. The entry is kept so the
    # failing audit stays reproducible.
    "ortner_snc_da": {"cvode": False, "dt_ms": 0.00625},
    "tomko_ca1": {"cvode": False, "dt_ms": 0.025},
}


def solver_kwargs(model_id: str) -> dict:
    """The production solver contract.  **Fails closed** for unknown models.

    This used to fall back from ``CAMPAIGN_SOLVER`` to the superseded
    ``numerical_qc.json`` and then to a guessed fixed 0.025 ms.  The canonical
    audit already failed closed, so the repository could run a full production
    campaign for a newly registered model at a time step nothing had ever
    validated, while the audit tool refused to touch it -- the dangerous
    direction of the two (2026-08-06 audit, section 5.3).

    A new model must therefore appear in ``CAMPAIGN_SOLVER`` with a setting its
    own feature-level convergence audit accepts before it can be simulated.
    """
    if model_id in CAMPAIGN_SOLVER:
        return dict(CAMPAIGN_SOLVER[model_id])
    raise KeyError(
        f"{model_id!r} has no CAMPAIGN_SOLVER entry. Production refuses to guess "
        f"a solver contract: run tools/convergence_audit.py {model_id} and add the "
        f"accepted setting. Known: {sorted(CAMPAIGN_SOLVER)}"
    )


def normalised_family_defined(rheobase) -> bool:
    """Can a rheobase-normalised protocol be applied to this sample at all?

    ``rheobase = 0`` means the draw fires with no injected current, so "1.5x
    rheobase" is 0 nA -- the same measurement as rest, relabelled as a
    suprathreshold step.  This used to be suppressed only when the *model* was
    listed in ``SPONTANEOUSLY_ACTIVE``, which is a property of the canonical
    parameters, not of the draw.  A broad prior turns ordinarily silent models
    spontaneous all the time: 15,234 stored rows across hh_cable, hh_point,
    bbp_l5pc, mainen_l5pc, pospischil_* and dougalis_vlpag_da carry
    "step_rheo_1p5x" and "step_rheo_3p0x" features that were measured at 0 nA
    (2026-08-09 audit, 4.1).  Draw-level rheobase decides this, not a model-level
    list.
    """
    return rheobase is not None and float(rheobase) != 0.0


def _feature_row(adapter, kw, rheobase, protocols) -> tuple[dict, list[str]]:
    """Run the common-core protocol suite for one already-applied parameter set."""
    from compass.features.extract import extract, extract_rest

    row: dict[str, float] = {}
    flags: list[str] = []
    for spec in protocols:
        pid = spec.protocol_id
        if spec.kind == "rest":
            rec = adapter.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                                   total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract_rest(rec)
        else:
            if spec.amplitude_nA is not None:
                amp = spec.amplitude_nA
            elif not normalised_family_defined(rheobase):
                # rheobase-normalised protocols are undefined for a sample that
                # never spikes (no rheobase) or that spikes at zero current
                for k in ("spike_count", "mean_firing_rate_Hz", "first_isi_ms",
                          "adaptation_index", "ap_amplitude_mV", "ap_halfwidth_ms",
                          "ahp_depth_mV", "first_spike_latency_ms", "cv_isi",
                          "ap_threshold_mV", "ap_peak_mV", "last_isi_ms",
                          "depolarisation_block", "resting_potential_mV"):
                    row[f"{pid}.{k}"] = np.nan
                continue
            else:
                amp = rheobase * spec.amplitude_rheobase_mult
            rec = adapter.simulate(float(amp), delay_ms=spec.delay_ms,
                                   duration_ms=spec.duration_ms,
                                   total_ms=spec.total_ms, protocol_id=pid, **kw)
            feats = extract(rec)
        f = adapter.qc_flags(rec)
        if f:
            flags.extend(f"{pid}:{x}" for x in f)
        for k, v in feats.items():
            row[f"{pid}.{k}"] = float(v) if v is not None and np.isfinite(v) else np.nan
    return row, flags


# Bumped whenever the row-level validity classification changes, so a stored row
# always says which validator produced its verdict (audit 6.4.2).
# v3: `normalised_protocol_valid` is reported separately from `protocol_valid`.
VALIDATOR_VERSION = "row-validity-3"


def classify_row(row: dict, flags: list[str], rheobase, spontaneous: bool,
                 error: str | None) -> dict:
    """Separate the distinct meanings of "valid" (2026-08-06 audit, 6.4.2).

    Collapsing them into one ``simulation_status=ok`` field made two very
    different things indistinguishable.  The broad Tier C/D priors are
    *supposed* to produce noncanonical neurons: a finite but silent or
    depolarisation-blocked draw is a prior-predictive outcome that a frozen
    analysis policy should classify, not a corrupt simulation.  Conversely a
    numerically failed rheobase probe is corrupt even though downstream code
    could manufacture a finite feature row from the surviving protocols.
    """
    numerical = {"solver_failure", "nonfinite_voltage", "nonphysical_voltage"}
    bare = {f.split(":")[-1] for f in flags}
    execution_valid = error is None and not (bare & numerical)
    # `protocol_valid` is scoped to the ABSOLUTE family, which is the primary
    # endpoint and is applied as intended for every executed sample: a fixed
    # current in nA does not depend on what the draw does to excitability.
    #
    # The rheobase-normalised family is a different question and is now reported
    # as its own field.  The comment here used to assert that `protocol_valid`
    # already accounted for an unusable rheobase; it did not, and 24,000+ rows
    # were counted as "protocol applied as intended" while their normalised
    # protocols were undefined or run at 0 nA (2026-08-09 audit, 4.1).
    protocol_valid = execution_valid
    normalised_protocol_valid = bool(execution_valid
                                     and normalised_family_defined(rheobase))
    feature_valid = execution_valid and any(
        np.isfinite(v) for k, v in row.items()
        if k.startswith("step_abs_hi.") and isinstance(v, float))

    # Order matters, and getting it wrong misclassified 81% of a pacemaker's
    # rows.  `depolarised_rest` is raised whenever the pre-stimulus baseline
    # sits above -20 mV, which is true of EVERY spontaneously active cell,
    # because its baseline window contains spikes.  Reading that flag as
    # depolarisation block therefore labels a healthy pacemaker as blocked.
    # Spontaneity is checked first, and genuine block is taken from the
    # extractor's own `depolarisation_block` feature -- which tests whether the
    # cell stops firing while still depolarised -- rather than from a baseline
    # heuristic.
    blocked = any(
        float(v) >= 0.5 for k, v in row.items()
        if k.endswith(".depolarisation_block") and isinstance(v, (int, float))
        and np.isfinite(v))
    if error is not None:
        physiology = "simulation_error"
    elif bare & numerical:
        physiology = "unstable"
    elif spontaneous or rheobase == 0.0:
        physiology = "spontaneous_block" if blocked else "spontaneous"
    elif blocked or "depolarised_rest" in bare:
        physiology = "depolarisation_block"
    elif rheobase is None:
        physiology = "nonspiking"
    else:
        physiology = "canonical_plausible"

    return {
        "execution_valid": bool(execution_valid),
        "protocol_valid": bool(protocol_valid),
        # Eligibility for the rheobase-normalised endpoint specifically.  Kept
        # out of `analysis_eligible` on purpose: a silent or spontaneous draw is
        # a perfectly valid observation on the absolute family, which is the
        # primary endpoint, and dropping it there would bias the cohort towards
        # canonical-looking cells.
        "normalised_protocol_valid": normalised_protocol_valid,
        "feature_valid": bool(feature_valid),
        "physiology_class": physiology,
        # The preregistered inclusion rule for the primary analysis: the run
        # executed, the protocol was applied, and features exist.
        "analysis_eligible": bool(execution_valid and protocol_valid and feature_valid),
        "validator_version": VALIDATOR_VERSION,
        "failure_reason": error or (";".join(sorted(bare & numerical)) or ""),
    }


def run_block(
    model_id: str,
    scales: np.ndarray,
    param_ids: list[str],
    sample_ids: np.ndarray,
    regime: str,
    replicate: int,
    seed: int,
    prior_id: str,
    out_path: Path,
    identity: dict | None = None,
) -> dict:
    """Simulate one block of parameter samples and write an atomic parquet shard."""
    from compass.adapters.models import get_adapter
    from compass.protocols import common as P

    kw = solver_kwargs(model_id)
    # Forward the FULL solver settings (including cvode) to the rheobase search.
    # The bisection costs ~9 simulations per sample and dominates the campaign
    # budget; running it fixed-step while the protocols run under CVODE made the
    # search ~10x more expensive than everything else combined.
    rheo_kw = dict(kw)
    spontaneous = model_id in P.SPONTANEOUSLY_ACTIVE
    protocols = P.common_core(model_id)

    adapter = get_adapter(model_id).build()
    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids):
        adapter.reset()
        rec_meta = {
            "run_id": f"{model_id}:{regime}:{replicate}:{int(sid)}",
            "model_id": model_id,
            "model_variant_id": "canonical",
            "sampling_regime": regime,
            "prior_id": prior_id,
            "conditioning_id": "none",
            "replicate": replicate,
            "seed": seed,
            "parameter_sample_id": int(sid),
            "protocol_id": "common_core",
        }
        error = None
        flags: list[str] = []
        rheo = None
        try:
            adapter.apply(dict(zip(param_ids, scales[i], strict=True)))
            rheo = adapter.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                                         duration_ms=P.RHEO_DURATION_MS,
                                         total_ms=P.RHEO_TOTAL_MS, **rheo_kw)
            search = dict(adapter.last_rheobase_search or {})
            feats, flags = _feature_row(adapter, kw, rheo, protocols)
            rec_meta["simulation_status"] = "ok"
            rec_meta["qc_flags"] = ";".join(sorted(set(flags)))
            rec_meta["rheobase_nA"] = np.nan if rheo is None else float(rheo)
            rec_meta["non_spiking"] = bool(rheo is None)
            # Distinguishes "silent" from "in depolarisation block at the top of
            # the search range but firing below it" -- two states the probe used
            # to report identically (2026-08-09 audit, 4.1).
            rec_meta["rheobase_blocked_at_ceiling"] = bool(
                search.get("blocked_at_ceiling", False))
            rec_meta.update(feats)
        except adapter.RheobaseProbeFailure as exc:
            # A probe that could not be evaluated fails the whole sample.  It is
            # NOT "this cell does not spike" (gate N5).
            error = f"rheobase_probe_failure: {exc}"
            rec_meta["simulation_status"] = "error"
            rec_meta["qc_flags"] = "rheobase_probe_failure"
            rec_meta["error"] = error
            rec_meta["rheobase_nA"] = np.nan
            rec_meta["non_spiking"] = False   # unknown, not False-as-silence
            rec_meta["rheobase_blocked_at_ceiling"] = False
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            rec_meta["simulation_status"] = "error"
            rec_meta["qc_flags"] = f"exception:{type(exc).__name__}"
            rec_meta["error"] = f"{exc}\n{traceback.format_exc()[-400:]}"
            rec_meta["rheobase_nA"] = np.nan
            rec_meta["non_spiking"] = False
            rec_meta["rheobase_blocked_at_ceiling"] = False
        rec_meta.update(classify_row(rec_meta, flags, rheo, spontaneous, error))
        rows.append(rec_meta)

    df = pd.DataFrame(rows)
    for j, pid in enumerate(param_ids):
        df[f"param.{pid}"] = scales[:, j]
    # Bind the shard to the identity that produced it, in the shard itself.
    # A sidecar alone is separable from the data; these columns are not.
    for k, v in (identity or {}).items():
        df[f"identity.{k}"] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    if identity:
        write_sidecar(out_path, identity, n_rows=len(rows))
    return {
        "shard": out_path.name,
        "n": len(rows),
        "wall_s": round(time.time() - t_start, 1),
        "n_error": int((df["simulation_status"] == "error").sum()),
        "n_nonspiking": int(df["non_spiking"].sum()),
        "n_ineligible": int((~df["analysis_eligible"]).sum()),
    }


def sidecar_path(shard: Path) -> Path:
    return shard.with_suffix(".identity.json")


def write_sidecar(shard: Path, identity: dict, n_rows: int) -> Path:
    import hashlib
    import json as _json

    payload = dict(identity)
    payload["n_rows"] = int(n_rows)
    payload["shard"] = shard.name
    payload["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    path = sidecar_path(shard)
    path.write_text(_json.dumps(payload, indent=1, sort_keys=True, default=str))
    return path


# Identity differences that are DOCUMENTED and deliberately tolerated, with the
# justification and the condition under which the tolerance is valid.
#
# This is not an escape hatch: `validate_campaign.py` re-derives the
# "not load-bearing" claim from the analysis code rather than trusting the text
# here, so if a drifted feature ever becomes a conditioning feature, an RSI
# target or a perturbation readout, the allowance stops applying and the shards
# are quarantined.
DOCUMENTED_DRIFT: dict[str, dict] = {
    "feature_schema": {
        ("features-1", "features-2"): {
            "changed": ["cv_isi"],
            "detail": (
                "features-2 requires >=3 spikes for cv_isi. Under features-1 a "
                "two-spike train has one ISI, np.std of which is exactly 0, so "
                "2,127 stored rows report cv_isi = 0.0 -- 'perfectly regular "
                "firing' manufactured from a single interval. Found by gate D3."),
            "not_consumed_by": ["CONDITIONING_FEATURES",
                                "CONDITIONING_FEATURES_NORMALISED",
                                "RSI_TARGETS", "PERT_FEATURES"],
            "resolution": (
                "cleared at the next full campaign regeneration; re-simulating "
                "98,048 rows to repair a feature no analysis reads is not a "
                "defensible use of compute, and the affected values are "
                "enumerated rather than hidden"),
        },
    },
}


def _drift_is_documented(field: str, stored, wanted) -> bool:
    return (str(stored), str(wanted)) in DOCUMENTED_DRIFT.get(field, {})


def shard_is_reusable(shard: Path, identity: dict, expected_rows: int) -> tuple[bool, str]:
    """May an existing shard be reused, or must it be recomputed?

    The previous rule was "the path exists" (audit 5.2).  That reuses a shard
    written by different code, a different solver contract, a different
    mechanism build or a partially different design, and nothing downstream can
    tell.  Reuse now requires the identity to match exactly, the row count to
    match, and the file's checksum to match what was recorded when it was
    written.
    """
    import hashlib
    import json as _json

    if not shard.exists():
        return False, "absent"
    side = sidecar_path(shard)
    if not side.exists():
        return False, "identity_unknown"
    try:
        rec = _json.loads(side.read_text())
    except Exception:  # noqa: BLE001
        return False, "identity_unreadable"
    if int(rec.get("n_rows", -1)) != int(expected_rows):
        return False, f"row_count {rec.get('n_rows')} != {expected_rows}"
    if rec.get("sha256") != hashlib.sha256(shard.read_bytes()).hexdigest():
        return False, "checksum_mismatch"
    # `recorded_only.*` fields are provenance, not reuse criteria: they are
    # written into the sidecar so drift stays visible and auditable, but they
    # do not by themselves invalidate a shard.
    mismatched = [k for k, v in identity.items()
                  if not k.startswith("recorded_only.")
                  and str(rec.get(k)) != str(v)
                  and not _drift_is_documented(k, rec.get(k), v)]
    if mismatched:
        return False, "identity_mismatch:" + ",".join(sorted(mismatched))
    drifted = [k for k, v in identity.items()
               if not k.startswith("recorded_only.") and str(rec.get(k)) != str(v)]
    if drifted:
        return True, "match_with_documented_drift:" + ",".join(sorted(drifted))
    return True, "match"


def quarantine(shard: Path, reason: str) -> Path:
    """Move a non-reusable shard aside rather than overwrite it.

    Overwriting destroys the evidence needed to work out *why* an identity
    changed; quarantine keeps it and still forces recomputation.
    """
    from datetime import UTC, datetime

    qdir = shard.parent / "_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    dest = qdir / f"{shard.stem}_{tag}_{reason.split(':')[0]}{shard.suffix}"
    shard.replace(dest)
    side = sidecar_path(shard)
    if side.exists():
        side.replace(dest.with_suffix(".identity.json"))
    return dest


def _worker(args) -> dict:
    """Entry point executed in a fresh process (one model, one block)."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")
    (model_id, scales, param_ids, sample_ids, regime, replicate, seed,
     prior_id, out_path, identity) = args
    shard = Path(out_path)
    n_expected = len(sample_ids)
    reusable, why = shard_is_reusable(shard, identity or {}, n_expected)
    if reusable:
        return {"shard": shard.name, "n": 0, "wall_s": 0.0, "n_error": 0,
                "n_nonspiking": 0, "n_ineligible": 0, "skipped": True,
                "reuse": why}
    if shard.exists():
        moved = quarantine(shard, why)
        print(f"  quarantined {shard.name} ({why}) -> {moved.name}", flush=True)
    return run_block(model_id, np.asarray(scales), list(param_ids),
                     np.asarray(sample_ids), regime, replicate, seed,
                     prior_id, shard, identity)
