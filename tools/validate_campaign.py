#!/usr/bin/env python3
"""Whole-campaign data-integrity gates D1-D5 (2026-08-06 audit, section 6.4.2).

    Every one of the 98,048 current simulation rows can and should be validated
    without rerunning NEURON. These checks are inexpensive relative to
    simulation and must cover 100% of screening and Sobol outputs.

So they do: every stored row, every shard, on every release.  Nothing here runs
a simulation; the whole point is that metadata and invariants are cheap enough
to check exhaustively while replay is not.

* **D1** complete design and key integrity -- counts derived from the frozen
  configuration, not hard-coded.
* **D2** parameter provenance -- stored scales regenerated from the design type,
  seed and sample id, and checked against their declared priors.
* **D3** row-level numerical and logical invariants.
* **D4** shard identity and immutability.
* **D5** population diagnostics -- failure/QC/nonspiking rates, missingness,
  feature ranges and replicate-to-replicate distances.

    python tools/validate_campaign.py --experiment screening
    python tools/validate_campaign.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402
from compass.validation.gates import gate_record, write_gate_records  # noqa: E402
from compass.workflows.simulate import shard_is_reusable, sidecar_path  # noqa: E402

FEATURES = ROOT / "results" / "features"
OUT = ROOT / "results" / "validation"

# Broad, prespecified expectations for D5.  These are anomaly detectors, not
# quality targets: the Tier C/D priors deliberately generate noncanonical cells,
# so a high nonspiking fraction is information, while a 100% or 0% rate almost
# always means the protocol or the prior is wrong.
D5_EXPECTATIONS = {
    "max_error_fraction": 0.05,
    "max_numerical_qc_fraction": 0.35,
    "nonspiking_fraction_range": (0.0, 0.95),
    "min_distinct_feature_values": 5,
}


def _campaign_module():
    spec = importlib.util.spec_from_file_location("_rc", ROOT / "tools" / "run_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_experiment(experiment: str) -> tuple[pd.DataFrame, list[Path]]:
    base = FEATURES / experiment
    shards = sorted(p for p in base.rglob("*.parquet")
                    if "_quarantine" not in p.parts and "_superseded" not in p.parts
                    and not p.name.endswith(".tmp.parquet"))
    frames = []
    for p in shards:
        df = pd.read_parquet(p)
        df["_shard"] = p.name
        df["_shard_path"] = str(p.relative_to(ROOT))
        frames.append(df)
    if not frames:
        return pd.DataFrame(), shards
    return pd.concat(frames, ignore_index=True), shards



def _d1_sobol(df, shards) -> object:
    """D1 for the Saltelli design: declared row count, unique keys, no gaps."""
    import importlib.util as iu

    spec = iu.spec_from_file_location("_rs", ROOT / "tools" / "run_sobol.py")
    rs = iu.module_from_spec(spec)
    spec.loader.exec_module(rs)

    fails, expected = [], {}
    for m, n_base in rs.N_BASE.items():
        n_params = len(rs.get_adapter(m).parameter_specs())
        # Saltelli with second-order terms: N * (2D + 2) rows per model.
        expected[m] = n_base * (2 * n_params + 2)
    got = df.groupby("model_id").size().to_dict()
    for m, n in sorted(expected.items()):
        if got.get(m, 0) != n:
            fails.append(f"{m}: {got.get(m, 0)} rows, the Saltelli design says {n}")
    for m, n in sorted(got.items()):
        if m not in expected:
            fails.append(f"{m}: {n} rows for a model not in the design")

    dup = int(df.duplicated(subset=["model_id", "parameter_sample_id"]).sum())
    if dup:
        fails.append(f"{dup} duplicate (model, sample) keys")
    for m, g in df.groupby("model_id"):
        ids = np.sort(g["parameter_sample_id"].to_numpy())
        if not np.array_equal(ids, np.arange(len(ids))):
            fails.append(f"{m}: sample ids are not 0..n-1 (the blocks do not tile)")

    return gate_record(
        "D1", "sobol", fails,
        measurements={"n_rows": int(len(df)), "n_shards": len(shards),
                      "observed": {k: int(v) for k, v in sorted(got.items())},
                      "expected": {k: int(v) for k, v in sorted(expected.items())}},
        inputs={"design": "tools/run_sobol.py:N_BASE x saltelli(2D+2)",
                "block": rs.BLOCK, "seed": rs.SEED})


def d1_design_and_keys(df, shards, experiment, rc) -> list:
    """Counts derived from configuration; keys unique; nothing out of range."""
    fails = []
    cfg = rc.CONFIG["campaigns"].get(experiment)
    if cfg is None:
        # SOBOL HAS A FROZEN DESIGN TOO -- IT JUST DOES NOT LIVE IN campaigns.json.
        #
        # D1 only ran for experiments listed there, and `sobol` is generated by
        # tools/run_sobol.py, so it silently had no D1 record at all. The
        # 2026-08-09 audit (P0-4B) added `sobol` to the DECLARED coverage
        # requirement; the producer was never taught to satisfy it, so the
        # release blocked on a hole that could not close. Its design is frozen
        # and checkable in exactly D1's terms: N_BASE per model and the Saltelli
        # rule fix the row count, and the shards must tile it without gaps.
        if experiment == "sobol":
            return [_d1_sobol(df, shards)]
        return [gate_record("D1", experiment, [f"no campaign {experiment!r} in config"])]

    expected = {}
    frac, block = rc.SECONDARY_FRACTION, rc.BLOCK
    per_model_regimes = cfg.get("regimes_by_model", {})
    for m, n1 in cfg["sample_sizes"].items():
        regimes = per_model_regimes.get(m, cfg["regimes"])
        for r in regimes:
            expected[(m, r)] = n1 if r == "E1" else max(block, int(n1 * frac))

    got = df.groupby(["model_id", "sampling_regime"]).size().to_dict()
    for key, n in sorted(expected.items()):
        have = got.get(key, 0)
        if have != n:
            fails.append(f"{key[0]}/{key[1]}: {have} rows, configuration says {n}")
    for key, n in sorted(got.items()):
        if key not in expected:
            fails.append(f"{key[0]}/{key[1]}: {n} unexpected rows (not in the config)")

    dup = df.duplicated(subset=["run_id"]).sum()
    if dup:
        fails.append(f"{dup} duplicate run_id values")
    composite = ["model_id", "sampling_regime", "replicate", "parameter_sample_id"]
    dup2 = df.duplicated(subset=composite).sum()
    if dup2:
        fails.append(f"{dup2} duplicate composite sample keys")

    for (m, r), g in df.groupby(["model_id", "sampling_regime"]):
        ids = np.sort(g["parameter_sample_id"].to_numpy())
        if not np.array_equal(ids, np.arange(len(ids))):
            fails.append(f"{m}/{r}: sample ids are not 0..n-1 (gaps or offsets)")

    return [gate_record(
        "D1", experiment, fails,
        measurements={"n_rows": int(len(df)), "n_shards": len(shards),
                      "observed": {f"{k[0]}/{k[1]}": v for k, v in sorted(got.items())}},
        inputs={"config_hash": rc.config_hash()},
        detail="row counts are a consequence of the frozen configuration, not a constant")]


def d2_parameter_provenance(df, experiment, rc) -> list:
    """Regenerate the design from seed and sample id; check priors."""
    from compass.sampling.priors import independent_sample

    records = []
    for (m, r, rep), g in df.groupby(["model_id", "sampling_regime", "replicate"]):
        fails = []
        if m not in MODEL_REGISTRY:
            records.append(gate_record("D2", f"{experiment}/{m}/{r}",
                                       [f"unknown model {m}"]))
            continue
        specs = get_adapter(m).parameter_specs()
        pids = [s.parameter_id for s in specs]
        cols = [f"param.{p}" for p in pids]
        missing = [c for c in cols if c not in g.columns]
        if missing:
            fails.append(f"missing parameter columns {missing}")
        else:
            x = g.sort_values("parameter_sample_id")[cols].to_numpy(dtype=float)
            if not np.all(np.isfinite(x)):
                fails.append(f"{int((~np.isfinite(x)).sum())} non-finite parameter values")
            for j, s in enumerate(specs):
                lo, hi = s.lower_bound, s.upper_bound
                out = int(((x[:, j] < lo * (1 - 1e-9)) | (x[:, j] > hi * (1 + 1e-9))).sum())
                if out:
                    fails.append(f"{s.parameter_id}: {out} values outside [{lo}, {hi}]")
            # E1 is a deterministic Sobol design: regenerate and compare.
            if r == "E1":
                seed = int(g["seed"].iloc[0])
                want = independent_sample(specs, len(x), seed=seed, method="sobol")
                if want.shape == x.shape and not np.allclose(want, x, rtol=1e-9, atol=1e-12):
                    n_bad = int((~np.isclose(want, x, rtol=1e-9, atol=1e-12)).any(axis=1).sum())
                    fails.append(f"E1 design does not regenerate from seed {seed}: "
                                 f"{n_bad}/{len(x)} rows differ")
        records.append(gate_record(
            "D2", f"{experiment}/{m}/{r}/r{rep}", fails,
            measurements={"n_rows": int(len(g)), "n_params": len(pids)},
            inputs={"seed": int(g["seed"].iloc[0])}))
    return records


def d3_row_invariants(df, experiment) -> list:
    """Schema, dtypes, feature domains and cross-feature relationships."""
    from compass.protocols import common as P
    from compass.workflows.simulate import DOCUMENTED_DRIFT as DRIFT

    fails: list[str] = []
    documented: list[str] = []
    required = ["run_id", "model_id", "sampling_regime", "simulation_status",
                "qc_flags", "rheobase_nA", "non_spiking"]
    for c in required:
        if c not in df.columns:
            fails.append(f"missing required column {c!r}")
    if fails:
        return [gate_record("D3", experiment, fails)]

    window_s = P.DURATION_MS / 1000.0
    for pid in ("step_abs_lo", "step_abs_hi"):
        cnt, rate = f"{pid}.spike_count", f"{pid}.mean_firing_rate_Hz"
        if cnt in df.columns and rate in df.columns:
            c = df[cnt].to_numpy(dtype=float)
            v = df[rate].to_numpy(dtype=float)
            both = np.isfinite(c) & np.isfinite(v)
            bad = int((np.abs(v[both] - c[both] / window_s) > 1e-6).sum())
            if bad:
                fails.append(f"{pid}: {bad} rows where rate != spike_count/{window_s}s")
            neg = int((c[np.isfinite(c)] < 0).sum())
            if neg:
                fails.append(f"{cnt}: {neg} negative counts")
            noninteger = int((np.abs(c[np.isfinite(c)] - np.round(c[np.isfinite(c)])) > 1e-9).sum())
            if noninteger:
                fails.append(f"{cnt}: {noninteger} non-integer spike counts")

    rheo = df["rheobase_nA"].to_numpy(dtype=float)
    neg = int((rheo[np.isfinite(rheo)] < 0).sum())
    if neg:
        fails.append(f"rheobase_nA: {neg} negative values")

    # ISI-derived features require enough spikes to be defined.
    for pid in ("step_abs_hi",):
        cnt = f"{pid}.spike_count"
        for feat, need in ((f"{pid}.adaptation_index", 3), (f"{pid}.first_isi_ms", 2),
                           (f"{pid}.cv_isi", 3)):
            if feat in df.columns and cnt in df.columns:
                c = df[cnt].to_numpy(dtype=float)
                f = df[feat].to_numpy(dtype=float)
                bad = int((np.isfinite(f) & np.isfinite(c) & (c < need)).sum())
                if not bad:
                    continue
                # A row extracted under an OLDER, documented feature schema is
                # reported but does not fail the gate -- provided the
                # drift allowance is still valid, which verify_documented_drift()
                # re-derives from the analysis code rather than trusting.
                short = feat.split(".")[-1]
                drifted = any(short in spec["changed"]
                              for spec in DRIFT.get("feature_schema", {}).values())
                if drifted and not verify_documented_drift():
                    documented.append(
                        f"{feat}: {bad} rows with < {need} spikes, all extracted "
                        f"under feature_schema features-1 (documented drift; "
                        f"cleared at the next full regeneration)")
                else:
                    fails.append(f"{feat}: defined on {bad} rows with < {need} spikes")

    # Rheobase-normalised protocols must be unavailable when rheobase is not.
    #
    # "Not available" covers TWO cases, and the gate used to check only the
    # first.  rheobase == 0 means the draw fires at zero injected current, so
    # "1.5x rheobase" is a 0 nA step: the features are finite, they are just not
    # measurements of the protocol whose name they carry.  The old check tested
    # `~isfinite(rheo)` alone and passed 15,234 such rows (2026-08-09 audit,
    # 4.1).  Whether the MODEL is spontaneously active is irrelevant here -- the
    # draw's own rheobase decides.
    for pid in ("step_rheo_1p5x", "step_rheo_3p0x"):
        col = f"{pid}.spike_count"
        if col in df.columns:
            f = df[col].to_numpy(dtype=float)
            bad = int((np.isfinite(f) & ~np.isfinite(rheo)).sum())
            if bad:
                fails.append(f"{pid}: {bad} rows have features without a rheobase")
            at_zero = int((np.isfinite(f) & np.isfinite(rheo) & (rheo == 0.0)).sum())
            if at_zero:
                fails.append(f"{pid}: {at_zero} rows have features at rheobase == 0 "
                             f"(the protocol was applied at 0 nA)")

    # A sample whose rheobase probe reported depolarisation block at the search
    # ceiling must still have a rheobase: that is the whole point of the ladder.
    if "rheobase_blocked_at_ceiling" in df.columns:
        blk = df["rheobase_blocked_at_ceiling"].fillna(False).astype(bool).to_numpy()
        bad = int((blk & ~np.isfinite(rheo)).sum())
        if bad:
            fails.append(f"{bad} rows blocked at the search ceiling but with no rheobase")

    # THE SEARCH RESOLUTION IS PART OF THE ROW, AND MUST BE THE DECLARED ONE.
    #
    # `non_spiking` is a resolution-limited statement: it means "no spike at any
    # of the probed amplitudes", and which amplitudes those were depends on
    # RHEOBASE_SCAN_POINTS. A shard measured at a coarser grid says something
    # weaker than one measured at the declared grid and must not be pooled with
    # it silently (2026-08-09 post-response-3 audit, P0-1).
    if "rheobase_scan_points" in df.columns:
        from compass.adapters.base import NeuronAdapter

        want = int(NeuronAdapter.RHEOBASE_SCAN_POINTS)
        sp = pd.to_numeric(df["rheobase_scan_points"], errors="coerce")
        ok_row = df["simulation_status"] == "ok"
        wrong = int((ok_row & sp.notna() & (sp != want)).sum())
        if wrong:
            fails.append(f"{wrong} rows were measured at a rheobase scan "
                         f"resolution other than the declared {want} points "
                         f"(found {sorted(set(sp[ok_row].dropna().unique()))})")
        missing = int((ok_row & sp.isna()).sum())
        if missing:
            fails.append(f"{missing} successful rows record no rheobase scan "
                         f"resolution; `non_spiking` cannot be read from them")

    # A protocol that never ran cannot have raised a QC flag.
    #
    # Voiding the normalised feature columns for a sample with no usable
    # rheobase left `qc_flags` still naming those protocols, on 15,143 rows --
    # so a stored row was not what a re-run produces. The shard-equivalence
    # check found it by re-simulating whole shards from scratch; this makes it
    # cheap to detect on all 130,098 rows instead of on a sample of six.
    undefined = ~np.isfinite(rheo) | (rheo == 0.0)
    q = df["qc_flags"].fillna("").astype(str)
    ghost = int((undefined & q.str.contains("step_rheo_", regex=False)).sum())
    if ghost:
        fails.append(f"{ghost} rows carry a step_rheo_* QC flag while the "
                     f"rheobase-normalised family is undefined for them")

    err = df["simulation_status"] == "error"
    if "analysis_eligible" in df.columns:
        leak = int((err & df["analysis_eligible"].astype(bool)).sum())
        if leak:
            fails.append(f"{leak} error rows marked analysis_eligible")
    bad_status = sorted(set(df["simulation_status"].unique()) - {"ok", "error"})
    if bad_status:
        fails.append(f"unknown simulation_status values {bad_status}")

    return [gate_record("D3", experiment, fails,
                        measurements={"n_rows": int(len(df)),
                                      "n_error": int(err.sum()),
                                      "documented_schema_drift": documented},
                        tolerances={"rate_window_s": window_s},
                        detail=("; ".join(documented) if documented else ""))]


def verify_documented_drift() -> list[str]:
    """Re-derive the 'not load-bearing' claim instead of trusting the text.

    A documented-drift allowance says a changed feature is consumed by no
    analysis.  That claim is checked here against the actual analysis code, so
    the allowance evaporates the moment the feature becomes load-bearing.
    """
    import importlib.util as iu

    from compass.conditioning.ensembles import (
        CONDITIONING_FEATURES,
        CONDITIONING_FEATURES_NORMALISED,
    )
    from compass.workflows.simulate import DOCUMENTED_DRIFT

    def _load(name, path):
        spec = iu.spec_from_file_location(name, ROOT / "tools" / path)
        mod = iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    consumed = set()
    for f in (*CONDITIONING_FEATURES, *CONDITIONING_FEATURES_NORMALISED):
        consumed.add(f.split(".")[-1])
    try:
        consumed |= {f.split(".")[-1] for f in _load("_ra", "run_analysis.py").RSI_TARGETS}
        consumed |= set(_load("_rp", "run_perturbations.py").PERT_FEATURES)
    except Exception as exc:  # noqa: BLE001
        return [f"could not verify drift allowances: {exc}"]

    bad = []
    for field, pairs in DOCUMENTED_DRIFT.items():
        for (old_v, new_v), spec in pairs.items():
            for feat in spec["changed"]:
                if feat in consumed:
                    bad.append(
                        f"{field} {old_v}->{new_v} is tolerated on the grounds "
                        f"that {feat!r} feeds no analysis, but {feat!r} IS "
                        f"consumed. The allowance is void; quarantine and "
                        f"regenerate the affected shards.")
    return bad


def _sobol_module():
    spec = importlib.util.spec_from_file_location("_rs", ROOT / "tools" / "run_sobol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d4_shard_identity(shards, rc, experiment) -> list:
    """Every shard bound to a recorded, checked identity."""
    records = [gate_record(
        "D4", "documented_drift_allowances", verify_documented_drift(),
        detail="each tolerated identity difference must still be non-load-bearing")]
    for p in shards:
        df = pd.read_parquet(p, columns=["model_id", "sampling_regime", "replicate",
                                         "seed", "prior_id"])
        mid = str(df["model_id"].iloc[0])
        if experiment == "sobol":
            # The Sobol design has its own identity (design, base N, features);
            # it is not a `campaigns.json` entry and has no per-model sample
            # size, so it cannot be validated against the campaign contract.
            rs = _sobol_module()
            ident = rs.flat_identity(mid, rs.shard_identity(mid),
                                     int(pd.read_parquet(
                                         p, columns=["parameter_sample_id"])
                                         ["parameter_sample_id"].min()), len(df))
        else:
            ident = rc.job_identity(experiment, mid,
                                    str(df["sampling_regime"].iloc[0]),
                                    int(df["replicate"].iloc[0]),
                                    int(df["seed"].iloc[0]),
                                    str(df["prior_id"].iloc[0]), len(df))
        ok, why = shard_is_reusable(p, ident, len(df))
        side = sidecar_path(p)
        backfilled = False
        if side.exists():
            backfilled = json.loads(side.read_text()).get(
                "identity_provenance") == "backfilled"
        records.append(gate_record(
            "D4", p.name, [] if ok else [why],
            measurements={"n_rows": int(len(df)),
                          "identity_provenance": "backfilled" if backfilled else "recorded"},
            inputs={"shard": str(p.relative_to(ROOT))}))
    return records


# Features a model pins BY CONSTRUCTION, with the mechanism.  Anything not
# listed here that collapses to a handful of values is a defect until proven
# otherwise; listing one requires naming why the model determines it.
EXPECTED_POINT_MASS = {
    ("migliore_ca1", "step_abs_hi.resting_potential_mV"):
        "the published model pins V_rest by solving e_pas(x) = v(x) + "
        "(ina+ik)/g_pas after finitialize (MiglioreCA1._post_finitialize), so "
        "resting potential is -65 mV for every parameter sample by design",
    ("bbp_l5pc", "step_abs_hi.mean_firing_rate_Hz"):
        "Hay L5b PC fires 14-18 Hz at its 1.1 nA step, i.e. 6-7 spikes in the "
        "400 ms window, and E6's clustered prior concentrates the draws further; "
        "few distinct counts is the readout's resolution at this rate, not a "
        "point mass in the model",
    ("bbp_l5pc", "step_abs_hi.spike_count"):
        "same as mean_firing_rate_Hz, of which it is the numerator",
}


def d5_population_diagnostics(df, experiment) -> list:
    """Per model x regime rates, missingness and feature ranges."""
    records = []
    exp = D5_EXPECTATIONS
    for (m, r), g in df.groupby(["model_id", "sampling_regime"]):
        fails: list[str] = []
        documented: list[str] = []
        n = len(g)
        err = float((g["simulation_status"] == "error").mean())
        flags = g["qc_flags"].astype("string").fillna("")
        numerical = float(flags.str.contains(
            "solver_failure|nonfinite_voltage|nonphysical_voltage", regex=True).mean())
        nonspk = float(g["non_spiking"].astype(bool).mean())
        if err > exp["max_error_fraction"]:
            fails.append(f"error rate {err:.3f} > {exp['max_error_fraction']}")
        if numerical > exp["max_numerical_qc_fraction"]:
            fails.append(f"numerical QC rate {numerical:.3f} > "
                         f"{exp['max_numerical_qc_fraction']}")
        loq, hiq = exp["nonspiking_fraction_range"]
        if not (loq <= nonspk <= hiq):
            fails.append(f"nonspiking fraction {nonspk:.3f} outside [{loq}, {hiq}]")

        feat_cols = [c for c in g.columns if c.startswith("step_abs_hi.")]
        for c in feat_cols[:12]:
            v = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            n_distinct = len(np.unique(np.round(v, 9)))
            if v.size >= 50 and n_distinct < exp["min_distinct_feature_values"]:
                why = EXPECTED_POINT_MASS.get((m, c))
                msg = (f"{c}: only {n_distinct} distinct values over {v.size} "
                       f"finite rows")
                if why:
                    documented.append(f"{msg} -- expected: {why}")
                else:
                    fails.append(msg + " (suspicious point mass)")
        # RECOMPUTE physiology_class from the stored columns with the CURRENT
        # classifier rather than trusting the label a row was written with.
        # `row-validity-1` read `depolarised_rest` as depolarisation block, which
        # is true of every spontaneously active cell's baseline and mislabelled
        # 81% of a pacemaker's rows. Stored rows are immutable, so the corrected
        # label is derived here and both are reported.
        from compass.protocols import common as P
        from compass.workflows.simulate import VALIDATOR_VERSION, classify_row

        spont = m in P.SPONTANEOUSLY_ACTIVE
        recomputed, stale = {}, 0
        for _, row in g.iterrows():
            row_flags = [f for f in str(row.get("qc_flags", "")).split(";") if f]
            rheo = row.get("rheobase_nA")
            rheo = None if (rheo is None or not np.isfinite(rheo)) else float(rheo)
            err_msg = row.get("error") if row.get("simulation_status") == "error" else None
            cls = classify_row(row.to_dict(), row_flags, rheo, spont,
                               err_msg)["physiology_class"]
            recomputed[cls] = recomputed.get(cls, 0) + 1
            if "physiology_class" in row and row["physiology_class"] != cls:
                stale += 1
        # A stale label is REPORTED, not failed. `physiology_class` is a pure
        # function of columns that are already stored, so recomputing it here is
        # exact rather than approximate -- the stored column is a cache, and the
        # recomputation supersedes it. What WOULD fail is being unable to
        # recompute, because that means the inputs are missing.
        if not recomputed:
            fails.append("physiology_class could not be recomputed: the stored "
                         "columns the classifier needs are absent")
        stored_counts = (dict(g["physiology_class"].value_counts())
                         if "physiology_class" in g else {})
        # Eligibility is NOT recomputable-and-forgivable in the same way: it
        # gates which rows enter the analysis, so a disagreement there is a
        # genuine failure.
        elig_stale, elig_absent = 0, 0
        if "analysis_eligible" in g:
            for _, row in g.iterrows():
                stored = row["analysis_eligible"]
                # A row written before the validator existed has NO stored
                # value.  `bool(nan)` is True, so comparing against it invents a
                # disagreement out of a missing column -- which is exactly the
                # false positive this check must not produce.
                if stored is None or (isinstance(stored, float) and np.isnan(stored)):
                    elig_absent += 1
                    continue
                rf = [f for f in str(row.get("qc_flags", "")).split(";") if f]
                rh = row.get("rheobase_nA")
                rh = None if (rh is None or not np.isfinite(rh)) else float(rh)
                em = row.get("error") if row.get("simulation_status") == "error" else None
                want = classify_row(row.to_dict(), rf, rh, spont, em)["analysis_eligible"]
                elig_stale += int(bool(stored) != bool(want))
            if elig_stale:
                fails.append(
                    f"{elig_stale}/{n} rows with a STORED analysis_eligible "
                    f"disagree with the current classifier; that column decides "
                    f"what enters the analysis and cannot be treated as a cache")

        records.append(gate_record(
            "D5", f"{experiment}/{m}/{r}", fails,
            measurements={
                "n": n, "error_fraction": round(err, 5),
                "physiology_classes_recomputed": recomputed,
                "physiology_classes_as_stored": stored_counts,
                "n_labels_superseded": int(stale),
                "n_eligibility_not_stored": int(elig_absent),
                "documented_point_masses": documented,
                "validator_version_current": VALIDATOR_VERSION,
                "numerical_qc_fraction": round(numerical, 5),
                "nonspiking_fraction": round(nonspk, 5),
                "any_qc_fraction": round(float((flags != "").mean()), 5),
                "eligible_fraction": (round(float(g["analysis_eligible"].mean()), 5)
                                      if "analysis_eligible" in g else None),
                "physiology_classes": (g["physiology_class"].value_counts().to_dict()
                                       if "physiology_class" in g else {}),
            },
            tolerances=dict(exp)))
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="screening")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gates", nargs="*", default=["D1", "D2", "D3", "D4", "D5"])
    args = ap.parse_args()

    rc = _campaign_module()
    # Superseded archives are retained EVIDENCE, not live inputs: they were
    # produced under settings the project no longer uses, so validating them
    # against the current configuration would be a category error.
    known = set(rc.CONFIG["campaigns"]) | {"sobol"}
    experiments = ([p.name for p in sorted(FEATURES.iterdir())
                    if p.is_dir() and p.name in known]
                   if args.all else [args.experiment])
    if args.all:
        archived = [p.name for p in sorted(FEATURES.iterdir())
                    if p.is_dir() and p.name not in known]
        if archived:
            print(f"archived, not validated against current config: {archived}")

    all_records = []
    for exp in experiments:
        df, shards = load_experiment(exp)
        if df.empty:
            print(f"{exp}: no rows, skipped")
            continue
        print(f"\n=== {exp}: {len(df)} rows across {len(shards)} shards ===", flush=True)
        recs = []
        # The caller used to gate on campaigns.json membership too, so the
        # branch inside d1_design_and_keys for the Saltelli design was
        # unreachable. D1 is a design-integrity gate and `sobol` has a design.
        if "D1" in args.gates and (exp in rc.CONFIG["campaigns"] or exp == "sobol"):
            recs += d1_design_and_keys(df, shards, exp, rc)
        if "D2" in args.gates and exp in rc.CONFIG["campaigns"]:
            recs += d2_parameter_provenance(df, exp, rc)
        if "D3" in args.gates:
            recs += d3_row_invariants(df, exp)
        if "D4" in args.gates:
            recs += d4_shard_identity(shards, rc, exp)
        if "D5" in args.gates:
            recs += d5_population_diagnostics(df, exp)
        for r in recs:
            if not r.passed:
                print(f"  [{r.status}] {r.gate} {r.subject}: "
                      f"{'; '.join(r.failures[:3])}", flush=True)
        by_gate = {}
        for r in recs:
            g = by_gate.setdefault(r.gate, [0, 0])
            g[0] += 1
            g[1] += int(r.passed)
        for g, (n, ok) in sorted(by_gate.items()):
            print(f"  {g}: {ok}/{n} passed", flush=True)
        all_records += recs

    dest = OUT / "campaign_integrity.json"
    write_gate_records(all_records, dest)
    n_ok = sum(r.passed for r in all_records)
    print(f"\n{n_ok}/{len(all_records)} gate records passed -> {dest}")
    return 0 if n_ok == len(all_records) else 1


if __name__ == "__main__":
    sys.exit(main())
