"""Feature-level numerical convergence over PRIOR SAMPLES (review F1, F12).

``tools/numerical_qc.py`` tests a single canonical parameter point and accepts a
setting on spike count, first-spike time and peak voltage.  That is too narrow
for a campaign whose whole purpose is to sweep a broad prior: threshold
crossings, silence and depolarisation block are parameter-dependent, so
convergence at the canonical point does not transfer.

This tool instead:

* draws a stratified set of parameter samples from the model's own prior;
* runs the full common-core protocol suite at each (dt, mesh) setting;
* compares **every conditioning feature** against the finest setting;
* accepts a setting only if all features agree within tolerance on all samples.

Output: ``data/provenance/convergence_audit_<model>.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import get_adapter  # noqa: E402
from compass.conditioning.ensembles import CONDITIONING_FEATURES  # noqa: E402
from compass.features.extract import extract, extract_rest  # noqa: E402
from compass.protocols import common as P  # noqa: E402
from compass.sampling.priors import independent_sample  # noqa: E402

OUT = ROOT / "data" / "provenance"

# A feature counts as reproduced when it matches the reference within EITHER a
# relative tolerance OR an absolute floor (features near zero cannot be judged
# relatively).
REL_TOL = 0.05

# A convergence tolerance may never be finer than the readout's own quantisation
# step, or it demands a precision the measurement cannot express and no mesh can
# satisfy.  `mean_firing_rate_Hz` is a spike COUNT over the 400 ms analysis
# window, so its quantum is 1000/400 = 2.5 Hz; the tolerance was 2.0 Hz.  A
# single threshold-adjacent spike therefore scored as a convergence failure at
# every rate below 40 Hz, while the identical difference in `spike_count`
# (ABS_TOL 1.0, one spike) scored as agreement.  The same physical event was
# both a pass and a fail depending on the units it was reported in.
#
# Raised to exactly one quantum on principle, decided from the protocol geometry
# and not from any model's result.  Note that it does NOT change the verdict for
# `mainen_l5pc`, which still fails on `adaptation_index`: this correction was not
# a route to a wanted answer.  See D27 and RESULTS 5.1.
_RATE_QUANTUM_HZ = 1000.0 / 400.0

ABS_TOL = {
    "rest.resting_potential_mV": 0.5,
    "hyperpol_abs.input_resistance_Mohm": 1.0,
    "hyperpol_abs.membrane_tau_ms": 0.5,
    "hyperpol_abs.sag_ratio": 0.02,
    "rheobase_nA": 0.01,
    "step_abs_hi.first_spike_latency_ms": 1.0,
    "step_abs_hi.ap_amplitude_mV": 2.0,
    "step_abs_hi.ap_halfwidth_ms": 0.1,
    "step_abs_hi.ahp_depth_mV": 2.0,
    "step_abs_hi.first_isi_ms": 2.0,
    "step_abs_hi.mean_firing_rate_Hz": _RATE_QUANTUM_HZ,   # was 2.0: below the quantum
    "step_abs_hi.adaptation_index": 0.05,
    "step_abs_hi.spike_count": 1.0,
}


def set_mesh(adapter, factor: float) -> int:
    """Set every section's nseg to ``factor`` x the d_lambda(100 Hz) count.

    ``adapter.reset()`` FIRST is not optional.  ``cm`` and ``Ra`` are themselves
    sampled parameters in several models, so calling this while a parameter
    sample is applied would (a) compute the electrotonic length constant from
    perturbed passive properties, giving a different mesh for every setting, and
    (b) re-snapshot the perturbed values as the new canonical baseline, so every
    subsequent ``apply`` would scale an already-scaled profile.  Both happened in
    the first version of this tool: the mesh drifted 1106 -> 1474 -> 2002 ->
    10546 segments across settings and agreement degraded monotonically for
    reasons that had nothing to do with discretisation.
    """
    adapter.reset()
    secs = adapter._sections.get("all")
    if secs is None:
        # Point models have no "all" section list and nothing to refine.  The
        # earlier version raised KeyError here and, because the caller filtered
        # stdout, four models were silently skipped.
        secs = [adapter.soma]
    secs = list(secs)
    if factor is not None:
        for sec in secs:
            lam = 1e5 * np.sqrt(sec.diam / (4.0 * np.pi * 100.0 * sec.Ra * sec.cm))
            base = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1
            sec.nseg = max(1, int(round(base * factor)) // 2 * 2 + 1)
        adapter._canonical.clear()
        adapter._snapshot_canonical()
    return sum(s.nseg for s in secs)


def feature_vector(adapter, kw) -> dict:
    """Full common-core feature row for the currently applied parameters."""
    row = {}
    rheo = adapter.find_rheobase(delay_ms=P.RHEO_DELAY_MS,
                                 duration_ms=P.RHEO_DURATION_MS,
                                 total_ms=P.RHEO_TOTAL_MS, **kw)
    row["rheobase_nA"] = np.nan if rheo is None else float(rheo)
    for spec in P.common_core(adapter.manifest.model_id):
        if spec.amplitude_rheobase_mult is not None:
            continue                      # normalised family not audited here
        if spec.kind == "rest":
            rec = adapter.simulate(0.0, delay_ms=0.0, duration_ms=0.0,
                                   total_ms=spec.total_ms, protocol_id="c", **kw)
            f = extract_rest(rec)
        else:
            rec = adapter.simulate(float(spec.amplitude_nA), delay_ms=spec.delay_ms,
                                   duration_ms=spec.duration_ms,
                                   total_ms=spec.total_ms, protocol_id="c", **kw)
            f = extract(rec)
        for k, v in f.items():
            row[f"{spec.protocol_id}.{k}"] = float(v) if v is not None and np.isfinite(v) else np.nan
    return row


def agrees(a: float, b: float, name: str) -> bool:
    """Does feature value ``a`` reproduce reference ``b``?"""
    if not np.isfinite(a) and not np.isfinite(b):
        return True                        # both undefined: consistent
    if not np.isfinite(a) or not np.isfinite(b):
        return False                       # one defined, one not: a real change
    tol = ABS_TOL.get(name, 1e-6)
    return abs(a - b) <= max(tol, REL_TOL * abs(b))


def rescore(model_id: str) -> int:
    """Re-evaluate a stored audit under the current tolerances.

    Every setting keeps its raw feature rows, so a tolerance correction does not
    require re-running the simulations -- which for `mainen_l5pc` cost 75 minutes
    on 7 cores.  Storing the rows was a deliberate choice; this is what makes it
    pay off, and it is why a tolerance can be fixed without the fix being
    entangled with a decision about whether to spend compute.
    """
    path = OUT / f"convergence_audit_{model_id}.json"
    if not path.exists():
        print(f"no stored audit at {path}")
        return 1
    d = json.loads(path.read_text())
    ref_rows = d["reference"]["rows"]
    feats = list(CONDITIONING_FEATURES)
    print(f"{model_id}: re-scoring {len(d['settings'])} stored settings against "
          f"reference dt={d['reference']['dt_ms']} "
          f"mesh={d['reference'].get('mesh_factor')}x "
          f"({d['reference'].get('total_nseg')} seg)")
    for key, s in d["settings"].items():
        per = {f: float(np.mean([agrees(a.get(f, np.nan), b.get(f, np.nan), f)
                                 for a, b in zip(s["rows"], ref_rows, strict=True)]))
               for f in feats}
        worst = min(per, key=per.get)
        was, s["fraction_features_reproduced"] = s["fraction_features_reproduced"], \
            float(np.mean(list(per.values())))
        s["min_feature_agreement"] = per[worst]
        s["worst_feature"] = worst
        s["accepted"] = bool(min(per.values()) == 1.0)
        s["per_feature_agreement"] = per
        failing = [k.split(".")[-1] for k, v in per.items() if v < 1.0]
        print(f"  {key:26s} mean {was:.3f} -> {s['fraction_features_reproduced']:.3f}  "
              f"{'ACCEPT' if s['accepted'] else 'reject'}  failing={failing}")
    ok = {k: v for k, v in d["settings"].items() if v["accepted"]}
    d["chosen"] = min(ok, key=lambda k: ok[k]["wall_s"]) if ok else None
    d["tolerances"] = {"rel_tol": REL_TOL, "abs_tol": dict(ABS_TOL)}
    d["rescored_utc"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(d, indent=1, default=float))
    print(f"\nchosen: {d['chosen']}")
    return 0


def _audit_setting(job):
    """Evaluate one (dt, mesh) setting in a FRESH process.

    A fresh adapter per setting removes the last source of coupling between
    settings: changing ``nseg`` makes NEURON re-interpolate segment values, so
    repeatedly re-meshing one adapter leaves ~1.6% hysteresis in the canonical
    spatial profile.  Building from scratch also lets the settings run in
    parallel, which is what makes a prior-sample audit affordable at all.
    """
    import sys as _s

    _s.path.insert(0, str(ROOT / "src"))
    model_id, dt, mesh, points, pids = job
    from compass.adapters.models import get_adapter as _ga

    adapter = _ga(model_id).build()
    nseg = set_mesh(adapter, mesh)
    t0 = time.time()
    rows = []
    for p in points:
        adapter.reset()
        adapter.apply(dict(zip(pids, p, strict=True)))
        rows.append(feature_vector(adapter, {"cvode": False, "dt_ms": dt}))
    return {"dt_ms": dt, "mesh_factor": mesh, "total_nseg": nseg,
            "wall_s": round(time.time() - t0, 1), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--dts", nargs="*", type=float, default=[0.05, 0.025, 0.0125])
    ap.add_argument("--mesh", nargs="*", default=["as_built", "1.0", "2.0"],
                    help="'as_built' keeps the adapter's own nseg; numbers are "
                         "multiples of the d_lambda(100 Hz) mesh")
    ap.add_argument("--ref-dt", type=float, default=0.00625)
    ap.add_argument("--ref-mesh", default="2.0",
                    help="'as_built' or a d_lambda multiple; use as_built for "
                         "point models, where nseg 1->3 is a different model "
                         "(spatially resolved mechanisms) not a refinement")
    ap.add_argument("--workers", type=int, default=7,
                    help="parallel settings evaluated at once (default 7). This "
                         "pool used to be hard-coded, so an audit run alongside "
                         "a campaign silently oversubscribed the machine")
    ap.add_argument("--rescore", action="store_true",
                    help="re-evaluate the stored audit under the CURRENT "
                         "tolerances, reusing its saved feature rows; no "
                         "simulation is run")
    args = ap.parse_args()

    if args.rescore:
        return rescore(args.model)

    probe = get_adapter(args.model)
    specs = probe.parameter_specs()
    pids = [s.parameter_id for s in specs]
    # deterministic stratified prior sample + the canonical point
    scales = independent_sample(specs, args.samples, seed=20260805, method="sobol")
    points = [np.ones(len(pids)).tolist()] + [scales[i].tolist() for i in range(len(scales))]

    meshes = [None if str(m) == "as_built" else float(m) for m in args.mesh]
    settings = [(d, m) for m in meshes for d in args.dts]
    ref_mesh = None if str(args.ref_mesh) == "as_built" else float(args.ref_mesh)
    ref = (args.ref_dt, ref_mesh)
    if ref in settings:
        settings.remove(ref)
    jobs = [(args.model, d, m, points, pids) for d, m in [ref, *settings]]

    results: dict = {"model_id": args.model, "n_points": len(points),
                     "reference": {"dt_ms": ref[0], "mesh_factor": ref[1]},
                     "rel_tol": REL_TOL, "abs_tol": ABS_TOL, "settings": {}}

    t_all = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(jobs), args.workers), maxtasksperchild=1) as pool:
        done = list(pool.imap_unordered(_audit_setting, jobs))
    print(f"all {len(jobs)} settings evaluated in {(time.time()-t_all)/60:.1f} min", flush=True)

    by_key = {(r["dt_ms"], r["mesh_factor"]): r for r in done}
    ref_res = by_key[ref]
    ref_rows = ref_res["rows"]
    results["reference"].update({"total_nseg": ref_res["total_nseg"],
                                 "wall_s": ref_res["wall_s"],
                                 "rows": ref_rows})
    print(f"reference dt={ref[0]} mesh={ref[1]}x ({ref_res['total_nseg']} seg) "
          f"{ref_res['wall_s']}s", flush=True)

    feats = list(CONDITIONING_FEATURES)
    for dt, mesh in settings:
        r = by_key[(dt, mesh)]
        per_feature = {}
        for f in feats:
            ok_f = [agrees(a.get(f, np.nan), b.get(f, np.nan), f)
                    for a, b in zip(r["rows"], ref_rows, strict=True)]
            per_feature[f] = float(np.mean(ok_f))
        worst = min(per_feature, key=per_feature.get)
        key = f"dt{dt}_mesh{'as_built' if mesh is None else str(mesh)+'x'}"
        results["settings"][key] = {
            "dt_ms": dt, "mesh_factor": mesh, "total_nseg": r["total_nseg"],
            "wall_s": r["wall_s"],
            "fraction_features_reproduced": float(np.mean(list(per_feature.values()))),
            "min_feature_agreement": per_feature[worst],
            "worst_feature": worst,
            "accepted": bool(min(per_feature.values()) == 1.0),
            "per_feature_agreement": per_feature,
            # raw feature rows retained so a later audit can add settings
            # without paying for the (dominant) reference run again
            "rows": r["rows"],
        }
        rr = results["settings"][key]
        print(f"  dt={dt:<7} mesh={'as_built' if mesh is None else f'{mesh}x':<8} "
              f"({r['total_nseg']:>5} seg) {rr['wall_s']:>7.1f}s  "
              f"mean_agree={rr['fraction_features_reproduced']:.3f}  "
              f"worst={worst.split('.')[-1]}@{per_feature[worst]:.2f}  "
              f"{'ACCEPT' if rr['accepted'] else 'reject'}", flush=True)

    ok = {k: v for k, v in results["settings"].items() if v["accepted"]}
    results["chosen"] = min(ok, key=lambda k: ok[k]["wall_s"]) if ok else None
    results["tolerances"] = {"rel_tol": REL_TOL, "abs_tol": dict(ABS_TOL)}
    results["argv"] = sys.argv[1:]
    results["written_utc"] = datetime.now(UTC).isoformat()

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"convergence_audit_{args.model}.json"
    # An audit output is EVIDENCE.  This file previously justified the Mainen
    # mesh refinement in a source comment, and a later `--mesh as_built` run
    # overwrote it: the accepted production combination vanished and `chosen`
    # became null, so the code cited a file that no longer contained the claim.
    # Supersede rather than overwrite (audit C1 / D26); nothing is ever lost.
    if dest.exists():
        prev = json.loads(dest.read_text())
        tag = prev.get("written_utc", datetime.fromtimestamp(
            dest.stat().st_mtime, UTC).isoformat()).replace(":", "").replace("-", "")
        old_dir = OUT / "_superseded"
        old_dir.mkdir(parents=True, exist_ok=True)
        moved = old_dir / f"convergence_audit_{args.model}_{tag}.json"
        dest.rename(moved)
        print(f"previous audit preserved at {moved.relative_to(ROOT)}", flush=True)
    dest.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nchosen: {results['chosen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
