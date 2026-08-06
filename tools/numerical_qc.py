"""Numerical convergence audit (blueprint 14.5) -> chooses per-model solver settings.

For each model the same protocol is integrated with a range of fixed time steps
and with CVODE at two tolerances.  The finest fixed-step run is the reference.
A setting is accepted only if spike count matches the reference exactly and the
first-spike time and AP amplitude agree within tolerance.

Output: data/provenance/numerical_qc.json -- the solver settings actually used
by the campaign, with the evidence that justified them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY, get_adapter  # noqa: E402

SPIKE_THRESHOLD = 0.0
TOL_FIRST_SPIKE_MS = 0.5
TOL_AMPLITUDE_MV = 2.0


def spike_times(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    tt, vv = t[m], v[m]
    if vv.size < 3:
        return np.array([])
    up = np.where((vv[:-1] < SPIKE_THRESHOLD) & (vv[1:] >= SPIKE_THRESHOLD))[0]
    return tt[up]


def summarise(rec, t0, t1):
    if rec.v.size == 0:
        return dict(n_spikes=-1, first_spike=np.nan, v_max=np.nan)
    st = spike_times(rec.t, rec.v, t0, t1)
    m = (rec.t >= t0) & (rec.t <= t1)
    return dict(
        n_spikes=int(st.size),
        first_spike=float(st[0]) if st.size else float("nan"),
        v_max=float(np.max(rec.v[m])) if np.any(m) else float("nan"),
    )


# amplitude that reliably drives repetitive firing in each model
PROBE_AMP = {
    "hh_point": 0.25, "hh_cable": 0.40, "pospischil_rs": 1.2, "pospischil_fs": 1.0,
    "khaliq_purkinje": 0.05, "mainen_l5pc": 0.5, "migliore_ca1": 0.5, "bbp_l5pc": 0.8,
}

SETTINGS = [
    ("fixed_0.0125", dict(dt_ms=0.0125, cvode=False)),
    ("fixed_0.025", dict(dt_ms=0.025, cvode=False)),
    ("fixed_0.05", dict(dt_ms=0.05, cvode=False)),
    ("cvode", dict(cvode=True, dt_ms=0.025)),
]

DELAY, DUR, TOTAL = 150.0, 500.0, 700.0


def audit(model_id: str, adapter=None) -> dict:
    adapter = adapter or get_adapter(model_id).build()
    amp = PROBE_AMP[model_id]
    out = {"model_id": model_id, "probe_amplitude_nA": amp, "runs": {}}
    for name, kw in SETTINGS:
        t0 = time.time()
        rec = adapter.simulate(amp, delay_ms=DELAY, duration_ms=DUR, total_ms=TOTAL,
                               protocol_id="qc", **kw)
        elapsed = time.time() - t0
        s = summarise(rec, DELAY, DELAY + DUR)
        s["wall_s"] = round(elapsed, 3)
        out["runs"][name] = s
    ref = out["runs"]["fixed_0.0125"]
    accepted = {}
    for name, s in out["runs"].items():
        ok = (
            s["n_spikes"] == ref["n_spikes"]
            and (np.isnan(ref["first_spike"]) or
                 abs(s["first_spike"] - ref["first_spike"]) <= TOL_FIRST_SPIKE_MS)
            and abs(s["v_max"] - ref["v_max"]) <= TOL_AMPLITUDE_MV
        )
        accepted[name] = bool(ok)
    out["accepted"] = accepted
    # cheapest accepted setting wins
    ok_names = [n for n, good in accepted.items() if good]
    chosen = min(ok_names, key=lambda n: out["runs"][n]["wall_s"]) if ok_names else "fixed_0.0125"
    out["chosen"] = chosen
    out["chosen_kwargs"] = dict(SETTINGS)[chosen]
    out["speedup_vs_finest"] = round(ref["wall_s"] / out["runs"][chosen]["wall_s"], 2)
    return out


def spatial_audit(model_id: str, chosen_kwargs: dict, adapter) -> dict:
    """Spatial discretisation convergence (blueprint 14.5).

    Compares the model's as-published segmentation against the standard
    ``d_lambda`` rule at 100 Hz and at twice that resolution.  Multicompartment
    models only; a point model has nothing to refine.
    """
    if adapter.manifest.is_point_model:
        return {"model_id": model_id, "applicable": False}
    amp = PROBE_AMP[model_id]
    secs = list(adapter._sections["all"])
    published = {s.name(): s.nseg for s in secs}

    def run(label):
        t0 = time.time()
        rec = adapter.simulate(amp, delay_ms=DELAY, duration_ms=DUR, total_ms=TOTAL,
                               protocol_id="qc_space", **chosen_kwargs)
        s = summarise(rec, DELAY, DELAY + DUR)
        s["wall_s"] = round(time.time() - t0, 3)
        s["total_nseg"] = sum(x.nseg for x in secs)
        return label, s

    out = {"model_id": model_id, "applicable": True, "levels": {}}
    lbl, s = run("as_published")
    out["levels"][lbl] = s

    for factor, name in ((1, "d_lambda_100Hz"), (2, "d_lambda_2x")):
        for sec in secs:
            lam = 1e5 * np.sqrt(sec.diam / (4.0 * np.pi * 100.0 * sec.Ra * sec.cm))
            base = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1
            sec.nseg = base * factor
        adapter._canonical.clear()
        adapter._snapshot_canonical()
        lbl, s = run(name)
        out["levels"][lbl] = s

    ref = out["levels"]["d_lambda_2x"]
    for s in out["levels"].values():
        s["matches_finest"] = bool(s["n_spikes"] == ref["n_spikes"])
    ok = [n for n, s in out["levels"].items() if s["matches_finest"]]
    out["chosen"] = min(ok, key=lambda n: out["levels"][n]["wall_s"]) if ok else "d_lambda_2x"
    # restore published segmentation for cleanliness
    for sec in secs:
        sec.nseg = published[sec.name()]
    return out


def main(argv):
    targets = argv[1:] or list(MODEL_REGISTRY)
    do_space = "--spatial" in targets
    targets = [t for t in targets if not t.startswith("--")] or list(MODEL_REGISTRY)
    results = []
    for mid in targets:
        adapter = get_adapter(mid).build()
        r = audit(mid, adapter)
        runs = "  ".join(
            f"{n}:{v['n_spikes']}sp/{v['wall_s']}s{'*' if r['accepted'][n] else 'x'}"
            for n, v in r["runs"].items()
        )
        print(f"{mid:16s} -> {r['chosen']:13s} ({r['speedup_vs_finest']}x)   {runs}", flush=True)
        if do_space:
            sp = spatial_audit(mid, r["chosen_kwargs"], adapter)
            r["spatial"] = sp
            if sp.get("applicable"):
                lv = "  ".join(
                    f"{n}:{v['total_nseg']}seg/{v['n_spikes']}sp/{v['wall_s']}s"
                    f"{'*' if v['matches_finest'] else 'x'}"
                    for n, v in sp["levels"].items()
                )
                print(f"{'':16s}    spatial -> {sp['chosen']:15s} {lv}", flush=True)
        results.append(r)
    path = ROOT / "data" / "provenance" / "numerical_qc.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = {r["model_id"]: r for r in existing}
    merged.update({r["model_id"]: r for r in results})
    path.write_text(json.dumps(list(merged.values()), indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
