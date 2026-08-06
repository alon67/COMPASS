"""Compile NEURON mod files into per-model mechanism libraries.

Each model gets an isolated build directory under ``data/models/<model_id>/mod``
so that mechanism name collisions between independently developed models (e.g.
two different ``na.mod`` implementations) cannot occur.  Simulation workers are
pinned to one model and load only that model's library.

The compiled library hash is recorded for provenance (blueprint section 20).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "data" / "models"
ARCH = platform.machine()  # arm64 / x86_64

# model_id -> list of directories (relative to data/raw) whose *.mod files are needed
MOD_SOURCES: dict[str, list[str]] = {
    "hh_point": [],  # NEURON built-in hh + pas only
    "pospischil_rs": ["modeldb_123623/PospischilEtAl2008"],
    "pospischil_fs": ["modeldb_123623/PospischilEtAl2008"],
    "pospischil_ib": ["modeldb_123623/PospischilEtAl2008"],
    "khaliq_purkinje": ["modeldb_48332/purkinje"],
    "mainen_l5pc": ["modeldb_2488/cells"],
    "migliore_ca1": ["modeldb_2796/ca1"],
    "bbp_l5pc": ["bbp_bluepyopt/examples/l5pc/mechanisms"],
    "dopaminergic_sn": ["modeldb_267306"],
    "stg_lp": ["modeldb_267621"],
}


def collect_mods(model_id: str) -> list[Path]:
    out: dict[str, Path] = {}
    for rel in MOD_SOURCES[model_id]:
        base = RAW / rel
        if not base.exists():
            print(f"  !! missing source dir {base}")
            continue
        for p in sorted(base.rglob("*.mod")):
            if "__MACOSX" in p.parts:
                continue
            out.setdefault(p.name, p)  # first occurrence wins, deterministic by sort
    return list(out.values())


def build(model_id: str) -> dict:
    dest = MODELS / model_id / "mod"
    rec = {"model_id": model_id, "arch": ARCH}
    mods = collect_mods(model_id)
    if not mods:
        rec.update(status="ok", n_mod=0, note="built-in mechanisms only")
        return rec
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for p in mods:
        shutil.copy2(p, dest / p.name)
    # copy any .inc files referenced by INCLUDE statements
    for rel in MOD_SOURCES[model_id]:
        for inc in (RAW / rel).rglob("*.inc"):
            if "__MACOSX" not in inc.parts:
                shutil.copy2(inc, dest / inc.name)
    proc = subprocess.run(
        ["nrnivmodl"], cwd=dest, capture_output=True, text=True, timeout=1800
    )
    lib = dest / ARCH / ".libs" / "libnrnmech.so"
    if not lib.exists():
        alt = list((dest / ARCH).glob("**/libnrnmech*")) if (dest / ARCH).exists() else []
        lib = alt[0] if alt else None
    if lib and lib.exists():
        rec.update(
            status="ok",
            n_mod=len(mods),
            lib=str(lib.relative_to(ROOT)),
            sha256=hashlib.sha256(lib.read_bytes()).hexdigest()[:16],
        )
    else:
        rec.update(status="error", n_mod=len(mods),
                   stderr=proc.stderr[-1500:], stdout=proc.stdout[-800:])
    return rec


def main(argv: list[str]) -> int:
    targets = argv[1:] or list(MOD_SOURCES)
    out = []
    for mid in targets:
        print(f"building {mid} ...", flush=True)
        rec = build(mid)
        out.append(rec)
        print(f"  [{rec['status']}] mods={rec.get('n_mod')} {rec.get('sha256','')}", flush=True)
        if rec["status"] == "error":
            print(rec.get("stderr", "")[-800:])
    path = ROOT / "data" / "provenance" / "mechanism_builds.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    by_id = {r["model_id"]: r for r in existing}
    by_id.update({r["model_id"]: r for r in out})
    path.write_text(json.dumps(list(by_id.values()), indent=1))
    ok = sum(r["status"] == "ok" for r in out)
    print(f"\n{ok}/{len(out)} mechanism libraries built")
    return 0 if ok == len(out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
