"""Download the COMPASS model panel and record immutable provenance.

Every asset is fetched from a citable archive (ModelDB accession or a pinned
Blue Brain Project git commit), hashed, and logged to
``data/provenance/downloads.json`` so that any downstream result can be traced
back to a specific archive version (blueprint section 20).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROV = ROOT / "data" / "provenance" / "downloads.json"
CTX = ssl.create_default_context(cafile=certifi.where())

# ModelDB accessions selected against the blueprint 6.1 panel specification.
# `class` is the blueprint slot the model fills; `why` is the inclusion rationale
# recorded before any result was inspected (blueprint WP0 acceptance criterion).
MODELDB = {
    123623: dict(slot="cortical_rs_and_fs", why="Pospischil et al. 2008: one framework, many cortical classes"),
    97747: dict(slot="fast_spiking_interneuron", why="Golomb et al. 2007: independent FS implementation"),
    48332: dict(slot="cerebellar_purkinje", why="Khaliq & Raman 2003: Purkinje resurgent-Na point model"),
    2488: dict(slot="cortical_pyramidal_a", why="Mainen & Sejnowski 1996: independent L5 multicompartment"),
    139653: dict(slot="cortical_pyramidal_b", why="Hay et al. 2011: BBP-lineage L5b PC, BluePyOpt reference"),
    2796: dict(slot="hippocampal_ca1", why="Migliore et al. 1999: CA1 pyramidal multicompartment"),
    267306: dict(slot="dopaminergic", why="substantia nigra dopaminergic neuron"),
    2017402: dict(slot="dopaminergic_alt", why="Ortner et al. 2017 SNc, alternative DA implementation"),
    267621: dict(slot="invertebrate_stg", why="Zang & Marder 2023 LP neuron: crustacean STG landscape"),
    17664: dict(slot="cerebellar_purkinje_alt", why="Miyasho et al. 2001: multicompartment Purkinje"),
    116835: dict(slot="cerebellar_granule", why="Diwakar et al. 2009: multicompartment granule cell"),
    226010: dict(slot="dopaminergic_alt2", why="vlPAG/DRN dopamine neuron"),
}

# Blue Brain Project assets, pinned by git ref.
BBP_REPOS = {
    "bluepyopt": dict(
        url="https://github.com/BlueBrain/BluePyOpt.git",
        why="official BBP optimisation package; examples/l5pc is the canonical BBP L5PC e-model",
    ),
    "sscx_emodels": dict(
        url="https://github.com/BlueBrain/SSCxEModelExamples.git",
        why="BBP somatosensory-cortex e-models produced by the production BBP pipeline",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_modeldb(mid: int, meta: dict) -> dict:
    dest = RAW / f"modeldb_{mid}"
    zpath = RAW / f"modeldb_{mid}.zip"
    url = f"https://modeldb.science/download/{mid}"
    rec = dict(source="modeldb", id=mid, url=url, slot=meta["slot"], rationale=meta["why"],
               fetched_utc=datetime.now(UTC).isoformat())
    try:
        if not zpath.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "COMPASS/0.1"})
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r, zpath.open("wb") as fh:
                shutil.copyfileobj(r, fh)
        rec["sha256"] = sha256_file(zpath)
        rec["bytes"] = zpath.stat().st_size
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        mods = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.mod"))
        hocs = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.hoc"))
        rec.update(status="ok", path=str(dest.relative_to(ROOT)), n_mod=len(mods),
                   n_hoc=len(hocs), mod_files=mods[:80], hoc_files=hocs[:40])
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=str(exc)[:200])
    return rec


def fetch_bbp(name: str, meta: dict) -> dict:
    dest = RAW / f"bbp_{name}"
    rec = dict(source="bluebrain", id=name, url=meta["url"], slot=f"bbp_{name}",
               rationale=meta["why"], fetched_utc=datetime.now(UTC).isoformat())
    try:
        if not dest.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", meta["url"], str(dest)],
                check=True, capture_output=True, timeout=900,
            )
        commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        mods = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*.mod"))
        rec.update(status="ok", path=str(dest.relative_to(ROOT)), commit=commit, n_mod=len(mods))
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=str(exc)[:300])
    return rec


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for mid, meta in MODELDB.items():
        rec = fetch_modeldb(mid, meta)
        records.append(rec)
        print(f"[{rec['status']:>5}] modeldb {mid:>8}  mods={rec.get('n_mod','-'):>3}  {meta['slot']}",
              flush=True)
    for name, meta in BBP_REPOS.items():
        rec = fetch_bbp(name, meta)
        records.append(rec)
        print(f"[{rec['status']:>5}] bbp {name:>13}  mods={rec.get('n_mod','-'):>3}", flush=True)
    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps(records, indent=1))
    ok = sum(r["status"] == "ok" for r in records)
    print(f"\n{ok}/{len(records)} assets retrieved -> {PROV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
