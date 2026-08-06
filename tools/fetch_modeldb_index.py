"""Fetch a complete local index of ModelDB model metadata.

Produces data/provenance/modeldb_index.json: one record per ModelDB accession with
name, cell types, currents, simulation environments, model concepts and paper info.
The index is used for criteria-based, auditable model selection (blueprint WP0/WP1:
"No model has been selected because it supports a preferred result").
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

API = "https://modeldb.science/api/v1"
CTX = ssl.create_default_context(cafile=certifi.where())
OUT = Path(__file__).resolve().parents[1] / "data" / "provenance" / "modeldb_index.json"


def _get(url: str, retries: int = 3):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45, context=CTX) as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001 - network flakiness is expected
            last = exc
    raise RuntimeError(f"{url}: {last}")


def _flatten(field):
    """ModelDB wraps most attributes as {'value': ..., 'attr_id': int}."""
    if isinstance(field, dict) and "value" in field:
        field = field["value"]
    if isinstance(field, list):
        out = []
        for item in field:
            if isinstance(item, dict):
                out.append(item.get("object_name", item.get("name", str(item))))
            else:
                out.append(str(item))
        return out
    return field


KEEP = [
    "neurons",
    "currents",
    "modeling_application",
    "model_concepts",
    "model_type",
    "brain_regions",
    "receptors",
    "transmitters",
    "genes",
    "implementers",
    "references",
]


def record(mid: int) -> dict:
    try:
        d = _get(f"{API}/models/{mid}")
    except Exception as exc:  # noqa: BLE001
        return {"id": mid, "error": str(exc)[:120]}
    rec = {"id": mid, "name": d.get("name", ""), "ver_date": d.get("ver_date", "")}
    for key in KEEP:
        if key in d:
            rec[key] = _flatten(d[key])
    return rec


def main() -> int:
    ids = _get(f"{API}/models")
    print(f"ModelDB reports {len(ids)} models; fetching metadata...", flush=True)
    out: list[dict] = []
    with cf.ThreadPoolExecutor(12) as ex:
        for i, rec in enumerate(ex.map(record, ids), 1):
            out.append(rec)
            if i % 200 == 0:
                print(f"  {i}/{len(ids)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    n_err = sum("error" in r for r in out)
    print(f"wrote {OUT} ({len(out)} records, {n_err} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
