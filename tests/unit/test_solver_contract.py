"""There must be exactly one authoritative solver contract.

Regression guard for audit finding C2.  `tools/canonical_audit.py` used to take
its solver settings from `data/provenance/numerical_qc.json` (the first-pass QC
sweep: CVODE at dt = 0.025 ms for six of eight models) while the campaign ran
fixed-step at per-model time steps from `CAMPAIGN_SOLVER`.  The canonical
manifests define the conditioning target that every ensemble is accepted
against, so the targets were produced by a different integrator than the
ensembles they anchor.

Nothing in the pipeline failed when that happened -- both files were internally
consistent and both were written by working code.  These tests make the
divergence itself the failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass.workflows.simulate import CAMPAIGN_SOLVER

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "results" / "manifests"


def _manifests() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(MANIFESTS.glob("canonical_*.json"))]


def test_campaign_solver_covers_every_registered_model() -> None:
    from compass.adapters.models import MODEL_REGISTRY

    missing = set(MODEL_REGISTRY) - set(CAMPAIGN_SOLVER)
    assert not missing, f"models with no declared solver contract: {sorted(missing)}"


def test_canonical_audit_reads_the_campaign_contract() -> None:
    """The manifest builder must not have its own private source of settings."""
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from canonical_audit import solver_kwargs

    for mid, expected in CAMPAIGN_SOLVER.items():
        assert solver_kwargs(mid) == dict(expected), (
            f"{mid}: canonical_audit would build the manifest with "
            f"{solver_kwargs(mid)} but the campaign runs {expected}")


def test_canonical_audit_refuses_to_guess() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from canonical_audit import solver_kwargs

    with pytest.raises(KeyError):
        solver_kwargs("a_model_that_does_not_exist")


@pytest.mark.skipif(not MANIFESTS.exists() or not any(MANIFESTS.glob("canonical_*.json")),
                    reason="no canonical manifests built yet")
def test_stored_manifests_match_the_campaign_contract() -> None:
    """The manifests on disk must have been built under the campaign's solver.

    This is the check that would have caught C2 the day it appeared: it compares
    the artefact, not the code that writes it.
    """
    wrong = []
    for r in _manifests():
        mid = r["model_id"]
        if mid not in CAMPAIGN_SOLVER:
            continue
        stored = dict(r.get("solver") or {})
        expected = dict(CAMPAIGN_SOLVER[mid])
        if stored != expected:
            wrong.append(f"{mid}: manifest={stored} campaign={expected}")
    assert not wrong, "manifests built under a different solver contract:\n" + "\n".join(wrong)


def _nseg(model_id: str) -> int:
    """Build one model and report its segment count. Runs in its own process."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from compass.adapters.models import get_adapter

    return int(get_adapter(model_id).build().manifest.n_compartments)


@pytest.mark.slow
@pytest.mark.skipif(not MANIFESTS.exists() or not any(MANIFESTS.glob("canonical_*.json")),
                    reason="no canonical manifests built yet")
def test_stored_manifest_compartment_counts_match_the_adapters() -> None:
    """A manifest may not record a mesh the adapter no longer builds.

    `mainen_l5pc` recorded 754 segments -- the pre-refinement "no compartment
    longer than 50 um" mesh -- after the adapter had been changed to refine to
    twice the d_lambda(100 Hz) mesh (2,304 segments).  Every geometry-derived
    covariate in the cross-model analysis read the stale number.

    NEURON keeps one global section list per process and the adapters refuse to
    build a second model in a process that already holds one, so each model is
    built in a fresh spawned process.
    """
    import multiprocessing as mp

    rows = _manifests()
    ids = [r["model_id"] for r in rows]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(ids), 4), maxtasksperchild=1) as pool:
        built = pool.map(_nseg, ids)

    wrong = [f"{r['model_id']}: manifest={r.get('n_compartments')} adapter={b}"
             for r, b in zip(rows, built, strict=True)
             if int(r.get("n_compartments", -1)) != b]
    assert not wrong, "manifests record a mesh the adapter does not build:\n" + "\n".join(wrong)


# --------------------------------------------------------------------------- #
# The feature schema must be DECLARED, not inferred from a byte hash
# --------------------------------------------------------------------------- #
# Shard reuse compares the declared `FEATURE_SCHEMA_VERSION`, not a hash of
# `extract.py`, because a byte hash would discard 98,048 correct simulations
# when a comment changed. That trade is only safe if an undeclared change to the
# extractor is caught somewhere else -- here.
#
# The recorded digest below is the content hash of the extraction module at the
# moment FEATURE_SCHEMA_VERSION was last set. Changing the module without
# bumping the version fails this test, which is a code-review failure rather
# than a data-discarding one.
FEATURE_MODULE_DIGEST = {
    "features-2": "7f7efafe586169eb",
}


def test_feature_schema_version_is_declared_when_the_extractor_changes():
    """An undeclared extractor change must fail review, not silently ship."""
    import hashlib
    from pathlib import Path as _P

    from compass.features.extract import FEATURE_SCHEMA_VERSION

    src = _P(__file__).resolve().parents[2] / "src" / "compass" / "features" / "extract.py"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    recorded = FEATURE_MODULE_DIGEST.get(FEATURE_SCHEMA_VERSION)
    if recorded is None:
        # First run under a newly declared version: record it and pass. The
        # value is written into this file by the developer, so the NEXT change
        # without a version bump fails.
        pytest.skip(
            f"FEATURE_SCHEMA_VERSION={FEATURE_SCHEMA_VERSION} has no recorded "
            f"module digest yet. Set FEATURE_MODULE_DIGEST[{FEATURE_SCHEMA_VERSION!r}] "
            f"= {digest!r} in this file to arm the check.")
    assert digest == recorded, (
        f"src/compass/features/extract.py changed (digest {digest}) while "
        f"FEATURE_SCHEMA_VERSION is still {FEATURE_SCHEMA_VERSION!r}. Either the "
        f"change cannot alter a stored feature -- in which case update "
        f"FEATURE_MODULE_DIGEST -- or it can, in which case bump the version so "
        f"shards extracted under the two definitions can never be pooled.")
