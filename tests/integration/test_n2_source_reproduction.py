"""Gate N2: quantitative agreement with sourced published values.

N2 reported ``0/0`` through three audits.  A gate with no evidence blocks a
release, which is correct, but it also says nothing about whether these
implementations land anywhere near the neurons they claim to be.  The
2026-08-09 post-response-2 audit (3.2) gave two acceptable dispositions:
implement N2 with model-specific sourced targets, or remove every claim that
requires source-level reproduction.  Both are done here.

The targets are declared in ``data/provenance/n2_source_targets.json``, one
scalar per (model, feature), each naming its publication and the place in this
repository where that source statement is recorded.  Models with no sourceable
scalar are declared ``evaluable: false`` with the reason; those declarations are
tested too, so a model cannot silently disappear from the gate.

**This is not trace reproduction and does not claim to be.**  Reproducing a
published figure needs the published stimulus, temperature and recording site,
and the distributed archives carry mechanisms and morphology rather than the
validation protocol.  What N2 tests is that each implementation's canonical
operating point matches the scalar its source publishes -- a strictly stronger
statement than N1's unit-error interval and a strictly weaker one than trace
reproduction.  The report says exactly that and no more.

Measurements come from ``results/manifests/canonical_<model>.json``, produced by
``tools/canonical_audit.py`` under each model's production solver contract.  The
gate FAILS CLOSED when that manifest is absent: an unmeasured model is not a
passing one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from compass.adapters.models import MODEL_REGISTRY  # noqa: E402

TARGETS = ROOT / "data" / "provenance" / "n2_source_targets.json"
MANIFESTS = ROOT / "results" / "manifests"


def _spec() -> dict:
    assert TARGETS.exists(), (
        f"{TARGETS.relative_to(ROOT)} is absent. Gate N2 fails closed: without a "
        f"declared target set there is nothing to reproduce and nothing to "
        f"withdraw.")
    return json.loads(TARGETS.read_text())


SPEC = _spec()
EVALUABLE = sorted(m for m, v in SPEC["models"].items() if v.get("evaluable"))
NOT_EVALUABLE = sorted(m for m, v in SPEC["models"].items() if not v.get("evaluable"))


def _canonical(model_id: str) -> dict:
    p = MANIFESTS / f"canonical_{model_id}.json"
    assert p.is_file(), (
        f"no canonical manifest for {model_id}; run tools/canonical_audit.py. "
        f"N2 cannot pass on a model that was never measured.")
    return json.loads(p.read_text())


def _measured(canon: dict, feature: str) -> float:
    """Pull one scalar out of the canonical manifest."""
    if feature in canon:
        return float(canon[feature])
    feats = canon.get("canonical_features", {})
    for key in (feature, f"hyperpol_abs.{feature}", f"rest.{feature}"):
        if key in feats:
            return float(feats[key])
    raise AssertionError(
        f"canonical manifest has no {feature!r}; N2 cannot be scored from it")


def test_n2_declaration_covers_every_registered_model():
    """A model must be either targeted or explicitly declared not evaluable."""
    declared = set(SPEC["models"])
    missing = sorted(set(MODEL_REGISTRY) - declared)
    assert not missing, (
        f"models with no N2 declaration at all: {missing}. Add a target set or "
        f"an explicit `evaluable: false` with a reason -- an absent declaration "
        f"is how N2 became 0/0 in the first place.")


@pytest.mark.parametrize("model_id", NOT_EVALUABLE)
def test_n2_known_failures_state_a_reason_and_withdraw_the_claim(model_id):
    d = SPEC["models"][model_id]
    assert d.get("reason"), f"{model_id}: not evaluable but no reason recorded"
    assert d.get("claims_withdrawn"), (
        f"{model_id}: not evaluable, so the report must say that no source-level "
        f"reproduction is claimed for it")


@pytest.mark.parametrize("model_id", EVALUABLE)
def test_n2_canonical_measurement_matches_the_published_value(model_id):
    canon = _canonical(model_id)
    failures = []
    for tgt in SPEC["models"][model_id]["targets"]:
        feat = tgt["feature"]
        got = _measured(canon, feat)
        assert np.isfinite(got), f"{model_id}/{feat}: canonical value is not finite"
        if "published_range" in tgt:
            lo, hi = tgt["published_range"]
            slack = tgt.get("tolerance_frac", 0.0)
            lo, hi = lo * (1 - slack), hi * (1 + slack)
            ok = lo <= got <= hi
            want = f"[{lo:.4g}, {hi:.4g}]"
        else:
            pub = float(tgt["published"])
            if "tolerance_abs" in tgt:
                tol = float(tgt["tolerance_abs"])
            else:
                tol = abs(pub) * float(tgt["tolerance_frac"])
            ok = abs(got - pub) <= tol
            want = f"{pub:.4g} +/- {tol:.4g}"
        if not ok:
            failures.append(
                f"{feat}: measured {got:.6g}, published {want} "
                f"[{tgt['source']}]")
    assert not failures, f"{model_id} N2 disagreement:\n  " + "\n  ".join(failures)
