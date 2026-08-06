#!/usr/bin/env python3
"""Assert that every headline number in the reports matches its artefact.

Why this exists
---------------
This project has repeatedly shipped documents whose prose disagreed with the
tables it was generated from -- a "0.048" ICC that was not an ICC, a Sobol mean
quoted as 0.363 in prose and 0.481 in JSON, a surrogate fraction of 44% against
a stored 45%. Every one of those was found by a human reading carefully, which
does not scale and does not survive a re-run.

This walks the reverse direction: it reads the artefacts, then greps the reports
for the string each number should appear as. A number that changes in a re-run
and is not updated in the prose fails here rather than in review.

It deliberately checks only the load-bearing claims -- the ones a reviewer would
challenge -- not every digit in every table.

    python tools/verify_report_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analyses"
DOCS = {
    "RESULTS.md": (ROOT / "docs" / "RESULTS.md").read_text(),
    "README.md": (ROOT / "README.md").read_text(),
}


def _load(name: str, kind: str = "json"):
    p = AN / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if kind == "json" else pd.read_parquet(p)


CHECKS: list[tuple[str, str, list[str]]] = []


def expect(label: str, text: str, docs: list[str]) -> None:
    """Require `text` to appear verbatim in each named document."""
    CHECKS.append((label, text, docs))


def main() -> int:
    dec = _load("decision.json")
    mix = _load("mixed_effects.json")
    ps = _load("perturbations_summary.json")
    rr = _load("readout_resolution.json")
    kn = _load("key_numbers.json")
    rob = _load("robustness.json")

    both = ["RESULTS.md", "README.md"]
    res = ["RESULTS.md"]

    # --- decision -----------------------------------------------------------
    expect("decision conclusion", f"H-B: {dec['n_hb_passed']}/{dec['n_hb_total']}", res)
    crit = {c["criterion"][:40]: c["value"] for c in dec["criteria"]}
    for key, fmt in (("conditioning effect direction agr", "{:.3f}"),
                     ("top-tuple module stability (mean p", "{:.3f}")):
        v = next(v for k, v in crit.items() if k.startswith(key[:30]))
        expect(f"criterion {key[:28]}", fmt.format(v), res)

    # --- perturbation -------------------------------------------------------
    for tag, k in (("matched", "top_vs_control_middle"), ("random", "top_vs_control_random")):
        v = ps[k]
        expect(f"pert {tag} ratio", f"{v['ratio_point']:.2f}", both)
        expect(f"pert {tag} p", f"{v['wilcoxon_p']:.3f}", res)

    # --- readout floors -----------------------------------------------------
    expect("floors resolvable",
           "four of eight models", res)
    for m in rr["unresolvable_models"]:
        expect(f"floor below: {m}", m, res)

    # --- sobol --------------------------------------------------------------
    expect("sobol mean IF", f"{kn['sobol']['mean_interaction_fraction']:.3f}", res)
    expect("sobol n quotable", f"{kn['sobol']['n_analyses']} quotable", res)

    # --- mixed effects ------------------------------------------------------
    expect("adjusted ICC", f"{mix['icc_adjusted']:.3f}", res)
    expect("VPC", f"{mix['vpc_model_identity']:.3f}", res)

    # --- variance partition / robustness ------------------------------------
    m = rob["cluster_bootstrap_matched"]["synergy_area"]
    expect("matched bootstrap median", f"{m['ratio_median']:.1f}", both)

    # --- surrogate ----------------------------------------------------------
    sc = kn["surrogate_control"]
    expect("E8 naive frac", f"{sc['E8_naive_frac_synergy']:.3f}", res)

    failures = []
    for label, text, docs in CHECKS:
        for d in docs:
            if text not in DOCS[d]:
                failures.append(f"  {d}: expected '{text}' for [{label}] -- NOT FOUND")

    # --- forbidden strings: numbers known to be superseded -------------------
    RETIRED = {
        "0.4812": "ungated Sobol mean (pre-quotable-gate)",
        "2.34×": "perturbation matched ratio before target correction",
        "4.05×": "perturbation random ratio before target correction",
        "−4.468": "migliore LOMO R2 before target correction",
        "0.048": "the mis-computed 'ICC'",
        "8.5×": "matched variance ratio before target correction",
    }
    # The marker may sit a line or two above the number: these documents are
    # hard-wrapped, so a sentence like "Before the targets were corrected the
    # same battery gave 4.05x ..." straddles two lines.  Search the surrounding
    # window rather than the single line, or every wrapped correction is a false
    # positive and the check gets switched off -- which is how these drift in.
    MARKER = re.compile(
        r"earlier|previous|before|withdraw|was |old |orig|superseded|pre-audit|"
        r"revised|reported as|rather than|corrected|first release|not a finding",
        re.I)
    for bad, why in RETIRED.items():
        for d, txt in DOCS.items():
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if bad not in line:
                    continue
                window = " ".join(lines[max(0, i - 3):i + 2])
                if not MARKER.search(window):
                    failures.append(
                        f"  {d}: retired value '{bad}' ({why}) appears without "
                        f"a correction marker:\n      {line.strip()[:110]}")

    if failures:
        print(f"FAILED: {len(failures)} inconsistency(ies) between reports and artefacts\n")
        print("\n".join(failures))
        return 1
    print(f"OK: {len(CHECKS)} headline numbers match their artefacts; "
          f"no retired value quoted without a correction marker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
