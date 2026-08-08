#!/usr/bin/env python3
"""Generate a claim manifest and check every load-bearing statement against it.

Why this exists
---------------
This project has repeatedly shipped documents whose prose disagreed with the
tables it was generated from -- a "0.048" ICC that was not an ICC, a Sobol mean
quoted as 0.363 in prose and 0.481 in JSON, a surrogate fraction of 44% against
a stored 45%.  Every one was found by a human reading carefully, which does not
scale and does not survive a re-run.

Extension required by the 2026-08-06 audit (section 5.8)
--------------------------------------------------------
The previous version checked 18 known values by string equality and missed:
the test-count contradictions (README 97, Results 86, observed 98), a false
``ruff check .`` instruction, the "eight independent families" wording, a false
table-provenance claim, the stale model audit, and the invalid resolution-floor
interpretation.  String equality cannot catch any of those, because they are
claims about *wording* and about *facts other than numbers*.

So verification now has three layers:

1. a generated **claim manifest** -- every checked value with its artifact,
   selector, aggregation, units, uncertainty and the string it must appear as,
   written to ``results/analyses/claim_manifest.json`` so a reader can audit the
   provenance of each number rather than trusting the checker;
2. **forbidden wording** -- the specific phrasings the audit says must not
   appear, each with the reason and the replacement; and
3. **fact checks** -- assertions about the repository itself (the test count,
   the lint command, which files exist) that no number-matching can express.

Narrative scientific claims still require human review.  This tool bounds the
damage; it does not replace the reviewer.

    python tools/verify_report_numbers.py
    python tools/verify_report_numbers.py --write-manifest
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analyses"
DOC_PATHS = {
    "RESULTS.md": ROOT / "docs" / "RESULTS.md",
    "README.md": ROOT / "README.md",
}
DOCS = {k: p.read_text() for k, p in DOC_PATHS.items() if p.exists()}
MANIFEST = AN / "claim_manifest.json"


def _load(name: str, kind: str = "json"):
    p = AN / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if kind == "json" else pd.read_parquet(p)


CLAIMS: list[dict] = []


def claim(label: str, text: str, docs: list[str], *, artifact: str,
          selector: str, aggregation: str = "point value", units: str = "",
          uncertainty: str = "", note: str = "") -> None:
    """Register one checkable claim with its full provenance."""
    CLAIMS.append({
        "label": label, "expected_string": text, "documents": docs,
        "artifact": artifact, "selector": selector, "aggregation": aggregation,
        "units": units, "uncertainty": uncertainty, "note": note,
    })


# --------------------------------------------------------------------------- #
# 2. Forbidden wording (audit section 11 "Avoid")
# --------------------------------------------------------------------------- #
FORBIDDEN = [
    (r"eight independent (model )?(families|lineages)",
     "the eight model ids represent ten source lineages across the whole panel "
     "and six in the discovery cohort; say 'implementations' or name the "
     "lineage count"),
    (r"(four|4) (of eight |of 8 )?models are (below|unresolvable)",
     "the 2.5 Hz measurement-floor analysis was withdrawn (audit section 4)"),
    (r"irreducible measurement floor",
     "withdrawn: a deterministic quantisation grid is not a stochastic "
     "detection limit"),
    (r"\b4\s*[-–]\s*5\s*(sigma|σ)\b",
     "these are conditional bootstrap signal-to-noise ratios, not sigma "
     "significances (audit 7.3)"),
    (r"biological synergy modules were established",
     "not supported: 20.8% pass functional support and the paired tests are "
     "inconclusive"),
    (r"Sobol interactions are pervasive(?!.{0,120}(precision|convergence|qualif))",
     "requires a precision qualification (audit 7.8)"),
    (r"ruff check \.\s*$",
     "`ruff check .` includes generated notebooks and fails; the checked "
     "command is `ruff check src tests tools`"),
]

# --------------------------------------------------------------------------- #
# 3. Repository facts that no number-matching can express
# --------------------------------------------------------------------------- #


def repo_facts() -> list[tuple[str, bool, str]]:
    """(description, ok, detail) for claims about the repository itself."""
    out = []

    # (a) the documented test count must equal the observed one
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "-m", "not slow", str(ROOT / "tests")],
            capture_output=True, text=True, timeout=900, cwd=ROOT)
        # "N/M tests collected (K deselected)" -> N is the SELECTED count.
        m = re.search(r"(\d+)/(\d+) tests collected", proc.stdout)
        if m:
            observed = int(m.group(1))
        else:
            m = re.search(r"(\d+)\s+tests? collected", proc.stdout)
            observed = int(m.group(1)) if m else None
    except Exception as exc:  # noqa: BLE001
        observed = None
        out.append(("test collection", False, f"could not collect: {exc}"))
    if observed is not None:
        quoted = set()
        for txt in DOCS.values():
            # Only a claim of the form "N tests" that is clearly about the
            # suite as a whole; "30 tests" inside a sentence about one file is
            # not a claim about the count and must not trip the check.
            quoted |= {int(x) for x in re.findall(
                r"(?:^|[^.\d])(\d{2,4})\s+(?:passing |passed )?tests?"
                r"(?=[ ,:;.]|$)(?![^\n]{0,40}\bin\b)", txt, re.M)}
        bad = {q for q in quoted if q != observed}
        out.append((
            f"documented fast-test count == observed ({observed})",
            not bad,
            f"documents quote {sorted(quoted)}; observed {observed}" if bad
            else f"{observed} fast tests, consistently quoted"))

    # (b) the lint command in the docs must actually pass
    proc = subprocess.run(["ruff", "check", "src", "tests", "tools"],
                          capture_output=True, text=True, cwd=ROOT)
    out.append(("`ruff check src tests tools` passes", proc.returncode == 0,
                proc.stdout.strip()[-200:] or "clean"))

    # (c) withdrawn artifacts must not be referenced as evidence
    withdrawn = ["readout_resolution.json", "readout_resolution.parquet"]
    for w in withdrawn:
        refs = [d for d, txt in DOCS.items()
                if w in txt and "withdraw" not in txt[max(0, txt.find(w) - 400):
                                                      txt.find(w) + 400].lower()]
        out.append((f"withdrawn artifact {w} not cited as evidence", not refs,
                    f"cited in {refs}" if refs else "not cited, or cited as withdrawn"))

    # (d) the model audit must be newer than the convergence evidence it reads
    audit = ROOT / "docs" / "model_audits" / "model_audit.md"
    convs = sorted((ROOT / "data" / "provenance").glob("convergence_audit_*.json"))
    if audit.exists() and convs:
        newest = max(p.stat().st_mtime for p in convs)
        out.append(("model audit is not stale", audit.stat().st_mtime >= newest,
                    f"audit {datetime.fromtimestamp(audit.stat().st_mtime, UTC):%Y-%m-%d %H:%M} "
                    f"vs newest convergence file "
                    f"{datetime.fromtimestamp(newest, UTC):%Y-%m-%d %H:%M}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    dec = _load("decision.json")
    mix = _load("mixed_effects.json")
    ps = _load("perturbations_summary.json")
    kn = _load("key_numbers.json")
    rob = _load("robustness.json")
    pinf = _load("perturbation_inference.json")

    both = ["RESULTS.md", "README.md"]
    res = ["RESULTS.md"]

    if dec:
        claim("decision H-B tally", f"H-B: {dec['n_hb_passed']}/{dec['n_hb_total']}",
              res, artifact="decision.json", selector="n_hb_passed/n_hb_total",
              aggregation="count of passed criteria",
              note="primary sensitivity arm only")
        crit = {c["criterion"]: c["value"] for c in dec["criteria"]}
        for key in ("conditioning effect direction agreement across LINEAGES",):
            v = next((v for k, v in crit.items() if k.startswith(key[:40])), None)
            if v is not None:
                claim("conditioning direction agreement", f"{v:.3f}", res,
                      artifact="decision.json",
                      selector=f"criteria[criterion={key!r}].value",
                      aggregation="fraction of lineages agreeing in sign",
                      units="fraction")
        es = dec.get("effective_sample", {})
        if es:
            claim("source lineage count",
                  f"{es['n_source_lineages_primary']} source lineages", both,
                  artifact="decision.json",
                  selector="effective_sample.n_source_lineages_primary",
                  aggregation="distinct source_lineage_id in the primary cohort",
                  note="NOT the number of model ids")

    if ps:
        for tag, k in (("matched", "top_vs_control_middle"),
                       ("random", "top_vs_control_random")):
            v = ps.get(k)
            if not v:
                continue
            claim(f"perturbation {tag} ratio", f"{v['ratio_point']:.2f}", both,
                  artifact="perturbations_summary.json",
                  selector=f"{k}.ratio_point",
                  aggregation="mean |eta| of top tuples / mean |eta| of controls",
                  units="dimensionless ratio of feature-SD units",
                  uncertainty=f"model-bootstrap 95% CI {v['ratio_ci95']}")
            claim(f"perturbation {tag} p", f"{v['wilcoxon_p']:.3f}", res,
                  artifact="perturbations_summary.json", selector=f"{k}.wilcoxon_p",
                  aggregation="paired Wilcoxon across models")

    if pinf:
        ip = pinf.get("inferential_precision", {})
        if "n_top_tuples_ci_excludes_zero" in ip:
            claim("top tuples distinguishable from zero",
                  f"{ip['n_top_tuples_ci_excludes_zero']}/{ip['n_top_tuples_total']}",
                  res, artifact="perturbation_inference.json",
                  selector="inferential_precision.n_top_tuples_ci_excludes_zero",
                  aggregation="count of tuples whose background bootstrap CI excludes 0",
                  note="replaces the withdrawn resolvable/floor comparison")

    if kn and "sobol" in kn:
        claim("sobol mean interaction fraction",
              f"{kn['sobol']['mean_interaction_fraction']:.3f}", res,
              artifact="key_numbers.json",
              selector="sobol.mean_interaction_fraction",
              aggregation="mean over quotable summaries", units="fraction",
              note="quotable now requires completeness AND precision AND "
                   "physical plausibility (audit M8)")
        claim("sobol quotable count", f"{kn['sobol']['n_analyses']} quotable", res,
              artifact="key_numbers.json", selector="sobol.n_analyses",
              aggregation="count")

    if mix:
        flat = mix.get("flat", mix)
        if "icc_adjusted" in flat:
            claim("adjusted ICC", f"{flat['icc_adjusted']:.3f}", res,
                  artifact="mixed_effects.json", selector="flat.icc_adjusted",
                  aggregation="var_group / (var_group + var_residual)",
                  note="descriptive only; withdrawn as a decision criterion")
        if "vpc_model_identity" in flat:
            claim("VPC", f"{flat['vpc_model_identity']:.3f}", res,
                  artifact="mixed_effects.json", selector="flat.vpc_model_identity",
                  aggregation="var_group / total variance")

    if rob and "cluster_bootstrap_matched" in rob:
        m = rob["cluster_bootstrap_matched"]["synergy_area"]
        claim("matched bootstrap median ratio", f"{m['ratio_median']:.1f}", both,
              artifact="robustness.json",
              selector="cluster_bootstrap_matched.synergy_area.ratio_median",
              aggregation="median over cluster-bootstrap replicates",
              uncertainty=str(m.get("ratio_ci95", "")))

    if kn and "surrogate_control" in kn:
        sc = kn["surrogate_control"]
        claim("E8 naive synergy fraction", f"{sc['E8_naive_frac_synergy']:.3f}", res,
              artifact="key_numbers.json",
              selector="surrogate_control.E8_naive_frac_synergy", units="fraction")

    failures = []
    for c in CLAIMS:
        for d in c["documents"]:
            if d in DOCS and c["expected_string"] not in DOCS[d]:
                failures.append(
                    f"  [value] {d}: expected '{c['expected_string']}' for "
                    f"[{c['label']}] from {c['artifact']}:{c['selector']} -- NOT FOUND")

    for pattern, why in FORBIDDEN:
        rx = re.compile(pattern, re.I | re.M)
        for d, txt in DOCS.items():
            for m in rx.finditer(txt):
                start = txt.rfind("\n", 0, max(0, m.start() - 400))
                window = txt[max(0, start):m.end() + 200]
                if re.search(r"withdraw|must not|avoid|forbidden|no longer|"
                             r"incorrect|retired|do not", window, re.I):
                    continue
                failures.append(f"  [wording] {d}: forbidden phrasing "
                                f"{m.group(0)!r} -- {why}")

    RETIRED = {
        "0.4812": "ungated Sobol mean (pre-quotable-gate)",
        "2.34×": "perturbation matched ratio before target correction",
        "4.05×": "perturbation random ratio before target correction",
        "−4.468": "migliore LOMO R2 before target correction",
        "0.048": "the mis-computed 'ICC'",
        "8.5×": "matched variance ratio before target correction",
        "12.743": "median intrinsic dimension under the biased TwoNN",
        "18.57": "unique-R2 ratio under the biased TwoNN",
    }
    MARKER = re.compile(
        r"earlier|previous|before|withdraw|was |old |orig|superseded|pre-audit|"
        r"revised|reported as|rather than|corrected|first release|not a finding|"
        r"biased|retired", re.I)
    for bad, why in RETIRED.items():
        # Numeric retired values must match as WHOLE numbers. Plain substring
        # matching flagged "0.0483" as the retired ICC "0.048", which is the
        # kind of false positive that gets a check switched off.
        numeric = re.fullmatch(r"[-−]?[\d.]+", bad) is not None
        rx = (re.compile(r"(?<![\d.])" + re.escape(bad) + r"(?![\d])")
              if numeric else None)
        for d, txt in DOCS.items():
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if (rx.search(line) is None) if numeric else (bad not in line):
                    continue
                window = " ".join(lines[max(0, i - 3):i + 2])
                if not MARKER.search(window):
                    failures.append(
                        f"  [retired] {d}: '{bad}' ({why}) appears without a "
                        f"correction marker:\n      {line.strip()[:110]}")

    facts = repo_facts()
    for desc, ok, detail in facts:
        if not ok:
            failures.append(f"  [fact] {desc}: {detail}")

    if args.write_manifest or True:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            "written_utc": datetime.now(UTC).isoformat(),
            "n_claims": len(CLAIMS),
            "claims": CLAIMS,
            "forbidden_wording": [{"pattern": p, "reason": w} for p, w in FORBIDDEN],
            "retired_values": RETIRED,
            "repository_facts": [{"check": d, "ok": o, "detail": t}
                                 for d, o, t in facts],
            "n_failures": len(failures),
        }, indent=1, default=str))

    if failures:
        print(f"FAILED: {len(failures)} inconsistency(ies)\n")
        print("\n".join(failures))
        print(f"\nclaim manifest -> {MANIFEST}")
        return 1
    print(f"OK: {len(CLAIMS)} claims match their artefacts; "
          f"{len(FORBIDDEN)} forbidden phrasings absent; "
          f"{len(facts)} repository facts hold.")
    print(f"claim manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
