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
# THE DOCUMENTS THAT GO TO THE REVIEWER WERE THE ONES NOT CHECKED.
#
# The forbidden-phrasing rules and the retired-value scan exist to stop the
# project asserting what its artifacts do not support, and until now they ran
# over RESULTS.md and README.md only -- while every audit response, whose whole
# purpose is to assert things to a reviewer and which is a `reports`-class entry
# in the release inventory, was outside their scope. Globbed rather than listed
# so the next response is covered without anyone remembering to add it.
DOC_PATHS.update({p.name: p for p in
                  sorted((ROOT / "docs").glob("COMPASS_AUDIT_RESPONSE_*.md"))})
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
# A prefix marking a rule that only applies while the decision WITHHOLDS the
# H-B criteria. Some bans encode a transient state of the release rather than a
# permanent error; when the state changes they forbid true statements, so they
# are switched off from the artifact instead of being remembered by hand.
CONDITIONAL_ON_H_B_WITHHELD = "@@HB_WITHHELD@@"

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
    # --- semantic contradictions the number checker cannot see ------------- #
    #
    # The 2026-08-09 post-response-2 audit (3.2, 5.6) found RESULTS.md still
    # asserting that ten implementations over seven lineages "pass every gate",
    # three sections after a table correctly calling the same cohort
    # "N3-passing, N4 unresolved".  Every number in the document verified; the
    # sentence was simply false.  These patterns are the class of statement that
    # cannot be true while any gate has a failing or missing record, so they are
    # forbidden outright rather than checked against a value.
    (r"pass(es|ing)? (every|all) gates?",
     "no cohort passes every gate: N1 fails for 2 models, N4 for 2, and N2 is "
     "evaluable only for the 6 models with a sourceable published scalar. Use "
     "compass.analysis.cohorts.cohort_label(), which names the gates actually "
     "enforced"),
    (r"all gates pass",
     "as above -- state which gates, for which cohort"),
    (r"gate[- ]passing cohort",
     "the phrase is generated by cohort_label() and must not be written by "
     "hand; it is only true when every N4 failure is excluded"),
    (r"full N1\s*[-–]\s*N7 (enforcement|compliance)",
     "N2 is evaluable for 6 of 14 models; 'full N1-N7' overstates it. Say "
     "N1/N3/N4-enforced and name N2's coverage"),
    (r"(confirmatory|confirms) (that )?(the )?(interaction structure|higher[- ]"
     r"order structure) is (generally )?non[- ]biological",
     "not established: the supportable statement is that sampling, "
     "conditioning, covariance and geometry materially influence the sign and "
     "magnitude of the measured interactions (audit 2026-08-09, executive "
     "verdict)"),
    (r"H-B is (refuted|disproven|absent|excluded)",
     "H-B is UNRESOLVED, not refuted: the numerically repeatable subset is "
     "three lineages and the matched-control interval crosses zero"),
    (r"five independent (confirmations|criteria|lines of evidence)",
     "the five H-G criteria are correlated derivatives of the same small panel "
     "and the same endpoints; describe the vote, not five independent tests"),
    # --- 2026-08-09 post-response-3 audit, P1-3 ---------------------------- #
    #
    # Each of these was an ACTIVE sentence sitting a few hundred lines from its
    # own contradiction, in a document whose every number verified.  A retained
    # historical table is fine; an active instruction that disagrees with the
    # current one is not, so the phrasings are forbidden and the current
    # statement is generated.
    (r"N2 (is )?not implemented",
     "N2 is implemented for the sourceable subset. Quote "
     "compass.validation.gates.n2_coverage()['statement'] -- "
     "'source-referenced canonical scalar agreement: 6/14 evaluable "
     "implementations' -- rather than restating the pre-implementation status"),
    (r"N2[^.\n]{0,40}\b(7/7|7 of 7)\b",
     "the N2 roll-up is 6 model tests plus one declaration-coverage test, not "
     "seven reproduced models (post-response-3 audit, P1-2)"),
    (r"strict reproduction of published traces is not implemented",
     "true of trace-level reproduction and misleading as an N2 summary; say "
     "what N2 does cover"),
    # STATE-DEPENDENT: forbidden only while the decision withholds H-B. The rule
    # was written when the perturbation inputs were stale and the tally read
    # 0/0, and it is switched off from `decision.json` rather than by hand --
    # otherwise it outlives the condition it guards and starts forbidding a true
    # statement, which is how a correctness check becomes a source of error.
    (CONDITIONAL_ON_H_B_WITHHELD + r"the H-B (quantities|criteria) are (now )?scored",
     "H-B is withheld while its perturbation inputs are stale; the values are "
     "printed beside the table but not scored"),
    (r"archives? (are|is) (written |stored )?(at |in )?float32",
     "trace archives are float64; a float32 archive can only support "
     "agreement 'to within rounding', which is not the claim the gate makes"),
    (r"validate_replay\.py --experiment screening\s*$",
     "the manual release-validation command must replay every declared "
     "campaign, not screening alone -- all six are mandatory"),
    (r"\b154\b[^.\n]{0,40}(flagged|waveform|of 914)",
     "the flagged-sample count is generated; quote n_with_problems from "
     "results/validation/retained_waveforms_*.json and never sum findings, "
     "which exceed samples because one sample can raise several"),
    # Naming the withdrawn arm while recounting what a previous revision did is
    # history and is allowed. Pointing a reader at its OUTPUTS, or listing it as
    # a live arm, is not: `decision_n4_projected.json` still asserts an overall
    # H-G conclusion the project no longer holds.
    (r"(results/analyses/[^\s`)]*n4_projected|SENSITIVITY_ARMS\[[\"']n4_projected"
     r"|--suffix\s+_?n4_projected)",
     "the n4_projected arm was withdrawn as a duplicate of `primary`; its "
     "outputs are quarantined under results/analyses/_withdrawn and the "
     "release manifest blocks on any that reappear"),
]

# --------------------------------------------------------------------------- #
# 3. Repository facts that no number-matching can express
# --------------------------------------------------------------------------- #


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# Criterion phrases that must carry ONE value throughout the documents.
#
# String equality against a single expected occurrence cannot catch a stale
# TABLE: the reviewed revision carried a second, eight-model decision table
# whose every row named a real criterion and quoted a superseded number, and
# nothing failed, because the CURRENT number also appeared elsewhere in the file
# (post-response audit, section 3.4). The rule below is the one that catches it:
# find every line that talks about a criterion, and require each of them either
# to quote the current value or to be explicitly marked as superseded.
#
# Keyed by a distinctive phrase; the value is pulled from decision.json by
# matching the start of the criterion name, so the check cannot drift from the
# artifact.
UNIQUE_VALUE = {
    "conditioning effect direction agreement",
    "high-covariance subset effect direction agreement",
    "unique R2(sampling+geometry)",
    "out-of-sample R2 for a held-out",
    "sampling regime predictable across unseen lineages",
    "statistical synergy modules validated by direct perturbation",
    "top-synergy tuples more non-additive than matched controls",
}


def criterion_consistency(dec: dict | None, marker: re.Pattern) -> list[str]:
    """Every line naming a criterion must quote its CURRENT value, or be marked."""
    if not dec:
        return []
    out = []
    entries = [(c["criterion"], c["value"]) for c in dec.get("criteria", [])]
    entries += [(q["quantity"], q["value"]) for q in dec.get("descriptive", [])]
    for name, value in entries:
        key = next((k for k in UNIQUE_VALUE if name.lower().startswith(k.lower())),
                   None)
        if key is None or not isinstance(value, int | float):
            continue
        # Accept any rounding the documents plausibly use -- but never a
        # zero-decimal one. `f"{0.8:.0f}"` is "1", and "1" is a substring of
        # almost every table row, so including it made the check pass on the
        # very rows it exists to catch.
        allowed = {f"{value:.{p}f}" for p in range(1, 5)}
        allowed |= {f"{value:.{p}f}".rstrip("0").rstrip(".") for p in range(1, 5)}
        if float(value).is_integer():
            allowed.add(str(int(value)))
        # A ratio may be written 2.97x or 2.971
        allowed |= {f"{value:.2f}x", f"{value:.1f}x"}
        # Guard against an accepted form so short it matches by accident.
        allowed = {a for a in allowed if len(a) >= 3}
        for doc, txt in DOCS.items():
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if key.lower() not in line.lower():
                    continue
                if any(a in line for a in allowed):
                    continue
                # A line that quotes no number at all is prose, not a table row.
                if not re.search(r"\d\.\d", line):
                    continue
                window = " ".join(lines[max(0, i - 3):i + 2])
                if marker.search(window):
                    continue
                out.append(
                    f"  [stale] {doc}: a line about '{key}' quotes a number that "
                    f"is not the current {value:.4g} and carries no correction "
                    f"marker:\n      {line.strip()[:110]}")
    return out


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
    # `--write-manifest` used to be guarded by `if args.write_manifest or True`,
    # i.e. the flag did nothing and the manifest was rewritten on every run
    # whatever the caller asked for.  Writing it by default is right -- the
    # release manifest hashes it as evidence -- but the flag has to mean
    # something, so the default is now explicit and the opt-OUT is real
    # (post-response audit, section 3.4).
    ap.add_argument("--no-write-manifest", action="store_true",
                    help="check only; do not rewrite claim_manifest.json. Use "
                         "when re-verifying a released tree, so the artifact "
                         "hash recorded in release_manifest.json still matches.")
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

    # WAVEFORM SAMPLE ARITHMETIC, ASSERTED RATHER THAN MERELY POLICED.
    #
    # The only protection this number had was a FORBIDDEN rule banning the one
    # value a previous response got wrong (`154`). A ban on one wrong answer
    # cannot detect the next one, and the count is regenerated by every
    # waveform run, so it is now a positive claim: summed over the campaign
    # records the battery actually wrote, and required to appear in the
    # documents verbatim.
    wf = sorted((ROOT / "results" / "validation").glob("retained_waveforms_*.json"))
    if wf:
        recs = [json.loads(p.read_text()) for p in wf]
        n_sel = sum(r["n_selected"] for r in recs)
        n_flag = sum(r["n_with_problems"] for r in recs)
        n_rec = sum(r["n_features_recomputed"] for r in recs)
        n_bad = sum(r["n_features_mismatched"] for r in recs)
        claim("waveform samples retained", f"{n_sel} traces", res,
              artifact="retained_waveforms_*.json", selector="sum(n_selected)",
              aggregation=f"summed over {len(recs)} campaign records", units="traces")
        claim("waveform samples flagged", f"{n_flag}", res,
              artifact="retained_waveforms_*.json",
              selector="sum(n_with_problems)",
              aggregation="a SAMPLE count, not a finding count", units="samples")
        claim("waveform features recomputed", f"{n_rec:,}", res,
              artifact="retained_waveforms_*.json",
              selector="sum(n_features_recomputed)",
              aggregation="recomputed from the reloaded float64 archive")
        # Non-zero mismatches would make the recomputation claim false; the
        # documents must not be able to say "zero mismatches" when they are not.
        if n_bad:
            CLAIMS.append({"label": "waveform feature mismatches", "documents": res,
                           "expected_string": f"{n_bad} mismatched",
                           "artifact": "retained_waveforms_*.json",
                           "selector": "sum(n_features_mismatched)",
                           "aggregation": "", "units": "", "uncertainty": "",
                           "note": "non-zero: the archive does not reproduce the row"})

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

    h_b_withheld = not dec.get("n_hb_total")
    for pattern, why in FORBIDDEN:
        if pattern.startswith(CONDITIONAL_ON_H_B_WITHHELD):
            if not h_b_withheld:
                continue
            pattern = pattern[len(CONDITIONAL_ON_H_B_WITHHELD):]
        rx = re.compile(pattern, re.I | re.M)
        for d, txt in DOCS.items():
            for m in rx.finditer(txt):
                start = txt.rfind("\n", 0, max(0, m.start() - 400))
                window = txt[max(0, start):m.end() + 200]
                # A phrase QUOTED while being retracted is not the project
                # making the claim. The window has to admit that, or the
                # correction narrative itself trips the check it exists to
                # enforce -- and the usual response to that is to delete the
                # narrative rather than the claim.
                if re.search(r"withdraw|must not|avoid|forbidden|no longer|"
                             r"incorrect|retired|do not|once read|used to "
                             r"(read|say)|still said|was false|is false|"
                             r"wrongly|forbidden phrasing", window, re.I):
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
        # Values the post-response audit found still standing in RESULTS.md
        # after the regeneration. Each is a real number from a real earlier run;
        # what makes it a defect is appearing without a correction marker beside
        # the value that superseded it.
        "1843.55": "unique-R2 ratio before the regenerated screening tables",
        "0.7310": "lineage LOMO median R2 before the true lineage folds",
        "0.4352": "regime-classifier accuracy under model-id folds",
        "0.2381": "functional-support fraction before the regenerated battery",
        "2.1250": "perturbation matched ratio before the regenerated battery",
        "7/42": "Sobol quotable count before the expansion campaign",
        "0/5 models": "perturbation repeatability before the half-step reruns",
    }
    MARKER = re.compile(
        r"earlier|previous|before|withdraw|was |old |orig|superseded|pre-audit|"
        r"revised|reported as|rather than|corrected|first release|not a finding|"
        r"biased|retired", re.I)
    # A RETIRED VALUE OF ONE QUANTITY CAN BE THE CURRENT VALUE OF ANOTHER.
    #
    # `2.34x` was the perturbation MATCHED ratio before a target correction and
    # is retired under that name; on the probe-4 battery it is the current
    # RANDOM ratio, and the check flagged the line that reports it correctly.
    # A number that some CLAIM asserts as current cannot simultaneously be a
    # stale number left lying in prose, so those strings are exempt -- and the
    # exemption is derived from the claims this run just built, not listed by
    # hand, so it cannot outlive the coincidence that created it.
    current_strings = {c["expected_string"] for c in CLAIMS}
    current_strings |= {s + "×" for s in current_strings}
    for bad, why in RETIRED.items():
        if bad in current_strings:
            continue
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

    failures.extend(criterion_consistency(dec, MARKER))

    facts = repo_facts()
    for desc, ok, detail in facts:
        if not ok:
            failures.append(f"  [fact] {desc}: {detail}")

    if not args.no_write_manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            # No wall-clock timestamp. This file is hashed into
            # release_manifest.json, and a timestamp made its hash change on
            # every run -- so a re-verification of an unchanged tree
            # invalidated the release manifest that had just been built from
            # it, which is how the reviewed revision came to have a
            # claim_manifest.json whose recorded hash no longer matched. The
            # revision identifies the state; the content is now a pure function
            # of the inputs.
            "git_revision": _git_rev(),
            "n_claims": len(CLAIMS),
            "claims": CLAIMS,
            "forbidden_wording": [{"pattern": p, "reason": w} for p, w in FORBIDDEN],
            "retired_values": RETIRED,
            "unique_value_claims": sorted(UNIQUE_VALUE),
            # check + verdict only. Two of these details embed file mtimes,
            # which change on a checkout without any content changing, and this
            # file has to hash deterministically. The details are printed and,
            # when a check fails, appear in the failure list.
            "repository_facts": [{"check": d, "ok": o} for d, o, _t in facts],
            "n_failures": len(failures),
        }, indent=1, default=str))

    where = (f"claim manifest -> {MANIFEST}" if not args.no_write_manifest
             else "claim manifest NOT rewritten (--no-write-manifest)")
    if failures:
        print(f"FAILED: {len(failures)} inconsistency(ies)\n")
        print("\n".join(failures))
        print(f"\n{where}")
        return 1
    print(f"OK: {len(CLAIMS)} claims match their artefacts; "
          f"{len(FORBIDDEN)} forbidden phrasings absent; "
          f"{len(UNIQUE_VALUE)} criteria carry a single value; "
          f"{len(facts)} repository facts hold.")
    print(where)
    return 0


if __name__ == "__main__":
    sys.exit(main())
