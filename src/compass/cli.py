"""``compass`` command-line entry point (blueprint §12).

Thin dispatcher over the pipeline stages in ``tools/``, so that every scientific
result is regenerable from a versioned experiment configuration:

    compass validate       --experiment screening
    compass canonical      --all-models
    compass simulate       --experiment screening
    compass analyze-info   --experiment screening
    compass analyze-sobol
    compass analyze-perturbations
    compass report         --experiment screening
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _run(script: str, *args: str) -> int:
    path = TOOLS / script
    if not path.exists():
        print(f"missing pipeline script {path}", file=sys.stderr)
        return 2
    argv = [str(path), *args]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = old


def cmd_validate(args) -> int:
    """Validate manifests, parameter tables, protocols and target data."""
    from compass.adapters.models import MODEL_REGISTRY
    from compass.protocols import common as P
    from compass.registry.schema import ValidationError

    problems = []
    for mid, cls in MODEL_REGISTRY.items():
        try:
            adapter = cls()
            specs = adapter.parameter_specs()          # schema validates eagerly
            manifest = adapter.manifest
            if mid not in P.ABSOLUTE_AMPLITUDES:
                problems.append(f"{mid}: no absolute protocol amplitudes declared")
            if mid not in P.CANONICAL_RHEOBASE:
                problems.append(f"{mid}: no canonical rheobase recorded")
            if not manifest.source_reference or not manifest.license:
                problems.append(f"{mid}: missing provenance or license")
            ids = [s.parameter_id for s in specs]
            if len(set(ids)) != len(ids):
                problems.append(f"{mid}: duplicate parameter ids")
            tiers = {s.evidence_tier.value for s in specs}
            if tiers & {"A", "B"}:
                problems.append(f"{mid}: claims tier A/B evidence; verify the source data")
            print(f"  ok  {mid:16s} {len(specs)} parameters, tiers={sorted(tiers)}")
        except ValidationError as exc:
            problems.append(f"{mid}: {exc}")
    prov = ROOT / "data" / "provenance" / "downloads.json"
    if prov.exists():
        for rec in json.loads(prov.read_text()):
            if rec["status"] != "ok":
                problems.append(f"asset {rec['id']}: {rec.get('error', 'not retrieved')}")
    for p in problems:
        print(f"  FAIL {p}", file=sys.stderr)
    print(f"\n{len(MODEL_REGISTRY) - len({p.split(':')[0] for p in problems})}"
          f"/{len(MODEL_REGISTRY)} models valid")
    return 1 if problems else 0


def cmd_canonical(args) -> int:
    return _run("canonical_audit.py", *(args.models or []))


def cmd_simulate(args) -> int:
    extra = ["--experiment", args.experiment]
    if args.models:
        extra += ["--models", *args.models]
    if args.workers:
        extra += ["--workers", str(args.workers)]
    return _run("run_campaign.py", *extra)


def cmd_analyze_info(args) -> int:
    return _run("run_analysis.py", "--experiment", args.experiment)


def cmd_analyze_sobol(args) -> int:
    return _run("run_sobol.py", *(["--analyse-only"] if args.analyse_only else []))


def cmd_analyze_perturbations(args) -> int:
    return _run("run_perturbations.py", "--experiment", args.experiment)


def cmd_benchmarks(args) -> int:
    return _run("run_benchmarks.py")


def cmd_report(args) -> int:
    rc = _run("run_crossmodel.py", "--experiment", args.experiment)
    if rc:
        return rc
    rc = _run("summarise_results.py")
    if rc:
        return rc
    rc = _run("build_model_audit.py")
    if rc:
        return rc
    rc = _run("build_notebook.py")
    if rc or not args.execute:
        return rc
    nb = ROOT / "notebooks" / "COMPASS_results.ipynb"
    return subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--execute",
         "--to", "notebook", "--inplace", str(nb)], check=False).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compass", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate manifests, units and provenance")
    p.add_argument("--experiment", default="screening")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("canonical", help="reproduce canonical traces and features")
    p.add_argument("--models", nargs="*")
    p.add_argument("--all-models", action="store_true")
    p.set_defaults(fn=cmd_canonical)

    p = sub.add_parser("simulate", help="run a simulation campaign")
    p.add_argument("--experiment", default="screening")
    p.add_argument("--models", nargs="*")
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("analyze-info", help="ensembles, O-information, RSI")
    p.add_argument("--experiment", default="screening")
    p.set_defaults(fn=cmd_analyze_info)

    p = sub.add_parser("analyze-sobol", help="variance-based sensitivity")
    p.add_argument("--analyse-only", action="store_true")
    p.set_defaults(fn=cmd_analyze_sobol)

    p = sub.add_parser("analyze-perturbations", help="direct factorial perturbations")
    p.add_argument("--experiment", default="screening")
    p.set_defaults(fn=cmd_analyze_perturbations)

    p = sub.add_parser("benchmarks", help="estimator validation and comparators")
    p.set_defaults(fn=cmd_benchmarks)

    p = sub.add_parser("report", help="cross-model decision, audit and notebook")
    p.add_argument("--experiment", default="screening")
    p.add_argument("--execute", action="store_true", help="also execute the notebook")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
