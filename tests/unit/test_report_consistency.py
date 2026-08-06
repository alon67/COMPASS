"""The reports must agree with the artefacts they were generated from.

This project has shipped, more than once, a document whose prose contradicted
the table beside it: an ICC of 0.048 that was not an ICC, a Sobol mean quoted as
0.363 in prose while the JSON said 0.481, a surrogate fraction of 44% against a
stored 45%. Each was caught by a human reading carefully. That does not scale,
and it does not survive the next re-run.

`tools/verify_report_numbers.py` reads the artefacts and checks that each
load-bearing number appears in the reports, and that no retired value appears
without a correction marker. This runs it as a test so the check happens
whenever the suite does, not only when someone remembers to run the pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "verify_report_numbers.py"
ANALYSES = ROOT / "results" / "analyses"


@pytest.mark.skipif(
    not (ANALYSES / "decision.json").exists(),
    reason="no analysis artefacts present; nothing to check the reports against")
def test_reports_match_artefacts() -> None:
    r = subprocess.run([sys.executable, str(TOOL)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "reports disagree with the artefacts they were generated from:\n"
        + r.stdout + r.stderr)
