#!/usr/bin/env python3
"""Is the running environment the one the published numbers were produced in?

``pyproject.toml`` declares lower bounds, so "the dependencies are satisfied"
and "this is the environment that produced the results" are different
statements, and only the first was checkable (2026-08-09 post-response-3 audit,
P1-4).  ``requirements.lock`` records the exact versions; this tool compares the
interpreter it is running under against that file and writes the comparison as
release provenance.

    python tools/check_environment_lock.py            # report, exit 1 on drift
    python tools/check_environment_lock.py --write    # also record the result

Drift is a *blocker*, not a crash: a reviewer re-running the pipeline on a newer
SciPy should get a clear statement of what differs from the archived run rather
than a refusal to start.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock"
OUT = ROOT / "data" / "provenance" / "environment_lock_check.json"

PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#]+)\s*$")


def read_lock(path: Path = LOCK) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing {path.relative_to(ROOT)}")
    out = {}
    for line in path.read_text().splitlines():
        m = PIN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    if not out:
        raise SystemExit(f"{path.name} declares no pins")
    return out


def compare(pins: dict[str, str]) -> dict:
    matched, drifted, absent = {}, {}, []
    for name, want in sorted(pins.items()):
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            absent.append(name)
            continue
        (matched if got == want else drifted)[name] = (
            got if got == want else {"locked": want, "installed": got})
    return {
        "locked_packages": len(pins),
        "matched": matched,
        "drifted": drifted,
        "not_installed": absent,
        "python": {
            "locked": "3.12.4",
            "installed": platform.python_version(),
            "matches": platform.python_version() == "3.12.4",
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conforms": not drifted and not absent
                    and platform.python_version() == "3.12.4",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="record the comparison under data/provenance")
    args = ap.parse_args()

    result = compare(read_lock())
    result["checked_utc"] = datetime.now(UTC).isoformat()
    result["lock_file"] = LOCK.name
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")

    print(f"lock: {result['locked_packages']} pins, "
          f"{len(result['matched'])} matched, {len(result['drifted'])} drifted, "
          f"{len(result['not_installed'])} absent")
    for name, v in sorted(result["drifted"].items()):
        print(f"  DRIFT {name}: locked {v['locked']}, installed {v['installed']}")
    for name in result["not_installed"]:
        print(f"  ABSENT {name}")
    if not result["python"]["matches"]:
        print(f"  DRIFT python: locked {result['python']['locked']}, "
              f"installed {result['python']['installed']}")
    print("conforms to the archived environment" if result["conforms"]
          else "DOES NOT conform to the archived environment")
    return 0 if result["conforms"] else 1


if __name__ == "__main__":
    sys.exit(main())
