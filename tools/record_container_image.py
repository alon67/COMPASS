#!/usr/bin/env python3
"""Record the built container image's digest as release provenance.

A Dockerfile is a recipe, not an environment. Two builds of the same Dockerfile
weeks apart differ in base image, apt packages and any dependency the recipe
does not pin (2026-08-09 post-response-3 audit, P1-4). What identifies the
environment a result was produced in is the IMAGE DIGEST, and it has to be
recorded at build time because it cannot be recovered afterwards.

    docker build --build-arg BASE_IMAGE=python@sha256:<d> -t compass:0.1.0 .
    python tools/record_container_image.py --tag compass:0.1.0

Writes ``data/provenance/container_image.json``: the image's own digest, its
base digest, the pinned dependency set, and the recorded build time. If Docker
is not available the tool says so and records nothing, rather than inventing a
provenance entry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "provenance" / "container_image.json"


def _docker(*args: str) -> str | None:
    try:
        r = subprocess.run(["docker", *args], capture_output=True, text=True,
                           timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="the built image tag")
    args = ap.parse_args()

    probe = _docker("version", "--format", "{{.Server.Version}}")
    if probe is None:
        print("docker is not available here; nothing recorded. Run this on the "
              "machine that built the release image.", file=sys.stderr)
        return 1

    inspect = _docker("image", "inspect", args.tag,
                      "--format", "{{json .}}")
    if not inspect:
        print(f"no local image {args.tag!r}", file=sys.stderr)
        return 1
    meta = json.loads(inspect)
    base = _docker("image", "inspect", args.tag, "--format",
                   "{{index .Config.Labels \"base.digest\"}}") or None

    rec = {
        "tag": args.tag,
        "image_id": meta.get("Id"),
        "repo_digests": meta.get("RepoDigests", []),
        "base_digest_label": base,
        "created": meta.get("Created"),
        "architecture": meta.get("Architecture"),
        "os": meta.get("Os"),
        "docker_server_version": probe,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "dependency_lock": "requirements.lock",
        "_doc": ("The image digest is the environment identity. A Dockerfile "
                 "alone does not pin the base image or the apt set, so two "
                 "builds of it are two environments (2026-08-09 "
                 "post-response-3 audit, P1-4)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {rec['image_id']}")
    if not rec["repo_digests"]:
        print("NOTE: the image has not been pushed, so it has no repo digest. "
              "Push it before depositing the archive, or the digest identifies "
              "nothing a reader can pull.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
