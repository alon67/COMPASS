# COMPASS reproducible environment (blueprint WP9: archived code and containers).
#
# Build:  docker build -t compass:0.1.0 .
# Run:    docker run --rm -v "$PWD/results:/opt/compass/results" compass:0.1.0 \
#             bash -lc "python tools/fetch_models.py && python tools/build_mechanisms.py && pytest -q"
#
# Model archives are NOT baked into the image: they carry their own licenses and
# are fetched at run time by tools/fetch_models.py, which records a SHA-256 for
# every asset in data/provenance/downloads.json.

# BASE IMAGE: buildable from a tag, ARCHIVED by digest.
#
# `python:3.12-slim-bookworm` is a moving target -- rebuilding under the same tag
# a year from now can pick up a different Python patch release and a different
# libc, and none of the published numbers are claimed to be invariant to that
# (2026-08-09 post-response-3 audit, P1-4). The tag is kept as the default so an
# ordinary `docker build .` works, and the RELEASE build pins the digest:
#
#   docker build --build-arg BASE_IMAGE=python@sha256:<digest> -t compass:0.1.0 .
#   python tools/record_container_image.py --tag compass:0.1.0
#
# `tools/record_container_image.py` writes the built image's own digest and its
# base digest to data/provenance/container_image.json, which is release content.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    NEURON_MODULE_OPTIONS=-nogui \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# NEURON needs a C compiler and bison/flex to run nrnivmodl on the .mod files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        bison \
        flex \
        git \
        ca-certificates \
        libncurses-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/compass

COPY pyproject.toml README.md LICENSE requirements.lock ./
COPY src ./src

# THE LOCK, then the package with --no-deps.
#
# Installing `-e .` alone resolves the lower bounds in pyproject.toml and can
# select a materially different NEURON, eFEL, SciPy or statsmodels than the one
# every published number was produced under. requirements.lock is those exact
# versions; `--no-deps` stops the editable install from re-resolving them.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip install --no-cache-dir -e . --no-deps \
    && python -m pip install --no-cache-dir jupyter nbconvert ipykernel

COPY tools ./tools
COPY tests ./tests
COPY configs ./configs
COPY docs ./docs
COPY notebooks ./notebooks

# Fail the build if the environment cannot import NEURON, does not match the
# archived lock, or cannot pass the fast suite.
RUN python -c "import neuron, efel, SALib; print('NEURON', neuron.__version__)" \
    && python tools/check_environment_lock.py \
    && python -m pytest tests/unit tests/numerical -q

CMD ["bash"]
