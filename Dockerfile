# COMPASS reproducible environment (blueprint WP9: archived code and containers).
#
# Build:  docker build -t compass:0.1.0 .
# Run:    docker run --rm -v "$PWD/results:/opt/compass/results" compass:0.1.0 \
#             bash -lc "python tools/fetch_models.py && python tools/build_mechanisms.py && pytest -q"
#
# Model archives are NOT baked into the image: they carry their own licenses and
# are fetched at run time by tools/fetch_models.py, which records a SHA-256 for
# every asset in data/provenance/downloads.json.

FROM python:3.12-slim-bookworm

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

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir pytest ruff jupyter nbconvert ipykernel

COPY tools ./tools
COPY tests ./tests
COPY configs ./configs
COPY docs ./docs
COPY notebooks ./notebooks

# Fail the build if the environment cannot import NEURON or compile a mechanism.
RUN python -c "import neuron, efel, SALib; print('NEURON', neuron.__version__)" \
    && python -m pytest tests/unit tests/numerical -q

CMD ["bash"]
