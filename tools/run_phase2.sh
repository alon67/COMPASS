#!/usr/bin/env bash
# Phase 2: the corrective campaign demanded by the 2026-08-05 review.
#
#   migliore rerun -> re-analysis -> size matching -> robustness
#   -> perturbation rerun -> cross-model decision -> reports
#
# Each stage depends on the one before it, so this runs strictly in order.
# DO NOT edit anything under src/ while this is running (decision log D13).
#
# Failure handling: `set -euo pipefail` is mandatory here.  Every stage used to
# be piped through `tail`, so the shell saw tail's exit status (always 0) and a
# Python traceback in stage C would let stages D..H run happily on stale inputs
# and print "phase 2 complete".  The 2026-08-06 audit (M10) found 13 such pipes
# in this file alone.  Output is now teed to a log instead: the console stays
# readable, the exit status stays real, and the full output survives.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export NEURON_MODULE_OPTIONS=-nogui

LOGDIR="results/reports/phase2_logs"
mkdir -p "$LOGDIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

fail() {
    local stage=$1
    log "FAILED at stage ${stage}; full output in ${LOGDIR}/${RUN_ID}_${stage}.log"
    exit 1
}

# run <stage-tag> <tail-lines> <command...>
# Streams everything to a per-stage log, echoes the last N lines, and propagates
# the command's real exit status.
run() {
    local stage=$1 keep=$2
    shift 2
    local out="${LOGDIR}/${RUN_ID}_${stage}.log"
    if ! "$@" >"$out" 2>&1; then
        tail -n 40 "$out" || true
        fail "$stage"
    fi
    tail -n "$keep" "$out" || true
}

log "stage A: re-simulate the four models whose settings the audit rejected"
# Cheapest first, so results accumulate; mainen_l5pc (6.1x cost) runs last.
run A_hh_point 3 python3 tools/run_campaign.py --experiment screening \
    --models hh_point --regimes E1 E2 E6 --workers 12
run A_khaliq 3 python3 tools/run_campaign.py --experiment screening \
    --models khaliq_purkinje --regimes E1 E2 E6 --workers 12
run A_bbp 3 python3 tools/run_campaign.py --experiment screening \
    --models bbp_l5pc --regimes E1 E2 E6 --workers 12
run A_mainen 3 python3 tools/run_campaign.py --experiment screening \
    --models mainen_l5pc --regimes E1 --workers 12

log "stage B: verify no failed simulations anywhere"
# Non-zero exit if any shard holds a failed simulation, so a silent numerical
# regression cannot slip past into the analysis stages.
python3 - <<'PY'
import glob
import sys

import pandas as pd

bad = 0
for f in sorted(glob.glob("results/features/screening/*/*.parquet")):
    d = pd.read_parquet(f, columns=["simulation_status"])
    bad += int((d.simulation_status != "ok").sum())
print(f"failed simulations across all shards: {bad}")
sys.exit(1 if bad else 0)
PY

log "stage C: information analysis (primary QC policy)"
run C_analysis 12 python3 tools/run_analysis.py --experiment screening

log "stage D: size-matched cross-model recomputation"
run D_matched 6 python3 tools/run_matched.py --experiment screening --draws 20

log "stage F: perturbation battery (E3 backgrounds, in-prior, 4 features)"
run F_perturb 30 python3 -u tools/run_perturbations.py --experiment screening --workers 8

log "stage F2: readout measurement floors (audit M5)"
run F2_resolution 10 python3 tools/readout_resolution.py

log "stage G: cross-model analysis and preregistered decision"
run G_crossmodel 20 python3 tools/run_crossmodel.py --experiment screening

log "stage G2: robustness (AFTER the decision, which regenerates lomo.parquet)"
run G2_robust 4 python3 tools/run_robustness.py --experiment screening --boot 2000

log "stage H: key numbers, model audit, notebook"
run H_summary 2 python3 tools/summarise_results.py
run H_audit 2 python3 tools/build_model_audit.py
run H_notebook 1 python3 tools/build_notebook.py
run H_execute 1 python3 -m jupyter nbconvert --execute --to notebook --inplace \
    notebooks/COMPASS_results.ipynb

log "phase 2 complete; logs in ${LOGDIR}/${RUN_ID}_*.log"
