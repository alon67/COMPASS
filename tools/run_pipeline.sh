#!/usr/bin/env bash
# COMPASS analysis pipeline: repair -> information analysis -> Sobol ->
# perturbations -> cross-model decision -> report artefacts.
#
# Every stage skips work whose output already exists, so the pipeline is
# resumable and safe to restart.
#
# IMPORTANT: do not edit anything under src/ while this is running. Workers are
# spawned fresh per block and re-import the package, so a source tree that is
# momentarily inconsistent (e.g. a call added before the method it calls) is
# picked up by whichever blocks happen to start in that window. See decision
# log D13.
#
# Failure handling: `set -euo pipefail` is mandatory.  Piping a stage through
# `tail` hands the shell tail's exit status, so a stage could die and the
# pipeline would still print "pipeline complete" -- with downstream stages
# reading stale inputs.  The 2026-08-06 audit (M10) found 8 such pipes here.
# Stages now tee to per-stage logs and propagate their real exit status.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export NEURON_MODULE_OPTIONS=-nogui

LOGDIR="results/reports/pipeline_logs"
mkdir -p "$LOGDIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# run <stage-tag> <tail-lines> <command...>
run() {
    local stage=$1 keep=$2
    shift 2
    local out="${LOGDIR}/${RUN_ID}_${stage}.log"
    if ! "$@" >"$out" 2>&1; then
        tail -n 40 "$out" || true
        log "FAILED at stage ${stage}; full output in ${out}"
        exit 1
    fi
    tail -n "$keep" "$out" || true
}

log "stage 0: repair any missing/corrupted shards"
run 0_repair_a 3 python3 tools/run_campaign.py --experiment screening \
    --models hh_point hh_cable pospischil_rs pospischil_fs khaliq_purkinje bbp_l5pc \
    --regimes E1 E2 E6 --workers 12
run 0_repair_b 3 python3 tools/run_campaign.py --experiment screening \
    --models mainen_l5pc migliore_ca1 --regimes E1 --workers 12

log "stage 1: verify no failed simulations remain"
# Exits non-zero if anything failed, so the analysis stages cannot silently
# consume a partially broken campaign.
python3 - <<'PY'
import glob
import sys

import pandas as pd

bad = 0
for f in sorted(glob.glob("results/features/screening/*/*.parquet")):
    d = pd.read_parquet(f, columns=["simulation_status"])
    bad += int((d.simulation_status != "ok").sum())
print(f"remaining failed simulations across all shards: {bad}")
sys.exit(1 if bad else 0)
PY

log "stage 2: information analysis"
run 2_analysis 12 python3 tools/run_analysis.py --experiment screening

log "stage 3: Sobol campaign"
run 3_sobol 12 python3 tools/run_sobol.py --workers 12 \
    --models hh_point hh_cable pospischil_rs pospischil_fs khaliq_purkinje bbp_l5pc

log "stage 4: perturbation battery"
run 4_perturb 14 python3 tools/run_perturbations.py --experiment screening --workers 8

log "stage 4b: readout measurement floors (audit M5)"
run 4b_resolution 10 python3 tools/readout_resolution.py

log "stage 5: cross-model analysis and decision"
run 5_crossmodel 30 python3 tools/run_crossmodel.py --experiment screening

log "stage 6: key numbers and model audit"
run 6_summary 3 python3 tools/summarise_results.py
run 6_audit 2 python3 tools/build_model_audit.py

log "stage 7: reports must agree with the artefacts they were generated from"
run 7_verify 5 python3 tools/verify_report_numbers.py

log "pipeline complete; logs in ${LOGDIR}/${RUN_ID}_*.log"
