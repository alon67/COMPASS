#!/usr/bin/env bash
# Repair cascade for the 2026-08-06 audit blockers C1-C4.
#
# Rebuilding the canonical manifests under the campaign's own solver contract
# (D24) changes the conditioning target, which changes ensemble membership,
# which changes every downstream analysis and the perturbation backgrounds.
# This runs that cascade in dependency order.
#
# DO NOT edit anything under src/ while this is running (decision log D13).
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export NEURON_MODULE_OPTIONS=-nogui

LOGDIR="results/reports/repair_logs"
mkdir -p "$LOGDIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

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

# R1 is expensive (~30 min) and idempotent, so skip it when the stored
# manifests already satisfy the contract.  The test is the authority on that,
# not a flag: if it passes, the artefacts are current by definition.
if python3 -m pytest tests/unit/test_solver_contract.py -q -m "not slow" >/dev/null 2>&1; then
    log "stage R1: manifests already match CAMPAIGN_SOLVER, skipping rebuild"
else
    log "stage R1: rebuild canonical manifests under CAMPAIGN_SOLVER (C2 / D24)"
    run R1_manifests 20 python3 tools/canonical_audit.py
fi

log "stage R2: confirm the artefacts now match the one contract"
run R2_contract 10 python3 -m pytest tests/unit/test_solver_contract.py -q -m "not slow"

log "stage R3: re-derive ensembles against the corrected targets"
run R3_analysis 12 python3 tools/run_analysis.py --experiment screening

log "stage R4: size-matched cross-model recomputation"
run R4_matched 6 python3 tools/run_matched.py --experiment screening --draws 20

log "stage R5: perturbation battery (E3 backgrounds move with the targets)"
run R5_perturb 30 python3 -u tools/run_perturbations.py --experiment screening --workers 8

log "stage R6: readout measurement floors"
run R6_resolution 10 python3 tools/readout_resolution.py

log "stage R7: cross-model analysis and preregistered decision"
run R7_crossmodel 20 python3 tools/run_crossmodel.py --experiment screening

log "stage R8: robustness (after the decision regenerates lomo.parquet)"
run R8_robust 4 python3 tools/run_robustness.py --experiment screening --boot 2000

log "stage R9: key numbers, model audit, notebook"
run R9_summary 3 python3 tools/summarise_results.py
run R9_audit 2 python3 tools/build_model_audit.py
run R9_notebook 1 python3 tools/build_notebook.py
run R9_execute 1 python3 -m jupyter nbconvert --execute --to notebook --inplace \
    notebooks/COMPASS_results.ipynb

log "stage R10: reports must agree with the artefacts they were generated from"
run R10_verify 5 python3 tools/verify_report_numbers.py

log "repair cascade complete; logs in ${LOGDIR}/${RUN_ID}_*.log"
