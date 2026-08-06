#!/usr/bin/env bash
# Phase 2: the corrective campaign demanded by the 2026-08-05 review.
#
#   migliore rerun -> re-analysis -> size matching -> robustness
#   -> perturbation rerun -> cross-model decision -> reports
#
# Each stage depends on the one before it, so this runs strictly in order.
# DO NOT edit anything under src/ while this is running (decision log D13).
set -u
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export NEURON_MODULE_OPTIONS=-nogui

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "stage A: re-simulate the four models whose settings the audit rejected"
# Cheapest first, so results accumulate; mainen_l5pc (6.1x cost) runs last.
python3 tools/run_campaign.py --experiment screening --models hh_point \
    --regimes E1 E2 E6 --workers 12 2>&1 | tail -3
python3 tools/run_campaign.py --experiment screening --models khaliq_purkinje \
    --regimes E1 E2 E6 --workers 12 2>&1 | tail -3
python3 tools/run_campaign.py --experiment screening --models bbp_l5pc \
    --regimes E1 E2 E6 --workers 12 2>&1 | tail -3
python3 tools/run_campaign.py --experiment screening --models mainen_l5pc \
    --regimes E1 --workers 12 2>&1 | tail -3

log "stage B: verify no failed simulations anywhere"
python3 - <<'PY'
import glob, pandas as pd
bad = 0
for f in sorted(glob.glob("results/features/screening/*/*.parquet")):
    d = pd.read_parquet(f, columns=["simulation_status"])
    bad += int((d.simulation_status != "ok").sum())
print(f"failed simulations across all shards: {bad}")
PY

log "stage C: information analysis (primary QC policy)"
python3 tools/run_analysis.py --experiment screening 2>&1 | tail -12

log "stage D: size-matched cross-model recomputation"
python3 tools/run_matched.py --experiment screening --draws 20 2>&1 | tail -6

log "stage F: perturbation battery (E3 backgrounds, in-prior, 4 features)"
python3 -u tools/run_perturbations.py --experiment screening --workers 8 2>&1 | tail -30

log "stage G: cross-model analysis and preregistered decision"
python3 tools/run_crossmodel.py --experiment screening 2>&1 | tail -20

log "stage G2: robustness (AFTER the decision, which regenerates lomo.parquet)"
python3 tools/run_robustness.py --experiment screening --boot 2000 2>&1 | tail -4

log "stage H: key numbers, model audit, notebook"
python3 tools/summarise_results.py 2>&1 | tail -2
python3 tools/build_model_audit.py 2>&1 | tail -2
python3 tools/build_notebook.py 2>&1 | tail -1
cd notebooks && python3 -m jupyter nbconvert --execute --to notebook --inplace \
    COMPASS_results.ipynb 2>&1 | tail -1
cd ..

log "phase 2 complete"
