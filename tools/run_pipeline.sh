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
set -u
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export NEURON_MODULE_OPTIONS=-nogui

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "stage 0: repair any missing/corrupted shards"
python3 tools/run_campaign.py --experiment screening \
    --models hh_point hh_cable pospischil_rs pospischil_fs khaliq_purkinje bbp_l5pc \
    --regimes E1 E2 E6 --workers 12 2>&1 | tail -3
python3 tools/run_campaign.py --experiment screening \
    --models mainen_l5pc migliore_ca1 --regimes E1 --workers 12 2>&1 | tail -3

log "stage 1: verify no failed simulations remain"
python3 - <<'PY'
import glob, pandas as pd
bad = 0
for f in sorted(glob.glob("results/features/screening/*/*.parquet")):
    d = pd.read_parquet(f, columns=["simulation_status"])
    bad += int((d.simulation_status != "ok").sum())
print(f"remaining failed simulations across all shards: {bad}")
PY

log "stage 2: information analysis"
python3 tools/run_analysis.py --experiment screening 2>&1 | tail -12

log "stage 3: Sobol campaign"
python3 tools/run_sobol.py --workers 12 \
    --models hh_point hh_cable pospischil_rs pospischil_fs khaliq_purkinje \
             bbp_l5pc 2>&1 | tail -12

log "stage 4: perturbation battery"
python3 tools/run_perturbations.py --experiment screening --workers 8 2>&1 | tail -14

log "stage 5: cross-model analysis and decision"
python3 tools/run_crossmodel.py --experiment screening 2>&1 | tail -30

log "stage 6: key numbers and model audit"
python3 tools/summarise_results.py 2>&1 | tail -3
python3 tools/build_model_audit.py 2>&1 | tail -2

log "pipeline complete"
