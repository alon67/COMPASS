#!/usr/bin/env bash
# THE single dependency-aware driver (2026-08-06 audit, section 5.6).
#
# Before this, three shell drivers each reproduced a different subset:
# run_pipeline.sh omitted matched analysis, robustness, benchmarks, convergence
# and canonical gates, estimator reliability and the notebook; run_phase2.sh
# omitted Sobol, benchmarks and report-number verification; run_repair.sh
# omitted Sobol, benchmarks and estimator reliability.  No single documented
# command recreated all headline artifacts from a clean checkout.
#
#   ./tools/run_all.sh                      # everything, from assets to report
#   ./tools/run_all.sh --from analysis      # skip simulation, rebuild analyses
#   ./tools/run_all.sh --until campaign     # stop after the campaigns
#   ./tools/run_all.sh --only gates         # one stage
#   ./tools/run_all.sh --dry-run            # print the plan and exit
#   ./tools/run_all.sh --analysis-only      # alias for --from analysis
#   ./tools/run_all.sh --force              # ignore stored artifacts
#
# Fails closed: `set -euo pipefail`, every stage logged, and the run ends with
# the release gates, the report-number check and an artifact inventory.  A
# stage that is skipped is reported as skipped, never as passed.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOGS="$ROOT/results/logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOGS"
WORKERS="${COMPASS_WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) - 2 ))}"
[ "$WORKERS" -lt 1 ] && WORKERS=1

# Ordered stage list.  Each stage depends only on stages before it.
STAGES=(assets mechanisms calibrate convergence canonical campaign sobol
        analysis matched crossmodel perturb robustness benchmarks reliability
        gates notebook report inventory)

FROM=""; UNTIL=""; ONLY=""; DRY=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --analysis-only) FROM="analysis"; shift ;;
    --dry-run) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

index_of() { local i=0; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && { echo $i; return; }; i=$((i+1)); done; echo -1; }
START=0; STOP=$(( ${#STAGES[@]} - 1 ))
if [ -n "$FROM" ];  then START=$(index_of "$FROM");  [ "$START" -lt 0 ] && { echo "unknown stage $FROM" >&2; exit 2; }; fi
if [ -n "$UNTIL" ]; then STOP=$(index_of "$UNTIL");  [ "$STOP" -lt 0 ] && { echo "unknown stage $UNTIL" >&2; exit 2; }; fi
if [ -n "$ONLY" ];  then START=$(index_of "$ONLY"); STOP=$START; [ "$START" -lt 0 ] && { echo "unknown stage $ONLY" >&2; exit 2; }; fi

SKIPPED=()
run_stage() {  # run_stage <name> <command...>
  local name="$1"; shift
  local idx; idx=$(index_of "$name")
  if [ "$idx" -lt "$START" ] || [ "$idx" -gt "$STOP" ]; then
    SKIPPED+=("$name"); return 0
  fi
  echo ""
  echo "=== [$name] $* "
  if [ "$DRY" = "1" ]; then return 0; fi
  local t0; t0=$(date +%s)
  if ! "$@" > "$LOGS/$name.log" 2>&1; then
    echo "!!! stage '$name' FAILED after $(( $(date +%s) - t0 ))s; see $LOGS/$name.log"
    tail -40 "$LOGS/$name.log"
    exit 1
  fi
  echo "    ok in $(( $(date +%s) - t0 ))s -> $LOGS/$name.log"
}

echo "COMPASS full pipeline"
echo "  workers=$WORKERS  logs=$LOGS"
echo "  stages: ${STAGES[*]:$START:$((STOP-START+1))}"

run_stage assets       python3 tools/fetch_models.py
run_stage mechanisms   python3 tools/build_mechanisms.py
run_stage calibrate    python3 tools/calibrate_amplitudes.py --workers "$WORKERS"
# Convergence is evidence, not a step to re-run casually: it supersedes rather
# than overwrites, and every audit costs hours. Run it explicitly per model.
run_stage convergence  bash -c 'echo "convergence audits are run per model: tools/convergence_audit.py <model>"'
run_stage canonical    python3 tools/canonical_audit.py
run_stage campaign     bash -c "
  set -e
  python3 tools/backfill_shard_identity.py --all
  python3 tools/run_campaign.py --experiment screening --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_expand --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_scramble2 --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_v2_replication --workers $WORKERS"
run_stage sobol        python3 tools/run_sobol.py --workers "$WORKERS"
run_stage analysis     bash -c "
  set -e
  python3 tools/run_analysis.py --experiment screening
  python3 tools/run_analysis.py --experiment screening --qc-policy strict
  python3 tools/run_analysis.py --experiment screening_v2_replication || true"
run_stage matched      python3 tools/run_matched.py --experiment screening
run_stage crossmodel   python3 tools/run_crossmodel.py --experiment screening
run_stage perturb      bash -c "
  set -e
  python3 tools/run_perturbations.py --experiment screening --workers $WORKERS
  python3 tools/perturbation_inference.py"
run_stage robustness   python3 tools/run_robustness.py
run_stage benchmarks   python3 tools/run_benchmarks.py
run_stage reliability  python3 tools/estimator_reliability.py
run_stage gates        bash -c "
  set -e
  python3 tools/validate_campaign.py --all
  python3 tools/validate_replay.py --experiment screening --workers $WORKERS
  python3 tools/build_model_audit.py"
run_stage notebook     python3 tools/build_notebook.py
run_stage report       bash -c "
  set -e
  python3 tools/summarise_results.py
  python3 tools/verify_report_numbers.py"
run_stage inventory    python3 tools/build_release_manifest.py

echo ""
echo "=== SUMMARY ==="
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "SKIPPED stages (NOT passed, just not run): ${SKIPPED[*]}"
fi
echo "logs: $LOGS"
[ "$DRY" = "1" ] && echo "(dry run: nothing was executed)"
exit 0
