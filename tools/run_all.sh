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
#   ./tools/run_all.sh --force              # quarantine derived artifacts first
#   ./tools/run_all.sh --force-simulations  # also quarantine feature shards
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
#
# `perturb` MUST precede `crossmodel`.  It did not, and the consequence was not
# cosmetic: run_crossmodel scores two H-B criteria from perturbations.parquet
# and now also reads the numerical-repeatability verdict in
# perturbation_inference.json, so on a clean tree the decision was computed from
# perturbation artifacts that did not exist yet, and on a dirty one from
# whatever the previous run had left behind (post-response audit, section 3.3).
#
# `numerical` is separated from `gates` because the two fail for different
# reasons.  A campaign-integrity or replay failure is a broken pipeline and must
# stop it; a failing N1/N2/N4 gate is the expected, recorded state of this
# release and must NOT stop it, or the pipeline can never reach the report and
# release-manifest stages whose whole job is to describe that state.
STAGES=(assets mechanisms calibrate convergence canonical campaign sobol
        analysis matched perturb crossmodel robustness benchmarks reliability
        gates waveforms numerical notebook report inventory)

FROM=""; UNTIL=""; ONLY=""; DRY=0; FORCE=0; FORCE_SIM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --until) UNTIL="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --analysis-only) FROM="analysis"; shift ;;
    --dry-run) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    --force-simulations) FORCE=1; FORCE_SIM=1; shift ;;
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

# CLEAN-RUN SEMANTICS (2026-08-09 audit, 4.4).
#
# `--force` used to be parsed and never read, so the only thing it did was let a
# caller believe the run had ignored stored artifacts.  It now has a definition:
#
#   --force              quarantine every DERIVED artifact (analyses, validation
#                        records, figures, generated reports) before starting, so
#                        no stage can consume a previous run's output.  Feature
#                        shards survive: their reuse is already identity-gated,
#                        and discarding 1.1 GB of correct simulation is not a
#                        reproducibility improvement.
#   --force-simulations  additionally quarantine the feature shards and the
#                        perturbation cache: a genuine from-zero reconstruction.
#
# Both are recorded in the run receipt, and `tools/run_receipt.py check` reports
# any manifest artifact the run did not itself produce.
quarantine_derived() {
  local dest="$ROOT/results/_superseded_run_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dest"
  for d in analyses validation figures reports; do
    [ -d "$ROOT/results/$d" ] && mv "$ROOT/results/$d" "$dest/$d"
  done
  if [ "$FORCE_SIM" = "1" ]; then
    [ -d "$ROOT/results/features" ] && mv "$ROOT/results/features" "$dest/features"
    [ -d "$ROOT/results/traces" ] && mv "$ROOT/results/traces" "$dest/traces"
  fi
  echo "  --force: derived artifacts moved to ${dest#"$ROOT"/}"
}

SKIPPED=()
receipt() { [ "$DRY" = "1" ] && return 0; python3 tools/run_receipt.py "$@" >/dev/null 2>&1 || true; }

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
    receipt stage "$name" 1 "$(( $(date +%s) - t0 ))"
    echo "!!! stage '$name' FAILED after $(( $(date +%s) - t0 ))s; see $LOGS/$name.log"
    tail -40 "$LOGS/$name.log"
    exit 1
  fi
  receipt stage "$name" 0 "$(( $(date +%s) - t0 ))"
  echo "    ok in $(( $(date +%s) - t0 ))s -> $LOGS/$name.log"
}

run_stage_soft() {  # like run_stage, but returns the status instead of exiting
  local name="$1"; shift
  local idx; idx=$(index_of "$name")
  if [ "$idx" -lt "$START" ] || [ "$idx" -gt "$STOP" ]; then
    SKIPPED+=("$name"); return 0
  fi
  echo ""
  echo "=== [$name] $* "
  if [ "$DRY" = "1" ]; then return 0; fi
  local t0; t0=$(date +%s)
  local rc=0
  "$@" > "$LOGS/$name.log" 2>&1 || rc=$?
  receipt stage "$name" "$rc" "$(( $(date +%s) - t0 ))"
  tail -20 "$LOGS/$name.log"
  echo "    stage '$name' exited $rc after $(( $(date +%s) - t0 ))s -> $LOGS/$name.log"
  return "$rc"
}

echo "COMPASS full pipeline"
echo "  workers=$WORKERS  logs=$LOGS"
echo "  stages: ${STAGES[*]:$START:$((STOP-START+1))}"
if [ "$FORCE" = "1" ] && [ "$DRY" = "0" ]; then quarantine_derived; fi
receipt begin --stages "${STAGES[@]:$START:$((STOP-START+1))}" $( [ "$FORCE" = "1" ] && echo --force )

run_stage assets       python3 tools/fetch_models.py
run_stage mechanisms   python3 tools/build_mechanisms.py
run_stage calibrate    python3 tools/calibrate_amplitudes.py --workers "$WORKERS"
# Convergence is evidence, not a step to re-run casually: it supersedes rather
# than overwrites, and every audit costs hours (the panel costs days). This
# stage therefore VERIFIES the stored evidence against the solver contract and
# records `regenerated: false` with the reason -- instead of printing a sentence
# and reporting "ok", which is what it used to do. Regenerate explicitly, per
# model: python3 tools/convergence_audit.py <model_id>
run_stage convergence  python3 tools/verify_convergence_evidence.py
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
  python3 tools/run_analysis.py --experiment screening_v2_replication
  python3 tools/run_analysis.py --experiment screening_scramble2
  python3 tools/run_analysis.py --experiment screening --suffix _dropfeat \
      --drop-features step_abs_hi.adaptation_index
  python3 tools/run_analysis.py --experiment screening --suffix _n4proj \
      --drop-unidentifiable-parameters
  python3 tools/run_analysis.py --experiment screening_v2_replication \
      --suffix _n4proj --drop-unidentifiable-parameters"
# The matched design pools the expansion campaign; without --pool it silently
# reproduces the pre-expansion result and E7 stays below MIN_MATCH_N.
run_stage matched      python3 tools/run_matched.py --experiment screening --pool screening_expand
# Perturbations BEFORE cross-model scoring: run_crossmodel reads both
# perturbations.parquet and the repeatability verdict in
# perturbation_inference.json, and withholds its two H-B criteria when the
# verdict is absent.
run_stage perturb      bash -c "
  set -e
  python3 tools/run_perturbations.py --experiment screening --workers $WORKERS
  python3 tools/run_perturbations.py --experiment screening --workers $WORKERS \
      --dt-scale 0.5 --suffix _dt_half
  python3 tools/perturbation_inference.py"
run_stage crossmodel   python3 tools/run_crossmodel.py --experiment screening
run_stage robustness   python3 tools/run_robustness.py
run_stage benchmarks   python3 tools/run_benchmarks.py
run_stage reliability  python3 tools/estimator_reliability.py
# REPLAY MUST COVER EVERY CAMPAIGN WITH SHARDS.
#
# It covered screening and the replication cohort only, so the scramble
# controls, the expansion campaign and the whole Sobol design were release
# artifacts with no determinism evidence at all -- and because the release
# manifest aggregated by gate NAME, one passing REPLAY record satisfied the
# mandatory gate for all of them (2026-08-09 audit, 3.1 and 5.7). The manifest
# now requires a passing record per named campaign, so an omission here fails
# the release instead of disappearing into the roll-up.
run_stage gates        bash -c "
  set -e
  python3 tools/validate_campaign.py --all
  for exp in screening screening_expand screening_scramble2 \
             screening_v2_replication sobol pilot; do
    python3 tools/validate_replay.py --experiment \$exp --workers $WORKERS
  done
  python3 tools/build_model_audit.py"
# Replay checks that a stored FEATURE ROW reproduces. It cannot see whether the
# voltage behind it is plausible, because a trace that is wrong the same way
# twice reproduces perfectly. This retains a stratified waveform sample and runs
# the trace-level battery on it. Soft, for the same reason as `numerical`: a
# flagged trace is a recorded finding, and the manifest is what blocks.
# Coverage extends to the REPLICATION campaign, which had no retained waveforms
# at all (2026-08-09 audit, 3.3), and the declared stratification policy is
# itself a gate subject.
WAVE_RC=0
run_stage_soft waveforms bash -c "
  rc=0
  for exp in screening screening_expand screening_scramble2 \
             screening_v2_replication sobol; do
    python3 tools/retain_waveforms.py --experiment \$exp --workers $WORKERS || rc=\$?
  done
  exit \$rc" || WAVE_RC=$?
# The neuronal-validity gates are EVIDENCE-PRODUCING, and this release is known
# to fail N1, N2 and N4.  Under the hard runner that non-zero exit aborted the
# pipeline before the model audit, notebook, report and release manifest -- so
# the special handling for a blocked release, three stages later, was
# unreachable and the release could never be described (post-response audit,
# section 3.3).  The status is captured and reported; the manifest is what
# blocks the release.
GATES_RC=0
run_stage_soft numerical python3 tools/record_numerical_gates.py --slow || GATES_RC=$?
if [ "$GATES_RC" -ne 0 ]; then
  echo "    (numerical gates reported failures; recorded and carried forward to"
  echo "     the release manifest, which is what fails the release)"
fi
run_stage notebook     python3 tools/build_notebook.py
run_stage report       bash -c "
  set -e
  python3 tools/summarise_results.py
  python3 tools/verify_report_numbers.py"
# The release manifest exits non-zero when the verdict is BLOCKED.  That is a
# real signal that has to reach the caller, but it is not a pipeline crash: the
# manifest was built correctly and simply reports that the release is not
# releasable.  So capture the status rather than aborting here, print the
# summary, and then exit with it.
RELEASE_RC=0
run_stage_soft inventory python3 tools/build_release_manifest.py || RELEASE_RC=$?
receipt finish
if [ "$DRY" = "0" ]; then
  echo ""; echo "=== [clean-build check] which artifacts did THIS run produce?"
  python3 tools/run_receipt.py check || true
fi

echo ""
echo "=== SUMMARY ==="
if [ ${#SKIPPED[@]} -gt 0 ]; then
  echo "SKIPPED stages (NOT passed, just not run): ${SKIPPED[*]}"
fi
echo "logs: $LOGS"
[ "$DRY" = "1" ] && echo "(dry run: nothing was executed)"
if [ "$GATES_RC" -ne 0 ]; then
  echo "numerical gates: FAILURES RECORDED (exit $GATES_RC) -- see"
  echo "results/validation/numerical_gates.json. The pipeline continued so the"
  echo "release manifest could describe the failure; it did not pass."
fi
if [ "$WAVE_RC" -ne 0 ]; then
  echo "retained waveforms: FINDINGS RECORDED (exit $WAVE_RC) -- see"
  echo "results/validation/retained_waveforms_screening.json and"
  echo "data/provenance/waveform_findings.json."
fi
if [ "$RELEASE_RC" -ne 0 ]; then
  echo "RELEASE BLOCKED -- see the blockers above and in"
  echo "results/validation/release_manifest.json. Exiting $RELEASE_RC so a"
  echo "caller cannot mistake a blocked release for a successful one."
fi
exit "$RELEASE_RC"
