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
STAGES=(environment assets mechanisms calibrate convergence canonical campaign
        sobol resolution analysis matched perturb crossmodel robustness
        benchmarks reliability gates waveforms numerical notebook report
        inventory)

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
  # RETAINED HISTORY IS NOT A DERIVED ARTIFACT.
  #
  # `results/analyses/_withdrawn` holds artifacts that were deliberately
  # withdrawn and are kept so the record is legible -- the obsolete
  # `*_n4_projected` outputs that still assert a conclusion the project no
  # longer holds, and the withdrawn readout-resolution analysis. No stage
  # regenerates them, by design, so sweeping them into a superseded-run
  # directory would silently destroy the history the 2026-08-09
  # post-response-3 audit (P0-4C) asked to have preserved rather than deleted.
  # They are restored in place, and the release manifest keeps blocking on any
  # copy that reappears OUTSIDE this directory.
  if [ -d "$dest/analyses/_withdrawn" ]; then
    mkdir -p "$ROOT/results/analyses"
    mv "$dest/analyses/_withdrawn" "$ROOT/results/analyses/_withdrawn"
    echo "  --force: retained results/analyses/_withdrawn (history, not output)"
  fi
  # THE PERTURBATION CACHE IS IDENTITY-GATED SIMULATION, NOT A STALE ANALYSIS.
  #
  # It lives under results/analyses/, so the sweep above took it -- 25 hours of
  # NEURON with every reuse already bound to the backgrounds, candidate rows,
  # selection code and sim code that produced it (P0-3). That is the argument
  # already written below for feature shards, and it applies here more
  # strongly, not less: discarding correct, identity-checked simulation is not
  # a reproducibility improvement, it is just a slower way to get the same
  # bytes. `--force-simulations` is what discards it, with everything else.
  if [ "$FORCE_SIM" = "0" ] && [ -d "$dest/analyses/_perturbation_cache" ]; then
    mkdir -p "$ROOT/results/analyses"
    mv "$dest/analyses/_perturbation_cache" "$ROOT/results/analyses/_perturbation_cache"
    echo "  --force: retained results/analyses/_perturbation_cache (identity-gated)"
  fi
  if [ "$FORCE_SIM" = "1" ]; then
    [ -d "$ROOT/results/features" ] && mv "$ROOT/results/features" "$dest/features"
    [ -d "$ROOT/results/traces" ] && mv "$ROOT/results/traces" "$dest/traces"
    # The corpus snapshots are the before/after evidence for the regeneration
    # itself; a from-zero run must not erase the "before".
    [ -d "$ROOT/results/corpus_snapshots" ] && \
      echo "  --force-simulations: retained results/corpus_snapshots"
  fi
  echo "  --force: derived artifacts moved to ${dest#"$ROOT"/}"
}

# Which QUARANTINE-VULNERABLE artifacts each stage owns -- the files under
# results/{analyses,validation,figures,reports} that exist only because that
# stage ran.  Stages that write nowhere in those four directories declare "",
# with the reason; tests/unit/test_audit_regressions.py fails if a stage in
# STAGES has no entry here, so the map cannot fall behind the pipeline.
#
# WHY THIS EXISTS.  `--force` discards every derived artifact and `--from`
# rebuilds only the stages in the plan.  Combined, they delete the outputs of
# every earlier stage and never replace them, and nothing between the
# quarantine and the release manifest notices.  That is not hypothetical: the
# 2026-08-10 rebuild ran `--force --from resolution`, which removed
# results/analyses/sobol_*.parquet and results/validation/convergence_evidence
# .json -- a whole required campaign's analysis and the solver-convergence
# record -- from a release whose inventory declares both.  It was found by
# reading the superseded directory, not by the pipeline.  The check below turns
# that into a refusal before anything is moved.
stage_outputs() {
  case "$1" in
    environment|assets|mechanisms|calibrate|resolution)
      echo "" ;;                     # write only to data/provenance/
    canonical|campaign)
      echo "" ;;                     # write only to results/{manifests,features}/
    convergence) echo "results/validation/convergence_evidence.json" ;;
    sobol)       echo "results/analyses/sobol_summary.parquet
                       results/analyses/sobol_indices.parquet
                       results/analyses/sobol_convergence.parquet" ;;
    analysis)    echo "results/analyses/endpoints_screening*.parquet
                       results/analyses/order_curves_screening*.parquet
                       results/analyses/tuple_scores_screening*.parquet
                       results/analyses/rsi_screening*.parquet" ;;
    matched)     echo "results/analyses/matched_design_screening.parquet
                       results/analyses/endpoints_matched_screening.parquet" ;;
    perturb)     echo "results/analyses/perturbations.parquet
                       results/analyses/perturbations_raw.parquet
                       results/analyses/perturbation_inference.json" ;;
    crossmodel)  echo "results/analyses/decision.json
                       results/analyses/lomo.parquet
                       results/analyses/classification.parquet" ;;
    robustness)  echo "results/analyses/robustness*.parquet" ;;
    benchmarks)  echo "results/analyses/benchmarks*.json" ;;
    reliability) echo "results/analyses/estimator_reliability*.json" ;;
    gates)       echo "results/validation/campaign_integrity.json
                       results/validation/replay_gate_*.json" ;;
    waveforms)   echo "results/validation/retained_waveforms_*.json
                       results/validation/waveform_gate_*.json" ;;
    numerical)   echo "results/validation/numerical_gates.json" ;;
    notebook)    echo "results/figures/*" ;;
    report)      echo "results/reports/*" ;;
    inventory)   echo "results/validation/release_manifest.json" ;;
    *) echo "__UNDECLARED__" ;;
  esac
}

# Refuse a plan whose quarantine would delete what it will not rebuild.
check_force_plan() {
  local orphaned="" i=0 s g f
  for s in "${STAGES[@]}"; do
    if [ "$i" -lt "$START" ] || [ "$i" -gt "$STOP" ]; then
      for g in $(stage_outputs "$s"); do
        [ "$g" = "__UNDECLARED__" ] && { echo "!!! stage '$s' has no stage_outputs entry"; exit 2; }
        for f in $ROOT/$g; do
          [ -e "$f" ] && orphaned="$orphaned  ${f#"$ROOT"/}  (only stage '$s' rebuilds it)
"
        done
      done
    fi
    i=$((i+1))
  done
  if [ -n "$orphaned" ]; then
    echo "!!! --force would discard derived artifacts that no stage in this plan rebuilds:"
    printf "%s" "$orphaned"
    echo "Run the earlier stage too (--from $(printf '%s' "${STAGES[$START]}") is too late),"
    echo "or move those files aside deliberately if they are meant to leave the release."
    exit 2
  fi
}

SKIPPED=()
# A RECEIPT FAILURE IS A PIPELINE FAILURE.
#
# This was `|| true`, so every receipt operation could fail silently and the run
# would still report success while producing no attribution at all. The receipt
# is the only artifact that says which stage produced which file; a release run
# that cannot write one has not demonstrated a clean build (2026-08-09
# post-response-3 audit, P1-1).
receipt() {
  [ "$DRY" = "1" ] && return 0
  if ! python3 tools/run_receipt.py "$@" > "$LOGS/receipt.log" 2>&1; then
    echo "!!! run receipt operation failed: run_receipt.py $*"
    tail -20 "$LOGS/receipt.log"
    exit 1
  fi
}

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

# A STAGE THAT RUNS AFTER THE RECEIPT CLOSES IS OUTSIDE IT, BY CONSTRUCTION.
#
# `inventory` builds the release manifest, and the manifest must be the last
# thing written -- it hashes results/validation/run_receipt.json, so any receipt
# write after the build invalidates the manifest that just hashed it (the
# ordering fix of the 2026-08-09 audit, P1-1). But `inventory` was still run
# through `run_stage_soft`, which calls `receipt stage` -- into a receipt that
# `finish` had already closed. Receipt operations are hard failures by design,
# so the 2026-08-11 rebuild ran for twenty hours and then died on its final
# call with `KeyError: '_index'`. Nothing had caught it because no previous run
# had ever reached this line: proving that a clean run completes is exactly
# what P1-1 asked for, and the first attempt found this.
#
# The stage is therefore run WITHOUT touching the receipt, and the receipt
# records that it did so, instead of the exclusion being an accident.
run_stage_unreceipted() {
  local name="$1"; shift
  local idx; idx=$(index_of "$name")
  if [ "$idx" -lt "$START" ] || [ "$idx" -gt "$STOP" ]; then
    SKIPPED+=("$name"); return 0
  fi
  echo ""
  echo "=== [$name] $* (outside the receipt: it runs after the receipt closes)"
  if [ "$DRY" = "1" ]; then return 0; fi
  local t0; t0=$(date +%s)
  local rc=0
  "$@" > "$LOGS/$name.log" 2>&1 || rc=$?
  tail -20 "$LOGS/$name.log"
  echo "    stage '$name' exited $rc after $(( $(date +%s) - t0 ))s -> $LOGS/$name.log"
  return "$rc"
}

echo "COMPASS full pipeline"
echo "  workers=$WORKERS  logs=$LOGS"
echo "  stages: ${STAGES[*]:$START:$((STOP-START+1))}"
if [ "$FORCE" = "1" ] && [ "$DRY" = "0" ]; then check_force_plan; quarantine_derived; fi
receipt begin --stages "${STAGES[@]:$START:$((STOP-START+1))}" \
  $( [ "$FORCE" = "1" ] && echo --force ) \
  $( [ "$FORCE_SIM" = "1" ] && echo --force-simulations )

# The environment the run happens in is evidence, and it is the FIRST thing
# recorded: a pipeline that produces results under a drifted SciPy should say
# so before it produces them, not after (post-response-3 audit, P1-4).
run_stage environment  python3 tools/check_environment_lock.py --write
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
# EVERY DECLARED CAMPAIGN, INCLUDING THE PILOT.
#
# `pilot` was absent here while `--force-simulations` quarantined its shards, so
# a from-zero run silently produced a corpus with one fewer campaign than the
# release inventory declares -- and, because required campaigns used to be
# derived from what was on disk, nothing reported the hole (2026-08-09
# post-response-3 audit, P1-1). The declared set now lives in
# configs/release_inventory.json:required_campaigns and this list must match it;
# tests/unit/test_audit_regressions.py fails if they diverge.
run_stage campaign     bash -c "
  set -e
  python3 tools/backfill_shard_identity.py --all
  python3 tools/run_campaign.py --experiment pilot --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_expand --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_scramble2 --workers $WORKERS
  python3 tools/run_campaign.py --experiment screening_v2_replication --workers $WORKERS"
run_stage sobol        python3 tools/run_sobol.py --workers "$WORKERS"
# Is the declared rheobase resolution fine enough for the firing windows these
# models actually have? Measured on a stratified sample at 8x the campaign
# grid, so the contract in RHEOBASE_SEARCH_CONTRACT is evidence-backed rather
# than asserted (post-response-3 audit, P0-1).
# SOFT: this stage produces EVIDENCE about the measurement, not an input to
# anything downstream, so a failure here must be recorded and carried to the
# release manifest rather than aborting a multi-hour rebuild -- the same
# treatment `numerical` and `waveforms` get, and for the same reason.
#
# --refine 4 rather than 8: a 4x grid resolves windows down to ceiling/256,
# which is what is needed to say whether windows narrower than the campaign
# grid exist at all, and costs half as much. The exhaustive scan does not
# stop at the first hit, so its cost is linear in the refinement factor.
RESOLUTION_RC=0
run_stage_soft resolution python3 tools/audit_rheobase_resolution.py \
                          --experiment screening --per-model 6 --refine 4 \
                          --workers "$WORKERS" || RESOLUTION_RC=$?
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
      --suffix _n4proj --drop-unidentifiable-parameters
  python3 tools/run_analysis.py --experiment screening --suffix _wideband_n4proj \
      --drop-unidentifiable-parameters \
      --readmit-keys data/provenance/voltage_band_tightened_rows.json"
# The matched design pools the expansion campaign; without --pool it silently
# reproduces the pre-expansion result and E7 stays below MIN_MATCH_N.
run_stage matched      python3 tools/run_matched.py --experiment screening --pool screening_expand
# Perturbations BEFORE cross-model scoring: run_crossmodel reads both
# perturbations.parquet and the repeatability verdict in
# perturbation_inference.json, and withholds its two H-B criteria when the
# verdict is absent.
#
# `--resume` MAKES THE IDENTITY CHECK, NOT THE CLOCK, DECIDE.
#
# Without it the battery re-simulates unconditionally: 25 hours, every time,
# even when every cached slice's backgrounds, candidate rows, selection-code
# hash and simulation-code hash are identical to what this run would produce.
# That is not caution, it is the reason a full clean rebuild was avoided rather
# than run. With `--resume` a slice is reused only when `slice_is_reusable()`
# says the identity matches, and a mismatch re-simulates that model and says
# why -- the same policy the campaign stage already applies to feature shards.
run_stage perturb      bash -c "
  set -e
  python3 tools/run_perturbations.py --experiment screening --workers $WORKERS --resume
  python3 tools/run_perturbations.py --experiment screening --workers $WORKERS \
      --resume --dt-scale 0.5 --suffix _dt_half
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
# THE NOTEBOOK IS EXECUTED, NOT ONLY WRITTEN.
#
# `build_notebook.py` emits the .ipynb; nothing ran it, so the figure cells
# never executed and `results/figures/**` -- a declared inventory class with
# min_files 6 -- had no producer at all. The published PNG/PDF pairs had been
# exported by hand, `--force` quarantined them, and the 2026-08-11 release was
# built with zero figures. Executing the notebook makes the class reproducible
# and, incidentally, is the only check that the notebook still runs against the
# current artifacts.
run_stage notebook     bash -c "
  set -e
  python3 tools/build_notebook.py
  python3 -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 notebooks/COMPASS_results.ipynb"
run_stage report       bash -c "
  set -e
  python3 tools/summarise_results.py
  python3 tools/verify_report_numbers.py"
# The release manifest exits non-zero when the verdict is BLOCKED.  That is a
# real signal that has to reach the caller, but it is not a pipeline crash: the
# manifest was built correctly and simply reports that the release is not
# releasable.  So capture the status rather than aborting here, print the
# summary, and then exit with it.
#
# ORDER: FINISH THE RECEIPT, THEN CHECK IT, THEN BUILD THE MANIFEST.
#
# The manifest used to be built while the receipt was still open. `run_receipt
# .json` lives under results/validation and is therefore IN the inventory, so
# the subsequent `receipt stage inventory` and `receipt finish` writes changed a
# file the manifest had just hashed -- leaving every nominal full run with a
# manifest that fails its own verification (2026-08-09 post-response-3 audit,
# P1-1). The manifest build is now the last operation, and nothing mutates an
# inventoried file after it.
#
# The receipt's own stage record is written before `finish` from the timings
# already collected, so closing early costs no attribution.
receipt finish
RELEASE_RC=0
run_stage_unreceipted inventory python3 tools/build_release_manifest.py || RELEASE_RC=$?

# THE CHECK RUNS AFTER THE MANIFEST, BECAUSE IT READS THE MANIFEST.
#
# It used to run before, and on a `--force` run the quarantine had just taken
# the previous manifest away -- so the check printed "no release manifest to
# check against", returned 1, and the driver reported that as "artifacts this
# run did not produce". Two errors in one line: the check had not run at all,
# and the message named a cause it had not established. Observed on the
# 2026-08-11 rebuild, which is what turned it from an argument into a defect.
#
# Moving it after `inventory` is safe: `check` only READS. The ordering
# constraint above is that nothing may MUTATE an inventoried file after the
# manifest is hashed, and this does not. The authoritative list also now lives
# in the manifest itself (clean_build_receipt.artifacts_not_produced_by_this_run,
# computed from the same run_receipt.unattributed() the check calls), so this
# invocation is a human-readable echo of a number the release already carries.
RECEIPT_RC=0
if [ "$DRY" = "0" ]; then
  echo ""; echo "=== [clean-build check] which artifacts did THIS run produce?"
  python3 tools/run_receipt.py check || RECEIPT_RC=$?
  if [ "$RECEIPT_RC" -ne 0 ]; then
    echo "    (the manifest lists artifacts this run did not itself produce;"
    echo "     see clean_build_receipt in results/validation/release_manifest.json)"
  fi
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
if [ "$RESOLUTION_RC" -ne 0 ]; then
  echo "rheobase resolution audit: exited $RESOLUTION_RC -- see"
  echo "data/provenance/rheobase_resolution_audit.json"
fi
if [ "$WAVE_RC" -ne 0 ]; then
  echo "retained waveforms: FINDINGS RECORDED (exit $WAVE_RC) -- see"
  echo "results/validation/retained_waveforms_screening.json and"
  echo "data/provenance/waveform_findings.json."
fi
if [ "$RECEIPT_RC" -ne 0 ]; then
  echo "run receipt: CHECK FAILED (exit $RECEIPT_RC) -- artifacts in the manifest"
  echo "were not produced by this run. See results/validation/run_receipt.json."
fi
if [ "$RELEASE_RC" -ne 0 ]; then
  echo "RELEASE BLOCKED -- see the blockers above and in"
  echo "results/validation/release_manifest.json. Exiting $RELEASE_RC so a"
  echo "caller cannot mistake a blocked release for a successful one."
fi
[ "$RELEASE_RC" -eq 0 ] && [ "$RECEIPT_RC" -ne 0 ] && exit "$RECEIPT_RC"
exit "$RELEASE_RC"
