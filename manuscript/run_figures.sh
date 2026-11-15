#!/usr/bin/env bash
# Regenerate the manuscript figures and the summary tables they use, from stored
# analysis tables in this repository only (no simulation is run).
#   bash manuscript/run_figures.sh [OUTDIR]     default OUTDIR: results/figures/manuscript
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
OUT="${1:-$ROOT/results/figures/manuscript}"; TAB="$ROOT/results/analyses/manuscript"
mkdir -p "$OUT" "$TAB"
python3 "$HERE/variance_partition_bootstrap.py" "$TAB/variance_partition.json" > /dev/null
python3 "$HERE/summarise_review_analyses.py" "$TAB/review_numbers.json"
python3 "$HERE/make_figures.py" "$OUT" "$TAB/variance_partition.json" \
        "$TAB/review_numbers_null_table.csv" "$TAB/review_numbers_dropout_table.csv"
python3 "$HERE/make_figure_s1.py" "$OUT/figS1.png"
echo "figures in $OUT; tables in $TAB"
