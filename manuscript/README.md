# Manuscript figures

Scripts that draw the figures of the manuscript *Selection for a shared behavior
creates parameter dependence read as synergy in populations of neuron models*
(Degani-Katzav and Korngreen) and compute the summary tables quoted in it.

They read only analysis tables stored in this repository (`results/analyses/`,
`results/analyses/review_v5/`, `results/analyses/review_v17/`,
`results/analyses/review_v18/` and `results/analyses/review_v19/`). No simulation is run, and the per-simulation
feature tables (`results/features/`, not distributed) are not needed.

```bash
bash manuscript/run_figures.sh
```

| Script | What it does | Output |
|---|---|---|
| `variance_partition_bootstrap.py` | **Withdrawn, kept for the record; not run by `run_figures.sh`.** The V17 attribution, whose geometry block contained measures computed from the covariance matrix. Fig. 5C-D now read `results/analyses/review_v18/attribution.json` (see below). | (none) |
| `summarise_review_analyses.py` | Summarises the analyses added after review: each accepted population compared with same-size unselected and shuffled populations, each electrical feature left out of the acceptance rule in turn, the nearest-neighbour estimate, and the most synergistic triplets. | `results/analyses/manuscript/review_numbers.json`, `review_numbers_null_table.csv`, `review_numbers_dropout_table.csv` |
| `make_figures.py` | Figures 1 to 5 of the main text (Fig. 5C-D from `results/analyses/review_v18/attribution.json`). | `results/figures/manuscript/fig1.png` … `fig5.png` |
| `make_table3.py` | Table 3 of the main text: for each implementation, at 3 and 0.5 SD with the root-mean-square distance and fixed-amplitude steps, the accepted population size and the synergy part, redundancy part and net O-information of the selected population against the mean and 95% range of 100 same-size unselected samples (and the synergy part of 100 same-size feature-complete samples); the Hay model also on 8,192 random draws. Reads `results/analyses/review_v18/two_stage.parquet` and `results/analyses/review_v22/hay_two_stage.parquet`. | `results/analyses/manuscript/table3_per_implementation.csv` |
| `make_figure_s1.py` | The figure of the direct perturbation test. That test is no longer part of the Supplementary Material (it could not answer its question); the script is kept for the record. | `results/figures/manuscript/figS1.png` |

The figures stored under `results/figures/manuscript/` are the ones in the
submitted manuscript; rerunning `run_figures.sh` reproduces them pixel for pixel
(checked on macOS with matplotlib and the Arial font installed).

The review analyses themselves (`results/analyses/review_v5/`) were produced by
`tools/run_review_analyses.py` and `tools/run_review_knn_ranked.py`. Those two
scripts do read the per-simulation feature tables, so rerunning them requires
regenerating those tables first with `tools/run_campaign.py --experiment screening`
(the parameter samples are fixed by the recorded seeds).

The analyses added after the review of manuscript V17 (`results/analyses/review_v17/`)
were produced by three scripts. `tools/run_review_random_draws.py` repeats the
selection on parameter sets drawn with independent pseudo-random draws instead of
the Sobol design (campaign `screening_random`: `tools/run_campaign.py --experiment
screening_random`) and compares the nearest-neighbour estimate with same-size
unselected samples, shuffled copies and same-rank-correlation synthetic populations
(Fig. 2C-D, Table S4). `tools/run_review_selection_bands.py` adds the estimator test
systems that resemble selection on positive parameters (Fig. 2A-B).
`tools/run_review_regression_variants.py` repeats the attribution analysis without
the geometry measures computed from the covariance matrix (Fig. 5C-D, Table S1).

The analyses added after the review of manuscript V18 (`results/analyses/review_v18/`)
are produced by `tools/run_review_v18.py`, which reads the per-simulation feature
tables of the `screening` and `screening_random` campaigns and the canonical targets
in `results/manifests/`. Outputs: `two_stage.parquet` and `pools.parquet` (selected
populations against same-size unselected and feature-complete samples; Fig. 3A-C,
Fig. 4C), `dropout_decomposed.parquet` (Fig. 3F, Table S3), `rate_dedup.parquet` and
`eligibility.parquet` (firing rate removed from the distance; spike-count eligibility),
`pairs_calibrated.parquet`, `gauss_common_scale.parquet`, `ksg_pairs.parquet`
(Fig. 2D, Table S4), `sobol_box.parquet` (spacing control), `clustered_interior.parquet`
(clustered populations without rows at the prior bounds), `attribution.json`
(Fig. 5C-D, Table S1) and `run_record.json` (features, transforms, seeds). The
rheobase-normalised selections of the Khaliq model are excluded everywhere: its
canonical rheobase is 0, so that family is undefined for it (`tools/run_analysis.py`,
`family_defined`).

The analyses added after the review of manuscript V20 (`results/analyses/review_v19/`)
are produced by `tools/run_review_v19.py`, which also reads the per-simulation feature
tables of the `screening` campaign:

```bash
python tools/run_review_v19.py
```

Outputs: `estimator_agreement.json` (agreement between the Gaussian and
Gaussian-copula triplet estimates, under three counting conventions; Methods and
Section S1), `lolo_two_baselines.json` (leave-one-source-model-out prediction scored
against the training-model mean and against the left-out model's own mean; Fig. 5D,
Section S1) and `top_triplet_thinning.parquet` (overlap of the most synergistic
triplets between tolerances, against random thinning of the 3-SD population;
Section S2).

The analyses for the review of manuscript V22 (`results/analyses/review_v22/`) are
produced by `tools/run_review_v22.py` in two parts. The first reads stored tables and the
per-simulation feature tables of the `screening` and `screening_random` campaigns; the
second needs the Hay model in `screening_random` (8,192 random draws, added to
`configs/experiments/campaigns.json`):

```bash
python tools/run_campaign.py --experiment screening_random --models bbp_l5pc --workers 13
python tools/run_review_v22.py stored
python tools/run_review_v22.py hay --workers 3
```

Outputs of the first part: `effect_with_chance_ranges.parquet` and
`table3_conditions.csv` (synergy part, redundancy part and net O-information of every
selected population with the chance range of same-size unselected and feature-complete
samples), `ratios.csv` and `ratio_summary.csv` (per-model ratios and their median, range
and geometric mean), `geometry_excluded.parquet` (summaries with capacitance, diameter and
axial resistance left out; Table S5), `top_triplets.csv` and `triplet_shares.csv` (most
synergistic triplets and the share of synergy carried by those parameters; Section S2)
and `rank_one.json` / `rank_one_check.csv` (numerical check of the exact result for a
single selection constraint; Section S1). Outputs of the second part: `hay_two_stage.parquet`
and `hay_pools.parquet` (Table 3), `hay_random_null.parquet`, `hay_random_knn.parquet`,
`hay_random_pairs.parquet`, `hay_pairs_calibrated.parquet`, `hay_gauss_common_scale.parquet`
and `hay_ksg_pairs.parquet` (Fig. 2C-D, Table S4), with the same code, seeds and reference
sizes as the other models.

