# Manuscript figures

Scripts that draw the figures of the manuscript *Selection for a shared behavior
creates parameter dependence read as synergy in populations of neuron models*
(Degani-Katzav and Korngreen) and compute the summary tables quoted in it.

They read only analysis tables stored in this repository (`results/analyses/`,
`results/analyses/review_v5/` and `results/analyses/review_v17/`). No simulation is run, and the per-simulation
feature tables (`results/features/`, not distributed) are not needed.

```bash
bash manuscript/run_figures.sh
```

| Script | What it does | Output |
|---|---|---|
| `variance_partition_bootstrap.py` | Splits the variation in the synergy area between the sampling design, the population geometry and the source lineage (the original published model a code version descends from), with a bootstrap over the five lineages used for the main decision. | `results/analyses/manuscript/variance_partition.json` |
| `summarise_review_analyses.py` | Summarises the analyses added after review: each accepted population compared with same-size unselected and shuffled populations, each electrical feature left out of the acceptance rule in turn, the nearest-neighbour estimate, and the most synergistic triplets. | `results/analyses/manuscript/review_numbers.json`, `review_numbers_null_table.csv`, `review_numbers_dropout_table.csv` |
| `make_figures.py` | Figures 1 to 5 of the main text. | `results/figures/manuscript/fig1.png` … `fig5.png` |
| `make_figure_s1.py` | Supplementary Figure S1 (the perturbation test). | `results/figures/manuscript/figS1.png` |

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
