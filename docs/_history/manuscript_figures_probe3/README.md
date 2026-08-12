# Manuscript figure set — WITHDRAWN, probe-3

`COMPASS_figures.ipynb` builds six composite manuscript figures and
`figure_legends.md` describes them. **Both are withdrawn from the release.**
They are kept because the record of what was drawn, and why it was withdrawn,
is part of the evidence — not because they support any current number.

## Why

1. **The legend for Figure 6 asserts a claim the current data retract.** It
   reads *"statistically synergistic tuples show more measured non-additivity
   than matched and random controls"*. On the probe-4 corpus the matched
   contrast is **0.99×**, with a lineage-clustered 95% CI of [−0.023, +0.037]:
   on the three numerically repeatable lineages the top-synergy tuples are
   **indistinguishable** from their matched controls. Shipping that legend
   would state the opposite of the release's own finding.

2. **Every quantity in that legend is probe-3.** It describes 7 models and 42
   tuples per class, with "`mainen_l5pc` was not regenerated"; the probe-4
   battery covers 8 models and 48 tuples and includes `mainen_l5pc`. Mean |η|
   was 0.306 / 0.103 / 0.065 (top / matched / random) and is now
   0.189 / 0.115 / 0.068. Functional support was 31.0% / 7.1% / 2.4% and is now
   20.8% / 8.3% / 2.1%.

3. **The notebook cannot be re-executed against the current tables.** It reads
   `lomo.held_out_model` and `lomo.model_id`. Neither column exists: the schema
   became `held_out_group` / `group_col` when leave-one-lineage-out was
   corrected to hold out lineages rather than model ids. Its last successful
   execution was **2026-08-08**, before the probe-4 corpus existed. It also
   hard-codes the seven-model battery.

4. **It silently overwrote the shipped figures.** It saved to
   `results/figures/Figure_{n}` — the same names the results notebook used — so
   running it replaced Figure_1 through Figure_4 with manuscript composites
   before failing on the schema error. That is how the collision was found. The
   results notebook now writes `panel_{n}`, leaving `Figure_{n}` free for a
   repaired manuscript set.

## What the release ships instead

`notebooks/COMPASS_results.ipynb`, executed against the probe-4 artifacts, and
the thirteen `results/figures/panel_*.png` / `.pdf` pairs it saves. Those are
exploratory panels, not manuscript figures, and they carry no legends.

## To revive this set

Fix the two `lomo` column references and the seven-model assumption,
re-execute, and rewrite all six legends from the probe-4 values — Figure 6's
conclusion first, since its direction has changed. Until that is done, this
directory is history and supports nothing.
