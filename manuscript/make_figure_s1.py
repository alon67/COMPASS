"""Supplementary Figure S1, in the same style as the main figures.

Individual implementations in grey (filled: reproduced at half the time step; open: not
reproduced), means as red bars, plain panel letters, axes offset, units in [ ].
Usage: make_figure_s1.py OUT.png
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
AN = str(Path(__file__).resolve().parents[1] / 'results' / 'analyses') + '/'
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False, 'savefig.dpi': 300})
RED, BLUE, GREY, LGREY = '#d62728', '#1f4e9c', '#6e6e6e', '#c8c8c8'
SHORT = {'hh_point': 'Hodgkin–Huxley, one compartment', 'hh_cable': 'Hodgkin–Huxley, soma and dendrite',
         'pospischil_rs': 'Pospischil, regular-spiking', 'pospischil_fs': 'Pospischil, fast-spiking',
         'khaliq_purkinje': 'Khaliq, Purkinje', 'migliore_ca1': 'Migliore, CA1', 'bbp_l5pc': 'Hay, layer 5',
         'mainen_l5pc': 'Mainen, layer 5 (not converged)'}


def lab(ax, s, x=-0.3):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=12, va='bottom')


def offset(ax, which=('left', 'bottom')):
    for w in which:
        ax.spines[w].set_position(('outward', 4))


rep = json.load(open(AN + 'perturbation_inference.json'))['numerical_repeatability']['per_model']
REP = {m: bool(v['numerically_repeatable']) for m, v in rep.items()}
p = pd.read_parquet(AN + 'perturbations.parquet'); p = p[p.primary_feature].copy(); p['absm'] = p.mean_nonadditivity_sd.abs()
pm = p.groupby(['kind', 'model_id']).agg(a=('absm', 'mean'), s=('functionally_supported', 'mean')).reset_index()
KINDS = [('top_synergy', 'Most\nsynergistic'), ('control_middle', 'Middle\nranked'), ('control_random', 'Random')]
fig = plt.figure(figsize=(7.2, 2.8))
axs = [fig.add_axes([0.08, 0.3, 0.19, 0.6]), fig.add_axes([0.37, 0.3, 0.19, 0.6]), fig.add_axes([0.82, 0.3, 0.16, 0.6])]
for k, (col, yl, sc) in enumerate([('a', 'Mean |non-additivity|\n[SD of firing rate]', 1),
                                   ('s', 'Triplets meeting the rule\nfixed in advance [%]', 100)]):
    ax = axs[k]; lab(ax, 'AB'[k], -0.45)
    for i, (kind, _) in enumerate(KINDS):
        v = pm[pm.kind == kind].set_index('model_id')[col] * sc
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        for (m, val), j in zip(v.items(), jit):
            if m == 'mainen_l5pc':
                ax.scatter(i + j, val, s=18, zorder=3, marker='x', color=LGREY, lw=1.0)
            else:
                ax.scatter(i + j, val, s=14, zorder=3, facecolor=GREY if REP[m] else 'white', edgecolor=GREY, lw=0.8)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=RED, lw=1.8, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in KINDS], fontsize=6.5); ax.set_ylabel(yl)
    ax.set_xlim(-0.6, 2.6); offset(ax, ('left',)); ax.tick_params(axis='x', length=0)
    if col == 'a':
        ax.axhline(0.25, color='k', lw=0.5, ls=(0, (2, 2)))
ax = axs[2]; lab(ax, 'C', -1.6)
med = p[p.kind == 'top_synergy'].groupby('model_id').absm.median().sort_values(ascending=False)
yy = np.arange(len(med))
for y, (m, v) in zip(yy, med.items()):
    c = LGREY if m == 'mainen_l5pc' else BLUE
    ax.barh(y, v, 0.62, color=c if REP[m] else 'white', edgecolor=c, lw=0.9)
ax.set_yticks(yy); ax.set_yticklabels([SHORT[m] for m in med.index], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('Median |non-additivity|,\nmost synergistic triplets\n[SD of firing rate]')
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
h = [Line2D([], [], ls='', marker='o', ms=4, mfc=GREY, mec=GREY, label='Reproduced at half the time step'),
     Line2D([], [], ls='', marker='o', ms=4, mfc='white', mec=GREY, label='Not reproduced'),
     Line2D([], [], ls='', marker='x', ms=4, color=LGREY, label='Mainen (not converged)'),
     Line2D([], [], color=RED, lw=1.8, label='Mean of the eight implementations')]
fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12))
fig.savefig(sys.argv[1], bbox_inches='tight'); print('ok')
