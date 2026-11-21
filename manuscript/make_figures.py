"""Main-text Figures 1-5 of the manuscript (labels in American spelling).

Style taken from Korngreen (2026, PLoS Comput Biol; 2026, Commun Biol) and Almog and
Korngreen (2014, 2016): axes offset from the data with outward ticks, no top or right
frame, plain (not bold) panel letters, axis labels with units in square brackets,
individual cases in grey with the population mean in red, unframed legends, and a
boxed top-to-bottom pipeline with a bold coloured heading and a grey sub-line.

Reads stored COMPASS analysis tables only (results/analyses/, results/analyses/review_v5/ and
results/analyses/review_v17/).
Usage: make_figures.py OUTDIR VP_JSON NULL_CSV DROPOUT_CSV  (see run_figures.sh)
"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from pathlib import Path
AN = str(Path(__file__).resolve().parents[1] / 'results' / 'analyses') + '/'
OUT = sys.argv[1]; VP = sys.argv[2]
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'legend.frameon': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic'})

RED, BLUE, LBLUE, GREY, LGREY = '#d62728', '#1f4e9c', '#8fb3e0', '#8c8c8c', '#c8c8c8'

MODELS = ['hh_point', 'hh_cable', 'pospischil_rs', 'pospischil_fs', 'khaliq_purkinje',
          'migliore_ca1', 'bbp_l5pc', 'mainen_l5pc']
PRIM = MODELS[:-1]
SHORT = {'hh_point': 'Hodgkin–Huxley, one compartment', 'hh_cable': 'Hodgkin–Huxley, soma and dendrite',
         'pospischil_rs': 'Pospischil, regular-spiking', 'pospischil_fs': 'Pospischil, fast-spiking',
         'khaliq_purkinje': 'Khaliq, Purkinje', 'migliore_ca1': 'Migliore, CA1',
         'bbp_l5pc': 'Hay, layer 5', 'mainen_l5pc': 'Mainen, layer 5 (not converged)'}


def lab(ax, s, x=-0.2, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight='normal', va='bottom', ha='left')


def offset(ax, which=('left', 'bottom'), d=4):
    for w in which:
        ax.spines[w].set_position(('outward', d))


e = pd.read_parquet(AN + 'endpoints_annotated_screening_n4proj.parquet')
g = e[(e.estimator == 'gaussian') & (e.protocol_family == 'absolute')]
TOLS = [('E1', 'none'), ('E3', 'weighted_euclidean'), ('E3b', 'weighted_euclidean'),
        ('E4', 'weighted_euclidean'), ('E4b', 'weighted_euclidean')]
RV = AN + 'review_v5/'
NULL = pd.read_csv(sys.argv[3]); KNN = pd.read_parquet(RV + 'knn_check.parquet'); DROP = pd.read_csv(sys.argv[4])
kn = json.load(open(AN + 'key_numbers.json')); bm = json.load(open(AN + 'benchmarks.json'))

# ---------------- Figure 1: design as a pipeline + toy illustration ----------------
fig = plt.figure(figsize=(7.0, 3.9))
ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis('off')
lab(ax, 'A', 0.0, 0.955)
steps = [
    ('Independent prior', 'Parameters drawn independently, ¼ to 4 times the published value', '#1f4e9c', '#eef3fb'),
    ('Simulation', 'Eight published models in NEURON; 13 electrical features per parameter set', '#1f4e9c', '#eef3fb'),
    ('Selection gate', 'Keep sets within 3, 2, 1 or 0.5 SD of the published behavior', '#c46a00', '#fdf1e2'),
    ('Accepted population', 'An ensemble of models that share the target behavior', '#1f4e9c', '#eef3fb'),
    ('Higher-order dependence', 'O-information of every group of 3–5 parameters, compared\nwith unselected samples of the same size and shuffled copies', '#2a7a3b', '#edf6ee'),
]
h, gap, top = 1.55, 0.45, 9.9
for i, (t, s, ec, fc) in enumerate(steps):
    y = top - (i + 1) * h - i * gap
    ax.add_patch(FancyBboxPatch((0.6, y), 8.8, h, boxstyle='round,pad=0.02,rounding_size=0.18', fc=fc, ec=ec, lw=0.9))
    ax.text(0.95, y + h * 0.66, t, fontsize=8.5, fontweight='bold', color=ec, va='center')
    ax.text(0.95, y + h * (0.24 if '\n' in s else 0.28), s, fontsize=6.6, color='#555555', va='center')
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((5.0, y - 0.03), (5.0, y - gap + 0.03), arrowstyle='-|>', mutation_scale=8, lw=0.9, color='#444'))
ax = fig.add_axes([0.66, 0.2, 0.31, 0.64]); lab(ax, 'B', -0.3, 1.05)
rng = np.random.default_rng(1); x = rng.uniform(0, 1, (3000, 2)); keep = np.abs(x.sum(1) - 1) < 0.15
ax.scatter(x[~keep, 0], x[~keep, 1], s=1.2, c=LGREY, lw=0, rasterized=True, label='Rejected')
ax.scatter(x[keep, 0], x[keep, 1], s=1.2, c=RED, lw=0, rasterized=True, label='Accepted')
r_all = np.corrcoef(x.T)[0, 1]; r_acc = np.corrcoef(x[keep].T)[0, 1]
ax.set_xlabel('Parameter 1 [a.u.]'); ax.set_ylabel('Parameter 2 [a.u.]'); ax.set_aspect('equal')
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); offset(ax)
ax.text(0.12, 1.07, f'All: r = {abs(r_all):.2f}    Accepted: r = −{abs(r_acc):.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=6.8)
ax.legend(loc='upper right', markerscale=5, handletextpad=0.2, borderpad=0.1, bbox_to_anchor=(1.05, 1.0))
fig.savefig(OUT + '/fig1.png'); plt.close(fig)

# ---------------- Figure 2: estimator check ----------------
# A-B: the seven four-variable systems of the estimator benchmark plus four systems that
# look like selection on positive parameters (review of V17; tools/run_review_selection_bands.py).
# C-D: nearest-neighbour check on populations drawn with independent random draws
# (campaign screening_random; tools/run_review_random_draws.py).
ev = pd.DataFrame(bm['estimator_validation'])
SB = pd.read_parquet(AN + 'review_v17/selection_bands.parquet')
SB = SB[SB.estimator.isin(['gaussian', 'copula', 'knn'])]
SYS = [('independent', 'Independent (0)'), ('linear_correlated', 'Linear correlation (+)'),
       ('duplicated', 'Duplicated variable (+)'), ('mixture_pooled', 'Pooled mixture (+)'),
       ('sum_constrained', 'Sum constraint (−)'), ('xor_like', 'Product of two, zero-mean (−)'),
       ('multiplicative', 'Product of three, zero-mean (−)'),
       ('product of three positive variables, log scale', 'Product of three, positive (−)'),
       ('band on the sum of log-parameters', 'Band on sum of logs (−)'),
       ('band on the sum of squared log-parameters', 'Band on sum of squared logs (−)'),
       ('band on x1 + |x2 - x3|', r'Band on $x_1 + |x_2 - x_3|$ (−)')]
NEW = {s_ for s_, _ in SYS[7:]}


def est_value(sys_, es, n):
    if sys_ in NEW:
        return SB[(SB.system == sys_) & (SB.estimator == es) & (SB.n == n)].o_information.mean()
    return ev[(ev.system == sys_) & (ev.estimator == es) & (ev.n == n)].o_information.iloc[0]


def est_correct(sys_, es, n):
    if sys_ in NEW:
        return SB[(SB.system == sys_) & (SB.estimator == es) & (SB.n == n)].correct.mean()
    return float(ev[(ev.system == sys_) & (ev.estimator == es) & (ev.n == n)].correct.iloc[0])


EST = [('gaussian', 'Gaussian', BLUE), ('copula', 'Gaussian copula', LBLUE), ('knn', 'Nearest-neighbor', RED)]
fig = plt.figure(figsize=(7.2, 6.0)); gs2 = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.95, width_ratios=[1.25, 1.0])
ax = fig.add_subplot(gs2[0, 0]); lab(ax, 'A', -0.62); nmax = ev.n.max(); y = np.arange(len(SYS))
for j, (es, nm, col) in enumerate(EST):
    ax.barh(y + (j - 1) * 0.26, [est_value(s_, es, nmax) for s_, _ in SYS], height=0.25, color=col, label=nm)
ax.axvline(0, color='k', lw=0.6); ax.axhline(6.5, color=LGREY, lw=0.6, ls=(0, (2, 2)))
ax.set_yticks(y); ax.set_yticklabels([t for _, t in SYS], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('O-information [nats]'); offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
hA, lA = ax.get_legend_handles_labels(); fig.legend(hA, lA, loc='upper center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 0.97))
ax = fig.add_subplot(gs2[0, 1]); lab(ax, 'B', -0.32)
ns = sorted(ev.n.unique())
for es, nm, col in EST:
    ls, mk, ms_ = ('-', 's', 6) if es == 'gaussian' else (('--', 'o', 3.5) if es == 'copula' else ('-', 'o', 3.5))
    ax.plot(ns, [np.mean([est_correct(s_, es, n) for s_, _ in SYS]) * 100 for n in ns], ls=ls, marker=mk, ms=ms_,
            mfc='none' if es == 'gaussian' else col, color=col, label=nm)
ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels([f'{n/1000:g}k' if n >= 1000 else str(n) for n in ns]); ax.minorticks_off(); ax.set_ylim(0, 105)
ax.set_xlabel('Sample size [points]'); ax.set_ylabel('Systems classified correctly [%]'); offset(ax)
KM = ['hh_point', 'hh_cable', 'pospischil_rs', 'pospischil_fs', 'khaliq_purkinje']
KS = {'hh_point': 'HH, one comp.', 'hh_cable': 'HH, soma + dendrite', 'pospischil_rs': 'Pospischil, RS',
      'pospischil_fs': 'Pospischil, FS', 'khaliq_purkinje': 'Khaliq, Purkinje'}
RK = pd.read_parquet(AN + 'review_v17/random_knn.parquet')
RK = RK[(RK.population == 'selected') & (RK.metric == 'weighted_euclidean') & (RK.tolerance == 'permissive')]
RP = pd.read_parquet(AN + 'review_v17/random_pairs.parquet')
KR = pd.read_parquet(RV + 'knn_ranked_check.parquet')
REFS = [('unselected_same_size', 'Unselected, same size', RED, 'o'), ('shuffled', 'Shuffled copies', GREY, 's'),
        ('copula_surrogate', 'Synthetic, same rank correlations', BLUE, 'D')]
ax = fig.add_subplot(gs2[1, 0]); lab(ax, 'C', -0.62); yy = np.arange(len(KM))
for i, m in enumerate(KM):
    g_ = RK[RK.model_id == m]; sel = g_[g_.kind == 'selected'].omega_knn.mean()
    for j, (kind, nm, col, mk) in enumerate(REFS):
        d = sel - g_[g_.kind == kind].groupby('draw').omega_knn.mean()
        yj = i + (j - 1) * 0.2
        ax.plot([d.min(), d.max()], [yj, yj], color=col, lw=0.8)
        ax.plot(d.mean(), yj, marker=mk, ms=3.5, color=col, lw=0, label=nm if i == 0 else None)
    k_ = KR[KR.model_id == m]
    old = k_[k_.kind == 'selected'].omega_knn_ranked.mean() - k_[k_.kind == 'unselected_same_size'].groupby('draw').omega_knn_ranked.mean().mean()
    ax.plot(old, i + 0.38, marker='o', ms=4, mfc='none', mec='k', lw=0, label='Evenly spaced design (earlier check)' if i == 0 else None)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([KS[m] for m in KM], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('Selected minus reference,\nmean over triplets [nats]')
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax.legend(loc='upper center', fontsize=6.2, bbox_to_anchor=(0.35, -0.36), ncol=2, handletextpad=0.3, columnspacing=1.0)
ax = fig.add_subplot(gs2[1, 1]); lab(ax, 'D', -0.75)
for i, m in enumerate(KM):
    for j, (pop, met, tol, col, nm) in enumerate([('selected', 'weighted_euclidean', 'permissive', RED, 'Selected (3 SD)'),
                                                   ('unselected', 'none', 'none', GREY, 'Unselected')]):
        g_ = RP[(RP.model_id == m) & (RP.population == pop) & (RP.metric == met) & (RP.tolerance == tol)]
        d = g_[g_.kind == 'selected'].mi_knn.mean() - g_[g_.kind == 'copula_surrogate'].groupby('draw').mi_knn.mean()
        yj = i + (j - 0.5) * 0.3
        ax.plot([d.min(), d.max()], [yj, yj], color=col, lw=0.8)
        ax.plot(d.mean(), yj, marker='o', ms=3.5, color=col, lw=0, label=nm if i == 0 else None)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([KS[m] for m in KM], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('Pairwise information minus synthetic\nwith same rank correlations [nats]')
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax.legend(loc='lower right', fontsize=6.0, handletextpad=0.3)
fig.savefig(OUT + '/fig2.png', bbox_inches='tight', pad_inches=0.12); plt.close(fig)

# ---------------- Figure 3: selection compared with unselected samples of the same size ----------------
fig = plt.figure(figsize=(7.2, 5.0)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.62)
gb = fig.add_gridspec(2, 3, hspace=0.62, wspace=1.15, width_ratios=[1, 0.95, 0.5], left=0.125, right=0.93)
TL = ['permissive', 'moderate', 'strict', 'very_strict']; xs = np.arange(4); xt = ['3', '2', '1', '0.5']
NU = NULL[(NULL.metric == 'weighted_euclidean') & (NULL.null == 'unselected_same_size')]
def series(mid, col):
    out = []
    for t in TL:
        r = NU[(NU.model_id == mid) & (NU.tolerance == t)]
        out.append(r[col].iloc[0] if len(r) else np.nan)
    return np.array(out, float)
PAN = [('mean_abs_pearson', 'Mean |r| between\nparameter pairs', 1, False),
       ('frac_synergy_o3', 'Synergy-dominated\ntriplets [%]', 100, False),
       ('synergy_area', 'Synergy part of net\nO-information [nats]', 1, True)]
for k, (col, yl, sc, logy) in enumerate(PAN):
    ax = fig.add_subplot(gs[0, k]); lab(ax, 'ABC'[k], -0.45)
    S = {m: series(m, col + '_sel') * sc for m in PRIM}
    N_ = {m: series(m, col + '_null_mean') * sc for m in PRIM}
    for m in PRIM:
        ax.plot(xs, S[m], color='k' if m == 'bbp_l5pc' else GREY, lw=0.8 if m == 'bbp_l5pc' else 0.7, zorder=2)
    mn = np.nanmean([N_[m] for m in PRIM], axis=0)
    ax.plot(xs, mn, color=BLUE, lw=1.6, ls=(0, (3, 1.5)), marker='s', ms=3, zorder=3)
    ax.plot(xs, np.nanmean([S[m] for m in PRIM], axis=0), color=RED, lw=1.8, marker='o', ms=3.5, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(xt); ax.set_xlabel('Tolerance [SD]'); ax.set_ylabel(yl); offset(ax)
    if col == 'frac_synergy_o3':
        ax.axhline(50, color='k', lw=0.5, ls=(0, (2, 2))); ax.set_ylim(0, 100)
    if logy:
        ax.set_yscale('log')
# D: acceptance
ax = fig.add_subplot(gb[1, 0]); lab(ax, 'D', -0.45)
for mid in MODELS:
    ys = [g[(g.model_id == mid) & (g.regime == r) & (g.metric == 'weighted_euclidean')].acceptance_rate for r in ('E3', 'E3b', 'E4', 'E4b')]
    ys = [v.iloc[0] * 100 if len(v) else np.nan for v in ys]
    if mid == 'mainen_l5pc':
        ax.plot(xs, ys, color=LGREY, lw=0.8, ls=(0, (1.5, 1.5)))
    else:
        ax.plot(xs, ys, color='k' if mid == 'bbp_l5pc' else GREY, lw=0.8 if mid == 'bbp_l5pc' else 0.7)
ax.set_yscale('log'); ax.set_ylim(0.5, 150); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(['1', '10', '100']); ax.minorticks_off()
ax.set_xticks(xs); ax.set_xticklabels(xt); ax.set_xlabel('Tolerance [SD]'); ax.set_ylabel('Parameter sets\naccepted [%]'); offset(ax)
# E: non-neuronal systems
ax = fig.add_subplot(gb[1, 1]); lab(ax, 'E', -0.5)
xs5 = np.arange(5)
for c, nm, colr, mk in [(bm['comparators'][0], 'Repressilator', RED, 's'), (bm['comparators'][1], 'Lotka–Volterra', BLUE, 'D')]:
    ax.plot(xs5, [en['o_information_full'] for en in c['ensembles']], marker=mk, ms=3, color=colr, label=nm)
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(xs5); ax.set_xticklabels(['None', '3', '2', '1', '0.5']); ax.set_xlabel('Tolerance [SD]')
ax.set_ylabel('O-information of all six\nparameters [nats]'); ax.legend(loc='lower left', fontsize=6.6); offset(ax)
# F: which features do the selecting
ax = fig.add_subplot(gb[1, 2]); lab(ax, 'F', -1.75)
FN = {'hyperpol_abs.membrane_tau_ms': 'Membrane time constant', 'step_abs_hi.adaptation_index': 'Adaptation index',
      'step_abs_hi.ap_amplitude_mV': 'Spike amplitude', 'hyperpol_abs.input_resistance_Mohm': 'Input resistance',
      'rheobase_nA': 'Rheobase', 'step_abs_hi.first_spike_latency_ms': 'First-spike latency',
      'step_abs_hi.mean_firing_rate_Hz': 'Mean firing rate'}
d_ = DROP[(DROP.metric == 'weighted_euclidean') & (DROP.tolerance == 'permissive')]
fr = d_.groupby('dropped_feature').frac_changed.mean() * 100
others = fr[[f for f in fr.index if f not in FN]]
items = [(FN[f], fr[f]) for f in FN] + [('Other six features (each)', others.max())]
items = sorted(items[:-1], key=lambda t: -t[1]) + items[-1:]
yy = np.arange(len(items))
ax.barh(yy, [v for _, v in items], color=[BLUE] * (len(items) - 1) + [LGREY], height=0.6)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in items], fontsize=6.4); ax.invert_yaxis()
ax.set_xlabel('Accepted sets\nchanged [%]'); offset(ax, ('bottom',))
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
handles = [Line2D([], [], color=GREY, lw=0.7, label='Selected, individual implementations'),
           Line2D([], [], color='k', lw=0.8, label='Selected, Hay layer 5'),
           Line2D([], [], color=RED, lw=1.8, marker='o', ms=3.5, label='Selected, mean of seven'),
           Line2D([], [], color=BLUE, lw=1.6, ls=(0, (3, 1.5)), marker='s', ms=3, label='Unselected, same size, mean of seven'),
           Line2D([], [], color=LGREY, lw=0.8, ls=(0, (1.5, 1.5)), label='Mainen, not converged (D only)')]
fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, 1.04))
fig.savefig(OUT + '/fig3.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4: controls and reliability ----------------
fig = plt.figure(figsize=(7.2, 2.9))
axs = [fig.add_axes([0.085, 0.24, 0.17, 0.62]), fig.add_axes([0.37, 0.24, 0.19, 0.62]), fig.add_axes([0.80, 0.24, 0.14, 0.62])]
ax = axs[0]; lab(ax, 'A', -0.42)
oc = pd.read_parquet(AN + 'order_curves_screening_n4proj.parquet')
oc = oc[(oc.estimator == 'gaussian') & (oc.model_id.isin(PRIM))]
pm = oc[oc.regime.isin(['E3', 'E8'])].groupby(['regime', 'model_id'])[['frac_synergy', 'frac_synergy_sig']].mean() * 100
pos = {('E3', 'frac_synergy'): 0, ('E3', 'frac_synergy_sig'): 1, ('E8', 'frac_synergy'): 2.4, ('E8', 'frac_synergy_sig'): 3.4}
for reg in ('E3', 'E8'):
    t_ = pm.loc[reg]
    for m in PRIM:
        ax.plot([pos[(reg, 'frac_synergy')], pos[(reg, 'frac_synergy_sig')]], [t_.loc[m, 'frac_synergy'], t_.loc[m, 'frac_synergy_sig']],
                color='k' if m == 'bbp_l5pc' else GREY, lw=0.6, marker='o', ms=2.5, zorder=2)
    for c in ('frac_synergy', 'frac_synergy_sig'):
        ax.plot([pos[(reg, c)] - 0.3, pos[(reg, c)] + 0.3], [t_[c].mean()] * 2, color=RED, lw=1.8, zorder=3)
ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(['Sign', 'Beyond\ncontrol', 'Sign', 'Beyond\ncontrol'], fontsize=6.3)
ax.text(0.5, -30, 'Selected (3 SD)', ha='center', fontsize=6.8); ax.text(2.9, -30, 'Shuffled', ha='center', fontsize=6.8)
ax.set_ylabel('Groups of 3–5 parameters\ncalled synergistic [%]'); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 3.9)
offset(ax, ('left',)); ax.tick_params(axis='x', length=0)
ax = axs[1]; lab(ax, 'B', -0.34)
rel = json.load(open(AN + 'estimator_reliability.json'))['by_n']; ns = sorted(int(k) for k in rel)
ax.axvspan(75, 111, color='#ececec', lw=0); ax.text(115, 59, "Published\nbiological\ndatasets", ha='left', fontsize=6.3, va='top', color='#555')
ax.plot(ns, [rel[str(n)]['p_sign_flip_max'] * 100 for n in ns], marker='o', ms=3, color=GREY, label='Least favorable population')
ax.plot(ns, [rel[str(n)]['p_sign_flip_mean'] * 100 for n in ns], marker='o', ms=3.5, lw=1.8, color=RED, label='Mean of five populations')
ax.axhline(50, color='k', lw=0.5, ls=(0, (2, 2))); ax.set_ylim(0, 60); ax.set_xlabel('Population size [parameter sets]')
ax.set_ylabel('Subsamples with opposite\nsign of net O-information [%]'); ax.legend(loc='lower left', fontsize=6.5); offset(ax)
ax = axs[2]; lab(ax, 'C', -1.25)
cs = pd.read_parquet(AN + 'conditioning_shift.parquet'); cs = cs[cs.estimator == 'gaussian'].groupby('model_id')['d_net_o_information'].mean() * 1000
order = [m for m in MODELS if m in cs.index]; yy = np.arange(len(order))
ax.barh(yy, [cs[m] for m in order], color=[LGREY if m == 'mainen_l5pc' else BLUE for m in order], height=0.62)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(yy)
ax.set_yticklabels([SHORT[m].replace(' (not converged)', '*') for m in order], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('Change in net\nO-information after\nselection [millinats]'); offset(ax, ('bottom',)); ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
fig.savefig(OUT + '/fig4.png', bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 5: construction and attribution ----------------
fig = plt.figure(figsize=(7.2, 5.1)); gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.75)
ax = fig.add_subplot(gs[0, 0]); lab(ax, 'A', -0.62)
REG = [('NULL', 'max_z', 'Unselected, same size\nas selected'), ('E2', 'none', 'Correlated prior'), ('E6', 'none', 'Clustered population'),
       ('E3', 'max_z', 'Selected (3 SD)'), ('E3m', 'max_z', 'Random subset of selected'), ('E7', 'max_z', 'High-covariance subset'),
       ('E8', 'none', 'Shuffled control'), ('E9', 'none', 'Synthetic, same covariance')]
for i, (reg, met, nm) in enumerate(REG):
    if reg == 'NULL':
        nn = NULL[(NULL.metric == 'max_z') & (NULL.tolerance == 'permissive') & (NULL.null == 'unselected_same_size')]
        v = nn.set_index('model_id')['frac_synergy_o3_null_mean'] * 100
    else:
        v = g[(g.regime == reg) & (g.metric == met) & (g.model_id.isin(PRIM))].set_index('model_id')['frac_synergy_o3'] * 100
    ax.scatter(v.values, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=9, c=GREY, zorder=3, lw=0)
    ax.plot([v.mean()] * 2, [i - 0.32, i + 0.32], color=RED, lw=1.8, zorder=4)
ax.set_yticks(range(len(REG))); ax.set_yticklabels([r[2] for r in REG], fontsize=6.8); ax.invert_yaxis()
ax.axvline(50, color='k', lw=0.5, ls=(0, (2, 2))); ax.set_xlabel('Synergy-dominated triplets [%]'); ax.set_xlim(-3, 100)
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax = fig.add_subplot(gs[0, 1]); lab(ax, 'B', -0.75)
cv = pd.read_parquet(AN + 'covariance_shift.parquet')
cv = cv[(cv.estimator == 'gaussian') & (cv.protocol_family == 'absolute') & (cv.contrast == 'E7_vs_E3m')].groupby('model_id')['d_net_o_information'].mean() * 1000
order = [m for m in PRIM if m in cv.index]; yy = np.arange(len(order))
ax.barh(yy, [cv[m] for m in order], color=BLUE, height=0.62); ax.axvline(0, color='k', lw=0.6)
ax.set_yticks(yy); ax.set_yticklabels([SHORT[m] for m in order], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel('Net O-information, high-covariance\nminus random subset [millinats]')
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax = fig.add_subplot(gs[1, 0]); lab(ax, 'C', -0.62)
# C-D without the covariance-based geometry predictors (review of V17;
# tools/run_review_regression_variants.py): geometry = the two intrinsic-dimension
# estimates only; prediction from sampling design alone.
RVAR = json.load(open(AN + 'review_v17/regression_variants.json'))['variants']
vp = RVAR['synergy_area:geometry_without_covariance']
bl = [('sampling', 'Sampling design'), ('geometry', 'Intrinsic dimension'), ('lineage', 'Source model')]
yy = np.arange(3); u = [vp['unique'][b] for b, _ in bl]
lo = [vp['unique_ci'][b][0] for b, _ in bl]; hi = [vp['unique_ci'][b][1] for b, _ in bl]
ax.barh(yy, u, color=[BLUE, LBLUE, RED], height=0.6)
ax.errorbar(u, yy, xerr=[np.clip(np.array(u) - lo, 0, None), np.clip(np.array(hi) - u, 0, None)], fmt='none', ecolor='k', lw=0.8, capsize=2)
ax.set_yticks(yy); ax.set_yticklabels([n for _, n in bl], fontsize=6.8); ax.invert_yaxis()
ax.set_xlabel('Variance explained uniquely [R²]'); offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax = fig.add_subplot(gs[1, 1]); lab(ax, 'D', -0.75)
LN = {'hay_2011': 'Hay', 'hodgkin_huxley_1952': 'Hodgkin–Huxley', 'khaliq_2003': 'Khaliq', 'migliore_1999': 'Migliore', 'pospischil_2008': 'Pospischil'}
lolo = RVAR['synergy_area:no_geometry']['lolo']['sampling_only']; keys = list(LN); yy = np.arange(len(keys))
vals = [lolo[k] for k in keys]; shown = [max(v, -1.0) for v in vals]
ax.barh(yy, shown, color=[RED if v < 0 else BLUE for v in vals], height=0.6); ax.axvline(0, color='k', lw=0.6)
ax.set_xlim(-1, 1); ax.set_yticks(yy); ax.set_yticklabels([LN[k] for k in keys], fontsize=6.8); ax.invert_yaxis()
ax.set_xlabel('Prediction of left-out source model [R²]')
offset(ax, ('bottom',)); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
fig.savefig(OUT + '/fig5.png', bbox_inches='tight'); plt.close(fig)
print('ok')
