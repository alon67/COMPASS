"""Summarise the review analyses (results/analyses/review_v5/) into one JSON file plus two CSV tables.

Usage: summarise_review_analyses.py OUT.json   (also writes OUT_null_table.csv and OUT_dropout_table.csv)
"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
D = str(Path(__file__).resolve().parents[1] / 'results' / 'analyses' / 'review_v5') + '/'
OUT = sys.argv[1]
TOL = ['permissive', 'moderate', 'strict', 'very_strict']
out = {}

# ---- A: size-matched null ---------------------------------------------------
n = pd.read_parquet(D + 'size_matched_null.parquet')
sel = n[n.kind == 'selected'].set_index(['model_id', 'metric', 'tolerance'])
rows = []
for (m, met, tol), r in sel.iterrows():
    for kind in ('unselected_same_size', 'shuffled_same_size'):
        nl = n[(n.model_id == m) & (n.metric == met) & (n.tolerance == tol) & (n.kind == kind)]
        d = {'model_id': m, 'metric': met, 'tolerance': tol, 'null': kind, 'n': r.n}
        for c in ('frac_synergy_o3', 'net_o_information', 'synergy_area', 'mean_abs_pearson'):
            v = nl[c].to_numpy()
            d[c + '_sel'] = r[c]; d[c + '_null_mean'] = v.mean()
            d[c + '_null_lo'] = np.percentile(v, 2.5); d[c + '_null_hi'] = np.percentile(v, 97.5)
            d[c + '_pct'] = float(np.mean(v < r[c]) * 100)   # percentile of selected in null
            sd = v.std(ddof=1)
            d[c + '_z'] = (r[c] - v.mean()) / sd if sd > 0 else np.nan
        rows.append(d)
S = pd.DataFrame(rows)
S.to_csv(OUT.replace('.json', '_null_table.csv'), index=False)
for met in ('weighted_euclidean', 'max_z'):
    for kind in ('unselected_same_size', 'shuffled_same_size'):
        x = S[(S.metric == met) & (S.null == kind)]
        g = x.groupby('tolerance')
        out[f'{met}|{kind}'] = {tol: {
            'n_models': int(len(g.get_group(tol))),
            'frac_sel': g.get_group(tol).frac_synergy_o3_sel.mean(),
            'frac_null': g.get_group(tol).frac_synergy_o3_null_mean.mean(),
            'net_sel': g.get_group(tol).net_o_information_sel.mean(),
            'net_null': g.get_group(tol).net_o_information_null_mean.mean(),
            'syn_sel': g.get_group(tol).synergy_area_sel.mean(),
            'syn_null': g.get_group(tol).synergy_area_null_mean.mean(),
            'n_models_net_below_null_band': int((g.get_group(tol).net_o_information_sel < g.get_group(tol).net_o_information_null_lo).sum()),
            'n_models_syn_above_null_band': int((g.get_group(tol).synergy_area_sel > g.get_group(tol).synergy_area_null_hi).sum()),
            'n_models_frac_above_null_band': int((g.get_group(tol).frac_synergy_o3_sel > g.get_group(tol).frac_synergy_o3_null_hi).sum()),
            'syn_ratio_median': float(np.median(g.get_group(tol).synergy_area_sel / g.get_group(tol).synergy_area_null_mean)),
            'r_sel': g.get_group(tol).mean_abs_pearson_sel.mean(),
            'r_null': g.get_group(tol).mean_abs_pearson_null_mean.mean(),
        } for tol in TOL if tol in g.groups}

# ---- B: leave one feature out ------------------------------------------------
f = pd.read_parquet(D + 'feature_dropout.parquet')
fn = sel.reset_index()
f = f.merge(fn[['model_id', 'metric', 'tolerance', 'net_o_information', 'synergy_area', 'frac_synergy_o3']],
            on=['model_id', 'metric', 'tolerance'], suffixes=('', '_full'))
f['d_syn_rel'] = (f.synergy_area - f.synergy_area_full) / f.synergy_area_full
f.to_csv(OUT.replace('.json', '_dropout_table.csv'), index=False)
for met in ('weighted_euclidean', 'max_z'):
    x = f[(f.metric == met) & (f.tolerance == 'permissive')]
    out[f'dropout|{met}'] = x.groupby('dropped_feature').agg(
        jaccard=('jaccard_accepted', 'mean'), frac_changed=('frac_changed', 'mean'),
        d_syn_rel_mean=('d_syn_rel', 'mean'), d_syn_rel_min=('d_syn_rel', 'min'), d_syn_rel_max=('d_syn_rel', 'max'),
        frac_syn=('frac_synergy_o3', 'mean'), frac_syn_full=('frac_synergy_o3_full', 'mean'),
        n_net_neg=('net_o_information', lambda s: int((s < 0).sum())), n=('model_id', 'size')).round(4).to_dict('index')
    r = x[x.dropped_feature == 'rheobase_nA']
    out[f'rheobase|{met}'] = r[['model_id', 'jaccard_accepted', 'frac_changed', 'net_o_information',
                                 'net_o_information_full', 'synergy_area', 'synergy_area_full']].round(6).to_dict('records')

# ---- C: nearest-neighbour check ----------------------------------------------
k = pd.read_parquet(D + 'knn_check.parquet')
kk = {}
for m, g in k.groupby('model_id'):
    s = g[g.kind == 'selected'].set_index('tuple')
    sh = g[g.kind == 'shuffled_same_size'].groupby('tuple')
    un = g[g.kind == 'unselected_same_size'].groupby('tuple')
    sh_lo = sh.omega_knn.quantile(0.0); un_mean = un.omega_knn.mean(); sh_mean = sh.omega_knn.mean()
    d_sh = s.omega_knn - sh_mean
    d_g = s.omega_gaussian - sh.omega_gaussian.mean()
    kk[m] = {'n': int(g.n.iloc[0]), 'n_triplets': int(len(s)),
             'knn_sel_mean': s.omega_knn.mean(), 'knn_shuffled_mean': sh_mean.mean(), 'knn_unselected_mean': un_mean.mean(),
             'knn_frac_neg_sel': float((s.omega_knn < 0).mean()),
             'knn_frac_below_shuffled_mean': float((d_sh < 0).mean()),
             'knn_frac_below_all_shuffled': float((s.omega_knn < sh_lo).mean()),
             'knn_mean_shift_vs_shuffled': float(d_sh.mean()),
             'gauss_mean_shift_vs_shuffled': float(d_g.mean()),
             'gauss_frac_below_shuffled_mean': float((d_g < 0).mean()),
             'r_knn_gauss_shift': float(np.corrcoef(d_sh, d_g)[0, 1]) if len(s) > 2 else np.nan,
             'shuffle_sd_of_mean': float(g[g.kind == 'shuffled_same_size'].groupby('draw').omega_knn.mean().std()),
             'sel_mean_vs_shuffle_draw_means_pct': float((g[g.kind == 'shuffled_same_size'].groupby('draw').omega_knn.mean() > s.omega_knn.mean()).mean()*100)}
out['knn'] = kk

# ---- D: top triplets -----------------------------------------------------------
t = pd.read_parquet(D + 'top_triplets.parquet')
out['top'] = t.groupby(['metric', 'tolerance'])[['overlap_with_3sd', 'n_top']].mean().round(2).reset_index().to_dict('records')
out['top_by_model'] = t[t.metric == 'weighted_euclidean'].to_dict('records')

json.dump(out, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:20000])
