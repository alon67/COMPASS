"""WITHDRAWN (V17 attribution; kept for the record). Its geometry block contains measures computed
from the covariance matrix, of which the synergy part is itself a function, and it is not run by
run_figures.sh. The current analysis is tools/run_review_v18.py (results/analyses/review_v18/attribution.json).

Variance partition of the synergy area by sampling design, population geometry and source lineage,
with a bootstrap over source lineages (the five lineages used for the main decision).

Usage: variance_partition_bootstrap.py OUT.json
"""
import sys, importlib.util, json, numpy as np, pandas as pd
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT+'/src')
spec=importlib.util.spec_from_file_location('rc', ROOT+'/tools/run_crossmodel.py'); rc=importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
from compass.analysis.hierarchical import variance_partition, PREDICTOR_BLOCKS
print('arms:', list(rc.SENSITIVITY_ARMS))
arm='discovery_only'; sp=rc.SENSITIVITY_ARMS[arm]; print(arm, sp)
src=sp.get('source_experiment','screening')
ep=pd.read_parquet(f'{ROOT}/results/analyses/endpoints_annotated_{src}_n4proj.parquet')
ep=ep[rc.primary_mask(ep, arm)].reset_index(drop=True)
fit=rc.drop_duplicate_scientific_rows(ep); g=fit[fit.estimator=='gaussian'].reset_index(drop=True)
print('rows', len(g), 'lineages', g.source_lineage_id.nunique(), 'impl', g.model_id.nunique())
blocks={'sampling':PREDICTOR_BLOCKS['sampling'],'geometry':PREDICTOR_BLOCKS['geometry'],'lineage':['source_lineage_id']}
old=variance_partition(g,'synergy_area'); print(old[['block','r2_alone','r2_unique']].round(4).to_string())
vp=variance_partition(g,'synergy_area',blocks); print(vp[['block','r2_alone','r2_unique','r2_full_model']].round(4).to_string())
def ratio(df):
    v=variance_partition(df,'synergy_area',blocks).set_index('block')['r2_unique']
    return v, (v['sampling']+v['geometry'])/v['lineage'] if v['lineage']>0 else np.inf
rng=np.random.default_rng(20260804); L=g.source_lineage_id.unique(); rs=[]; us=[]
for b in range(2000):
    pick=rng.choice(L,len(L),replace=True)
    parts=[]
    for k,l in enumerate(pick):
        s=g[g.source_lineage_id==l].copy(); s['source_lineage_id']=f'{l}__{k}'; s['model_id']=s['model_id']+f'__{k}'; parts.append(s)
    d=pd.concat(parts,ignore_index=True)
    if len(set(pick))<2: continue   # check the ORIGINAL lineages
    try: v,r=ratio(d)
    except Exception: continue
    rs.append(r); us.append(v.values)
rs=np.array(rs); fin=rs[np.isfinite(rs)]
pt=ratio(g)[1]
out={'n_rows':len(g),'n_lineages':int(g.source_lineage_id.nunique()),'n_impl':int(g.model_id.nunique()),
 'unique':vp.set_index('block')['r2_unique'].round(4).to_dict(),'alone':vp.set_index('block')['r2_alone'].round(4).to_dict(),'full':float(vp.r2_full_model.iloc[0]),
 'ratio_point':float(pt),'n_boot':len(rs),'n_inf':int((~np.isfinite(rs)).sum()),'ratio_median':float(np.median(fin)),'ratio_ci':[float(np.percentile(fin,2.5)),float(np.percentile(fin,97.5))],
 'unique_ci':{b:[float(np.percentile(np.array(us)[:,i],2.5)),float(np.percentile(np.array(us)[:,i],97.5))] for i,b in enumerate(['sampling','geometry','lineage'])}}
print(json.dumps(out,indent=1)); json.dump(out,open(sys.argv[1],'w'),indent=1)
