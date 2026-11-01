import sys,json,html
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_all import valid

def main(root):
 root=Path(root);rows=[];trans=[]
 for j in sorted(root.glob('*__*__seed*')):
  r=valid(j)
  if not r:continue
  v,m,s=j.name.split('__');seed=int(s[4:]);summary=json.loads((r/'summary.json').read_text());f=pd.read_csv(r/'fold_results.csv')
  rows.append(dict(variant=v,model=m,seed=seed,mae_ah=summary['cell_macro_mae_ah'],worst_cell_mae_ah=f.mae_ah.max(),cells_meeting_target=int((f.mae_ah<=.05).sum()),run=str(r.relative_to(root))))
  t=pd.read_csv(r/'transfer_cell_results.csv');t['variant']=v;t['model']=m;t['seed']=seed;trans.append(t)
 if not rows:return
 raw=pd.DataFrame(rows);raw.to_csv(root/'ALL_SEEDS_SOURCE.csv',index=False)
 agg=raw.groupby(['variant','model']).mae_ah.agg(['mean','std','count','min','max']).reset_index();agg.to_csv(root/'SOURCE_SEED_SUMMARY.csv',index=False)
 t=pd.concat(trans);t.to_csv(root/'ALL_TRANSFER_CELLS.csv',index=False)
 per=t.groupby(['variant','model','seed','temperature_c','donor_fraction']).mae_ah.agg(mean_mae_ah='mean',worst_cell_mae_ah='max',cells='size',cells_meeting_target=lambda x:int((x<=.05).sum())).reset_index();per.to_csv(root/'TRANSFER_BY_SEED.csv',index=False)
 avg=per.groupby(['variant','model','temperature_c','donor_fraction']).mean_mae_ah.agg(['mean','std','count']).reset_index();avg.to_csv(root/'TRANSFER_SEED_SUMMARY.csv',index=False)
 tp=per.pivot(index=['model','seed','temperature_c','donor_fraction'],columns='variant',values='mean_mae_ah')
 if {'original','drop_constant'}<=set(tp.columns):
  tp['ablated_minus_original_ah']=tp.drop_constant-tp.original;tp.to_csv(root/'PAIRED_ABLATION_TRANSFER.csv')
 paired=raw.pivot(index=['model','seed'],columns='variant',values='mae_ah')
 if {'original','drop_constant'}<=set(paired.columns):
  paired['ablated_minus_original_ah']=paired.drop_constant-paired.original;paired.to_csv(root/'PAIRED_ABLATION_SOURCE.csv')
 fig,ax=plt.subplots(figsize=(10,5))
 for v,g in agg.groupby('variant'):ax.errorbar(g.model,g['mean'],yerr=g['std'].fillna(0),fmt='o',capsize=4,label=v)
 ax.axhline(.05,color='gray',linestyle=':');ax.set_ylabel('Source cell-macro MAE (Ah); bars = seed SD');ax.tick_params(axis='x',rotation=25);ax.legend();fig.tight_layout();fig.savefig(root/'source_seed_comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,temp in zip(axes,[35,45]):
  for (v,m),g in avg[(avg.temperature_c==temp)&avg.model.isin(['mlp_small','mlp_large','svr','hist_boost'])].groupby(['variant','model']):
   ax.plot(g.donor_fraction*100,g['mean'],label=f'{m} {v}',linestyle='--' if v=='original' else '-')
  ax.axhline(.05,color='gray',ls=':');ax.set(title=f'{temp} C',xlabel='Donor cycles (%)',ylabel='Mean across seeds of cell-macro MAE (Ah)')
 axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(root/'transfer_comparison.png',dpi=180);plt.close(fig)
 text='''<html><meta charset="utf-8"><title>Battery validation comparison</title><style>body{font:16px Arial;margin:35px;max-width:1200px}td,th{padding:7px;border:1px solid #ccc}table{border-collapse:collapse}img{max-width:100%}</style><h1>Battery validation comparison</h1>'''
 text+=f'<p>{len(raw)} of 32 planned training jobs available. Check SUITE_STATUS.json for final completion. Smoke runs are tests, not report evidence.</p>'
 text+='<p>Seed SD measures training variability; it is not a confidence interval or new independent cells. One-seed models have undefined SD. Negative paired ablation differences favour feature removal. All tuning and feature filtering use training cells only. Repeated seeds are not extra batteries.</p>'+agg.to_html(index=False,float_format=lambda x:f'{x:.5f}')
 text+='<img src="source_seed_comparison.png"><img src="transfer_comparison.png"><h2>Cell-level transfer success</h2>'+per.to_html(index=False,float_format=lambda x:f'{x:.5f}')
 text+='<h2>Explanations and run records</h2><p>SHAP plots explain source-only final fits, not donor-retrained models. Approximate permutation rankings depend on the background and correlated predictors.</p><ul>'
 for x in rows:
  rel=Path(x['run']).as_posix();text+=f'<li><a href="{rel}/REPORT.html">{html.escape(rel)}</a> | <a href="{rel}/shap/">SHAP files (seed 42)</a></li>'
 (root/'COMPARISON.html').write_text(text+'</ul></html>',encoding='utf-8')
if __name__=='__main__':main(sys.argv[1])
