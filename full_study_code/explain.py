"""Sampled interventional permutation SHAP, exact feature replacement, no closeness shortcut."""
import sys,json,argparse
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import Neural
from preflight import check

def load(run):
 folder=run/'fitted_model';cfg=json.loads((run/'config.json').read_text());name=json.loads((run/'summary.json').read_text())['model']
 if not name.startswith('mlp'):return joblib.load(folder/'model.joblib')['model']
 import torch
 from torch import nn
 saved=torch.load(folder/'network.pt',map_location='cpu',weights_only=True)
 n=Neural(name,{},cfg);n.pre=joblib.load(folder/'preprocessor.joblib');n.device='cpu';n.mu=saved['mu'];n.sd=saved['sd'];layers=[];width=saved['input_dim']
 for w in saved['widths']:layers.extend([nn.Linear(width,w),nn.ReLU()]);width=w
 layers.append(nn.Linear(width,1));n.net=nn.Sequential(*layers);n.net.load_state_dict(saved['state_dict']);n.net.eval();return n

def main():
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--smoke',action='store_true');a=p.parse_args();run=Path(a.run);out=run/'shap';out.mkdir(exist_ok=True)
 d,features=check();model=load(run);rng=np.random.default_rng(42);source=d[d.temperature_c==25]
 # One row per selected source cell, common background across models/variants.
 bg=source.groupby('cell',sort=True).sample(n=1,random_state=42).sample(n=10,random_state=42)[features].to_numpy()
 pd.DataFrame(bg,columns=features).to_csv(out/'background.csv',index=False);baseline=float(model.predict(bg).mean());cycles=1 if a.smoke else 4;importance=[];maxerr=0
 for temp,g in d.groupby('temperature_c'):
  samples=pd.concat([c.sample(n=min(1 if a.smoke else 2,len(c)),random_state=42) for _,c in g.groupby('cell',sort=True)])
  if a.smoke:samples=samples.head(2)
  vals=[]
  for x in samples[features].to_numpy():
   phi=np.zeros(len(features))
   for _ in range(cycles):
    order=rng.permutation(len(features))
    for perm in [order,order[::-1]]:
     state=bg.copy();path=[state.copy()]
     for j in perm:state[:,j]=x[j];path.append(state.copy())
     pred=model.predict(np.concatenate(path)).reshape(len(features)+1,len(bg)).mean(axis=1)
     phi[perm]+=np.diff(pred)/(2*cycles)
   vals.append(phi)
  values=np.array(vals);prediction=model.predict(samples[features].to_numpy());err=np.max(np.abs(baseline+values.sum(1)-prediction));maxerr=max(maxerr,float(err))
  assert err<1e-4,f'SHAP reconstruction error {err}'
  samples.to_csv(out/f'samples_{temp:g}C.csv',index=False);pd.DataFrame(values,columns=features).to_csv(out/f'shap_values_{temp:g}C.csv',index=False)
  imp=np.abs(values).mean(0)
  for f,v in zip(features,imp):importance.append({'temperature_c':temp,'feature':f,'mean_abs_shap_ah':v,'samples':len(samples)})
  idx=np.argsort(imp)[-12:];fig,ax=plt.subplots(figsize=(8,5));ax.barh(np.array(features)[idx],imp[idx]);ax.set(title=f'{temp:g} C: source-only model',xlabel='Mean absolute permutation SHAP (Ah)');fig.tight_layout();fig.savefig(out/f'SHAP_{temp:g}C.png',dpi=180);plt.close(fig)
  print('SHAP complete',run.name,temp,flush=True)
 pd.DataFrame(importance).to_csv(out/'importance_by_temperature.csv',index=False)
 (out/'COMPLETE.json').write_text(json.dumps({'method':'Antithetic sampled permutation interventional SHAP','seed':42,'background_rows':10,'permutation_pairs':cycles,'max_additivity_error_ah':maxerr,'smoke':a.smoke,'scope':'Saved source-only fit; source samples in-sample; target samples zero-label transfer. Not donor-retrained explanations.'},indent=2))
if __name__=='__main__':main()
