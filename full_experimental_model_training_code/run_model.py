"""One independent, fully logged experiment per model. Windows launchers call this."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,csv,hashlib,html,json,platform,sys,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from models import SPECS,fit,metrics
ROOT=Path(__file__).resolve().parent

def write_json(p,x):p.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def xy(d,features):return d[features].to_numpy(float),d.capacity_ah.to_numpy(float)
def macro(d,p):
    t=d[['cell','capacity_ah']].copy();t['prediction_ah']=p
    return float(np.mean([metrics(g.capacity_ah,g.prediction_ah)['mae_ah'] for _,g in t.groupby('cell')]))

class Experiment:
    def __init__(self,name,cfg,out):
        self.name=name;self.cfg=cfg;self.out=out;self.history=[];self.tuning=[];self.folds=[];self.predictions=[]
        self.history_file=(out/'training_history.csv').open('w',newline='',encoding='utf-8')
        self.writer=csv.DictWriter(self.history_file,fieldnames=['context','candidate','fold','epoch','train_mae_ah','train_mse_ah2','validation_mae_ah'])
        self.writer.writeheader()
    def emit(self,context,candidate='',fold=''):
        def log(row):
            row={'context':context,'candidate':candidate,'fold':fold,**row}
            self.history.append(row);self.writer.writerow(row);self.history_file.flush()
        return log
    def train(self,train,features,params,context,valid=None,epochs=None):
        X,y=xy(train,features)
        model=fit(self.name,params,self.cfg,X,y,valid=None if valid is None else xy(valid,features),emit=self.emit(context),epochs=epochs)
        self.audit_mask(model,features,context)
        return model
    def audit_mask(self,model,features,context):
        pre=model.pre if hasattr(model,'pre') else model
        filt=pre.named_steps.get('exactconstantfilter')
        kept=features if filt is None else [f for f,k in zip(features,filt.keep_) if k]
        with (self.out/'feature_masks.jsonl').open('a') as out:
            out.write(json.dumps({'context':context,'kept':kept,'removed':[f for f in features if f not in kept]})+'\n')
    def tune(self,d,features,context):
        splits=list(GroupKFold(n_splits=min(3,d.cell.nunique())).split(d,groups=d.cell))
        results=[]
        for candidate,params in enumerate(SPECS[self.name][2]):
            print(f'  {context}: candidate {candidate+1}/{len(SPECS[self.name][2])}',flush=True)
            errs=[];epochs=[]
            for fold,(ti,vi) in enumerate(splits):
                train,valid=d.iloc[ti],d.iloc[vi]
                assert not set(train.cell)&set(valid.cell)
                X,y=xy(train,features);Xv,yv=xy(valid,features)
                model=fit(self.name,params,self.cfg,X,y,valid=(Xv,yv),emit=self.emit(context,candidate,fold))
                self.audit_mask(model,features,f'{context}_candidate{candidate}_fold{fold}')
                pred=model.predict(Xv)
                # Aggregate each validation CELL equally, not each fold.
                for cell in valid.cell.unique():
                    mask=(valid.cell==cell).to_numpy();errs.append(metrics(yv[mask],pred[mask])['mae_ah'])
                ep=getattr(model,'best_epoch',None)
                if ep:epochs.append(ep)
                self.tuning.append({'context':context,'candidate':candidate,'fold':fold,'params':json.dumps(params),
                    'validation_cell_macro_mae_ah':macro(valid,pred),'selected_epochs':ep,
                    'train_cells':','.join(sorted(train.cell.unique())),'validation_cells':','.join(sorted(valid.cell.unique()))})
            results.append((float(np.mean(errs)),params,int(np.median(epochs)) if epochs else None))
        pd.DataFrame(self.tuning).to_csv(self.out/'tuning_history.csv',index=False)
        return min(results,key=lambda x:x[0])
    def add_predictions(self,test,p,experiment,**extra):
        r=test[['cell','sequence','temperature_c','capacity_ah']].copy();r['prediction_ah']=p;r['experiment']=experiment
        for k,v in extra.items():r[k]=v
        self.predictions.append(r)
        path=self.out/'predictions.csv'
        r.to_csv(path,mode='a',header=not path.exists(),index=False)
    def run(self,d,features):
        source=d[d.temperature_c==25].copy()
        if source.cell.nunique()<4:raise ValueError('Need at least four room-temperature cells')
        for cell in sorted(source.cell.unique()):
            print('Held-out battery:',cell,flush=True)
            train=source[source.cell!=cell];test=source[source.cell==cell]
            score,params,epochs=self.tune(train,features,'loco_'+cell)
            model=self.train(train,features,params,'refit_'+cell,epochs=epochs)
            p=model.predict(xy(test,features)[0]);self.add_predictions(test,p,'room_loco')
            self.folds.append({'cell':cell,'inner_mae_ah':score,'parameters':json.dumps(params),'epochs':epochs,
                               **metrics(test.capacity_ah,p)})
            pd.DataFrame(self.folds).to_csv(self.out/'fold_results.csv',index=False)
        # Dedicated diagnostic curve: one validation CELL, never an outer test.
        # Fixed first candidate ensures curve is not selected using this holdout.
        val_cell=sorted(source.cell.unique())[-1];valid=source[source.cell==val_cell];pool=source[source.cell!=val_cell]
        curve=[]
        for frac in (0.1,0.25,0.5,1.0):
            small=pd.concat([g.iloc[:max(2,int(np.ceil(len(g)*frac)))] for _,g in pool.groupby('cell')])
            model=self.train(small,features,SPECS[self.name][2][0],f'learning_fraction_{frac}',valid=valid)
            curve.append({'fraction':frac,'training_rows':len(small),'training_cells':small.cell.nunique(),
                          'validation_cell':val_cell,'train_mae_ah':macro(small,model.predict(xy(small,features)[0])),
                          'validation_mae_ah':macro(valid,model.predict(xy(valid,features)[0]))})
        pd.DataFrame(curve).to_csv(self.out/'learning_curve.csv',index=False)
        # Tune final source model ONLY with 25C batteries.
        score,params,epochs=self.tune(source,features,'source_selection')
        model=self.train(source,features,params,'source_final',epochs=epochs)
        folder=self.out/'fitted_model';folder.mkdir()
        if self.name.startswith('mlp'):model.save(folder,features)
        else:joblib.dump({'model':model,'features':features},folder/'model.joblib')
        write_json(folder/'settings.json',{'parameters':params,'epochs':epochs,'trained_temperature_c':25,'nominal_ah':3.5})
        zero_models=model
        splits=[]
        for temp in sorted(set(d.temperature_c)-{25}):
            target=d[d.temperature_c==temp]
            if target.cell.nunique()<2:raise ValueError(f'Insufficient target batteries at {temp}C')
            for cell in sorted(target.cell.unique()):
                test=target[target.cell==cell];donors=target[target.cell!=cell]
                assert not set(source.cell)&set(target.cell)
                for frac in self.cfg['fractions']:
                    cal=pd.concat([g.sort_values('sequence').iloc[:0 if frac==0 else max(1,int(np.ceil(len(g)*frac)))] for _,g in donors.groupby('cell')])
                    train=pd.concat([source,cal])
                    assert cell not in set(train.cell)
                    if frac==0:model=zero_models
                    else:model=self.train(train,features,params,f'transfer_{temp}_{cell}_{frac}',epochs=epochs)
                    self.add_predictions(test,model.predict(xy(test,features)[0]),'transfer',donor_fraction=frac,labelled_target_cycles=len(cal))
                    splits.append({'temperature_c':temp,'test_cell':cell,'donor_fraction':frac,'labelled_target_cycles':len(cal),
                        'train_cells':','.join(sorted(train.cell.unique())),'donor_cells':','.join(sorted(cal.cell.unique()))})
                print(f'Transfer completed: {temp}C {cell}',flush=True)
        pd.DataFrame(splits).to_csv(self.out/'transfer_splits.csv',index=False)
        # Baseline predictions use exactly the same outer training cells.
        baseline=[]
        for cell in source.cell.unique():
            test=source[source.cell==cell];mu=source[source.cell!=cell].capacity_ah.mean()
            baseline.append(metrics(test.capacity_ah,np.full(len(test),mu))['mae_ah'])
        mae=float(np.mean([r['mae_ah'] for r in self.folds]));base=float(np.mean(baseline))
        rng=np.random.default_rng(self.cfg['seed']);values=np.array([r['mae_ah'] for r in self.folds])
        ci=np.quantile(rng.choice(values,(2000,len(values)),replace=True).mean(1),[.025,.975])
        summary={'model':self.name,'title':SPECS[self.name][0],'source_cells':source.cell.nunique(),
            'source_rows':len(source),'cell_macro_mae_ah':mae,'mean_baseline_mae_ah':base,
            'improvement_vs_mean_pct':100*(1-mae/base),'soh_mae_percentage_points':mae*100/3.5,
            'cell_bootstrap_95pct_low_ah':ci[0],'cell_bootstrap_95pct_high_ah':ci[1],
            'exploratory_goal_met':bool(mae<=.05 and mae<=base*.8),'selected_parameters':params,'selected_epochs':epochs}
        write_json(self.out/'summary.json',summary)
        pred=pd.concat(self.predictions);scores=[]
        for key,g in pred[pred.experiment=='transfer'].groupby(['temperature_c','donor_fraction','cell']):
            scores.append(dict(zip(['temperature_c','donor_fraction','cell'],key))|metrics(g.capacity_ah,g.prediction_ah))
        scores=pd.DataFrame(scores);scores.to_csv(self.out/'transfer_cell_results.csv',index=False)
        ts=scores.groupby(['temperature_c','donor_fraction'],as_index=False).agg(mae_ah=('mae_ah','mean'),cells=('cell','nunique'))
        ts.to_csv(self.out/'transfer_summary.csv',index=False)
        self.plot(pred,curve,ts)
        self.report(summary,ts,curve)
        self.history_file.close()
        # Commit the complete fold table again atomically before marking success.
        tmp=self.out/'fold_results.tmp'
        pd.DataFrame(self.folds).to_csv(tmp,index=False)
        os.replace(tmp,self.out/'fold_results.csv')
        assert len(pd.read_csv(self.out/'fold_results.csv'))==source.cell.nunique()
        return summary
    def plot(self,pred,curve,ts):
        fig,ax=plt.subplots(figsize=(8,4));c=pd.DataFrame(curve)
        ax.plot(c.training_rows,c.train_mae_ah,'o-',label='Training');ax.plot(c.training_rows,c.validation_mae_ah,'o-',label='Validation battery')
        ax.set(xlabel='Training cycles (chronological prefixes)',ylabel='MAE (Ah)',title='Diagnostic data-size history (not test performance)');ax.legend();fig.tight_layout();fig.savefig(self.out/'learning_curve.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for temp,g in ts.groupby('temperature_c'):ax.plot(g.donor_fraction*100,g.mae_ah,'o-',label=f'{temp:g} C')
        ax.set(xlabel='First labelled donor cycles (%)',ylabel='Mean test-cell MAE (Ah)',title='25 C source + target-temperature donor data');ax.legend();fig.tight_layout();fig.savefig(self.out/'temperature_transfer.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4))
        for cell,g in pred[pred.experiment=='room_loco'].groupby('cell'):ax.scatter(g.capacity_ah,g.prediction_ah,s=8,alpha=.4,label=cell)
        ax.plot([2.5,3.5],[2.5,3.5],'k--');ax.set(xlabel='Measured capacity (Ah)',ylabel='Held-out prediction (Ah)');ax.legend(fontsize=6,ncol=5,loc='upper center',bbox_to_anchor=(0.5,-0.18));fig.tight_layout();fig.savefig(self.out/'predicted_vs_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
        h=pd.DataFrame(self.history)
        if 'epoch' in h:
            h=h[(h.context=='source_selection')&(h.candidate==0)&(h.fold==0)]
            if len(h):
                fig,ax=plt.subplots(figsize=(8,4));ax.plot(h.epoch,h.train_mae_ah,label='Train');ax.plot(h.epoch,h.validation_mae_ah,label='Validation')
                ax.set(xlabel='Epoch',ylabel='MAE (Ah)',title='Neural training history: source CV, candidate 0, fold 0');ax.legend();fig.tight_layout();fig.savefig(self.out/'epoch_history.png',dpi=160);plt.close(fig)
    def report(self,s,ts,curve):
        worst=max(self.folds,key=lambda x:x['mae_ah']);last=curve[-1]
        text=f'''# {SPECS[self.name][0]} - experiment notes

## Why test this model?
{SPECS[self.name][1]}

## What actually happened
- Room-temperature leave-one-cell-out MAE: {s['cell_macro_mae_ah']:.4f} Ah ({s['soh_mae_percentage_points']:.2f} nominal SoH percentage points).
- Improvement relative to the matched mean baseline: {s['improvement_vs_mean_pct']:.1f}%.
- Worst held-out battery: {worst['cell']} ({worst['mae_ah']:.4f} Ah MAE).
- Exploratory target (<=0.05 Ah and >=20% improvement): {s['exploratory_goal_met']}.
- Diagnostic full-data training MAE: {last['train_mae_ah']:.4f} Ah; separate validation battery MAE: {last['validation_mae_ah']:.4f} Ah.
- Selected source parameters: {s['selected_parameters']}; neural refit epochs: {s['selected_epochs']}.

## Interpretation to investigate, not a causal conclusion
A large training/validation gap is consistent with overfitting OR differences between batteries/protocols. High errors on both can indicate insufficient features, model bias, measurement problems, or optimisation failure. Improvement after adding target-temperature data supports the value of calibration in this sample; it does not isolate temperature from protocol and aging differences. Neural epoch loss is optimisation history; classical models do not have comparable epochs. Data-size histories use chronological prefixes, so age coverage changes too.

## Transfer results (mean across held-out target cells)
'''
        for _,r in ts.iterrows():text+=f"- {r.temperature_c:g} C, {100*r.donor_fraction:g}% donor cycles: {r.mae_ah:.4f} Ah MAE.\n"
        text+='''
## Limitations and report wording
Independent sample size is the number of cells, not cycles. Bootstrap ranges are approximate intervals for aggregate MAE, not individual prediction intervals. The primary 2.5-3.5 Ah range follows the Dataset 1 author extraction and excludes severe degradation. Temperature and charging rate can be confounded. This predicts capacity from post-charge voltage relaxation, not drive time. The source model is retrained with donor data, not fine-tuned. Transfer parameters and epoch budgets are selected on source cells only.

Use comparisons to explain evidence, not to claim that the model with the lowest score is universally best. Review the comparison report's nested selection estimate and test on independent batteries before deployment.
'''
        (self.out/'EXPERIMENT_NOTES.md').write_text(text,encoding='utf-8')
        imgs=['learning_curve.png','predicted_vs_actual.png','temperature_transfer.png']
        if (self.out/'epoch_history.png').exists():imgs.insert(0,'epoch_history.png')
        page='<html><meta charset="utf-8"><title>Experiment report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f6f8fb;color:#15243b}pre{white-space:pre-wrap;line-height:1.6}img{max-width:100%;background:white;margin:15px 0}</style><h1>'+html.escape(SPECS[self.name][0])+'</h1><pre>'+html.escape(text)+'</pre>'+''.join(f'<img src="{i}">' for i in imgs)+'</html>'
        (self.out/'REPORT.html').write_text(page,encoding='utf-8')

def compare(root):
    runs=[];register=[]
    for p in root.glob('*/status.json'):
        status=json.loads(p.read_text());register.append({**status,'folder':p.parent.name})
        if status.get('status')=='completed':
            s=json.loads((p.parent/'summary.json').read_text());runs.append({**s,**status,'folder':p.parent.name})
    if register:pd.DataFrame(register).sort_values('started_utc').to_csv(root/'RUN_REGISTER.csv',index=False)
    if not runs:return
    allruns=pd.DataFrame(runs).sort_values('started_utc');allruns.to_csv(root/'ALL_RUNS.csv',index=False)
    # Only compare matching data, feature, search and training settings.
    latest=allruns.iloc[-1]['comparison_key']
    comparable=allruns[allruns.comparison_key==latest].drop_duplicates('model',keep='last').sort_values('cell_macro_mae_ah')
    comparable.to_csv(root/'LATEST_COMPARISON.csv',index=False)
    # Family selection across separate jobs: choose by INNER score per outer cell.
    fs=[];ps=[]
    for _,r in comparable.iterrows():
        f=pd.read_csv(root/r.folder/'fold_results.csv');f['model']=r.model;fs.append(f)
        pr=pd.read_csv(root/r.folder/'predictions.csv');pr=pr[pr.experiment=='room_loco'];pr['model']=r.model;ps.append(pr)
    f=pd.concat(fs);pr=pd.concat(ps);selected=[]
    for cell,g in f.groupby('cell'):
        win=g.sort_values(['inner_mae_ah','model']).iloc[0];rows=pr[(pr.cell==cell)&(pr.model==win.model)].copy();selected.append(rows)
    selected=pd.concat(selected);selected.to_csv(root/'NESTED_SELECTION_PREDICTIONS.csv',index=False)
    nested=macro(selected,selected.prediction_ah.to_numpy())
    table=comparable[['title','cell_macro_mae_ah','soh_mae_percentage_points','improvement_vs_mean_pct','seconds','device','folder']].to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')
    page=f'''<html><meta charset="utf-8"><title>Battery model comparison</title><style>body{{font:16px system-ui;margin:40px;color:#182b45}}table{{border-collapse:collapse}}td,th{{padding:10px;border:1px solid #ddd}}</style><h1>Completed model experiments</h1><p>Latest completed run per model, matching dataset and settings. Every historical run is retained in ALL_RUNS.csv.</p>{table}<p>Nested selection MAE across currently completed families: <b>{nested:.4f} Ah</b>. The family for each test battery was selected using only its inner-validation score. Candidate set: {html.escape(', '.join(comparable.model))}.</p><p>Lowest individual outer-test score is descriptive, not an unbiased estimate after choosing a winner. The nested estimate changes as families are added. Repeatedly adapting choices after inspecting these tests requires fresh external validation.</p><p>Read each run's REPORT.html for histories, errors and interpretation. Hardware time includes tuning, curves and transfer, not just one fit.</p></html>'''
    (root/'COMPARISON.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(SPECS));p.add_argument('--compare',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--force-cpu',action='store_true');p.add_argument('--results');p.add_argument('--seed',type=int,default=42);p.add_argument('--drop-constant',action='store_true');p.add_argument('--unattended',action='store_true');a=p.parse_args()
    results=Path(a.results) if a.results else ROOT/'results';results.mkdir(exist_ok=True,parents=True)
    if a.compare:compare(results);return
    if not a.model:p.error('--model required')
    cfg={'seed':a.seed,'drop_constant':a.drop_constant,'epochs':150,'trees':300,'boost_iterations':200,'fractions':[0,.01,.05,.1,.25,.5,1.0],'force_cpu':a.force_cpu}
    if a.smoke:cfg.update(epochs=2,trees=5,boost_iterations=5,fractions=[0,.1])
    runid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+a.model+'_'+uuid.uuid4().hex[:6]
    out=results/runid;out.mkdir();started=time.perf_counter()
    status={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'model':a.model,'smoke_test':a.smoke}
    write_json(out/'status.json',status)
    log=(out/'console.log').open('w',encoding='utf-8')
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,s):self.stream.write(s);log.write(s);log.flush()
        def flush(self):self.stream.flush();log.flush()
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    exp=None
    try:
        manifest=json.loads((ROOT/'data/manifest.json').read_text());path=ROOT/'data/features.csv'
        if sha(path)!=manifest['features_sha256']:raise RuntimeError('Prepared data checksum mismatch. Restore original package or rebuild data.')
        if manifest.get('dataset')!='Zenodo 6405084, Dataset 1 NCA only' or manifest.get('nominal_ah')!=3.5:raise RuntimeError('This runner requires Zenodo Dataset 1 NCA, 3.5 Ah.')
        features=json.loads((ROOT/'data/feature_names.json').read_text())
        if features!=manifest['model_features']:raise RuntimeError('Feature schema mismatch')
        d=pd.read_csv(path)
        if a.smoke:d=d.groupby('cell',group_keys=False).head(12)
        key=json.dumps({'data':sha(path),'features':features,'config':cfg,'runner':sha(Path(__file__)),'models':sha(ROOT/'models.py')},sort_keys=True)
        status['comparison_key']=hashlib.sha256(key.encode()).hexdigest()
        env={'python':sys.version,'platform':platform.platform(),'device':'CPU'}
        if a.model.startswith('mlp'):
            import torch
            if torch.cuda.is_available() and not a.force_cpu:
                t=torch.ones(4,device='cuda',requires_grad=True);(t*t).sum().backward()
                env.update(device=torch.cuda.get_device_name(0),torch=torch.__version__,hip=torch.version.hip)
            else:env.update(device='CPU',torch=torch.__version__,hip=torch.version.hip)
        status['device']=env['device']
        write_json(out/'config.json',cfg);write_json(out/'environment.json',env)
        import importlib.metadata
        write_json(out/'package_versions.json',{k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','matplotlib','joblib']})
        write_json(out/'data_manifest.json',manifest)
        # Snapshot code to make each run reproducible even after future edits.
        for file in ('run_model.py','models.py','prepare_data.py'):(out/file).write_bytes((ROOT/file).read_bytes())
        print('Model:',a.model,'Device:',env['device'],'Output:',out,flush=True)
        exp=Experiment(a.model,cfg,out);summary=exp.run(d,features)
        status.update(status='completed',seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'status.json',status)
        if not a.unattended:compare(results)
        print('COMPLETE. Open',out/'REPORT.html',flush=True)
        if os.name=='nt' and not a.unattended:os.startfile(str(out/'REPORT.html'))
    except BaseException as e:
        status.update(status='failed',seconds=time.perf_counter()-started,error=str(e))
        write_json(out/'status.json',status)
        if not a.unattended:compare(results)
        traceback.print_exc();raise
    finally:
        if exp and not exp.history_file.closed:exp.history_file.close()

if __name__=='__main__':main()
