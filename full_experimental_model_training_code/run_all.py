"""Sequential, restartable study; completed jobs are reused only with identical code/data."""
import os,sys,json,hashlib,subprocess,platform,zipfile,traceback,argparse,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
MODELS=['mean','ridge','svr','random_forest','extra_trees','hist_boost','mlp_small','mlp_large']
def plan():
 return [(v,m,s) for s in [42,43,44,45,46] for v in ['original','drop_constant'] for m in MODELS if s==42 or m.startswith('mlp')]
def signature(smoke):
 h=hashlib.sha256()
 for f in ['models.py','run_model.py','run_all.py','summarise.py','explain.py','data/features.csv','data/feature_names.json','data/manifest.json']:
  h.update((ROOT/f).read_bytes())
 import importlib.metadata as md
 h.update(json.dumps({k:md.version(k) for k in ['numpy','scipy','pandas','scikit-learn','torch','matplotlib','joblib']},sort_keys=True).encode())
 h.update(str(smoke).encode());return h.hexdigest()
def valid(job):
 for p in sorted(job.glob('*/status.json'),reverse=True):
  try:s=json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):continue
  if s.get('status')=='completed' and all((p.parent/f).exists() and (p.parent/f).stat().st_size>0 for f in ['summary.json','fold_results.csv','transfer_cell_results.csv','transfer_splits.csv','predictions.csv']):return p.parent
 return None
def lockfile(path):
 f=path.open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
 try:
  if os.name=='nt':
   import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except OSError:raise RuntimeError('This suite is already running in this folder.')
 return f

def main():
 a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a.add_argument('--limit',type=int);args=a.parse_args()
 os.chdir(ROOT);lock=lockfile(ROOT/'suite.lock')
 sig=signature(args.smoke);out=ROOT/('smoke_results' if args.smoke else 'validation_results')/sig[:12];out.mkdir(parents=True,exist_ok=True)
 jobs=plan();jobs=jobs[:args.limit] if args.limit else jobs
 manifest={'signature':sig,'smoke':args.smoke,'jobs':jobs,'platform':platform.platform(),'python':sys.version}
 (out/'study_plan.json').write_text(json.dumps(manifest,indent=2))
 awake=False
 if os.name=='nt':
  import ctypes;awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
 failures=[]
 try:
  from preflight import check
  check()
  for i,(variant,model,seed) in enumerate(jobs,1):
   job=out/f'{variant}__{model}__seed{seed}';job.mkdir(exist_ok=True)
   print(f'\n[{i}/{len(jobs)}] {variant} / {model} / seed {seed}',flush=True)
   if valid(job):print('Already complete; skipping.',flush=True);continue
   cmd=[sys.executable,'-u','run_model.py','--model',model,'--seed',str(seed),'--results',str(job),'--force-cpu','--unattended']
   if variant=='drop_constant':cmd+=['--drop-constant']
   if args.smoke:cmd+=['--smoke']
   with (job/'launcher.log').open('a',encoding='utf-8') as log:
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    try:
     for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
     rc=proc.wait()
    except KeyboardInterrupt:proc.terminate();proc.wait();raise
   if rc or not valid(job):failures.append(job.name);print('FAILED; continuing with other jobs.',flush=True)
   if subprocess.run([sys.executable,'summarise.py',str(out)]).returncode:
    print('Interim report failed; remaining training jobs will continue.',flush=True)
  # Seed 42 source-only SHAP for every family and variant, separate from training completion.
  for variant,model,seed in jobs:
   if seed!=42:continue
   run=valid(out/f'{variant}__{model}__seed{seed}')
   if not run:continue
   if (run/'shap'/'COMPLETE.json').exists():continue
   cmd=[sys.executable,'-u','explain.py',str(run)]
   if args.smoke:cmd+=['--smoke']
   if subprocess.run(cmd).returncode:failures.append('SHAP '+str(run))
  if subprocess.run([sys.executable,'summarise.py',str(out)]).returncode:failures.append('Final comparison report')
  complete=len(jobs)==32 and not args.smoke and not failures and all(valid(out/f'{v}__{m}__seed{s}') for v,m,s in plan())
  state={'complete':complete,'failures':failures,'finished_utc':datetime.now(timezone.utc).isoformat(),'smoke':args.smoke}
  (out/'SUITE_STATUS.json').write_text(json.dumps(state,indent=2))
  # Archive current study only, including incomplete-job evidence if any.
  dest=ROOT/('SMOKE_RESULTS.zip' if args.smoke else 'RESULTS_TO_UPLOAD.zip');tmp=dest.with_suffix('.tmp')
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
   for p in out.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(out.parent))
  os.replace(tmp,dest)
  print('\n'+('ALL 32 TRAINING JOBS AND EXPLANATIONS COMPLETE' if complete else 'PARTIAL / TEST RUN: inspect SUITE_STATUS.json')+'\nReport: '+str(out/'COMPARISON.html')+'\nUpload: '+str(dest),flush=True)
  if os.name=='nt' and (out/'COMPARISON.html').exists():os.startfile(str(out/'COMPARISON.html'))
  return 0 if complete else 1
 finally:
  if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  lock.close()
if __name__=='__main__':
 try:sys.exit(main())
 except KeyboardInterrupt:print('Stopped. Relaunch to resume completed jobs.');sys.exit(130)
 except Exception:traceback.print_exc();sys.exit(1)
