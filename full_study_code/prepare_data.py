"""Rebuild Zenodo 6405084 Dataset 1 features from the original, checksum-verified ZIP."""
import argparse,hashlib,io,json,re,urllib.request,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew,kurtosis
ROOT=Path(__file__).resolve().parent
URL='https://zenodo.org/records/6405084/files/Dataset_1_NCA_battery.zip?download=1'
MD5='cbd6d7eab2a296b52385e025e8d0f42a'
COLS=['time/s','control/V/mA','Ecell/V','<I>/mA','Q discharge/mA.h','cycle number']
def digest(path,kind='sha256'):
 h=hashlib.new(kind)
 with Path(path).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def rate(s):return float('0.'+s[1:] if '.' not in s and s.startswith('0') and len(s)>1 else s)
def extract_cycle(g,temp,charge_rate,discharge_rate):
 a=g[COLS[:-1]].to_numpy(float)
 if not np.isfinite(a).all():raise ValueError('nonfinite raw value')
 t,control,v,current,q=a.T
 if np.any(np.diff(t)<0):raise ValueError('time decreases within cycle')
 neg=np.flatnonzero(control < -1)
 if not len(neg):raise ValueError('no subsequent controlled discharge')
 discharge=int(neg[0])
 mask=(np.abs(control[:discharge])<1e-8)&(np.abs(current[:discharge])<=1)&(v[:discharge]>4.0)
 starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]]) if len(mask) else []
 chosen=None
 for start in starts:
  stop=start
  while stop<discharge and mask[stop]:stop+=1
  if stop-start>=14 and np.any(current[:start]>100):chosen=(int(start),int(stop));break
 if chosen is None:raise ValueError('no 14-point post-charge relaxation block above 4V')
 start,stop=chosen;rv=v[start:start+14];rt=t[start:start+14]
 if np.any(np.diff(rt)<=0):raise ValueError('nonincreasing relaxation times')
 cap=float(np.max(q[discharge:])/1000)
 if cap<=0:raise ValueError('nonpositive capacity')
 f={'ambient_c':float(temp),'planned_charge_c_rate':charge_rate,'planned_discharge_c_rate':discharge_rate,
    'relaxation_mean_v':float(rv.mean()),'relaxation_variance_v2':float(rv.var()),'relaxation_max_v':float(rv.max()),
    'relaxation_skewness':float(skew(rv,bias=True)),'relaxation_excess_kurtosis':float(kurtosis(rv,bias=True)),
    'relaxation_drop_v':float(rv[0]-rv[-1]),'relaxation_slope_v_per_s':float((rv[-1]-rv[0])/(rt[-1]-rt[0]))}
 f.update({f'relaxation_v_{j+1:02d}':float(x) for j,x in enumerate(rv)})
 if not np.isfinite(list(f.values())).all():raise ValueError('nonfinite derived feature')
 f.update(capacity_ah=cap,temperature_c=float(temp),relaxation_start_s=float(rt[0]),relaxation_end_s=float(rt[-1]),
          discharge_start_s=float(t[discharge]),relaxation_span_s=float(rt[-1]-rt[0]),primary_included=bool(2.5<=cap<=3.5))
 return f

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive');args=p.parse_args()
 path=Path(args.archive) if args.archive else ROOT/'Dataset_1_NCA_battery.zip'
 if not path.exists():print('Downloading Dataset 1 (~580 MB)...',flush=True);urllib.request.urlretrieve(URL,path)
 if digest(path,'md5')!=MD5:raise RuntimeError('Dataset 1 checksum does not match Zenodo record 6405084')
 out=ROOT/'data';out.mkdir(exist_ok=True);rows=[];excluded=[];inventory=[];seen=set()
 with zipfile.ZipFile(path) as z:
  for name in sorted(z.namelist()):
   if not name.lower().endswith('.csv'):continue
   cell=Path(name).stem;m=re.fullmatch(r'CY(\d+)-(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)-#(\d+)',cell)
   if not m:raise ValueError('Unexpected filename '+cell)
   if cell in seen:raise ValueError('Duplicate cell '+cell)
   seen.add(cell);temp,charge,discharge,_=m.groups();blob=z.read(name);raw=pd.read_csv(io.BytesIO(blob),usecols=COLS)
   inventory.append({'cell':cell,'file':name,'sha256':hashlib.sha256(blob).hexdigest(),'raw_rows':len(raw),'raw_cycles':raw['cycle number'].nunique()})
   for seq,g in raw.groupby('cycle number',sort=True):
    try:f=extract_cycle(g,int(temp),rate(charge),rate(discharge));f.update(cell=cell,sequence=float(seq));rows.append(f)
    except ValueError as e:excluded.append({'cell':cell,'sequence':seq,'reason':str(e)})
   print('Extracted',cell,flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'all_valid_pairs.csv',index=False)
 primary=df[df.primary_included].copy();counts=primary.groupby('cell').size();primary=primary[primary.cell.isin(counts[counts>=10].index)]
 primary.to_csv(out/'features.csv',index=False)
 meta={'cell','sequence','capacity_ah','temperature_c','relaxation_start_s','relaxation_end_s','discharge_start_s','relaxation_span_s','primary_included'}
 features=[c for c in primary if c not in meta]
 (out/'feature_names.json').write_text(json.dumps(features,indent=2))
 pd.DataFrame(excluded,columns=['cell','sequence','reason']).to_csv(out/'exclusions.csv',index=False)
 pd.DataFrame(inventory).to_csv(out/'source_files.csv',index=False)
 df.groupby(['cell','temperature_c']).agg(valid_cycles=('capacity_ah','size'),primary_range_cycles=('primary_included','sum')).to_csv(out/'all_cell_counts.csv')
 audit=primary.groupby('temperature_c').agg(cells=('cell','nunique'),cycles=('cell','size'));audit.to_csv(out/'audit_temperatures.csv')
 primary.groupby('cell').relaxation_span_s.agg(['min','median','max']).to_csv(out/'relaxation_timing_audit.csv')
 (out/'manifest.json').write_text(json.dumps({'dataset':'Zenodo 6405084, Dataset 1 NCA only','source_url':URL,'archive_md5':MD5,
  'archive_sha256':digest(path),'features_sha256':digest(out/'features.csv'),'extractor_sha256':digest(Path(__file__)),
  'nominal_ah':3.5,'source_temperature_c':25,'target_temperatures_c':sorted(set(primary.temperature_c)-{25}),
  'target':'Subsequent same-cycle discharge capacity (Ah)','range_filter':'2.5-3.5 Ah per authors Dataset 1 extraction; other valid cycles retained separately',
  'feature_rule':'First 14 contiguous pre-discharge open-circuit measurements above 4V after charging; |current|<=1mA',
  'authors_code_reference':'https://github.com/Yixiu-Wang/data-driven-capacity-estimation-from-voltage-relaxation/blob/main/Features_extraction_dataset1.py',
  'model_features':features},indent=2))
 print(audit.to_string());print('Primary cycles',len(primary),'features',len(features))
if __name__=='__main__':main()
