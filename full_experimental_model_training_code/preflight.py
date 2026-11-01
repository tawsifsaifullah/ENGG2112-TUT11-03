import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent

def check():
 m=json.loads((R/'data/manifest.json').read_text());f=json.loads((R/'data/feature_names.json').read_text());p=R/'data/features.csv'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==m['features_sha256'],'Data checksum mismatch'
 assert f==m['model_features'],'Feature schema mismatch'
 d=pd.read_csv(p);assert len(d)==22239 and d.cell.nunique()==66,'Unexpected dataset size'
 assert np.isfinite(d[f+['capacity_ah']].to_numpy()).all(),'Nonfinite data'
 assert d.groupby('cell').temperature_c.nunique().eq(1).all()
 assert not d.duplicated(['cell','sequence']).any()
 assert d.groupby('temperature_c').cell.nunique().to_dict()=={25.:35,35.:3,45.:28}
 print('Verified Dataset 1: 66 cells, 22,239 cycles; checksum and grouping OK.',flush=True)
 return d,f
if __name__=='__main__':check()
