import json,time
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ExactConstantFilter(BaseEstimator, TransformerMixin):
    """Fit on training data only; exact equality avoids removing small voltage variance."""
    def fit(self,X,y=None):
        X=np.asarray(X);self.keep_=np.any(X != X[0],axis=0)
        if not self.keep_.any(): raise ValueError("All training predictors are constant")
        return self
    def transform(self,X): return np.asarray(X)[:,self.keep_]

def preprocessing(cfg):
    steps=[SimpleImputer(keep_empty_features=True)]
    if cfg.get("drop_constant"): steps.append(ExactConstantFilter())
    return steps+[StandardScaler()]

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor

SPECS={
'mean':('Mean baseline','Establish whether any learned relationship beats the training-set average. It cannot track aging or temperature.',[{}]),
'ridge':('Ridge regression','Test whether regularised linear voltage-relaxation relationships are sufficient. Scaling and regularisation handle correlated measurements.',[{'alpha':1.0},{'alpha':100.0}]),
'svr':('RBF support vector regression','Test smooth nonlinear relationships on this modest dataset. Scaling matters; extrapolation to new temperatures can fail.',[{'C':1.0},{'C':10.0}]),
'random_forest':('Random Forest','Test nonlinear interactions with bagged trees. Averaging reduces variance, but trees extrapolate poorly.',[{'min_samples_leaf':2},{'min_samples_leaf':8}]),
'extra_trees':('Extra Trees','Test whether more randomised tree splits improve generalisation and speed relative to Random Forest.',[{'min_samples_leaf':2},{'min_samples_leaf':8}]),
'hist_boost':('Histogram gradient boosting','Test sequential correction of residual errors, matching the proposal\'s boosted-tree idea. Complexity can overfit a small number of cells.',[{'max_leaf_nodes':7},{'max_leaf_nodes':15}]),
'mlp_small':('Small neural network','Test learned nonlinear combinations with a modest 64/32 architecture. Compare held-out-cell error with training loss to assess overfitting.',[{'weight_decay':0.001},{'weight_decay':0.01}]),
'mlp_large':('Larger neural network','Test whether a 256/128/64 network improves over the small network. More parameters can lower training error without improving new-cell performance.',[{'weight_decay':0.001},{'weight_decay':0.01}]),
}

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {'mae_ah':float(np.mean(abs(y-p))),'rmse_ah':float(np.sqrt(np.mean((y-p)**2))),
            'bias_ah':float(np.mean(p-y)), 'soh_mae_pp':float(100/3.5*np.mean(abs(y-p)))}

class Neural:
    def __init__(self,name,params,cfg):self.name=name;self.params=params;self.cfg=cfg
    def fit(self,X,y,valid=None,emit=None,epochs=None):
        import torch
        from torch import nn
        torch.set_num_threads(4);torch.manual_seed(self.cfg['seed'])
        self.device='cuda' if torch.cuda.is_available() and not self.cfg.get('force_cpu') else 'cpu'
        if self.device=='cuda':torch.cuda.manual_seed_all(self.cfg['seed'])
        self.pre=make_pipeline(*preprocessing(self.cfg))
        x=self.pre.fit_transform(X).astype('float32');self.mu=float(np.mean(y));self.sd=max(float(np.std(y)),1e-6)
        self.widths=[64,32] if self.name=='mlp_small' else [256,128,64]
        layers=[];n=x.shape[1]
        for w in self.widths:layers.extend([nn.Linear(n,w),nn.ReLU()]);n=w
        layers.append(nn.Linear(n,1));self.net=nn.Sequential(*layers).to(self.device)
        xt=torch.tensor(x,device=self.device);yt=torch.tensor(((y-self.mu)/self.sd).astype('float32')[:,None],device=self.device)
        opt=torch.optim.AdamW(self.net.parameters(),lr=0.001,weight_decay=self.params['weight_decay'])
        rng=np.random.default_rng(self.cfg['seed']);best=float('inf');state=None;stale=0;self.best_epoch=1
        if valid is not None:
            xv=torch.tensor(self.pre.transform(valid[0]).astype('float32'),device=self.device)
        for epoch in range(1,(epochs or self.cfg['epochs'])+1):
            self.net.train()
            for idx in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/256)))):
                idx=torch.tensor(idx,device=self.device);opt.zero_grad(set_to_none=True)
                loss=((self.net(xt[idx])-yt[idx])**2).mean()
                if not torch.isfinite(loss):raise RuntimeError('Neural loss became nonfinite')
                loss.backward();opt.step()
            self.net.eval()
            with torch.no_grad():
                pred=self.net(xt).cpu().numpy().ravel()*self.sd+self.mu
                row={'epoch':epoch,'train_mae_ah':metrics(y,pred)['mae_ah'],'train_mse_ah2':float(np.mean((y-pred)**2))}
                if valid is not None:
                    vp=self.net(xv).cpu().numpy().ravel()*self.sd+self.mu
                    vm=metrics(valid[1],vp)['mae_ah'];row['validation_mae_ah']=vm
                    if vm<best-1e-7:
                        best=vm;self.best_epoch=epoch;stale=0
                        state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
                    else:stale+=1
                else:self.best_epoch=epoch
            if emit:emit(row)
            if valid is not None and stale>=25:break
        if state is not None:self.net.load_state_dict(state)
        return self
    def predict(self,X):
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(self.pre.transform(X).astype('float32'),device=self.device)).cpu().numpy().ravel()*self.sd+self.mu
    def save(self,path,features):
        import torch
        path=Path(path);path.mkdir(exist_ok=True)
        joblib.dump(self.pre,path/'preprocessor.joblib')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in self.net.state_dict().items()},'widths':self.widths,
                    'input_dim':self.net[0].in_features,'mu':self.mu,'sd':self.sd},path/'network.pt')
        (path/'features.json').write_text(json.dumps(features))

def fit(name,params,cfg,X,y,valid=None,emit=None,epochs=None):
    if name.startswith('mlp'):return Neural(name,params,cfg).fit(X,y,valid,emit,epochs)
    if name=='mean':m=DummyRegressor()
    elif name=='ridge':m=Ridge(**params)
    elif name=='svr':m=SVR(epsilon=0.02,**params)
    elif name=='random_forest':m=RandomForestRegressor(n_estimators=cfg['trees'],random_state=cfg['seed'],n_jobs=4,**params)
    elif name=='extra_trees':m=ExtraTreesRegressor(n_estimators=cfg['trees'],random_state=cfg['seed'],n_jobs=4,**params)
    elif name=='hist_boost':m=HistGradientBoostingRegressor(max_iter=cfg['boost_iterations'],learning_rate=0.05,l2_regularization=1,early_stopping=False,random_state=cfg['seed'],**params)
    else:raise ValueError(name)
    m=make_pipeline(*preprocessing(cfg),m).fit(X,y)
    if emit:
        row={'train_mae_ah':metrics(y,m.predict(X))['mae_ah']}
        if valid is not None:row['validation_mae_ah']=metrics(valid[1],m.predict(valid[0]))['mae_ah']
        emit(row)
    return m
