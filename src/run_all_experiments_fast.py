"""Fast experiment runner used to generate report tables."""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common import *
warnings.filterwarnings('ignore')
ROOT=project_root(); RES=ROOT/'results'; RES.mkdir(exist_ok=True)
def pp(scale=False):
    s=[('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold())]
    if scale: s.append(('scaler',StandardScaler()))
    return s
def reg_list():
    return {
      'DummyMean':Pipeline(pp(False)+[('model',DummyRegressor())]),
      'Ridge_log_y':TransformedTargetRegressor(regressor=Pipeline(pp(True)+[('model',Ridge(alpha=10))]), func=np.log1p, inverse_func=np.expm1),
      'RandomForest':Pipeline(pp(False)+[('model',RandomForestRegressor(n_estimators=40,min_samples_leaf=2,max_features='sqrt',random_state=RANDOM_STATE,n_jobs=1))]),
      'GradientBoosting':Pipeline(pp(False)+[('model',GradientBoostingRegressor(n_estimators=80,learning_rate=.05,max_depth=2,random_state=RANDOM_STATE))]),
    }
def clf_list():
    return {
      'DummyMostFrequent':Pipeline(pp(False)+[('model',DummyClassifier(strategy='most_frequent'))]),
      'LogisticRegression':Pipeline(pp(True)+[('model',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=RANDOM_STATE))]),
      'RandomForest':Pipeline(pp(False)+[('model',RandomForestClassifier(n_estimators=60,class_weight='balanced',max_features='sqrt',random_state=RANDOM_STATE,n_jobs=1))]),
      'GradientBoosting':Pipeline(pp(False)+[('model',GradientBoostingClassifier(n_estimators=80,learning_rate=.05,max_depth=2,random_state=RANDOM_STATE))]),
    }
def tune_gbr(Xtr,ytr,Xv,yv):
    params=[(60,.05,2),(100,.05,2),(80,.08,2),(100,.05,3)]
    best=None; rows=[]
    for n,lr,md in params:
        m=Pipeline(pp(False)+[('model',GradientBoostingRegressor(n_estimators=n,learning_rate=lr,max_depth=md,random_state=RANDOM_STATE))])
        m.fit(Xtr,ytr); pred=np.maximum(m.predict(Xv),0); met=regression_metrics(yv,pred); met.update({'n_estimators':n,'learning_rate':lr,'max_depth':md}); rows.append(met)
        if best is None or met['RMSE']<best[0]['RMSE']: best=(met,m)
    return best, pd.DataFrame(rows).sort_values('RMSE')
def tune_gbc(Xtr,ytr,Xv,yv):
    params=[(60,.05,1),(80,.05,2),(100,.05,2),(80,.08,2)]
    best=None; rows=[]
    for n,lr,md in params:
        m=Pipeline(pp(False)+[('model',GradientBoostingClassifier(n_estimators=n,learning_rate=lr,max_depth=md,random_state=RANDOM_STATE))])
        m.fit(Xtr,ytr); pred=m.predict(Xv); met=classification_metrics(yv,pred); met.update({'n_estimators':n,'learning_rate':lr,'max_depth':md}); rows.append(met)
        if best is None or met['F1']>best[0]['F1']: best=(met,m)
    return best, pd.DataFrame(rows).sort_values('F1',ascending=False)
def eval_reg(df,target):
    X,y=split_features_target(df,target); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
    rows=[]
    for name,m in reg_list().items():
        m.fit(Xtr,ytr); pred=np.maximum(m.predict(Xte),0); r={'task':f'regression_{target}','model':name}; r.update(regression_metrics(yte,pred)); rows.append(r)
    Xfit,Xval,yfit,yval=train_test_split(Xtr,ytr,test_size=.25,random_state=RANDOM_STATE)
    (best_met,best_m), tune_df=tune_gbr(Xfit,yfit,Xval,yval)
    best_m.fit(Xtr,ytr); pred=np.maximum(best_m.predict(Xte),0); r={'task':f'regression_{target}','model':'Tuned_GradientBoosting'}; r.update(regression_metrics(yte,pred)); rows.append(r)
    out=pd.DataFrame(rows).sort_values('RMSE')
    short=target.replace(', mM','').lower(); out.to_csv(RES/f'regression_{short}_metrics.csv',index=False); tune_df.to_csv(RES/f'regression_{short}_tuning.csv',index=False)
    return out.iloc[0].to_dict()
def eval_clf(df,target,mode,suffix):
    X,yr=split_features_target(df,target); y=make_binary_target(yr,mode); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE,stratify=y)
    rows=[]
    for name,m in clf_list().items():
        m.fit(Xtr,ytr); pred=m.predict(Xte); r={'task':suffix,'model':name}; r.update(classification_metrics(yte,pred)); rows.append(r)
    Xfit,Xval,yfit,yval=train_test_split(Xtr,ytr,test_size=.25,random_state=RANDOM_STATE,stratify=ytr)
    (best_met,best_m), tune_df=tune_gbc(Xfit,yfit,Xval,yval)
    best_m.fit(Xtr,ytr); pred=best_m.predict(Xte); r={'task':suffix,'model':'Tuned_GradientBoosting'}; r.update(classification_metrics(yte,pred)); rows.append(r)
    out=pd.DataFrame(rows).sort_values('F1',ascending=False); out.to_csv(RES/f'classification_{suffix}_metrics.csv',index=False); tune_df.to_csv(RES/f'classification_{suffix}_tuning.csv',index=False)
    return out.iloc[0].to_dict()
def main():
    df=load_data(); summ=[]
    for t in ['IC50, mM','CC50, mM','SI']:
        print('reg',t,flush=True); summ.append(eval_reg(df,t))
    for t,m,s in [('IC50, mM','median','ic50_gt_median'),('CC50, mM','median','cc50_gt_median'),('SI','median','si_gt_median'),('SI','si_gt_8','si_gt_8')]:
        print('clf',s,flush=True); summ.append(eval_clf(df,t,m,s))
    pd.DataFrame(summ).to_csv(RES/'summary_best_models.csv',index=False); print(pd.DataFrame(summ))
if __name__=='__main__': main()
