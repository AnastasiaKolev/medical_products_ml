"""Run all regression and classification experiments for the coursework.

This script performs train/test evaluation and a compact hyperparameter search.
The search grids are intentionally small so the code is reproducible on a laptop.
For deeper research, increase the grids in PARAM_GRIDS below.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from common import RANDOM_STATE, classification_metrics, load_data, make_binary_target, project_root, regression_metrics, split_features_target

warnings.filterwarnings("ignore")
ROOT = project_root()
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def pp(scale=False):
    steps=[("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
    if scale: steps.append(("scaler", StandardScaler()))
    return steps


def reg_models():
    return {
        "DummyMean": Pipeline(pp(False)+[("model", DummyRegressor())]),
        "Ridge_log_y": TransformedTargetRegressor(regressor=Pipeline(pp(True)+[("model", Ridge(alpha=10.0))]), func=np.log1p, inverse_func=np.expm1),
        "SVR_RBF_log_y": TransformedTargetRegressor(regressor=Pipeline(pp(True)+[("model", SVR(C=5.0, epsilon=0.1, gamma="scale"))]), func=np.log1p, inverse_func=np.expm1),
        "RandomForest": Pipeline(pp(False)+[("model", RandomForestRegressor(n_estimators=120, max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1))]),
        "GradientBoosting": Pipeline(pp(False)+[("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE))]),
    }


def clf_models():
    return {
        "DummyMostFrequent": Pipeline(pp(False)+[("model", DummyClassifier(strategy="most_frequent"))]),
        "LogisticRegression": Pipeline(pp(True)+[("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "SVC_RBF": Pipeline(pp(True)+[("model", SVC(C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline(pp(False)+[("model", RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"))]),
        "GradientBoosting": Pipeline(pp(False)+[("model", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE))]),
    }


def tune_reg(X_train, y_train):
    estimator=Pipeline(pp(False)+[("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    grid={"model__n_estimators":[100,180,260],"model__learning_rate":[0.03,0.05,0.08],"model__max_depth":[2,3],"model__min_samples_leaf":[1,3,5]}
    search=GridSearchCV(estimator, grid, cv=KFold(3, shuffle=True, random_state=RANDOM_STATE), scoring="neg_root_mean_squared_error", n_jobs=1)
    search.fit(X_train, y_train)
    return search


def tune_clf(X_train, y_train):
    estimator=Pipeline(pp(False)+[("model", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    grid={"model__n_estimators":[100,180,260],"model__learning_rate":[0.03,0.05,0.08],"model__max_depth":[1,2,3],"model__min_samples_leaf":[1,3,5]}
    search=GridSearchCV(estimator, grid, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), scoring="f1", n_jobs=1)
    search.fit(X_train, y_train)
    return search


def evaluate_regression(df, target):
    X,y=split_features_target(df,target)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE)
    rows=[]
    for name,model in reg_models().items():
        model.fit(X_train,y_train)
        pred=np.maximum(model.predict(X_test),0)
        row={"task":f"regression_{target}","model":name}; row.update(regression_metrics(y_test,pred)); rows.append(row)
    tuned=tune_reg(X_train,y_train)
    pred=np.maximum(tuned.predict(X_test),0)
    row={"task":f"regression_{target}","model":"Tuned_GradientBoosting"}; row.update(regression_metrics(y_test,pred)); rows.append(row)
    out=pd.DataFrame(rows).sort_values("RMSE")
    short=target.replace(", mM","").lower()
    out.to_csv(RESULTS_DIR/f"regression_{short}_metrics.csv",index=False)
    json.dump(tuned.best_params_, open(RESULTS_DIR/f"regression_{short}_best_params.json","w"), indent=2)
    return out.iloc[0].to_dict()


def evaluate_classification(df,target,mode,suffix):
    X,y_raw=split_features_target(df,target); y=make_binary_target(y_raw,mode)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
    rows=[]
    for name,model in clf_models().items():
        model.fit(X_train,y_train); pred=model.predict(X_test)
        proba=None
        if hasattr(model,'predict_proba'):
            try: proba=model.predict_proba(X_test)[:,1]
            except Exception: proba=None
        row={"task":suffix,"model":name}; row.update(classification_metrics(y_test,pred,proba)); rows.append(row)
    tuned=tune_clf(X_train,y_train); pred=tuned.predict(X_test)
    proba=tuned.predict_proba(X_test)[:,1]
    row={"task":suffix,"model":"Tuned_GradientBoosting"}; row.update(classification_metrics(y_test,pred,proba)); rows.append(row)
    out=pd.DataFrame(rows).sort_values("F1",ascending=False)
    out.to_csv(RESULTS_DIR/f"classification_{suffix}_metrics.csv",index=False)
    json.dump(tuned.best_params_, open(RESULTS_DIR/f"classification_{suffix}_best_params.json","w"), indent=2)
    return out.iloc[0].to_dict()


def main():
    df=load_data(); summary=[]
    for target in ["IC50, mM","CC50, mM","SI"]:
        print('Regression',target,flush=True); summary.append(evaluate_regression(df,target))
    for target,mode,suffix in [("IC50, mM","median","ic50_gt_median"),("CC50, mM","median","cc50_gt_median"),("SI","median","si_gt_median"),("SI","si_gt_8","si_gt_8")]:
        print('Classification',suffix,flush=True); summary.append(evaluate_classification(df,target,mode,suffix))
    pd.DataFrame(summary).to_csv(RESULTS_DIR/'summary_best_models.csv',index=False)
    print(pd.DataFrame(summary))
if __name__=='__main__': main()
