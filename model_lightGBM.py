# -*- coding: utf-8 -*-

import os
os.chdir('E:/develop/projects/hbc')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, ensemble, metrics, grid_search, model_selection, decomposition, linear_model
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from matplotlib.pyplot import rcParams
from sklearn.datasets import load_svmlight_file
import lightgbm as lgb
plt.rcParams['font.sans-serif']=['SimHei']
rcParams['figure.figsize'] = 80, 10

# load training&test set
df_train = pd.read_csv('./result/data_train_1.csv')
df_test = pd.read_csv('./result/data_test.csv')

idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]
df_train = df_train.fillna(-999)
df_test = df_test.fillna(-999)

x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])
X_test = np.array(df_test[x_tags])

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    #print('\n-------------------------------------')
    # fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')
    # predict train set
    #dtrain_pred = alg.predict(X_train)
    #dtrain_predprob = alg.predict_proba(X_train)[:,1]
    # print model report
    #print('score_auc:', round(metrics.roc_auc_score(y_train, dtrain_predprob), 5))
    #print('score_precision:', round(metrics.accuracy_score(y_train, dtrain_pred), 5))
    if useTrainCV:
        np.random.seed(314)
        scores_cv_auc = model_selection.cross_val_score(alg, X_train, y_train, cv=5, scoring='roc_auc')
        #print('score_cv_auc:', round(np.mean(scores_cv_auc), 5), 'std:', round(np.std(scores_cv_auc), 5))
    #print('-------------------------------------\n')
    return np.mean(scores_cv_auc)

model_gbm = lgb.LGBMClassifier(
        random_state=314,
        learning_rate=0.1,
        n_estimators=300,
        colsample_bytree=0.8,
        subsample=0.9
        )
print(modelfit(model_gbm, X_train, y_train))
