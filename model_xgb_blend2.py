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
plt.rcParams['font.sans-serif']=['SimHei']
rcParams['figure.figsize'] = 80, 10

# load training&test set
df_train = pd.read_csv('./result/data_train_m.csv')
df_test = pd.read_csv('./result/data_test_m.csv')
df_train = pd.read_csv('./result/data_train.csv')
df_test = pd.read_csv('./result/data_test.csv')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]

df_train = df_train.fillna(-999)
df_test = df_test.fillna(-999)

x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

df_featImp_sorted = pd.read_csv('./feature/feat_2018-01-02-1.csv', encoding='gb2312')
featcount = 200
x_tags = df_featImp_sorted[:featcount]['tags'].tolist()

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])
X_test = np.array(df_test[x_tags])


def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print('best n_estimators:', cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    # fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')
    # predict train set
    dtrain_pred = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    # print model report
    print('\nModel Report:')
    print("AUC Score (Train): %.5g" % metrics.roc_auc_score(y_train, dtrain_predprob))
    print("Accuracy: %.5g" % metrics.accuracy_score(y_train, dtrain_pred))

model_xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb1, X_train, y_train)
#model_xgb1.fit(X_train, y_train, eval_metric='auc')
print('score_AUC:', round(metrics.roc_auc_score(y_train, model_xgb1.predict_proba(X_train)[:,1]), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, model_xgb1.predict(X_train)), 5))
scores_cross = model_selection.cross_val_score(model_xgb1, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# split X_train and y_train
model_xgb_sub = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=314
        )
skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=314)
y_train_pred_list = []
y_test_pred_list = []
y_vali_pred_sub_list = []
for train_index, vali_index in skf.split(X_train, y_train):
   X_train_sub = X_train[train_index]
   y_train_sub = y_train[train_index]
   modelfit(model_xgb_sub, X_train_sub, y_train_sub)
   y_train_pred_sub = model_xgb_sub.predict_proba(X_train)[:,1]
   y_train_pred_list.append(y_train_pred_sub)
   y_test_pred_sub = model_xgb_sub.predict_proba(X_test)[:,1]
   y_test_pred_list.append(y_test_pred_sub)
   X_vali_sub = X_train[vali_index]
   y_vali_sub = y_train[vali_index]
   y_vali_pred_sub = model_xgb_sub.predict_proba(X_vali_sub)[:,1]
   y_vali_pred_sub_list.append(y_vali_pred_sub)
   print('score_AUC_vali:', round(metrics.roc_auc_score(y_vali_sub, y_vali_pred_sub), 5))
   
y_train_pred_final = np.zeros(len(y_train))
for i in range(len(y_train)):
    for j in range(len(y_train_pred_list)):
        y_train_pred_final[i] += y_train_pred_list[j][i]
y_train_pred_final = y_train_pred_final / len(y_train_pred_list)
print('score_AUC_train_final:', round(metrics.roc_auc_score(y_train, y_train_pred_final), 5))

y_test_pred_final = np.zeros(len(X_test))
for i in range(len(X_test)):
    for j in range(len(y_test_pred_list)):
        y_test_pred_final[i] += y_test_pred_list[j][i]
y_test_pred_final = y_test_pred_final / len(y_test_pred_list)
        
        
df_profile = pd.read_csv('./data_train_test/userProfile_test.csv')
restable = pd.DataFrame(np.concatenate((np.array(df_profile['userid']).reshape((-1,1)), y_test_pred_final.reshape((-1,1))), axis=1))
restable.loc[:,0] = restable.loc[:,0].astype(np.int64)
pd.DataFrame(restable).to_csv("./result/orderFuture_test-20180112-1.csv", header=['userid','orderType'], index=False)







