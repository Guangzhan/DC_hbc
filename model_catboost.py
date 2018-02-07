# -*- coding: utf-8 -*-

import os
os.chdir('E:/develop/projects/hbc')

import numpy as np
from catboost import CatBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, ensemble, metrics, grid_search, model_selection
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# load training&test set
df_train = pd.read_csv('./result/data_train.csv', encoding='gb2312')
df_test = pd.read_csv('./result/data_test.csv', encoding='gb2312')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]
#drop_tags = [idcol, target,
#             'actiontype_last_1', 'actiontype_last_2', 'actiontype_last_3', 'actiontype_last_4', 'actiontype_last_5', 'actiontype_last_6', 'actiontype_last_7', 'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
#             'actiontime_last_1', 'actiontime_last_2', 'actiontime_last_3', 'actiontime_last_4', 'actiontime_last_5', 'actiontime_last_6', 'actiontime_last_7', 'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20']
x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])

X_test = np.array(df_test[x_tags])


# specify the training parameters 
model_cb = CatBoostClassifier()
#train the model
model_cb.fit(X_train, y_train)
# make the prediction using the resulting model
y_train_pred = model_cb.predict(X_train)
y_train_predprob = model_cb.predict_proba(X_train)[:,1]
print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
scores_cross = model_selection.cross_val_score(model_cb, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# write out prediction result
y_test_predprob = model_cb.predict_proba(X_test)[:,1]
df_profile = pd.read_csv('./data_train_test/userProfile_test.csv')
restable = pd.DataFrame(np.concatenate((np.array(df_profile['userid']).reshape((-1,1)), y_test_predprob.reshape((-1,1))), axis=1))
restable.loc[:,0] = restable.loc[:,0].astype(np.int64)
pd.DataFrame(restable).to_csv("./result/orderFuture_test-20171226-3.csv", header=['userid','orderType'], index=False)
