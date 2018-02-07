#-*- coding:utf-8 -*-

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

# load training&test set
df_train = pd.read_csv('./result/data_train.csv', encoding='gb2312')
df_test = pd.read_csv('./result/data_test.csv', encoding='gb2312')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]
drop_tags = [idcol, target,
             'actiontype_last_8', 'actiontype_last_9', 'actiontype_last_10', 'actiontype_last_11', 'actiontype_last_12', 'actiontype_last_13', 'actiontype_last_14', 'actiontype_last_15', 'actiontype_last_16', 'actiontype_last_17', 'actiontype_last_18', 'actiontype_last_19', 'actiontype_last_20',
             'actiontime_last_8', 'actiontime_last_9', 'actiontime_last_10', 'actiontime_last_11', 'actiontime_last_12', 'actiontime_last_13', 'actiontime_last_14', 'actiontime_last_15', 'actiontime_last_16', 'actiontime_last_17', 'actiontime_last_18', 'actiontime_last_19', 'actiontime_last_20']
x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

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
    # plot feature importance
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    
# prediction model
# params tuning
# find n_estimators for a high learning rate
np.random.seed(314)
model_xgb = XGBClassifier(
        learning_rate=0.03,
        n_estimators=2000,
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
modelfit(model_xgb, X_train, y_train)
model_xgb.fit(X_train, y_train)
print('score_AUC:', round(metrics.roc_auc_score(y_train, model_xgb.predict_proba(X_train)[:,1]), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, model_xgb.predict(X_train)), 5))
scores_cross = model_selection.cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# xgboost编码原有特征
X_train_leaves = model_xgb.apply(X_train)
# 训练样本个数
train_rows = X_train_leaves.shape[0]
# 合并编码后的训练数据和测试数据
X_train_leaves = X_train_leaves.astype(np.int32)
(rows, cols) = X_train_leaves.shape
# 记录每棵树的编码区间
cum_count = np.zeros((1, cols), dtype=np.int32)
for j in range(cols):
    if j == 0:
        cum_count[0][j] = len(np.unique(X_train_leaves[:, j]))
    else:
        cum_count[0][j] = len(np.unique(X_train_leaves[:, j])) + cum_count[0][j-1]
print('Transform features genenrated by xgboost...')
# 对所有特征进行ont-hot编码
for j in range(cols):
    keyMapDict = {}
    if j == 0:
        initial_index = 1
    else:
        initial_index = cum_count[0][j-1]+1
    for i in range(rows):
        if X_train_leaves[i, j] not in keyMapDict:
            keyMapDict[X_train_leaves[i, j]] = initial_index
            X_train_leaves[i, j] = initial_index
            initial_index = initial_index + 1
        else:
            X_train_leaves[i, j] = keyMapDict[X_train_leaves[i, j]]

# 基于编码后的特征，将特征处理为libsvm格式且写入文件
print('Write xgboost learned features to file ...')
xgbFeatureLibsvm = open('./xgb_feature_libsvm', 'w')
for i in range(rows):
    if i < train_rows:
        xgbFeatureLibsvm.write(str(y_train[i]))
    for j in range(cols):
        xgbFeatureLibsvm.write(' '+str(X_train_leaves[i, j])+':1.0')
    xgbFeatureLibsvm.write('\n')
xgbFeatureLibsvm.close()

### lr start
X_train_xg = X_train_leaves
y_train_xg = y_train
# lr对load xgboost特征编码后的样本模型训练
lr = linear_model.LogisticRegression(C=0.1, penalty='l1')
lr.fit(X_train_xg, y_train_xg)
y_pred_train = lr.predict_proba(X_train_xg)[:, 1]
lr_test_auc = metrics.roc_auc_score(y_train_xg, y_pred_train)
print('基于Xgboost特征编码后的LR AUC: %.5f' % lr_test_auc)
