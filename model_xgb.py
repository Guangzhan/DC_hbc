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
df_train = pd.read_csv('./result/data_train.csv')
df_test = pd.read_csv('./result/data_test.csv')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]

df_train = df_train.fillna(-999)
df_test = df_test.fillna(-999)

x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

#df_featImp_sorted = pd.read_csv('./feature/feat_2018-01-02-1.csv', encoding='gb2312')
#featcount = 200
#x_tags = df_featImp_sorted[:featcount]['tags'].tolist()

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
        learning_rate=0.01,
        n_estimators=3500,
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

##############################################################################################

### params tuning
# grid search on max_depth and min_child_weight
param_test1 = {
        'max_depth': [3,5,7,9],
        'min_child_weight': [1,3,5]
        }
gsearch1 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test1, scoring='roc_auc', iid=False, cv=5
        )
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {
        'max_depth':[4,5,6],
        'min_child_weight':[1,2]
}
gsearch2 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5
        )
gsearch2.fit(X_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

param_test2b = {
        'min_child_weight':[6,8,10]
}
gsearch2b = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=7,
                                min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=314),
        param_grid=param_test2b, scoring='roc_auc', iid=False, cv=5
        )
gsearch2b.fit(X_train, y_train)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_

# tune gamma
param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test3, scoring='roc_auc', iid=False, cv=5
        )
gsearch3.fit(X_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

model_xgb2 = XGBClassifier(
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
modelfit(model_xgb2, X_train, y_train)

# tune subsample and colsample_bytree
param_test4 = {
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)]
}
gsearch4 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test4, scoring='roc_auc', iid=False, cv=5
        )
gsearch4.fit(X_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

param_test5 = {
        'subsample': [0.95],
        'colsample_bytree': [0.25,0.3,0.35]
}
gsearch5 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.3,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test5, scoring='roc_auc', iid=False, cv=5
        )
gsearch5.fit(X_train, y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# tuning Regularization Parameters
param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.3,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test6, scoring='roc_auc', iid=False, cv=5
        )
gsearch6.fit(X_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

param_test7 = {
        'reg_alpha':[0.3, 0.5, 0.7]
}
gsearch7 = grid_search.GridSearchCV(
        estimator=XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=5,
                                min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.3,
                                objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=314),
        param_grid=param_test7, scoring='roc_auc', iid=False, cv=5
        )
gsearch7.fit(X_train, y_train)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

model_xgb3 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.95,
        colsample_bytree=0.3,
        reg_alpha=0,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb3, X_train, y_train)
scores_cross = model_selection.cross_val_score(model_xgb3, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# reducing Learning Rate
model_xgb4 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.95,
        colsample_bytree=0.3,
        reg_alpha=0,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=314
        )
modelfit(model_xgb4, X_train, y_train)
scores_cross = model_selection.cross_val_score(model_xgb4, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

### final model
model_xgb_final = model_xgb1
#model_xgb_final.fit(X_train, y_train)
y_train_pred = model_xgb_final.predict(X_train)
y_train_predprob = model_xgb_final.predict_proba(X_train)[:,1]
importances = model_xgb_final.feature_importances_
df_featImp = pd.DataFrame({'tags': x_tags, 'importance': importances})
df_featImp_sorted = df_featImp.sort_values(by=['importance'], ascending=False)
#df_featImp_sorted.plot(x='tags', y='importance', kind='bar')
df_featImp_sorted.to_csv('./feature/feat_2018-01-29-1.csv', index=False)
featcount = 267
x_tags = df_featImp_sorted[:featcount]['tags'].tolist()


print('score_AUC:', round(metrics.roc_auc_score(y_train, y_train_predprob), 5))
print('score_precision:', round(metrics.accuracy_score(y_train, y_train_pred), 5))
scores_cross = model_selection.cross_val_score(model_xgb_final, X_train, y_train, cv=5, scoring='roc_auc')
print('score_cross:', round(np.mean(scores_cross), 5), 'std:', round(np.std(scores_cross), 5))

# write out prediction result
y_test_pred = model_xgb_final.predict_proba(X_test)[:,1]

df_profile = pd.read_csv('./data_train_test/userProfile_test.csv')
restable = pd.DataFrame(np.concatenate((np.array(df_profile['userid']).reshape((-1,1)), y_test_pred.reshape((-1,1))), axis=1))
#restable = pd.DataFrame(np.concatenate((np.array(df_profile['userid']).reshape((-1,1)), y_test_pred_super.reshape((-1,1))), axis=1))
restable.loc[:,0] = restable.loc[:,0].astype(np.int64)
pd.DataFrame(restable).to_csv("./result/orderFuture_test-20180112-4.csv", header=['userid','orderType'], index=False)
