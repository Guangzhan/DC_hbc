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
import tensorflow as tf
plt.rcParams['font.sans-serif']=['SimHei']
#rcParams['figure.figsize'] = 80, 10

# load training&test set
df_train = pd.read_csv('./result/data_train.csv')
df_test = pd.read_csv('./result/data_test.csv')
idcol = 'userid'
target = 'futureOrderType'
drop_tags = [idcol, target]

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

x_tags = [x for x in df_train.columns if x not in drop_tags]
y_tag = target

X_train = np.array(df_train[x_tags])
y_train = np.array(df_train[y_tag])
X_test = np.array(df_test[x_tags])

# get last 20 actiontype and actiontime of one user, transform it into 2D data (20*2)
actionMat = []
for i in range(len(df_train)):
    actionMatofOneUser = []
    actiontime_last_1 = df_train.loc[i, 'actiontime_last_1']
    for j in range(1,21):
        tag_actiontype = 'actiontype_last_'+str(j)
        tag_actiontime = 'actiontime_last_'+str(j)
        actiontype = df_train.loc[i, tag_actiontype]
        actiontimespan = actiontime_last_1 - df_train.loc[i, tag_actiontime]
        oneAction = [actiontype, actiontimespan]
        actionMatofOneUser.append(oneAction)
    actionMat.append(actionMatofOneUser)
actionMat = np.array(actionMat)

dtrain_x = actionMat
dtrain_y = np.array(pd.get_dummies(y_train))

############# hyper parameters #############
lr = 0.001
training_iters = 1000
_batch_size = int(0.3 * len(dtrain_x))
############# hyper parameters #############

n_inputs = 2
n_steps = 20
n_hidden_units = 10    # neurons in hidden layer
n_classes = 2

# tf graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
batch_size = tf.placeholder(tf.int32, [])   # 1024

# define weights and biases
weights = {
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),    # (2x128)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))   # (128x2)
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),   # (128,)
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))        # (2,)
        }

### define get_batch_data() ###
def get_batch_data(x, y, batch_size):
    rand_indices = np.random.randint(0, len(x), batch_size)
    return x[rand_indices], y[rand_indices]

### define RNN ###
def RNN(X, weights, biases, batch_size):
    ## hidden layer for input to cell
    # X ( batch_size, 20 steps, 2 inputs ) -> ( batch_size * 20, 2 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  # X_in -> ( batch_size * 20 steps, 128 hidden )
    ## cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # LSTM cell is divided into 2 parts ( c_state, m_state )
    _init_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(
            cell=lstm_cell, inputs=X_in, initial_state=_init_state, time_major=False)
    ## hidden layer for outputs as result
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases, batch_size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# start session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step < training_iters:
#        batch_xs, batch_ys = dtrain_x, dtrain_y
        batch_xs, batch_ys = get_batch_data(dtrain_x, dtrain_y, _batch_size)
        #batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys, batch_size:_batch_size})
        if step % 10 == 0:
            print('step:\t', step)
            print('accuracy_batch:\t', sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, batch_size:_batch_size}))
            print('accuracy_all:\t', sess.run(accuracy, feed_dict={x:dtrain_x, y:dtrain_y, batch_size:len(dtrain_x)}))
            print('---')
        step+=1

sess.close()
        
        

    
