# -*- coding: utf-8 -*-

import os
os.chdir('E:/develop/projects/hbc')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, ensemble, metrics, grid_search, model_selection
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from matplotlib.pyplot import rcParams
plt.rcParams['font.sans-serif']=['SimHei']
rcParams['figure.figsize'] = 80, 10


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(initial_value=tf.random_normal([in_size, out_size]))
    biases = tf.Variable(initial_value=tf.zeros([1, out_size])+0.1)
    z = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = z
    else:
        outputs = activation_function(z)
    return outputs

def compute_accuracy(sess, v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


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
y_train = np.array(pd.get_dummies(y_train))

X_test = np.array(df_test[x_tags])

dim_in = X_train.shape[1]
dim_out = 2
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=dim_in))
model.add(Dense(100, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=1000)
score = model.evaluate(X_train, y_train, batch_size=128)
print(score)

# define placeholder for inputs network
dim_in = X_train.shape[1]
dim_out = 2
xs = tf.placeholder(tf.float32, [None, dim_in])
ys = tf.placeholder(tf.float32, [None, dim_out])  # 0-1

# add output layer
layer1 = add_layer(xs, dim_in, 100, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 100, dim_out, activation_function=tf.nn.softmax)

# the error between prediction and true value
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
  sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
  if i % 50 == 0:
      print(sess.run(cross_entropy, feed_dict={xs: X_train, ys: y_train}))

sess.close()
