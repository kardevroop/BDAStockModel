# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:50:54 2019

@author: user
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
%matplotlib qt
import math, json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model, model_from_json

pathname = '.\\Data\\Daily\\daily_MSFT.csv'

dataset = pd.read_csv(pathname)
dataset = dataset.sort_values(by = 'timestamp')
dataset = dataset.drop(columns = ['timestamp'])
'''
dataset.index = dataset['timestamp']
plt.figure(figsize=(16,8))
plt.plot(dataset['close'], label='Close Price history')
'''
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#X = dataset.drop(columns = ['close']).values
#y = dataset.iloc[:, -2].values

y = dataset[:, -2]
X = np.delete(dataset, -2, axis = 1)

train_len = int(0.8 * len(X))
test_len = len(X) - train_len
X_train, X_test = X[0:train_len, :], X[train_len:len(X), :]
y_train, y_test = y[0:train_len], y[train_len:len(X)]

X_train_ = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test_ = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

model = Sequential()
model.add(LSTM(units=10, input_shape=(X_train.shape[1],1)))
#model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_, y_train, epochs=100, batch_size=1, verbose=2)

#Loading model 
with open('model.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model.h5')

scores = model.evaluate(X_test_, y_test, verbose = 0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

train_pred = model.predict(X_train_)
#X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
temp = np.concatenate((X_train[:,:-1], train_pred, np.reshape(X_train[:,-1], (X_train.shape[0], 1))), axis = 1)
temp = scaler.inverse_transform(temp)
train_pred = temp[:,-2]

test_pred = model.predict(X_test_)
#X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
temp = np.concatenate((X_test[:,:-1], test_pred, np.reshape(X_test[:,-1], (X_test.shape[0], 1))), axis = 1)
temp = scaler.inverse_transform(temp)
test_pred = temp[:,-2]

y_t = scaler.inverse_transform(dataset)[:,-2]

trainPredictPlot = np.zeros((X.shape[0], 1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:len(train_pred), :] = np.reshape(train_pred, (train_pred.shape[0], 1))
# shift test predictions for plotting
testPredictPlot = np.zeros((X.shape[0], 1))
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_pred):len(dataset), :] =  np.reshape(test_pred, (test_pred.shape[0], 1))

plt.figure(figsize=(16,8))
true, = plt.plot(y_t, label='Close Price history')
tr_pred, = plt.plot(trainPredictPlot, label='Train predictions')
tst_pred, = plt.plot(testPredictPlot, label='Test predictions')
plt.legend(handles=[true, tr_pred, tst_pred])
plt.show()
#plt.savefig('result.png')

data = pd.read_csv(pathname)
data = data.sort_values(by = 'timestamp').values
y = data[:, -2]
y_train_, y_test_ = y[0:train_len], y[train_len:len(X)]
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, test_pred)
print(str(score))

#Saving model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json.dump(model_json, json_file)
# serialize weights to HDF5
model.save_weights("model.h5")