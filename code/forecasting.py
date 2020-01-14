from datetime import datetime

import numpy as np
from data import getNinjaPvApi
from keras.layers import LSTM, Dropout, Activation
from util import constructTimeStamps
import pandas as pd

from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)

# param
look_back = 5  # we have a 5 point history in our input
part = 0.6  # we train on part of the set

# input datas : uncontrolable resource : solar production
timestamps = constructTimeStamps(
    datetime.strptime("2014-01-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
    datetime.strptime("2014-02-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
    datetime.strptime("01:00:00", "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
)

metadata, df = getNinjaPvApi(
    52.5170, 13.3889, timestamps
)

split = int(len(df) * part)
print(split)

df_train = df["electricity"][look_back:split].reset_index(drop=True)
for i in range(1, look_back):
    s = df["electricity"][look_back - i:split-i].reset_index(drop=True)
    df_train = pd.concat([df_train, s], axis=1, ignore_index=True)

df_train_label = df["electricity"][look_back + 1:split + 1]

print(df_train.shape)
print(df_train_label.shape)
assert len(df_train) == len(df_train_label)

print(df_train)

df_test = df["electricity"][602:len(df) - 2]
df_test_lebel = df["electricity"][603:len(df) - 1]
#
# model = Sequential()
# model.add(LSTM(256, input_shape=(1, look_back)))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('tanh'))
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # training it
# model.fit(trainx, trainy, epochs=20, batch_size=50, verbose=2)
#
# # testing it
# loss, accuracy = model.evaluate(df_test, df_test_lebel, verbose=0)
# print("Accuracy = {:.2f}".format(accuracy))
#
# # plotting
# predict = model.predict(df_test)
#
# plt.plot(predict)
# plt.plot(df_test_lebel)
#
# plt.show()
