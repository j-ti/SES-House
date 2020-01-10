from datetime import datetime

import numpy as np
from data import getNinjaPvApi
from util import constructTimeStamps

from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)

# input datas : uncontrolable resource : solar production
timestamps = constructTimeStamps(
    datetime.strptime("2014-01-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
    datetime.strptime("2014-02-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
    datetime.strptime("01:00:00", "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
)

metadata, df_pvPower = getNinjaPvApi(
    52.5170, 13.3889, timestamps
)

df_train = df_pvPower["electricity"][:600]
df_train_label = df_pvPower["electricity"][1:601]

df_test = df_pvPower["electricity"][602:len(df_pvPower) -2]
df_test_lebel = df_pvPower["electricity"][603:len(df_pvPower) -1]


model = Sequential()
model.add(Dense(16, input_shape=(1,), activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])


# training it
model.fit(df_train, df_train_label, epochs=100, batch_size=1, verbose=0)

# testing it
loss, accuracy = model.evaluate(df_test, df_test_lebel, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

# plotting
predict = model.predict(df_test)

plt.plot(predict)
plt.plot(df_test_lebel)

plt.show()
