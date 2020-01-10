from datetime import datetime

import numpy as np
from data import getNinjaPvApi
from util import constructTimeStamps

from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


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

df_train = df_pvPower["electricity"][0:600]




# building the model
def createModel(inputLayer, lLayer):
    model = Sequential()
    model.add(Dense(inputLayer[1], input_shape=inputLayer[0], activation='sigmoid'))
    for foo in lLayer:
        if foo[0] == 'D':
            if foo[2] == "sig":
                model.add(Dense(foo[1], activation='sigmoid'))
            elif foo[2] == "soft":
                model.add(Dense(foo[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


model = createModel([(4,), 16], [('D', 16, 'sig'), ('D', 2, 'soft')])

# training it
model.fit(train_X, train_y_ohe, epochs=100, batch_size=1, verbose=0)

# testing it
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

# plotting

