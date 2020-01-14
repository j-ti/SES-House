import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import getNinjaPvApi
from keras.engine.saving import model_from_json
from keras.layers import LSTM, Dropout, Activation
from keras.layers.core import Dense
from keras.models import Sequential
from util import constructTimeStamps
from util import makeShiftTest, makeShiftTrain

# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)
# param
look_back = 5  # we have a 5 point history in our input
part = 0.6  # we train on part of the set


def dataImport():
    timestamps = constructTimeStamps(
        datetime.strptime("2014-01-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
        datetime.strptime("2014-02-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
        datetime.strptime("01:00:00", "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )
    # input datas : uncontrolable resource : solar production
    _, df = getNinjaPvApi(
        52.5170, 13.3889, timestamps
    )
    return df


def buildSet(df, split):
    df_train = df["electricity"][look_back:split].reset_index(drop=True)
    df_train = makeShiftTrain(df, df_train, "electricity", look_back, split)
    df_train_label = df["electricity"][look_back + 1:split + 1]

    df_test = df["electricity"][split + look_back:].reset_index(drop=True)
    df_test = makeShiftTest(df, df_test, "electricity", look_back, split)
    df_test_label = df["electricity"][split + look_back:]

    return df_train, df_train_label, df_test, df_test_label


# building the model
def buildModel(trainx, trainy):
    model = Sequential()
    model.add(LSTM(256, input_shape=(1, look_back)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # training it
    model.fit(trainx, trainy, epochs=20, batch_size=50, verbose=2)
    saveModel(model)
    return model


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    ret = model.evaluate(testx, testy, verbose=0)
    print(ret)
    return ret


def plotPrediction(y, predict_y):
    plt.plot(y.reset_index(drop=True), label="actual", color="green")
    plt.plot(predict_y, label="predict", color="orange")
    plt.show()


def saveModel(model):
    model_json = model.to_json()
    with open(outputFolder + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(outputFolder + "model.h5")
    print("Saved model to disk")


def loadModel():
    # load json and create model
    json_file = open(outputFolder + "model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(outputFolder + "model.h5")

    # evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


def forecasting(load):
    # import data
    df = dataImport()
    split = int(len(df) * part)

    # split train / test
    df_train, df_train_label, df_test, df_test_label = buildSet(df, split)
    df_train_arr = np.array(df_train)
    df_test_arr = np.array(df_test)
    trainx = np.reshape(df_train_arr, (df_train_arr.shape[0], 1, df_train_arr.shape[1]))
    testx = np.reshape(df_test_arr, (df_test_arr.shape[0], 1, df_test_arr.shape[1]))

    if load:
        model = loadModel()
    else:
        model = buildModel(trainx, df_train_label)

    evalModel(model, testx, df_test_label)

    # plotting
    predict_test = pd.DataFrame(model.predict(testx))
    predict_train = pd.DataFrame(model.predict(trainx))

    plotPrediction(df_train_label, predict_train)
    plotPrediction(df_test_label, predict_test)


# if argv = 1, then we rebuild the model
def main(argv):
    load = False
    if len(argv) > 2:
        if argv[1] == '1':
            load = True
    global outputFolder
    outputFolder = (
            "output/"
            + "modelKeras"
            + "/"
    )
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)
    forecasting(load)


if __name__ == "__main__":
    main(sys.argv)
