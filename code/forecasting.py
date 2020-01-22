import os
import sys
from datetime import datetime

import pandas as pd
from data import getPecanstreetData
from keras import metrics
from keras.engine.saving import model_from_json
from keras.layers import LSTM, Dropout, Activation
from keras.layers.core import Dense
from keras.models import Sequential
from plot_forecast import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from util import constructTimeStamps, mean_absolute_percentage_error
from util import makeShift

# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)
# param
look_back = 10  # we have a 5 point history in our input
part = 0.6  # we train on part of the set


def dataImport():
    timestamps = constructTimeStamps(
        datetime.strptime("2018-02-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
        datetime.strptime("2018-03-01 00:00:00", "20%y-%m-%d %H:%M:%S"),
        datetime.strptime("00:15:00", "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )
    # input datas : uncontrolable resource : solar production
    df = getPecanstreetData(
        "./data/15minute_data_austin.csv", "local_15min", 1642, "solar", timestamps
    )
    # max = df.max()
    # df = df.multiply(1/max)

    return df.values, np.array(timestamps)


def splitData(config, loadsData):
    diff = loadsData.index[-1] - loadsData.index[0]
    endTrain = 96 * int(diff.days * config.TRAINPART)
    endValidation = endTrain + 96 * int(diff.days * config.VALIDATIONPART)
    return (
        loadsData[:endTrain],
        loadsData[endTrain:endValidation],
        loadsData[endValidation:],
    )


# we assume that data is either train, test or validation and is shape (nbPts, nbFeatures)
def buildSet(data, look_back, nbOutput, nbFeatures):
    X = makeShift(data, look_back, nbFeatures)
    col = []
    for i in range(len(data) - look_back):
        col.append(data[look_back + i: i + look_back + nbOutput])
    Y = np.array(col)
    return X, Y


# building the model
def buildModel(trainx, trainy):
    model = Sequential()
    nbfeatures = 0
    model.add(LSTM(256, input_shape=(look_back, nbfeatures)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    # model.add(Dense(3))
    # model.add(Dense(1))
    model.add(Activation("relu"))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[metrics.mae, metrics.mape, metrics.mse],
    )

    # training it
    history = model.fit(trainx, trainy, epochs=20, batch_size=20, verbose=2)
    saveModel(model)
    return model, history


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    ret = model.evaluate(testx, testy, verbose=0)
    print(ret)
    return ret


def saveModel(model):
    model_json = model.to_json()
    with open(outputFolder + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(outputFolder + "model.h5")
    print("Saved model to disk")


def loadModel():
    # load json and create model
    json_file = open(outputFolder + "model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(outputFolder + "model.h5")

    # evaluate loaded model
    loaded_model.compile(
        loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    return loaded_model


# first value must be an array with 5 pages
def forecast(model, nbToPredict, firstValue):
    pred = []
    val = firstValue
    for i in range(nbToPredict):
        pred.append(model.predict(val))
        val.pop(0)
        val.append(pred[-1])
    return pred


def forecasting(load):
    # import data
    df, timestamps = dataImport()
    split = int(len(df) * part)
    df = np.array(df)
    X, Y = buildSet(df, look_back, nb_out, nb_features)

    plotInput(df, timestamps)

    # split train / test
    df_train, df_train_label, df_test, df_test_label = buildSet(df, split)
    df_train_arr = np.array(df_train)
    df_test_arr = np.array(df_test)

    history = None
    if load:
        model = loadModel()
    else:
        model, history = buildModel(trainx, df_train_label)

    evalModel(model, testx, df_test_label)

    # plotting
    predict_test = pd.DataFrame(model.predict(testx))
    predict_train = pd.DataFrame(model.predict(trainx))

    if history is not None:
        plotHistory(history)

    plotPrediction(
        df_train_label, predict_train, df_test_label, predict_test, timestamps
    )
    plot100first(df_train_label, predict_train)
    plotEcart(
        np.array(df_train_label),
        np.array(predict_train),
        np.array(df_test_label),
        np.array(predict_test),
        timestamps,
    )
    print(
        "training\tMSE :\t{}".format(
            mean_squared_error(np.array(df_train_label), np.array(predict_train))
        )
    )
    print(
        "testing\t\tMSE :\t{}".format(
            mean_squared_error(np.array(df_test_label), np.array(predict_test))
        )
    )

    print(
        "training\tMAE :\t{}".format(
            mean_absolute_error(np.array(df_train_label), np.array(predict_train))
        )
    )
    print(
        "testing\t\tMAE :\t{}".format(
            mean_absolute_error(np.array(df_test_label), np.array(predict_test))
        )
    )

    print(
        "training\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(df_train_label), np.array(predict_train)
            )
        )
    )
    print(
        "testing\t\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(df_test_label), np.array(predict_test)
            )
        )
    )


# if argv = 1, then we rebuild the model
def main(argv):
    load = False
    if len(argv) == 2:
        if argv[1] == "1":
            load = True
    global outputFolder
    outputFolder = "output/" + "modelKeras" + "/"
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)
    forecasting(load)


if __name__ == "__main__":
    main(sys.argv)
