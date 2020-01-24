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
nbOut = 2
config = ""
nbFeatures = 1


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
def buildModel(trainX, trainY, valX, valY):
    model = Sequential()
    model.add(LSTM(256, input_shape=(look_back, nbFeatures)))
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
    history = model.fit(trainX, trainY, epochs=20, batch_size=20, verbose=2, validation_data=[valX, valY])
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


def train(config, model, trainX, trainY, validationX, validationY):
    history = model.fit(
        trainX,
        trainY,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(validationX, validationY),
        verbose=2,
        )
    return model, history


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

    df_train, df_validation, df_test = splitData(config, df)

    # here we have numpy array
    trainX, trainY = buildSet(df_train, look_back, nbOut, nbFeatures)
    validationX, validationY = buildSet(df_validation, look_back, nbOut, nbFeatures)
    testX, testY = buildSet(df_test, look_back, nbOut, nbFeatures)

    plotInput(df, timestamps)

    history = None
    if load:
        model = loadModel()
    else:
        model, history = buildModel(trainX, trainY, validationX, validationY)

    evalModel(model, testX, testY)

    # plotting
    testPrediction = model.predict(testX)
    trainPrediction = model.predict(trainX)

    if history is not None:
        plotHistory(history)

    plotPrediction(
        trainY, trainPrediction, testY, testPrediction, timestamps
    )
    plot100first(trainY, trainPrediction)
    plot100first(testY, testPrediction)
    plotEcart(
        np.array(trainY),
        np.array(trainPrediction),
        np.array(testY),
        np.array(testPrediction),
        timestamps,
    )
    print(
        "training\tMSE :\t{}".format(
            mean_squared_error(np.array(trainY), np.array(trainPrediction))
        )
    )
    print(
        "testing\t\tMSE :\t{}".format(
            mean_squared_error(np.array(testY), np.array(testPrediction))
        )
    )

    print(
        "training\tMAE :\t{}".format(
            mean_absolute_error(np.array(trainY), np.array(trainPrediction))
        )
    )
    print(
        "testing\t\tMAE :\t{}".format(
            mean_absolute_error(np.array(testY), np.array(testPrediction))
        )
    )

    print(
        "training\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(trainY), np.array(trainPrediction)
            )
        )
    )
    print(
        "testing\t\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(testY), np.array(testPrediction)
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
