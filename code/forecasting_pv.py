import os
import sys
from datetime import datetime

import numpy as np
from data import getPecanstreetData
from forecast_pv_conf import ForecastPvConfig
from forecasting import splitData, buildSet, evalModel, loadModel, saveModel, train
from keras import Sequential, metrics
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from plot_forecast import plotHistory, plotPrediction, plot100first, plotEcart, plotInput
from util import constructTimeStamps


def dataImport(config):
    timestamps = constructTimeStamps(
        datetime.strptime(config.DATE_START, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.DATE_END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime("00:15:00", "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )

    # input datas : uncontrolable resource : solar production
    df = getPecanstreetData(
        config.DATA_FILE, config.TIME_HEADER, config.DATAID, "solar", timestamps
    )
    return df, np.array(timestamps)


def buildModel(trainX, trainY, valX, valY, config, nbFeatures):
    model = Sequential()
    model.add(LSTM(config.NEURONS, input_shape=(config.LOOK_BACK, nbFeatures)))
    model.add(Dropout(config.DROPOUT))
    model.add(Dense(config.DENSE))
    model.add(Activation(config.ACTIVATION_FUNCTION))
    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=config.OPTIMIZE_FUNCTION,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )

    # training it
    model, history = train(config, model, trainX, trainY, valX, valY)
    saveModel(model)
    return model, history


def forecasting(config):
    # import data
    df, timestamps = dataImport(config)

    df_train, df_validation, df_test = splitData(config, df)

    nbFeatures = df_train.shape[1]

    # here we have numpy array
    trainX, trainY = buildSet(df_train, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)
    validationX, validationY = buildSet(df_validation, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)
    testX, testY = buildSet(df_test, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)

    plotInput(df, timestamps)

    history = None
    if config.LOAD_MODEL:
        model = loadModel()
    else:
        model, history = buildModel(trainX, trainY, validationX, validationY, config, nbFeatures)

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
    # printing error
    for _ in [1]:
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


def main(argv):
    config = ForecastPvConfig()
    global outputFolder
    outputFolder = config.OUTPUT_FOLDER
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    forecasting(config)


if __name__ == "__main__":
    main(sys.argv)