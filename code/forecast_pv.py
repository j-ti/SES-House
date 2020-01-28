import os
import sys
from datetime import datetime

import numpy as np
from data import getPecanstreetData
from forecast_conf import ForecastConfig
from forecast_pv_conf import ForecastPvConfig
from forecasting import splitData, buildSet, evalModel, loadModel, saveModel, train, addMinutes
from keras import Sequential, metrics
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from plot_forecast import plotHistory, plotPrediction, plot100first, plotEcart, plotInput
from sklearn.preprocessing import MinMaxScaler
from util import constructTimeStamps


def dataImport(config_main, config_pv):
    timestamps = constructTimeStamps(
        datetime.strptime(config_main.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_main.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_main.STEPSIZE, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )
    # input datas : uncontrollable resource : solar production
    df = getPecanstreetData(
        config_pv.DATA_FILE, config_pv.TIME_HEADER, config_pv.DATAID, "solar", timestamps
    )

    return addMinutes(df), np.array(timestamps)


def buildModel(trainX, trainY, valX, valY, config_pv, nbFeatures):
    model = Sequential()
    model.add(LSTM(config_pv.NEURONS, input_shape=(config_pv.LOOK_BACK, nbFeatures)))
    model.add(Dropout(config_pv.DROPOUT))
    model.add(Dense(config_pv.DENSE))
    model.add(Activation(config_pv.ACTIVATION_FUNCTION))
    model.add(Dense(config_pv.OUTPUT_SIZE))
    model.compile(
        loss=config_pv.LOSS_FUNCTION,
        optimizer=config_pv.OPTIMIZE_FUNCTION,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )

    # training it
    model, history = train(config_pv, model, trainX, trainY, valX, valY)
    saveModel(model)
    return model, history


def forecasting(config_main, config_pv):
    # import data, with all the features we want
    df, timestamps = dataImport(config_main, config_pv)
    df_train, df_validation, df_test = splitData(config_main, df)

    # datas are normalized
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_validation = scaler.transform(df_validation)
    df_test = scaler.transform(df_test)

    nbFeatures = df_train.shape[1]

    # here we have numpy array
    trainX, trainY = buildSet(np.array(df_train), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE, nbFeatures)
    validationX, validationY = buildSet(np.array(df_validation), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE, nbFeatures)
    testX, testY = buildSet(np.array(df_test), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE, nbFeatures)

    plotInput(df, timestamps)

    if config_pv.LOAD_MODEL:
        model = loadModel()
        history = None
    else:
        model, history = buildModel(trainX, trainY, validationX, validationY, config_pv, nbFeatures)

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
    config_pv = ForecastPvConfig()
    config_main = ForecastConfig()
    np.random.seed(config_main.SEED)
    global outputFolder
    outputFolder = config_pv.OUTPUT_FOLDER
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    forecasting(config_main, config_pv)


if __name__ == "__main__":
    main(sys.argv)
