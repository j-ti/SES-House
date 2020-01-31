import os
import sys
from datetime import datetime

import numpy as np
from data import getPecanstreetData
from forecast import splitData, buildSet, evalModel, loadModel, saveModel, train, addMinutes, addDayOfYear
from forecast_conf import ForecastConfig
from forecast_pv_conf import ForecastPvConfig
from keras import Sequential, metrics
from keras.layers import LSTM, Dropout, Dense, Activation
from plot_forecast import plotHistory, plotPrediction, plotEcart, plotPredictionPart
from sklearn.preprocessing import MinMaxScaler
from util import constructTimeStamps


def dataImport(config_main, config_pv):
    timestamps = constructTimeStamps(
        datetime.strptime(config_pv.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_pv.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_pv.STEP_SIZE, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )
    # input datas : uncontrollable resource : solar production
    df = getPecanstreetData(
        config_pv.DATA_FILE, config_pv.TIME_HEADER, config_pv.DATAID, "solar", timestamps
    )
    df = addMinutes(df)
    df = addDayOfYear(df, timestamps)

    return df, np.array(timestamps)


def buildModel(trainX, trainY, valX, valY, config_pv, nbFeatures):
    model = Sequential()
    model.add(LSTM(config_pv.NEURONS, input_shape=(config_pv.LOOK_BACK, nbFeatures)))
    model.add(Dropout(config_pv.DROPOUT))
    model.add(Activation(config_pv.ACTIVATION_FUNCTION))
    model.add(Dense(config_pv.OUTPUT_SIZE))
    model.compile(
        loss=config_pv.LOSS_FUNCTION,
        optimizer=config_pv.OPTIMIZE_FUNCTION,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )

    # training it
    history = train(config_pv, model, trainX, trainY, valX, valY)
    saveModel(config_pv, model)
    return model, history


def forecasting(config_main, config_pv):
    # import data, with all the features we want
    df, timestamps = dataImport(config_main, config_pv)

    df_train, df_validation, df_test = splitData(config_main, df, 24)

    # the SettingWithCopyWarning warning is there because df_train is a copy of the original data.
    # we force the date to have a 0 -> 365 range
    valMin = df_train.iloc[0, -1]
    df_train.iloc[0, -1] = 0
    valMax = df_train.iloc[1, -1]
    df_train.iloc[1, -1] = 365
    # datas are normalized
    scaler = MinMaxScaler()
    scaler.fit(df_train)

    df_train.iloc[0, -1] = valMin
    df_train.iloc[1, -1] = valMax

    df_train = scaler.transform(df_train)
    df_validation = scaler.transform(df_validation)
    df_test = scaler.transform(df_test)

    nbFeatures = df_train.shape[1]

    # here we have numpy array
    trainX, trainY = buildSet(np.array(df_train), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE)
    validationX, validationY = buildSet(np.array(df_validation), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE)
    testX, testY = buildSet(np.array(df_test), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE)

    # plotInputDay(timestamps, trainY[:, 0], config_pv)

    if config_pv.LOAD_MODEL:
        model = loadModel(config_pv)
        history = None
    else:
        model, history = buildModel(trainX, trainY, validationX, validationY, config_pv, nbFeatures)

    evalModel(model, testX, testY)

    # plotting
    trainPrediction = model.predict(trainX)
    testPrediction = model.predict(testX)
    valPrediction = model.predict(validationX)

    if history is not None:
        plotHistory(config_pv, history)

    plotPrediction(
        trainY, trainPrediction, testY, validationY, valPrediction, testPrediction, timestamps, config_pv
    )
    plotPredictionPart(
        trainY[0],
        trainPrediction[0],
        "1st day of train set",
        timestamps[: 24],
    )
    plotPredictionPart(
        validationY[0],
        valPrediction[0],
        "1st day of validation set",
        timestamps[len(trainX):len(trainX) + 24],
    )
    plotPredictionPart(
        testY[0],
        testPrediction[0],
        "1st day of test set",
        timestamps[len(trainX) + len(validationX): len(trainX) + len(validationX) + 24],
    )
    plotEcart(
        trainY,
        trainPrediction,
        validationY,
        valPrediction,
        testY,
        testPrediction,
        timestamps,
        config_pv
    )
    # # printing error
    # for _ in [1]:
    #     print(
    #         "training\tMSE :\t{}".format(
    #             mean_squared_error(np.array(trainY), np.array(trainPrediction))
    #         )
    #     )
    #     print(
    #         "testing\t\tMSE :\t{}".format(
    #             mean_squared_error(np.array(testY), np.array(testPrediction))
    #         )
    #     )
    #
    #     print(
    #         "training\tMAE :\t{}".format(
    #             mean_absolute_error(np.array(trainY), np.array(trainPrediction))
    #         )
    #     )
    #     print(
    #         "testing\t\tMAE :\t{}".format(
    #             mean_absolute_error(np.array(testY), np.array(testPrediction))
    #         )
    #     )
    #
    #     print(
    #         "training\tMAPE :\t{} %".format(
    #             mean_absolute_percentage_error(
    #                 np.array(trainY), np.array(trainPrediction)
    #             )
    #         )
    #     )
    #     print(
    #         "testing\t\tMAPE :\t{} %".format(
    #             mean_absolute_percentage_error(
    #                 np.array(testY), np.array(testPrediction)
    #             )
    #         )
    #     )


def main(argv):
    config_main = ForecastConfig()
    config_pv = ForecastPvConfig(config_main)
    np.random.seed(config_main.SEED)
    global outputFolder
    outputFolder = config_pv.OUTPUT_FOLDER
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    forecasting(config_main, config_pv)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv)
