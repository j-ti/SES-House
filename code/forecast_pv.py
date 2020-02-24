import os
import sys
from datetime import datetime

from sklearn.externals import joblib
import numpy as np
from data import getPecanstreetData
from forecast import (
    splitData,
    buildSet,
    evalModel,
    loadModel,
    saveModel,
    train,
    addMinutes,
    addMonthOfYear,
    buildModel,
)
from forecast_conf import ForecastConfig
from forecast_pv_conf import ForecastPvConfig
from plot_forecast import plotHistory, plotPrediction, plotEcart, plotPredictionPart
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from util import constructTimeStamps, mean_absolute_percentage_error


def dataImport(config_main, config_pv):
    timestamps = constructTimeStamps(
        datetime.strptime(config_pv.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_pv.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config_pv.STEP_SIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )
    # input datas : uncontrollable resource : solar production
    df = getPecanstreetData(
        config_pv.DATA_FILE,
        config_pv.TIME_HEADER,
        config_pv.DATAID,
        "solar",
        timestamps,
    )
    df = addMinutes(df)
    df = addMonthOfYear(df)

    return df, np.array(timestamps)


def buildModelPv(trainX, trainY, valX, valY, config_pv):
    model = buildModel(config_pv, trainX.shape)
    # training it
    history = train(config_pv, model, trainX, trainY, valX, valY)
    saveModel(config_pv, model)
    return model, history


def getParts(df, config_main, config_pv):
    df_train, df_validation, df_test = splitData(config_main, df)
    # datas are normalized
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    print(scaler.data_max_)
    joblib.dump(scaler, config_pv.MODEL_FILE_SC)
    df_train = scaler.transform(df_train)
    df_validation = scaler.transform(df_validation)
    df_test = scaler.transform(df_test)
    return df_train, df_validation, df_test, scaler


def forecasting(config_main, config_pv):
    df, timestamps = dataImport(config_main, config_pv)

    config_main.TIMESTAMPS = timestamps

    df_train, df_validation, df_test, scaler = getParts(df, config_main, config_pv)

    # here we have numpy array
    trainX, trainY = buildSet(
        np.array(df_train), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE
    )
    validationX, validationY = buildSet(
        np.array(df_validation), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE
    )
    testX, testY = buildSet(
        np.array(df_test), config_pv.LOOK_BACK, config_pv.OUTPUT_SIZE
    )

    # plotInputDay(timestamps, trainY[:, 0], config_pv)

    if config_pv.LOAD_MODEL:
        model = loadModel(config_pv)
        history = None
    else:
        model, history = buildModelPv(
            trainX, trainY, validationX, validationY, config_pv
        )

    evalModel(model, testX, testY)

    # plotting
    trainPrediction = model.predict(trainX)
    testPrediction = model.predict(testX)
    valPrediction = model.predict(validationX)

    if history is not None:
        plotHistory(config_pv, history)

    plotPrediction(
        trainY,
        trainPrediction,
        testY,
        validationY,
        valPrediction,
        testPrediction,
        timestamps,
        config_pv,
    )
    plotPredictionPart(
        config_pv,
        trainY[24],
        trainPrediction[24],
        "1st day of train set",
        timestamps[24 : config_pv.TIME_PER_DAY + 24],
        "train",
    )
    plotPredictionPart(
        config_pv,
        validationY[24],
        valPrediction[24],
        "3rd day of validation set",
        timestamps[len(trainX) + 24 : len(trainX) + 24 + config_pv.TIME_PER_DAY],
        "validation",
    )
    plotPredictionPart(
        config_pv,
        testY[24],
        testPrediction[24],
        "1st day of test set",
        timestamps[
            len(trainX)
            + len(validationX)
            + 24 : len(trainX)
            + 24
            + len(validationX)
            + config_pv.TIME_PER_DAY
        ],
        "test",
    )
    # plotPredictionPartMult(
    #     config_pv,
    #     testY[0],
    #     testPrediction,
    #     "1st day of test set",
    #     timestamps[len(trainX) + len(validationX): len(trainX) + len(validationX) + config_pv.TIME_PER_DAY],
    #     "test"
    # )

    plotEcart(
        trainY,
        trainPrediction,
        validationY,
        valPrediction,
        testY,
        testPrediction,
        timestamps,
        config_pv,
    )
    # printing error
    for _ in [1]:
        print(
            "training\tMSE :\t{}".format(
                mean_squared_error(np.array(trainY), np.array(trainPrediction))
            )
        )
        print(
            "validation\t\tMSE :\t{}".format(
                mean_squared_error(np.array(validationY), np.array(valPrediction))
            )
        )
        print(
            "testing\t\tMSE :\t{}".format(
                mean_squared_error(np.array(testY), np.array(testPrediction))
            )
        )
        ###
        print(
            "training\tMAE :\t{}".format(
                mean_absolute_error(np.array(trainY), np.array(trainPrediction))
            )
        )
        print(
            "validation\t\tMAE :\t{}".format(
                mean_absolute_error(np.array(validationY), np.array(valPrediction))
            )
        )
        print(
            "testing\t\tMAE :\t{}".format(
                mean_absolute_error(np.array(testY), np.array(testPrediction))
            )
        )
        ###
        print(
            "training\tMAPE :\t{} %".format(
                mean_absolute_percentage_error(
                    np.array(trainY), np.array(trainPrediction)
                )
            )
        )
        print(
            "validation\t\tMAPE :\t{} %".format(
                mean_absolute_percentage_error(
                    np.array(validationY), np.array(valPrediction)
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
    config_main = ForecastConfig()
    config_pv = ForecastPvConfig(config_main)
    np.random.seed(config_main.SEED)
    global outputFolder
    outputFolder = config_pv.OUTPUT_FOLDER
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    forecasting(config_main, config_pv)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv)
