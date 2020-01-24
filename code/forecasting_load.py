import sys
from datetime import datetime
from tensorflow import set_random_seed

from keras.engine.saving import model_from_json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import getPecanstreetData
from keras import metrics
from keras.layers import LSTM, Dropout, Activation
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from util import constructTimeStamps, mean_absolute_percentage_error
from util import makeTick

from forecasting import splitData, addMinutes, buildSet
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig

import time

set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


# building the model
def buildModel(state, trainXShape):
    model = Sequential()
    model.add(LSTM(state["neurons"], input_shape=(trainXShape[1], trainXShape[2])))
    model.add(Dropout(state["dropout"]))
    model.add(Dense(state["dense"]))
    model.add(Activation(state["activation_function"]))
    model.compile(
        loss=state["loss_function"],
        optimizer=state["optimize_function"],
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return model


def calcModel(config, trainX, trainY, validationX, validationY):
    model = buildModel(config, trainX.shape)
    history = model.fit(
        trainX,
        trainY,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=2,
    )

    return (model, history)


def getMeanSdDayBaseline(data):
    data = np.reshape(data, (96, int(len(data) / 96)))
    means = np.mean(data, axis=1)
    standard_dev = np.std(data, axis=1)
    return means, standard_dev


def plotDayBaseline(timestamps, realY, predictY):
    realMeans, realSd = getMeanSdDayBaseline(realY)
    x1 = list(range(96))

    plt.plot(x1, realMeans, label="actual", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )
    plt.plot(x1, predictY, label="predict Baseline", color="orange")

    time, tick = makeTick(timestamps[:96], "%H:%M")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def getMeanSdDay(data):
    nans = np.empty((Config().LOOK_BACK, 1))
    nans[:] = np.nan
    data = np.concatenate((nans, data), axis=0)
    data = np.reshape(data, (96, int(len(data) / 96)))
    means = np.nanmean(data, axis=1)
    standard_dev = np.nanstd(data, axis=1)
    return means, standard_dev


def plotDay(timestamps, realY, predictY):
    assert len(realY) == len(predictY)
    realMeans, realSd = getMeanSdDay(realY)
    predictedMeans, predictedSd = getMeanSdDay(predictY)
    x1 = list(range(96))

    plt.plot(x1, realMeans, label="actual", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )
    plt.plot(x1, predictedMeans, label="predict", color="orange")
    plt.fill_between(
        x1,
        predictedMeans - predictedSd * 0.5,
        predictedMeans + predictedSd * 0.5,
        color="orange",
        alpha=0.5,
    )

    time, tick = makeTick(timestamps[:96], "%H:%M")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotSets(timestamps, train, validation, test):
    time, tick = makeTick(timestamps)

    x1 = range(len(train))
    x2 = range(len(train), len(train) + len(validation))
    x3 = range(len(train) + len(validation), len(timestamps))
    plt.plot(x1, train, label="train set", color="green")
    plt.plot(x2, validation, label="validation set", color="blue")
    plt.plot(x3, test, label="test set", color="red")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def predictMean(train, test):
    assert len(train) % 96 == 0
    data = np.reshape(train, (96, int(len(train) / 96)))
    means = np.mean(data, axis=1)
    predictions = np.array(means)
    for i in range(int(len(test) / 96) - 1):
        predictions = np.concatenate((predictions, means))
    print(predictions)
    return predictions


def meanBaseline(train, test):
    predictions = predictMean(train, test)
    assert len(test) % 96 == 0
    mse = mean_squared_error(predictions, test)
    print("Baseline MSE: ", mse)
    return mse


def forecasting(
    config, timestamps, trainX, trainY, validationX, validationY, testX, testY, scaler
):
    modelConfig = INIT_MODEL_CONFIG
    if not config.LOAD:
        model, history = calcModel(
            modelConfig, trainX, trainY, validationX, validationY
        )
        saveModel(config, model)
        plotHistory(history)
    else:
        model = loadModel(config, modelConfig)

    predict_validation = pd.DataFrame(model.predict(validationX))
    predict_train = pd.DataFrame(model.predict(trainX))
    predict_test = pd.DataFrame(model.predict(testX))
    evalModel(model, testX, testY)

    mse = mean_squared_error(testY, predict_test)
    time.sleep(3)

    # plotDay(timestamps[-len(testY):], testY, predict_test)
    plotDay(timestamps[-len(testY) - config.LOOK_BACK :], testY, predict_test)

    plotPrediction(
        testY[:96],
        predict_test[:96],
        "Test Set 1st day",
        timestamps[-len(testY) : (-len(testY) + 96)],
    )
    plotPrediction(
        testY[96:192], predict_test[96:192], "Test Set 2nd day", timestamps[96:192]
    )
    plotPrediction(testY, predict_test, "Test Set", timestamps[: len(testY)])

    plotEcart(
        np.array(trainY),
        np.array(predict_train),
        np.array(validationY),
        np.array(predict_validation),
        np.array(testY),
        np.array(predict_test),
        timestamps,
    )
    print("training\tMSE :\t{}".format(mean_squared_error(trainY, predict_train)))
    print("testing\t\tMSE :\t{}".format(mean_squared_error(testY, predict_test)))

    print("training\tMAE :\t{}".format(mean_absolute_error(trainY, predict_train)))
    print(
        "testing\t\tMAE :\t{}".format(
            mean_absolute_error(np.array(testY), np.array(predict_test))
        )
    )

    print(
        "training\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(np.array(trainY), np.array(predict_train))
        )
    )
    print(
        "testing\t\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(np.array(testY), np.array(predict_test))
        )
    )



def main(argv):
    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )
    loadsData = getPecanstreetData(
        loadConfig.DATA_FILE, loadConfig.TIME_HEADER, loadConfig.DATAID, "grid", timestamps
    )

    input_data = addMinutes(loadsData)

    train_part, validation_part, test_part = splitData(config, input_data)

    scaler = MinMaxScaler()
    scaler.fit(train_part)
    scaler.transform(train_part)
    scaler.transform(validation_part)
    scaler.transform(test_part)

    train_x, train_y = buildSet(train_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE, train_part.shape[2])
    validation_x, validation_y = buildSet(validation_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE, validation_part.shape[2])
    test_x, test_y = buildSet(test_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE, test_part.shape[2])

    # train, val, test = splitData(config, loadsData)
    # meanBaseline(train.values, test.values)
    # plotDayBaseline(timestamps[-len(test):], test.values, predictMean(train.values, test.values)[:96])
    # plotPrediction(test.values[:96], predictMean(train.values, test.values)[:96], "test set 1st day", timestamps[-len(test):(-len(test) + 96)])

    forecasting(
        config, timestamps, trainX, trainY, validationX, validationY, testX, testY, sclaer
    )


if __name__ == "__main__":
    main(sys.argv)
