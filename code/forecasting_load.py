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
from util import constructTimeStamps, mean_absolute_percentage_error
from util import makeTick

from forecasting_load_config import Config, INIT_MODEL_CONFIG

import time


def getData(config, timestamps):
    # input datas : uncontrolable resource : solar production
    loadsData = getPecanstreetData(
        config.DATA_FILE, config.TIME_HEADER, config.DATAID, "grid", timestamps
    )
    return loadsData


def splitData(config, loadsData):
    diff = loadsData.index[-1] - loadsData.index[0]
    endTrain = 96 * int(diff.days * config.TRAINPART)
    endValidation = endTrain + 96 * int(diff.days * config.VALIDATIONPART)
    return (
        loadsData[:endTrain],
        loadsData[endTrain:endValidation],
        loadsData[endValidation:],
    )


def create_dataset(dataset, look_back):
    dataX, dataY = (
        np.empty((len(dataset) - look_back, look_back, 1)),
        np.empty((len(dataset) - look_back, 1)),
    )
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back)]
        # minutes = np.array([(i.hour * 60 + i.minute) / 1440 for i in a.index])
        # minutes = np.expand_dims(minutes, axis=1)
        a = a.values
        a = np.expand_dims(a, axis=1)
        dataX[i] = a  # np.concatenate((a, minutes), axis=1)
        dataY[i] = dataset.values[i + look_back]
    return dataX, dataY


def buildSets(config, loadsData):
    trainPart, validationPart, testPart = splitData(config, loadsData)

    trainX, trainY = create_dataset(trainPart, config.LOOK_BACK)
    validationX, validationY = create_dataset(validationPart, config.LOOK_BACK)
    testX, testY = create_dataset(testPart, config.LOOK_BACK)

    return trainX, trainY, validationX, validationY, testX, testY


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


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, x, y):
    ret = model.evaluate(x, y, verbose=0)
    return ret


def plotPrediction(real, predicted, nameOfSet, timestamps):
    time, tick = makeTick(timestamps)

    x1 = list(range(len(real)))

    plt.plot(x1, real, label="actual of " + nameOfSet, color="green")
    plt.plot(x1, predicted, label="predicted of " + nameOfSet, color="orange")

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotEcart(
    train_y,
    train_predict_y,
    validation_y,
    validation_predict_y,
    test_y,
    test_predict_y,
    timestamps,
):
    time, tick = makeTick(timestamps)

    x1 = list(range(len(train_y)))
    x2 = list(range(len(train_y), len(validation_y) + len(train_y)))
    x3 = list(
        range(
            len(train_y) + len(validation_y),
            len(test_y) + len(validation_y) + len(train_y),
        )
    )

    y1 = [train_predict_y[i] - train_y[i] for i in range(len(x1))]
    y2 = [validation_predict_y[i] - validation_y[i] for i in range(len(x2))]
    y3 = [test_predict_y[i] - test_y[i] for i in range(len(x3))]

    plt.plot(x1, y1, label="actual", color="green")
    plt.plot(x2, y2, label="actual", color="purple")
    plt.plot(x3, y3, label="actual", color="blue")

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Difference (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotHistory(history):
    plt.plot(history.history["loss"], label="train")
    # plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.plot(history.history["mean_absolute_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.tight_layout()
    plt.show()
    plt.plot(history.history["mean_absolute_percentage_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute percentage error")
    plt.tight_layout()
    plt.show()
    plt.plot(history.history["mean_squared_error"])
    plt.xlabel("Epoch")
    plt.ylabel("mean_squared_error")
    plt.tight_layout()
    plt.show()


def saveModel(config, model):
    model_json = model.to_json()
    with open(config.OUTPUT_FOLDER + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(config.OUTPUT_FOLDER + "model.h5")


def loadModel(config, modelConfig):
    # load json and create model
    json_file = open(config.OUTPUT_FOLDER + "model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.OUTPUT_FOLDER + "model.h5")

    # evaluate loaded model
    loaded_model.compile(
        loss=modelConfig["loss_function"],
        optimizer=modelConfig["optimize_function"],
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return loaded_model


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
    config, timestamps, trainX, trainY, validationX, validationY, testX, testY
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

    print(predict_test)
    mse = mean_squared_error(testY, predict_test)
    print("MSE: ", mse)
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
    config = Config()

    set_random_seed(23)
    np.random.seed(config.SEED)

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )

    loadsData = getData(config, timestamps)
    trainX, trainY, validationX, validationY, testX, testY = buildSets(
        config, loadsData
    )

    # train, val, test = splitData(config, loadsData)
    # meanBaseline(train.values, test.values)
    # plotDayBaseline(timestamps[-len(test):], test.values, predictMean(train.values, test.values)[:96])
    # plotPrediction(test.values[:96], predictMean(train.values, test.values)[:96], "test set 1st day", timestamps[-len(test):(-len(test) + 96)])

    forecasting(
        config, timestamps, trainX, trainY, validationX, validationY, testX, testY
    )


if __name__ == "__main__":
    main(sys.argv)
