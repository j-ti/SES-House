import sys
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
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

from forecasting_load_config import Config, modelOptimizationConfig, INIT_MODEL_CONFIG

from simpleai.search import SearchProblem
from simpleai.search.local import beam

import time
import random


def getData(config, timestamps):
    # input datas : uncontrolable resource : solar production
    loadsData = getPecanstreetData(
        config.DATA_FILE, config.TIME_HEADER, config.DATAID, "grid", timestamps
    )
    return loadsData


def splitData(config, loadsData):
    endTrain = int(len(loadsData) * config.TRAINPART)
    endValidation = endTrain + int(len(loadsData) * config.VALIDATIONPART)
    return (
        loadsData[:endTrain],
        loadsData[endTrain:endValidation],
        loadsData[endValidation:],
    )


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
            agg.dropna(inplace=True)
    return agg


def create_dataset(dataset, look_back):
    dataX = np.empty((len(dataset) - look_back, look_back, 1))
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back)]
        # minutes = np.array([i.hour * 60 + i.minute for i in a.index])
        a = a.values
        a = np.expand_dims(a, axis=1)
        # minutes = np.expand_dims(minutes, axis=1)
        dataX[i] = a # np.concatenate((a, a), axis=1)
    return dataX


def addMinutes(loadsData):
    dataset = np.array(loadsData.values)
    dataset = np.expand_dims(dataset, axis=1)
    minutes = np.array([i.hour * 60 + i.minute for i in loadsData.index])
    minutes = np.expand_dims(minutes, axis=1)
    return np.concatenate((dataset, minutes), axis=1)


def buildSets(config, loadsData):
    # dataset = addMinutes(loadsData)
    trainPart, validationPart, testPart = splitData(config, loadsData)

    trainX = create_dataset(trainPart, config.LOOK_BACK)
    trainY = trainPart[config.LOOK_BACK:]
    print(trainX.shape)
    print(trainY.shape)
    time.sleep(1)

    validationX = create_dataset(validationPart, config.LOOK_BACK)
    validationY = validationPart[config.LOOK_BACK:]

    testX = create_dataset(testPart, config.LOOK_BACK)
    testY = testPart[config.LOOK_BACK:]

    return trainX, trainY, validationX, validationY, testX, testY


# building the model
def buildModel(config, trainX):
    model = Sequential()
    print(trainX.shape)
    model.add(LSTM(config["neurons"], batch_input_shape=(config["batch_size"], trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(config["dense"]))
    model.add(Activation(config["activation_function"]))
    model.compile(
        loss=config["loss_function"],
        optimizer=config["optimize_function"],
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return model


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, x, y):
    ret = model.evaluate(x, y, verbose=0)
    print("!!!!!!!!!!!!!!!")
    print(ret)
    return ret


def getMeanDay(timestamps, data):
    pass

def plotDay(timestamps, realY, predictY):
    plt.xticks(tick, constructTimeStamps(""), rotation=20)

    plt.xlabel("Time of Day")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.show()


def plotPrediction(
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
    x3 = list(range(len(train_y) + len(validation_y), len(test_y) + len(validation_y) + len(train_y)))

    plt.plot(x1, train_y, label="actual", color="green")
    plt.plot(x1, train_predict_y, label="predict", color="orange")
    plt.plot(x2, validation_y, label="actual", color="purple")
    plt.plot(x2, validation_predict_y, label="predict", color="pink")
    plt.plot(x3, test_y, label="actual", color="blue")
    plt.plot(x3, test_predict_y, label="predict", color="red")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
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
    x3 = list(range(len(train_y) + len(validation_y), len(test_y) + len(validation_y) + len(train_y)))

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
    plt.show()


def plotHistory(history):
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.plot(history.history["mean_absolute_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.show()
    plt.plot(history.history["mean_absolute_percentage_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute percentage error")
    plt.show()
    plt.plot(history.history["mean_squared_error"])
    plt.xlabel("Epoch")
    plt.ylabel("mean_squared_error")
    plt.show()


def saveModel(model):
    model_json = model.to_json()
    with open(config.OUTPUT_FOLDER + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(config.OUTPUT_FOLDER + "model.h5")
    print("Saved model to disk")


class OptimizeLSTM(SearchProblem):
    def generate_random_state(self):
        random_state = dict(INIT_MODEL_CONFIG)
        for key in modelOptimizationConfig:
            random_state[key] = random.choice(modelOptimizationConfig[key])
        return random_state

    def actions(self, state):
        return state["next"]

    def result(self, state, action):
        state[state["next"]] = action
        state["next"] = (
            list(modelOptimizationConfig.keys()).index(state["next"]) + 1
        ) % len(modelOptimizationConfig)
        return state

    def value(self, state):
        _, _, value = calcModel(state)
        return value


def calcModel(config, trainX, trainY, validationX, validationY):
    model = buildModel(config, trainX)
    history = model.fit(
        trainX,
        trainY,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(validationX, validationY),
        verbose=2,
    )

    return (
        model,
        history,
        history.history['val_loss'][-1],
    )


def forecasting(
    config, timestamps, trainX, trainY, validationX, validationY, testX, testY
):
    if config.DO_PARAM_TUNING:
        lstmProblem = OptimizeLSTM()
        result = beam(lstmProblem)
        print(result.state)
        print(result.path())

    config = INIT_MODEL_CONFIG
    model, history, _ = calcModel(config, trainX, trainY, validationX, validationY)
    evalModel(model, testX, testY)
    predict_test = pd.DataFrame(model.predict(testX))
    predict_validation = pd.DataFrame(model.predict(validationX))
    predict_train = pd.DataFrame(model.predict(trainX))



    plotHistory(history)

    plotPrediction(
        trainY,
        predict_train,
        validationY,
        predict_validation,
        testY,
        predict_test,
        timestamps,
    )
    plotEcart(
        np.array(trainY),
        np.array(predict_train),
        np.array(validationY),
        np.array(predict_validation),
        np.array(testY),
        np.array(predict_test),
        timestamps,
    )
    print(
        "training\tMSE :\t{}".format(
            mean_squared_error(np.array(trainY), np.array(predict_train))
        )
    )
    print(
        "testing\t\tMSE :\t{}".format(
            mean_squared_error(np.array(testY), np.array(predict_test))
        )
    )

    print(
        "training\tMAE :\t{}".format(
            mean_absolute_error(np.array(trainY), np.array(predict_train))
        )
    )
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

    np.random.seed(config.SEED)

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )

    trainX, trainY, validationX, validationY, testX, testY = buildSets(
        config, getData(config, timestamps)
    )

    forecasting(
        config, timestamps, trainX, trainY, validationX, validationY, testX, testY
    )


if __name__ == "__main__":
    main(sys.argv)
