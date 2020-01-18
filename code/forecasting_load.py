import sys
from datetime import datetime

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
from util import makeShiftTest, makeShiftTrain, makeTick

from forecasting_load_config import Config

config = Config()

np.random.seed(config.SEED)


def dataImport():
    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )
    # input datas : uncontrolable resource : solar production
    df = getPecanstreetData(
        config.DATA_FILE, config.TIME_HEADER, config.DATAID, "grid", timestamps
    )
    df = df.reset_index(drop=True)
    return df, timestamps


def buildSet(df, split):
    df_train = df[config.LOOK_BACK : split].reset_index(drop=True)
    df_train = makeShiftTrain(df, df_train, config.LOOK_BACK, split)
    df_train_label = df[config.LOOK_BACK + 1 : split + 1]

    df_test = df[split + config.LOOK_BACK :].reset_index(drop=True)
    df_test = makeShiftTest(df, df_test, config.LOOK_BACK, split)
    df_test_label = df[split + config.LOOK_BACK :]

    return df_train, df_train_label, df_test, df_test_label


# building the model
def buildModel(trainx, trainy):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, config.LOOK_BACK)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation(config.ACTIVATION_FUNCTION))
    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=config.OPTIMIZE_FUNCTION,
        metrics=[metrics.mae, metrics.mape, metrics.mse],
    )

    # training it
    history = model.fit(
        trainx, trainy, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=2
    )
    saveModel(model)
    return model, history


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    ret = model.evaluate(testx, testy, verbose=0)
    print(ret)
    return ret


def plotPrediction(train_y, train_predict_y, test_y, test_predict_y, timestamps):
    time, tick = makeTick(timestamps)

    x1 = [i for i in range(len(train_y))]
    x2 = [i for i in range(len(train_y), len(test_y) + len(train_y))]
    plt.plot(x1, train_y.reset_index(drop=True), label="actual", color="green")
    plt.plot(x1, train_predict_y, label="predict", color="orange")
    plt.plot(x2, test_y.reset_index(drop=True), label="actual", color="blue")
    plt.plot(x2, test_predict_y, label="predict", color="red")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.show()


def plotEcart(train_y, train_predict_y, test_y, test_predict_y, timestamps):
    time, tick = makeTick(timestamps)

    x1 = [i for i in range(len(train_y))]
    x2 = [i for i in range(len(train_y), len(test_y) + len(train_y))]
    y1 = [train_predict_y[i] - train_y[i] for i in range(len(x1))]
    y2 = [test_predict_y[i] - test_y[i] for i in range(len(x2))]
    plt.plot(x1, y1, label="actual", color="green")
    plt.plot(x2, y2, label="actual", color="blue")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Difference (kW)")
    plt.legend()
    plt.show()


def plotHistory(history):
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


def forecasting():
    # import data
    df, timestamps = dataImport()
    split = int(len(df) * config.PART)

    # split train / test
    df_train, df_train_label, df_test, df_test_label = buildSet(df, split)
    df_train_arr = np.array(df_train)
    df_test_arr = np.array(df_test)
    trainx = np.reshape(df_train_arr, (df_train_arr.shape[0], 1, df_train_arr.shape[1]))
    testx = np.reshape(df_test_arr, (df_test_arr.shape[0], 1, df_test_arr.shape[1]))

    history = None
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
    forecasting()


if __name__ == "__main__":
    main(sys.argv)
