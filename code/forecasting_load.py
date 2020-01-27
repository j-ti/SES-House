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

from forecasting import splitData, addMinutes, buildSet, train, saveModel
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig
from plot_forecast import plotHistory, plotPredictionPart

from shutil import copyfile

import time

set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


def prepareData(config, loadConfig, timestamps):
    loadsData = getPecanstreetData(
        loadConfig.DATA_FILE,
        loadConfig.TIME_HEADER,
        loadConfig.DATAID,
        "grid",
        timestamps,
    )

    input_data = addMinutes(loadsData)
    input_data = add_day_of_week(input_data)

    train_part, validation_part, test_part = splitData(config, input_data)

    train_part = train_part.values
    validation_part = validation_part.values
    test_part = test_part.values

    scaler = MinMaxScaler()
    scaler.fit(train_part)
    train_part = scaler.transform(train_part)
    validation_part = scaler.transform(validation_part)
    test_part = scaler.transform(test_part)

    train_x, train_y = buildSet(
        train_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE
    )
    validation_x, validation_y = buildSet(
        validation_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE
    )
    test_x, test_y = buildSet(test_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE)

    return train_x, train_y, validation_x, validation_y, test_x, test_y, scaler


# building the model
def buildModel(loadConfig, trainXShape):
    model = Sequential()
    model.add(LSTM(loadConfig.NEURONS, input_shape=(trainXShape[1], trainXShape[2])))
    model.add(Dropout(loadConfig.DROPOUT))
    model.add(Dense(loadConfig.OUTPUT_SIZE))
    model.add(Activation(loadConfig.ACTIVATION_FUNCTION))
    model.compile(
        loss=loadConfig.LOSS_FUNCTION,
        optimizer=loadConfig.OPTIMIZE_FUNCTION,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return model


def add_day_of_week(data):
    daysOfWeek = pd.Series([i.weekday() for i in data.index], index=data.index)
    return pd.concat([data, daysOfWeek], axis=1)


def main(argv):
    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    copyfile("./code/forecast_conf.py", loadConfig.OUTPUT_FOLDER + "forecast_conf.py")
    copyfile(
        "./code/forecast_load_conf.py",
        loadConfig.OUTPUT_FOLDER + "forecast_load_conf.py",
    )

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )

    train_x, train_y, validation_x, validation_y, _, test_y, scaler = prepareData(
        config, loadConfig, timestamps
    )
    assert (
        len(timestamps)
        == len(train_y)
        + len(validation_y)
        + len(test_y)
        + loadConfig.LOOK_BACK * 3
        + loadConfig.OUTPUT_SIZE * 3
    )

    model = buildModel(loadConfig, train_x.shape)
    history = train(loadConfig, model, train_x, train_y, validation_x, validation_y)
    saveModel(loadConfig, model)
    plotHistory(loadConfig, history)
    validation_begin = len(train_y) + loadConfig.LOOK_BACK + loadConfig.OUTPUT_SIZE
    validation_timestamps = timestamps[
        validation_begin : validation_begin
        + len(validation_y)
        + loadConfig.OUTPUT_SIZE
        + loadConfig.LOOK_BACK
    ]
    validation_prediction = model.predict(validation_x)
    plotPredictionPart(
        validation_y[1, :],
        validation_prediction[1, :],
        "1st day of validation set",
        validation_timestamps[: loadConfig.OUTPUT_SIZE],
    )


if __name__ == "__main__":
    main(sys.argv)
