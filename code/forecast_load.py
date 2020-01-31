import sys
from datetime import datetime
from tensorflow import set_random_seed


import numpy as np
import pandas as pd
from data import getPecanstreetData
from sklearn.preprocessing import MinMaxScaler
from util import constructTimeStamps

from forecast import splitData, addMinutes, buildSet, train, saveModel, buildModel, get_split_indexes
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig
from plot_forecast import plotHistory, plotPredictionPart

from shutil import copyfile


set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


def getNormalizedParts(config, loadConfig, timestamps):
    loadsData = getPecanstreetData(
        loadConfig.DATA_FILE,
        loadConfig.TIME_HEADER,
        loadConfig.DATAID,
        "grid",
        timestamps,
    )
    assert len(timestamps) == len(loadsData)

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

    return train_part, validation_part, test_part, scaler


def prepareData(config, loadConfig, timestamps):
    train_part, validation_part, test_part, scaler = getNormalizedParts(
        config, loadConfig, timestamps
    )

    train_x, train_y = buildSet(
        train_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE
    )
    validation_x, validation_y = buildSet(
        validation_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE
    )
    test_x, test_y = buildSet(test_part, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE)

    return train_x, train_y, validation_x, validation_y, test_x, test_y, scaler


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

    train_x, train_y, validation_x, validation_y, _, test_y, scaler = prepareData(
        config, loadConfig, config.TIMESTAMPS
    )

    assert (
        len(config.TIMESTAMPS)
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
        validation_begin
        + loadConfig.OUTPUT_SIZE : validation_begin
        + len(validation_y)
        + loadConfig.LOOK_BACK
    ]
    assert len(validation_timestamps) == len(validation_y)
    validation_prediction = model.predict(validation_x)
    plotPredictionPart(
        validation_y[1, :],
        validation_prediction[1, :],
        "1st day of validation set",
        validation_timestamps[: loadConfig.OUTPUT_SIZE],
    )


if __name__ == "__main__":
    main(sys.argv)
