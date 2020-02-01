import sys
from tensorflow import set_random_seed

import numpy as np
import pandas as pd
from data import getPecanstreetData
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from forecast import (
    splitData,
    addMinutes,
    buildSet,
    train,
    saveModel,
    buildModel,
    get_split_indexes,
    add_day_of_week,
    add_weekend,
    loadModel,
)
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig
from plot_forecast import plotHistory, plotPredictionPart, plotPrediction

from shutil import copyfile


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
    input_data = add_weekend(input_data)

    for load in loadConfig.APPLIANCES:
        appliance_data = getPecanstreetData(
            loadConfig.DATA_FILE,
            loadConfig.TIME_HEADER,
            loadConfig.DATAID,
            load,
            timestamps,
        )
        input_data = pd.concat([input_data, appliance_data], axis=1)

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


def main(argv):
    set_random_seed(ForecastConfig().SEED)
    np.random.seed(ForecastConfig().SEED)

    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    train_x, train_y, validation_x, validation_y, test_x, test_y, scaler = prepareData(
        config, loadConfig, config.TIMESTAMPS
    )
    end_train, end_validation = get_split_indexes(config)

    if not loadConfig.LOAD_MODEL:
        copyfile(
            "./code/forecast_conf.py", loadConfig.OUTPUT_FOLDER + "forecast_conf.py"
        )
        copyfile(
            "./code/forecast_load_conf.py",
            loadConfig.OUTPUT_FOLDER + "forecast_load_conf.py",
        )
        model = buildModel(loadConfig, train_x.shape)
        history = train(loadConfig, model, train_x, train_y, validation_x, validation_y)
        saveModel(loadConfig, model)
        plotHistory(loadConfig, history)

        validation_timestamps = config.TIMESTAMPS[end_train:end_validation]
        validation_y_timestamps = validation_timestamps[loadConfig.LOOK_BACK :]
        assert (
            len(validation_y_timestamps) == len(validation_y) + loadConfig.OUTPUT_SIZE
        )
        validation_prediction = model.predict(validation_x)
        plotPredictionPart(
            loadConfig,
            validation_y[1, :],
            validation_prediction[1, :],
            "1st day of validation set",
            validation_y_timestamps[: loadConfig.OUTPUT_SIZE],
        )
    else:
        model = loadModel(loadConfig)
        test_prediction = model.predict(test_x)
        test_mse = mean_squared_error(test_y, test_prediction)
        print("test MSE: ", test_mse)

        plotPrediction(
            train_y,
            model.predict(train_x),
            validation_y,
            model.predict(validation_x),
            test_y,
            test_prediction,
            config.TIMESTAMPS,
            loadConfig,
        )

        test_timestamps = config.TIMESTAMPS[end_validation:]
        test_y_timestamps = test_timestamps[loadConfig.LOOK_BACK :]
        assert len(test_y_timestamps) == len(test_y) + loadConfig.OUTPUT_SIZE
        plotPredictionPart(
            loadConfig,
            test_y[1, :],
            test_prediction[1, :],
            "1st day of test set",
            test_y_timestamps[: loadConfig.OUTPUT_SIZE],
        )
        plotPredictionPart(
            loadConfig,
            test_y[2, :],
            test_prediction[2, :],
            "1st day of test set",
            test_y_timestamps[loadConfig.OUTPUT_SIZE : 2 * loadConfig.OUTPUT_SIZE],
        )


if __name__ == "__main__":
    main(sys.argv)
