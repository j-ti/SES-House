import sys
from datetime import datetime
from tensorflow import set_random_seed


import numpy as np
import pandas as pd
from data import getPecanstreetData
from util import constructTimeStamps

from forecast import splitData, addMinutes, buildSet, train, saveModel, buildModel
from forecast_load import getNormalizedParts
from forecast_baseline import one_step_persistence_model, one_day_persistence_model
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig

from shutil import copyfile


set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


def main(argv):
    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )

    train, validation, test, scaler = getNormalizedParts(config, loadConfig, timestamps)

    one_step_persistence_model(validation)
    one_step_persistence_model(test)

    one_day_persistence_model(loadConfig, validation)
    one_day_persistence_model(loadConfig, test)


if __name__ == "__main__":
    main(sys.argv)
