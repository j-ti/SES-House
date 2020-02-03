import sys
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed


import numpy as np
import pandas as pd
from data import getPecanstreetData
from util import constructTimeStamps

from forecast import splitData, addMinutes, buildSet, train, saveModel, buildModel
from forecast_pv import getParts, dataImport
from forecast_baseline import one_step_persistence_model, one_day_persistence_model, meanBaseline, predict_zero_one_day, \
    predict_zero_one_step, plot_baselines
from forecast_conf import ForecastConfig
from forecast_pv_conf import ForecastPvConfig

from shutil import copyfile


set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


def main(argv):
    config = ForecastConfig()
    pvConfig = ForecastPvConfig(config)
    timestamps = constructTimeStamps(
        datetime.strptime(pvConfig.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(pvConfig.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(pvConfig.STEP_SIZE, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )
    # input datas : uncontrollable resource : solar production
    df = getPecanstreetData(
        pvConfig.DATA_FILE, pvConfig.TIME_HEADER, pvConfig.DATAID, "solar", timestamps
    )
    df_train, df_validation, df_test = splitData(config, df)
    df_train, df_validation, df_test = np.array(df_train).reshape(-1, 1), np.array(df_validation).reshape(-1, 1), np.array(df_test).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_validation = scaler.transform(df_validation)
    df_test = scaler.transform(df_test)

    df_train = np.array([df_train[i, 0] for i in range(len(df_train))])
    df_validation = np.array([df_validation[i, 0] for i in range(len(df_validation))])
    df_test = np.array([df_test[i, 0] for i in range(len(df_test))])
    plot_baselines(config, df_train, df_test[:96], timestamps[len(df_train):len(df_train)+96])

    print("Validation:")
    one_step_persistence_model(df_validation)
    print("Test:")
    one_step_persistence_model(df_test)

    print("Validation:")
    one_day_persistence_model(config, df_validation)
    print("Test:")
    one_day_persistence_model(config, df_test)

    print("Validation:")
    meanBaseline(config, df_train, df_validation)
    print("Test:")
    meanBaseline(config, df_train, df_test)
    print("Train on test and predict for Test:")
    meanBaseline(config, df_test, df_test)

    print("Validation:")
    predict_zero_one_day(config, df_validation)
    print("Test:")
    predict_zero_one_day(config, df_test)

    print("Validation:")
    predict_zero_one_step(df_validation)
    print("Test:")
    predict_zero_one_step(df_test)



if __name__ == "__main__":
    main(sys.argv)
