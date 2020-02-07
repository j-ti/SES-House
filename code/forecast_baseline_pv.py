import sys
from datetime import datetime

import numpy as np
from data import getPecanstreetData
from forecast import splitData, loadModel, buildSet, addMinutes, addMonthOfYear
from forecast_baseline import one_step_persistence_model, meanBaseline, predict_zero_one_day, \
    predict_zero_one_step, plot_baselines, plotLSTM_Base_Real
from forecast_conf import ForecastConfig
from forecast_pv_conf import ForecastPvConfig
from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed
from util import constructTimeStamps

set_random_seed(ForecastConfig().SEED)
np.random.seed(ForecastConfig().SEED)


def main(argv):
    config = ForecastConfig()
    pvConfig = ForecastPvConfig(config)

    config.OUTPUT_FOLDER = pvConfig.OUTPUT_FOLDER
    timestamps = constructTimeStamps(
        datetime.strptime(pvConfig.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(pvConfig.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(pvConfig.STEP_SIZE, "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")
    )

    config.TIMESTAMPS = timestamps

    # input datas : uncontrollable resource : solar production
    df = getPecanstreetData(
        pvConfig.DATA_FILE, pvConfig.TIME_HEADER, pvConfig.DATAID, "solar", timestamps
    )
    df = addMinutes(df)
    df = addMonthOfYear(df, timestamps)

    df_train, df_validation, df_test = splitData(config, df)

    valMin = df_train.iloc[0, -1]
    df_train.iloc[0, -1] = 0
    valMax = df_train.iloc[1, -1]
    df_train.iloc[1, -1] = 11
    # datas are normalized
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    df_train.iloc[0, -1] = valMin
    df_train.iloc[1, -1] = valMax

    df_train = scaler.transform(df_train)
    df_validation = scaler.transform(df_validation)
    df_test = scaler.transform(df_test)

    X, y = buildSet(df_test, pvConfig.LOOK_BACK, pvConfig.OUTPUT_SIZE)

    df_train = np.array([df_train[i, 0] for i in range(len(df_train))])
    df_validation = np.array([df_validation[i, 0] for i in range(len(df_validation))])
    df_test = np.array([df_test[i, 0] for i in range(len(df_test))])



    model = loadModel(pvConfig)
    testPredictY = model.predict(X)

    import matplotlib.pyplot as plt
    plt.plot(df_test[:100])
    plt.show()
    plt.plot(y[0])
    plt.show()

    # plot_baselines(config, df_train, df_test[:96], timestamps[len(df_train):len(df_train) + 96])
    plotLSTM_Base_Real(config, df_train, testPredictY[72], "mean", y[72])
    # plotLSTM_Base_Real(config, df_train, testPredictY[0], "1step", y[0])

    print("Validation:")
    one_step_persistence_model(df_validation)
    print("Test:")
    one_step_persistence_model(df_test)

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
