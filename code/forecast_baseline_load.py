import sys

from forecast_load import getNormalizedParts
from forecast import get_split_indexes, buildSet, loadModel, get_timestamps_per_day
from forecast_baseline import (
    one_step_persistence_model,
    meanBaseline,
    mean_baseline_one_day,
    predict_zero_one_day,
    predict_zero_one_step,
    plotLSTM_Base_Real,
)
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig


def main(argv):
    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    train, validation, test, scaler = getNormalizedParts(
        config, loadConfig, config.TIMESTAMPS
    )

    baseline_train = train[:, 0]
    baseline_validation = validation[:, 0]
    baseline_test = test[:, 0]

    _, end_validation = get_split_indexes(config)

    # print("Validation:")
    one_step_persistence_model(baseline_validation)
    meanBaseline(config, baseline_train, baseline_validation)
    predict_zero_one_day(config, baseline_validation)
    predict_zero_one_step(baseline_validation)

    print("Test:")
    one_step_persistence_model(baseline_test)
    meanBaseline(config, baseline_train, baseline_test)
    mean_baseline_one_day(config, baseline_train, baseline_test)

    print("Train on test and predict for Test:")
    meanBaseline(config, baseline_test, baseline_test)
    mean_baseline_one_day(config, baseline_train, baseline_test)
    predict_zero_one_day(config, baseline_test)
    predict_zero_one_step(baseline_test)

    test_x, test_y = buildSet(test, loadConfig.LOOK_BACK, loadConfig.OUTPUT_SIZE)
    model = loadModel(loadConfig)
    test_predict = model.predict(test_x)

    if loadConfig.OUTPUT_SIZE == get_timestamps_per_day(config):
        plotLSTM_Base_Real(
            loadConfig, baseline_train, test_predict[0], "mean", test_y[0]
        )
    elif loadConfig.OUTPUT_SIZE == 1:
        plotLSTM_Base_Real(
            loadConfig, baseline_train, test_predict[:48], "", test_y[:48]
        )


if __name__ == "__main__":
    main(sys.argv)
