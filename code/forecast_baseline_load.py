import sys

from forecast_load import getNormalizedParts
from forecast import get_split_indexes
from forecast_baseline import (
    one_step_persistence_model,
    one_day_persistence_model,
    meanBaseline,
    predict_zero_one_day,
    predict_zero_one_step,
    plot_test_set,
    plot_days,
    plot_baselines,
)
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig


def main(argv):
    config = ForecastConfig()
    loadConfig = ForecastLoadConfig()

    train, validation, test, scaler = getNormalizedParts(
        config, loadConfig, config.TIMESTAMPS
    )

    train = train[:, 0]
    validation = validation[:, 0]
    test = test[:, 0]

    _, end_validation = get_split_indexes(config)
    test_timestamps = config.TIMESTAMPS[end_validation:]

    # plot_test_set(config, test)
    # plot_days(config, test[:96])
    plot_baselines(config, train, test[:96], test_timestamps[:96])

    print("Validation:")
    one_step_persistence_model(validation)
    one_day_persistence_model(config, validation)
    meanBaseline(config, train, validation)
    predict_zero_one_day(config, validation)
    predict_zero_one_step(validation)

    print("Test:")
    one_step_persistence_model(test)
    one_day_persistence_model(config, test)
    meanBaseline(config, train, test)
    print("Train on test and predict for Test:")
    meanBaseline(config, test, test)
    predict_zero_one_day(config, test)
    predict_zero_one_step(test)


if __name__ == "__main__":
    main(sys.argv)
