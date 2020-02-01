import sys

from forecast_load import getNormalizedParts
from forecast_baseline import (
    one_step_persistence_model,
    one_day_persistence_model,
    meanBaseline,
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

    print("Validation:")
    one_step_persistence_model(validation)
    print("Test:")
    one_step_persistence_model(test)

    print("Validation:")
    one_day_persistence_model(loadConfig, validation)
    print("Test:")
    one_day_persistence_model(loadConfig, test)

    print("Validation:")
    meanBaseline(config, train, validation)
    print("Test:")
    meanBaseline(config, train, test)


if __name__ == "__main__":
    main(sys.argv)
