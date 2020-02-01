import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from util import makeTick

from forecast import get_timestamps_per_day


def getMeanSdDayBaseline(config, data):
    timestamps_per_day = get_timestamps_per_day(config)
    data = np.reshape(data, (timestamps_per_day, int(len(data) / timestamps_per_day)))
    means = np.mean(data, axis=1)
    standard_dev = np.std(data, axis=1)
    return means, standard_dev


def plotDayBaseline(config, timestamps, realY, predictY):
    timestamps_per_day = get_timestamps_per_day(config)
    realMeans, realSd = getMeanSdDayBaseline(config, realY)
    x1 = list(range(timestamps_per_day))

    plt.plot(x1, realMeans, label="actual", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )
    plt.plot(x1, predictY, label="predict Baseline", color="orange")

    time, tick = makeTick(timestamps[:timestamps_per_day], "%H:%M")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotSets(timestamps, train, validation, test):
    time, tick = makeTick(timestamps)

    x1 = range(len(train))
    x2 = range(len(train), len(train) + len(validation))
    x3 = range(len(train) + len(validation), len(timestamps))
    plt.plot(x1, train, label="train set", color="green")
    plt.plot(x2, validation, label="validation set", color="blue")
    plt.plot(x3, test, label="test set", color="red")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def predictMean(config, train, test):
    timestamps_per_day = get_timestamps_per_day(config)
    data = np.reshape(train, (timestamps_per_day, int(len(train) / timestamps_per_day)))
    means = np.mean(data, axis=1)
    predictions = np.array(means)
    for i in range(int(len(test) / timestamps_per_day) - 1):
        predictions = np.concatenate((predictions, means))
    return predictions


def meanBaseline(config, train, test):
    timestamps_per_day = get_timestamps_per_day(config)
    predictions = predictMean(config, train, test)
    assert len(test) % timestamps_per_day == 0
    mse = mean_squared_error(predictions, test)
    print("mean baseline mse: ", mse)
    return mse


def predict_zero_one_step(part):
    assert len(part.shape) == 1
    predictions = [0 for i in range(len(part) - 1)]
    real = part[1:]
    mse = mean_squared_error(real, predictions)
    print("predict 0 for 1 step output MSE: ", mse)


def predict_zero_one_day(config, part):
    assert len(part.shape) == 1
    predictions = np.zeros((len(part) - 2 * config.OUTPUT_SIZE, config.OUTPUT_SIZE))
    real = np.empty((len(part) - 2 * config.OUTPUT_SIZE, config.OUTPUT_SIZE))
    for i in range(len(part) - 2 * config.OUTPUT_SIZE):
        real[i] = part[i + config.OUTPUT_SIZE : i + 2 * config.OUTPUT_SIZE]
    mse = mean_squared_error(real, predictions)
    print("predict 0 for day output MSE: ", mse)


def one_step_persistence_model(part):
    assert len(part.shape) == 1
    predictions = part[0:-1]
    real = part[1:]
    mse = mean_squared_error(real, predictions)
    print("1 Step Persistence Model MSE: ", mse)


def one_day_persistence_model(config, part):
    assert len(part.shape) == 1
    predictions = np.empty((len(part) - 2 * config.OUTPUT_SIZE, config.OUTPUT_SIZE))
    real = np.empty((len(part) - 2 * config.OUTPUT_SIZE, config.OUTPUT_SIZE))
    for i in range(len(part) - 2 * config.OUTPUT_SIZE):
        predictions[i] = part[i : i + config.OUTPUT_SIZE]
        real[i] = part[i + config.OUTPUT_SIZE : i + 2 * config.OUTPUT_SIZE]
    mse = mean_squared_error(real, predictions)
    print("1 Day Persistence Model MSE: ", mse)


def main(argv):
    pass


if __name__ == "__main__":
    main(sys.argv)
