import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data import getPecanstreetData
from util import constructTimeStamps, mean_absolute_percentage_error
from util import makeTick

from forecast import splitData, addMinutes, buildSet, train, saveModel
from forecast_conf import ForecastConfig
from forecast_load_conf import ForecastLoadConfig

import time


def getMeanSdDayBaseline(data):
    data = np.reshape(data, (96, int(len(data) / 96)))
    means = np.mean(data, axis=1)
    standard_dev = np.std(data, axis=1)
    return means, standard_dev


def plotDayBaseline(timestamps, realY, predictY):
    realMeans, realSd = getMeanSdDayBaseline(realY)
    x1 = list(range(96))

    plt.plot(x1, realMeans, label="actual", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )
    plt.plot(x1, predictY, label="predict Baseline", color="orange")

    time, tick = makeTick(timestamps[:96], "%H:%M")
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


def predictMean(train, test):
    assert len(train) % 96 == 0
    data = np.reshape(train, (96, int(len(train) / 96)))
    means = np.mean(data, axis=1)
    predictions = np.array(means)
    for i in range(int(len(test) / 96) - 1):
        predictions = np.concatenate((predictions, means))
    print(predictions)
    return predictions


def meanBaseline(train, test):
    predictions = predictMean(train, test)
    assert len(test) % 96 == 0
    mse = mean_squared_error(predictions, test)
    print("Baseline MSE: ", mse)
    return mse


def one_step_persistence_model(part):
    part_x = part[0:-1, 0]
    part_y = part[1:, 0]
    mse = mean_squared_error(part_x, part_y)
    print("1 Step Persistence Model MSE: ", mse)

    plot_x = range(len(part_x))
    plt.plot(plot_x[0:96], part_y[0:96], label="actual", color="green")
    plt.plot(
        plot_x[0:96], part_x[0:96], label="persistence model prediction", color="orange"
    )
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def one_day_persistence_model(part):
    pass


def main(argv):
    pass


if __name__ == "__main__":
    main(sys.argv)
