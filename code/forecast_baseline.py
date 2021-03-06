import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from forecast import get_timestamps_per_day
from sklearn.metrics import mean_squared_error
from util import makeTick


def getMeanSdDayBaseline(config, data):
    timestamps_per_day = get_timestamps_per_day(config)
    data = np.reshape(data, (timestamps_per_day, int(len(data) / timestamps_per_day)))
    means = np.mean(data, axis=1)
    standard_dev = np.std(data, axis=1)
    return means, standard_dev


def plot_days(config, test):
    timestamps_per_day = get_timestamps_per_day(config)
    x = list(range(timestamps_per_day))

    means = predictMean(config, test, test)
    for i in range(int(len(test) / timestamps_per_day)):
        label = "day ", i
        plt.plot(
            x,
            test[i * timestamps_per_day : i * timestamps_per_day + timestamps_per_day],
            label=label,
        )
    plt.plot(x, means[:timestamps_per_day], label="mean prediction", color="orange")
    plt.xlabel("Time of Day")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/plot_days.png")
    plt.show()


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
    plt.savefig(config.OUTPUT_FOLDER + "/plotDayBaseline.png")
    plt.show()


def plot_test_set(config, test):
    x3 = range(len(test))
    plt.plot(x3, test, label="test set", color="red")
    plt.plot(x3, predictMean(config, test, test), label="mean", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_following_days(config, matrix_values):
    times_per_day = get_timestamps_per_day(config)
    assert len(matrix_values) % times_per_day == 0

    follow_predicts = np.empty((matrix_values.shape[0] + times_per_day))

    for i in range(int(len(matrix_values) / times_per_day)):
        follow_predicts[
            i * times_per_day : i * times_per_day + times_per_day
        ] = matrix_values[i * times_per_day]
    return follow_predicts


def plot_baselines(config, train, test, timestamps):
    plt.plot(range(len(test)), test, label="test set")
    plt.plot(
        range(len(test)), predictMean(config, train, test), label="mean prediction"
    )
    plt.plot(range(1, len(test)), test[0:-1], label="next value persistence model")
    time, tick = makeTick(timestamps)
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/plot_baselines.png")
    plt.show()


def predictMean(config, train, test):
    timestamps_per_day = get_timestamps_per_day(config)
    data = np.reshape(train, (int(len(train) / timestamps_per_day), timestamps_per_day))
    means = np.mean(data, axis=0)
    predictions = np.array(means)
    assert len(test) % timestamps_per_day == 0
    for i in range(int(len(test) / timestamps_per_day) - 1):
        predictions = np.concatenate((predictions, means))
    return predictions


def mean_baseline_one_step(config, train, test):
    predictions = predictMean(config, train, test)
    assert len(test) == len(predictions)
    mse = mean_squared_error(predictions, test)
    print("mean baseline mse: ", mse)
    return mse


def mean_baseline_one_day(config, train, test):
    times_per_day = get_timestamps_per_day(config)
    data = np.reshape(train, (int(len(train) / times_per_day), times_per_day))
    means = np.mean(data, axis=0)
    real = np.full((len(test) - 2 * times_per_day + 1, times_per_day), np.nan)
    predictions = np.full(real.shape, np.nan)
    for i in range(len(real)):
        predictions[i] = means
        means = np.roll(means, -1)
        real[i] = test[i + times_per_day : i + 2 * times_per_day]
    mse = mean_squared_error(real, predictions)
    print("mean baseline 1 day mse: ", mse)
    return mse


def predict_zero_one_step(part):
    assert len(part.shape) == 1
    predictions = np.zeros((len(part) - 1))
    real = part[1:]
    mse = mean_squared_error(real, predictions)
    print("predict 0 for 1 step output MSE: ", mse)


def predict_zero_one_day(config, part):
    assert len(part.shape) == 1
    times_per_day = get_timestamps_per_day(config)
    real = np.full((len(part) - 2 * times_per_day + 1, times_per_day), np.nan)
    predictions = np.zeros(real.shape)
    for i in range(len(real)):
        real[i] = part[i + times_per_day : i + 2 * times_per_day]
    mse = mean_squared_error(real, predictions)
    print("predict 0 for day output MSE: ", mse)


def one_step_persistence_model(part):
    assert len(part.shape) == 1
    predictions = part[0:-1]
    real = part[1:]
    mse = mean_squared_error(real, predictions)
    print("1 Step Persistence Model MSE: ", mse)


def plotLSTM_Base_Real(config, train, lstm_predict, base, real):
    times_per_day = get_timestamps_per_day(config)
    plt.rc("font", size=17)  # controls default text sizes
    plt.rc("legend", fontsize=12.2)  # legend fontsize
    plt.plot(lstm_predict, label="LSTM prediction")
    plt.plot(real, label="real")
    if base == "mean":
        plt.plot(predictMean(config, train, real), label="mean prediction")
    else:
        x = np.array(list(range(len(real)))) + 1
        plt.plot(x, real, label="next value persistence model")
    plt.legend()
    time = [0, 5, 10, 15, 20]
    ticks = np.array(time) * int(times_per_day / 24)
    time = [
        datetime.strftime(j, "%H:%M")
        for j in [datetime.strptime(str(i), "%H") for i in time]
    ]
    plt.xticks(ticks, time)
    plt.xlabel("Time")
    plt.ylabel("Normalized Output Power")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/plot_lstm_" + base + "_real.png")
    plt.show()


def main(argv):
    pass


if __name__ == "__main__":
    main(sys.argv)
