import matplotlib.pyplot as plt
import numpy as np
from util import makeTick, getMeanSdDay
from sklearn.metrics import mean_squared_error

outputFolder = ""

def plotPrediction(train_y, train_predict_y, val_y, val_predict_y, test_y, test_predict_y, timestamps, config):
    y1, y1b = [], []
    time = []

    for i in range(min((len(train_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y1.extend(train_y[i * config.OUTPUT_SIZE])
        y1b.extend(train_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[i * config.OUTPUT_SIZE])

    y2, y2b = [], []
    for i in range(min((len(val_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y2.extend(val_y[i * config.OUTPUT_SIZE])
        y2b.extend(val_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[len(train_predict_y) + i * config.OUTPUT_SIZE])

    y3, y3b = [], []
    for i in range(min((len(test_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y3.extend(test_y[i * config.OUTPUT_SIZE])
        y3b.extend(test_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[len(train_predict_y) + len(val_predict_y) + i * config.OUTPUT_SIZE])

    y1, y1b, y2, y2b, y3, y3b = np.array(y1), np.array(y1), np.array(y2), np.array(y2b), np.array(y3), np.array(y3b)

    x1 = np.array(list(range(len(y1))))
    x2 = np.array(list(range(len(y2)))) + len(x1)
    x3 = np.array(list(range(len(y3)))) + len(x1) + len(x2)

    time, tick = makeTick(time)
    tick = [i * config.TIME_PER_DAY for i in tick]
    plt.plot(x1, y1, label="actual - train", color="green")
    plt.plot(x1, y1b, label="predict - train", color="orange")
    plt.plot(x2, y2, label="actual - val", color="blue")
    plt.plot(x2, y2b, label="predict - val", color="red")
    plt.plot(x3, y3, label="actual - test", color="green")
    plt.plot(x3, y3b, label="predict - test", color="orange")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/prediction.png")
    plt.show()


def plotPredictionPart(real, predicted, nameOfSet, timestamps):
    time, tick = makeTick(timestamps)
    x1 = list(range(len(real)))

    plt.plot(x1, real, label="actual of " + nameOfSet, color="green")
    plt.plot(x1, predicted, label="predicted of " + nameOfSet, color="orange")

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotDay(config, timestamps, realY, predictY):
    assert len(realY) == len(predictY)
    realMeans, realSd = getMeanSdDay(config, realY)
    predictedMeans, predictedSd = getMeanSdDay(config, predictY)
    x1 = list(range(96))

    plt.plot(x1, realMeans, label="actual", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )
    plt.plot(x1, predictedMeans, label="predict", color="orange")
    plt.fill_between(
        x1,
        predictedMeans - predictedSd * 0.5,
        predictedMeans + predictedSd * 0.5,
        color="orange",
        alpha=0.5,
    )

    time, tick = makeTick(timestamps[:96], "%H:%M")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot100first(train_y, train_predict_y):
    x1 = [i for i in range(len(train_y))]
    plt.plot(x1[:100], train_y[:100], label="actual", color="green")
    plt.plot(x1[:100], train_predict_y[:100], label="predict", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/100first.png")
    plt.show()


def plotInputDay(df, timestamps, config):
    realMeans, realSd = getMeanSdDay(config, df)
    x1 = list(range(96))

    plt.plot(x1, realMeans, label="mean", color="green")
    plt.fill_between(
        x1, realMeans - realSd * 0.5, realMeans + realSd * 0.5, color="green", alpha=0.5
    )

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


def plotHistory(config, history):
    plt.plot(history.history["mean_absolute_error"], label="train")
    plt.plot(history.history["val_mean_absolute_error"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/MAE.png")
    plt.show()
    plt.plot(history.history["mean_absolute_percentage_error"], label="train")
    plt.plot(history.history["val_mean_absolute_percentage_error"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute percentage error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/MAPE.png")
    plt.show()
    plt.plot(history.history["mean_squared_error"], label="train")
    plt.plot(history.history["val_mean_squared_error"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("mean_squared_error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUTPUT_FOLDER + "/MSE.png")
    plt.show()


def plotEcart(train_y, train_predict_y, val_y, val_predict_y, test_y, test_predict_y, timestamps, config):
    time = []

    y1, y1b = [], []
    for i in range(min((len(train_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y1.extend(train_y[i * config.OUTPUT_SIZE])
        y1b.extend(train_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[i * config.OUTPUT_SIZE])

    y2, y2b = [], []
    for i in range(min((len(val_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y2.extend(val_y[i * config.OUTPUT_SIZE])
        y2b.extend(val_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[len(train_predict_y) + i * config.OUTPUT_SIZE])

    y3, y3b = [], []
    for i in range(min((len(test_predict_y) // config.OUTPUT_SIZE) - 1, config.NB_PLOT)):
        y3.extend(test_y[i * config.OUTPUT_SIZE])
        y3b.extend(test_predict_y[i * config.OUTPUT_SIZE])
        time.append(timestamps[len(train_predict_y) + len(val_predict_y) + i * config.OUTPUT_SIZE])

    y1, y1b, y2, y2b, y3, y3b = np.array(y1), np.array(y1b), np.array(y2), np.array(y2b), np.array(y3), np.array(y3b)

    y1 = [y1[i] - y1b[i] for i in range(len(y1))]
    y2 = [y2[i] - y2b[i] for i in range(len(y2))]
    y3 = [y3[i] - y3b[i] for i in range(len(y3))]

    x1 = np.array(list(range(len(y1))))
    x2 = np.array(list(range(len(y2)))) + len(x1)
    x3 = np.array(list(range(len(y3)))) + len(x1) + len(x2)

    time, tick = makeTick(time)
    tick = [i * config.TIME_PER_DAY for i in tick]

    plt.plot(x1, y1, label="train", color="green")
    plt.plot(x2, y2, label="validation", color="blue")
    plt.plot(x3, y3, label="test", color="orange")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Difference (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/difference.png")
    plt.show()


def plotEcartMSE(train_y, train_predict_y, val_y, val_predict_y, test_y, test_predict_y, timestamps, config):

    y1 = [mean_squared_error(train_y[i], train_predict_y[i]) for i in range(len(train_y))]
    y2 = [mean_squared_error(val_y[i], val_predict_y[i]) for i in range(len(val_y))]
    y3 = [mean_squared_error(test_y[i], test_predict_y[i]) for i in range(len(test_y))]

    x1 = np.array(list(range(len(y1))))
    x2 = np.array(list(range(len(y2)))) + len(x1)
    x3 = np.array(list(range(len(y3)))) + len(x1) + len(x2)

    time, tick = makeTick(timestamps)
    plt.plot(x1, y1, label="train", color="green", linewidth=.5)
    plt.plot(x2, y2, label="validation", color="blue", linewidth=.5)
    plt.plot(x3, y3, label="test", color="orange", linewidth=.5)
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time of the prediction")
    plt.ylabel("Difference MSE")
    plt.legend()
    plt.savefig(outputFolder + "/differenceMSE.png")
    plt.show()
