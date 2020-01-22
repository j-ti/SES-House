import matplotlib.pyplot as plt
from util import makeTick
import numpy as np

outputFolder = ""

def plotPrediction(train_y, train_predict_y, test_y, test_predict_y, timestamps):
    time, tick = makeTick(timestamps)

    x1 = [i for i in range(len(train_y))]
    x2 = [i for i in range(len(train_y), len(test_y) + len(train_y))]
    plt.plot(x1, train_y.reset_index(drop=True), label="actual", color="green")
    plt.plot(x1, train_predict_y, label="predict", color="orange")
    plt.plot(x2, test_y.reset_index(drop=True), label="actual", color="blue")
    plt.plot(x2, test_predict_y, label="predict", color="red")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/prediction.png")
    plt.show()


def plot100first(train_y, train_predict_y):
    x1 = [i for i in range(len(train_y))]
    plt.plot(x1[:100], train_y.reset_index(drop=True)[:100], label="actual", color="green")
    plt.plot(x1[:100], train_predict_y[:100], label="predict", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Power output (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/100first.png")
    plt.show()


def plotEcart(train_y, train_predict_y, test_y, test_predict_y, timestamps):
    time, tick = makeTick(timestamps)

    x1 = [i for i in range(len(train_y))]
    x2 = [i for i in range(len(train_y), len(test_y) + len(train_y))]
    y1 = [train_predict_y[i] - train_y[i] for i in range(len(x1))]
    y2 = [test_predict_y[i] - test_y[i] for i in range(len(x2))]
    plt.plot(x1, y1, label="actual", color="green")
    plt.plot(x2, y2, label="actual", color="blue")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Difference (kW)")
    plt.legend()
    plt.savefig(outputFolder + "/difference.png")
    plt.show()


def plotInput(df, timestamps):
    time, tick = makeTick(timestamps)
    y = np.array(df)
    plt.plot(y, label="actual", color="green")
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Input datas")
    plt.legend()
    plt.savefig(outputFolder + "/input.png")
    plt.show()


def plotHistory(history):
    plt.plot(history.history["mean_absolute_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error")
    plt.savefig(outputFolder + "/MAE.png")
    plt.show()
    plt.plot(history.history["mean_absolute_percentage_error"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute percentage error")
    plt.savefig(outputFolder + "/MAPE.png")
    plt.show()
    plt.plot(history.history["mean_squared_error"])
    plt.xlabel("Epoch")
    plt.ylabel("mean_squared_error")
    plt.savefig(outputFolder + "/MSE.png")
    plt.show()
