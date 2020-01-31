import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from data import getPecanstreetData
from util import constructTimeStamps
from util import makeTick

from forecast_load_conf import ForecastLoadConfig

import time


def getData(config, timestamps):
    # input datas : uncontrolable resource : solar production
    loadsData = getPecanstreetData(
        config.DATA_FILE, config.TIME_HEADER, config.DATAID, "grid", timestamps
    )
    return loadsData


def splitData(config, loadsData):
    diff = loadsData.index[-1] - loadsData.index[0]
    endTrain = 96 * int(diff.days * config.TRAINPART)
    endValidation = endTrain + 96 * int(diff.days * config.VALIDATIONPART)
    return (
        loadsData[:endTrain],
        loadsData[endTrain:endValidation],
        loadsData[endValidation:],
    )


def create_dataset(dataset, look_back):
    dataX, dataY = (
        np.empty((len(dataset) - look_back, look_back, 1)),
        np.empty((len(dataset) - look_back, 1)),
    )
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back)]
        # minutes = np.array([(i.hour * 60 + i.minute) / 1440 for i in a.index])
        # minutes = np.expand_dims(minutes, axis=1)
        a = a.values
        a = np.expand_dims(a, axis=1)
        dataX[i] = a  # np.concatenate((a, minutes), axis=1)
        dataY[i] = dataset.values[i + look_back]
    return dataX, dataY


def buildSets(config, loadsData):
    trainPart, validationPart, testPart = splitData(config, loadsData)

    trainX, trainY = create_dataset(trainPart, config.LOOK_BACK)
    validationX, validationY = create_dataset(validationPart, config.LOOK_BACK)
    testX, testY = create_dataset(testPart, config.LOOK_BACK)
    print(testX.shape)
    print(testY.shape)
    time.sleep(3)

    return trainX, trainY, validationX, validationY, testX, testY


def plotPart(timestamps, real):
    time, tick = makeTick(timestamps, "%H:%M")

    x1 = list(range(len(real)))

    plt.plot(x1, real, label="house load", color="green")

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time on Day " + timestamps[0].strftime("%m-%d"))
    plt.ylabel("Power consumption (kW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def getMeanSdDay(data):
    data = np.reshape(data, (96, int(len(data) / 96)))
    means = np.nanmean(data, axis=1)
    standard_dev = np.nanstd(data, axis=1)
    return means, standard_dev


def plotDay(timestamps, realY):
    realMeans, realSd = getMeanSdDay(realY)
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


def main(argv):
    config = ForecastLoadConfig()

    timestamps = constructTimeStamps(
        datetime.strptime(config.BEGIN, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.END, "20%y-%m-%d %H:%M:%S"),
        datetime.strptime(config.STEPSIZE, "%H:%M:%S")
        - datetime.strptime("00:00:00", "%H:%M:%S"),
    )

    loadsData = getData(config, timestamps).values
    plotDay(timestamps, loadsData)

    plotPart(timestamps[:96], loadsData[:96])
    plotPart(timestamps[96 * 10 : 96 * 11], loadsData[96 * 10 : 96 * 11])
    plotPart(timestamps[96 * 100 : 96 * 101], loadsData[96 * 100 : 96 * 101])


if __name__ == "__main__":
    main(sys.argv)
