from datetime import timedelta, datetime

import numpy as np
import pandas as pd


def constructTimeStamps(start, end, stepsize):
    times = []
    timeIterator = start
    while timeIterator <= end:
        times.append(timeIterator)
        timeIterator += stepsize
    return times


def constructTimeStampsDaily(start, end, timeA, timeB, stepsize):
    times = []
    dayIterator = datetime(start.year, start.month, start.day)
    while dayIterator <= end:
        timeIterator = datetime(
            dayIterator.year,
            dayIterator.month,
            dayIterator.day,
            timeA.hour,
            timeA.minute,
            timeA.second,
        )
        dailyTimeB = datetime(
            dayIterator.year,
            dayIterator.month,
            dayIterator.day,
            timeB.hour,
            timeB.minute,
            timeB.second,
        )
        while timeIterator <= dailyTimeB:
            times.append(timeIterator)
            timeIterator += stepsize
        dayIterator += timedelta(days=1)

    return times


def getStepsize(timestamps):
    assert len(timestamps) >= 2
    stepsize = timestamps[1] - timestamps[0]
    for i in range(2, len(timestamps) - 1):
        assert timestamps[i] - timestamps[i - 1] == stepsize

    return stepsize


def getTimeIndexRange(timestamps, timeA, timeB):
    return list(range(timestamps.index(timeA), timestamps.index(timeB) + 1))


def getTimeIndexRangeDaily(timestamps, timeA, timeB, varA=0, varB=1):
    indexList = []
    start = timestamps[0]
    end = timestamps[-1]
    dayIterator = datetime(start.year, start.month, start.day)
    while dayIterator <= end:
        indexList.append(
            list(
                range(
                    timestamps.index(getConcatDateTime(dayIterator, timeA)) + varA,
                    timestamps.index(getConcatDateTime(dayIterator, timeB)) + varB,
                )
            )
        )
        dayIterator += timedelta(days=1)

    indexList = np.reshape(indexList, (1, -1))[0, :].tolist()
    return indexList


def diffIndexList(timestampsA, timestampsB):
    return list(set(timestampsA) - set(timestampsB))


def getConcatDateTime(date, time):
    return datetime(
        date.year, date.month, date.day, time.hour, time.minute, time.second
    )


def makeTick(timestamps, present="%m-%d %H:%M"):
    step = int(len(timestamps) / 14)
    time = [timestamps[i].strftime(present) for i in range(len(timestamps))][::step]
    tick = [i for i in range(len(timestamps))][::step]
    return time, tick


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true + np.finfo(float).eps
    y_pred = y_pred + np.finfo(float).eps
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def getMeanSdDay(config, data):
    nans = np.empty((config.LOOK_BACK, 1))
    nans[:] = np.nan
    data = np.concatenate((nans, data), axis=0)
    data = np.reshape(data, (96, int(len(data) / 96)))
    means = np.nanmean(data, axis=1)
    standard_dev = np.nanstd(data, axis=1)
    return means, standard_dev
