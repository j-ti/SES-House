import numpy as np
import pandas as pd
from datetime import timedelta, datetime


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


def getTimeIndexRangeDaily(timestamps, timeA, timeB):
    indexList = []
    start = timestamps[0]
    end = timestamps[-1]
    dayIterator = datetime(start.year, start.month, start.day)
    while dayIterator <= end:
        indexList.append(
            list(
                range(
                    timestamps.index(getConcatDateTime(dayIterator, timeA)),
                    timestamps.index(getConcatDateTime(dayIterator, timeB)) + 1,
                )
            )
        )
        dayIterator += timedelta(days=1)

    indexList = np.reshape(indexList, (1, -1))[0, :].tolist()
    return indexList


def getConcatDateTime(date, time):
    return datetime(
        date.year, date.month, date.day, time.hour, time.minute, time.second
    )


def makeShiftTrain(df_base, df, look_back, split):
    for i in range(1, look_back):
        s = df_base[look_back - i : split - i].reset_index(drop=True)
        df = pd.concat([df, s], axis=1, ignore_index=True)
    return df


def makeShiftTest(df_base, df, look_back, split):
    for i in range(1, look_back):
        s = df_base[split + look_back + i : len(df_base) - look_back + i].reset_index(
            drop=True
        )
        df = pd.concat([df, s], axis=1, ignore_index=True)
    return df


def makeTick(timestamps):
    step = int(len(timestamps) / 14)
    time = [timestamps[i].strftime("%m-%d %H:%M") for i in range(len(timestamps))][
        ::step
    ]
    tick = [i for i in range(len(timestamps))][::step]
    return time, tick


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true + np.finfo(float).eps
    y_pred = y_pred + np.finfo(float).eps
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
