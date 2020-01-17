from datetime import timedelta, datetime
import numpy as np


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

    indexList = np.reshape(indexList, (1, -1,))[0, :].tolist()
    return indexList


def getConcatDateTime(date, time):
    return datetime(
        date.year, date.month, date.day, time.hour, time.minute, time.second
    )
