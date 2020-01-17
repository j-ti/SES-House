import pandas as pd


def constructTimeStamps(start, end, stepsize):
    times = []
    timeIterator = start
    while timeIterator <= end:
        times.append(timeIterator)
        timeIterator += stepsize
    assert times[-1] == end
    return times


def getStepsize(timestamps):
    assert len(timestamps) >= 2
    stepsize = timestamps[1] - timestamps[0]
    for i in range(2, len(timestamps) - 1):
        assert timestamps[i] - timestamps[i - 1] == stepsize

    return stepsize


def getTimeIndexRange(timestamps, timeA, timeB):
    return list(range(timestamps.index(timeA), timestamps.index(timeB) + 1))


def makeShiftTrain(df_base, df, look_back, split):
    for i in range(1, look_back):
        s = df_base[look_back - i:split - i].reset_index(drop=True)
        df = pd.concat([df, s], axis=1, ignore_index=True)
    return df


def makeShiftTest(df_base, df, look_back, split):
    for i in range(1, look_back):
        s = df_base[split + look_back + i:len(df_base) - look_back + i].reset_index(drop=True)
        df = pd.concat([df, s], axis=1, ignore_index=True)
    return df


def makeTick(timestamps) :
    step = int(len(timestamps) / 14)
    time = [timestamps[i].strftime("%m-%d %H:%M") for i in range(len(timestamps))][::step]
    tick = [i for i in range(len(timestamps))][::step]
    return time, tick
