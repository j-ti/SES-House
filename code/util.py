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
    return list(range(timestamps.index(timeA), timestamps.index(timeB)))
