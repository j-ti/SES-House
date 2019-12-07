def constructTimeStamps(start, end, stepsize):
    times = []
    timeIterator = start
    while timeIterator <= end:
        times.append(timeIterator)
        timeIterator += stepsize
    return times
