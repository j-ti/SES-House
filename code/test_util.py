import unittest
from datetime import datetime, timedelta

from util import (
    constructTimeStamps,
    getStepsize,
    getTimeIndexRange,
    getTimeIndexRangeDaily,
    getConcatDateTime,
)


class Test(unittest.TestCase):
    def testConstructTimeStamps(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 23, 0, 0)
        stepsize = timedelta(hours=1)
        timestamps = constructTimeStamps(start, end, stepsize)
        self.assertEqual(24, len(timestamps))
        self.assertEqual(start, timestamps[0])
        self.assertEqual(datetime(2014, 1, 2, 23, 0, 0), timestamps[23])

    def testConstructTimeStampsWithUnfittingEnd(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 5, 5, 5)
        stepsize = timedelta(hours=2)
        timestamps = constructTimeStamps(start, end, stepsize)
        self.assertEqual(3, len(timestamps))
        self.assertEqual(datetime(2014, 1, 2, 4, 0, 0), timestamps[-1])

    def testStepsize(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 23, 0, 0)
        stepsize = timedelta(hours=1)
        timestamps = constructTimeStamps(start, end, stepsize)
        self.assertEqual(getStepsize(timestamps), stepsize)

    def testTimeIndexRange(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 23, 0, 0)
        stepsize = timedelta(hours=1)
        timestamps = constructTimeStamps(start, end, stepsize)
        indexList = getTimeIndexRange(timestamps, timestamps[0], timestamps[7])
        self.assertEqual(indexList, [0, 1, 2, 3, 4, 5, 6, 7])

    def testTimeIndexRangeDaily(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 4, 23, 0, 0)
        cutStart = datetime.strptime("10:00:00", "%H:%M:%S")
        cutEnd = datetime.strptime("13:00:00", "%H:%M:%S")
        stepsize = timedelta(hours=1)
        timestamps = constructTimeStamps(start, end, stepsize)
        indexList = getTimeIndexRangeDaily(timestamps, cutStart, cutEnd)
        self.assertEqual(
            indexList,
            [
                10,
                11,
                12,
                13,
                24 + 10,
                24 + 11,
                24 + 12,
                24 + 13,
                2 * 24 + 10,
                2 * 24 + 11,
                2 * 24 + 12,
                2 * 24 + 13,
            ],
        )

    def testConcatDateTime(self):
        date = datetime(2014, 1, 4, 23, 0, 0)
        time = datetime.strptime("12:34:56", "%H:%M:%S")
        DateAndTime = getConcatDateTime(date, time)
        self.assertEqual(datetime(2014, 1, 4, 12, 34, 56), DateAndTime)


if __name__ == "__main__":
    unittest.main()
