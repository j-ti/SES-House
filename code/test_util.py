import unittest
from datetime import datetime, timedelta

from util import constructTimeStamps, getStepsize, getTimeIndexRange


class Test(unittest.TestCase):
    def testConstructTimeStamps(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 23, 0, 0)
        stepsize = timedelta(hours=1)
        timestamps = constructTimeStamps(start, end, stepsize)
        self.assertEqual(24, len(timestamps))
        self.assertEqual(start, timestamps[0])
        self.assertEqual(datetime(2014, 1, 2, 23, 0, 0), timestamps[23])

    def testConstructTimeStampsError(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 5, 0, 0)
        stepsize = timedelta(hours=2)
        try:
            constructTimeStamps(start, end, stepsize)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

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


if __name__ == "__main__":
    unittest.main()
