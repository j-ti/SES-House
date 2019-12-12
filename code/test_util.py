import unittest

from datetime import datetime, timedelta

from util import constructTimeStamps, getStepsize


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


if __name__ == "__main__":
    unittest.main()
