import unittest

from datetime import datetime, timedelta

from util import constructTimeStamps


class Test(unittest.TestCase):
    def testGetSamplePv(self):
        start = datetime(2014, 1, 2, 0, 0, 0)
        end = datetime(2014, 1, 2, 23, 59, 59)
        stepsize = timedelta(hours=1)
        times = constructTimeStamps(start, end, stepsize)
        self.assertEqual(24, len(times))
        self.assertEqual(start, times[0])
        self.assertEqual(datetime(2014, 1, 2, 23, 0, 0), times[23])


if __name__ == "__main__":
    unittest.main()
