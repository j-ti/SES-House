import unittest

from datetime import datetime, timedelta

from load_loads import getLoadsData
from util import constructTimeStamps


class Test(unittest.TestCase):
    def testGetLoadsData(self):
        start = datetime(2014, 1, 1, 0, 0, 0)
        end = datetime(2014, 1, 1, 23, 59, 59)
        stepsize = timedelta(hours=1)
        loads = getLoadsData(
            "../sample/pecan-home-grid_solar-manipulated.csv", start, end, stepsize
        )
        self.assertEqual(len(constructTimeStamps(start, end, stepsize)), len(loads))
        [self.assertGreaterEqual(load, 0) for load in loads]


if __name__ == "__main__":
    unittest.main()
