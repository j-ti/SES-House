import unittest

from datetime import datetime, timedelta

from load_price import getPriceData
from util import constructTimeStamps


class Test(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)

    def testGetPriceDataDownsample(self):
        stepsize = timedelta(hours=2)
        prices = getPriceData(
            "./sample/pecan-iso_neiso-day_ahead_lmp_avg-20190602.csv",
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(prices)
        )
        [self.assertGreaterEqual(price_n, 0) for price_n in prices]
        self.assertEqual(prices[0], 137.5)
        self.assertEqual(prices[-1], 109)

    def testGetPriceDataOversample(self):
        stepsize = timedelta(minutes=1)
        prices = getPriceData(
            "./sample/pecan-iso_neiso-day_ahead_lmp_avg-20190602.csv",
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(prices)
        )
        [self.assertGreaterEqual(price_n, 0) for price_n in prices]
        self.assertEqual(prices[0], 4)
        self.assertEqual(prices[59], 4)
        self.assertEqual(prices[-1], 109)


if __name__ == "__main__":
    unittest.main()
