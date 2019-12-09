import unittest

from datetime import datetime, timedelta

from load_price import getPriceData
from util import constructTimeStamps


class Test(unittest.TestCase):
    def testGetPriceData(self):
        start = datetime(2014, 1, 1, 0, 0, 0)
        end = datetime(2014, 1, 1, 23, 59, 59)
        stepsize = timedelta(hours=1)
        price = getPriceData(
            "../sample/pecan-iso_neiso-day_ahead_lmp_avg-20190602.csv", start, end, stepsize
        )
        self.assertEqual(len(constructTimeStamps(start, end, stepsize)), len(price))
        [self.assertGreaterEqual(price_n, 0) for price_n in price]


if __name__ == "__main__":
    unittest.main()
