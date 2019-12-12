import unittest

from datetime import datetime, timedelta

from data import (
    getPriceData,
    getLoadsData,
    getSamplePvApi,
    getSamplePv,
    getSampleWind,
    NetworkException,
)
from util import constructTimeStamps


class LoadsTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)

    def testGetLoadsData(self):
        stepsize = timedelta(hours=1)
        loads = getLoadsData(
            "./sample/pecan-home-grid_solar-manipulated.csv",
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(len(constructTimeStamps(self.start, self.end, stepsize)), len(loads))
        [self.assertGreaterEqual(load, 0) for load in loads]


class PriceTest(unittest.TestCase):
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


class NinjaTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)
        self.windFile = "./sample/ninja_wind_52.5170_13.3889_corrected.csv"
        self.pvFile = "./sample/ninja_pv_52.5170_13.3889_corrected.csv"

    def testGetSampleWind(self):
        data = getSampleWind(
            self.windFile,
            constructTimeStamps(
                self.start,
                self.end,
                timedelta(hours=1),
            ),
        )
        self.assertEqual(len(data), 23)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetSampleWindOversample(self):
        stepsize = timedelta(minutes=1)
        data = getSampleWind(
            self.pvFile,
            constructTimeStamps(
                self.start,
                self.end,
                stepsize,
            ),
        )
        self.assertEqual(len(data), 22 * 60 + 1)

    def testGetSampleWindDownsample(self):
        stepsize = timedelta(hours=2)
        data = getSampleWind(
            self.windFile,
            constructTimeStamps(
                self.start,
                self.end,
                stepsize,
            ),
        )
        self.assertEqual(len(data), 12)

    def testGetSamplePv(self):
        data = getSamplePv(
            self.pvFile,
            constructTimeStamps(
                self.start,
                self.end,
                timedelta(hours=1),
            ),
        )
        self.assertEqual(len(data), 23)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetSamplePvApi(self):
        try:
            metadata, data = getSamplePvApi(
                52.5170,
                13.3889,
                constructTimeStamps(
                    datetime(2014, 1, 1, 0, 0, 0),
                    datetime(2014, 1, 1, 23, 00, 00),
                    timedelta(hours=1),
                ),
            )
            self.assertEqual(len(data), 24)
        except NetworkException:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
