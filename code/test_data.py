import unittest

from datetime import datetime, timedelta

from data import getPriceData, getLoadsData, getNinjaPvApi, getNinja, NetworkException
from util import constructTimeStamps


class LoadsTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)
        self.dataFile = "./sample/pecan-home-grid_solar-manipulated.csv"

    def testGetLoadsData(self):
        stepsize = timedelta(hours=1)
        loads = getLoadsData(
            self.dataFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(loads)
        )
        [self.assertGreaterEqual(load, 0) for load in loads]

    def testGetLoadsDataDownsample(self):
        stepsize = timedelta(hours=2)
        loads = getLoadsData(
            self.dataFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(len(loads), 12)
        self.assertAlmostEqual(loads[0], 0.430375)

    def testGetLoadsDataOversample(self):
        stepsize = timedelta(minutes=1)
        loads = getLoadsData(
            self.dataFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(len(loads), 22 * 60 + 1)
        self.assertEqual(loads[0], 0.500)
        for index in range(14):
            self.assertEqual(loads[index], loads[index + 1])


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
        self.assertEqual(prices[0], 275 * 1000)
        self.assertEqual(prices[-2], (384 + 629) * 1000)
        self.assertEqual(prices[-1], 109 * 1000)

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
        self.assertAlmostEqual(prices[0], (4 / 60) * 1000)
        self.assertAlmostEqual(prices[59], (4 / 60) * 1000)
        self.assertAlmostEqual(prices[-1], (109 / 60) * 1000)


class NinjaTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)
        self.windFile = "./sample/ninja_wind_52.5170_13.3889_corrected.csv"
        self.pvFile = "./sample/ninja_pv_52.5170_13.3889_corrected.csv"

    def testGetNinjaWindFile(self):
        data = getNinja(
            self.windFile, constructTimeStamps(self.start, self.end, timedelta(hours=1))
        )
        self.assertEqual(len(data), 23)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetNinjaWindFileOversample(self):
        stepsize = timedelta(minutes=1)
        data = getNinja(
            self.pvFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(len(data), 22 * 60 + 1)

    def testGetNinjaWindFileDownsample(self):
        stepsize = timedelta(hours=2)
        data = getNinja(
            self.windFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(len(data), 12)

    def testGetNinjaPvFile(self):
        data = getNinja(
            self.pvFile, constructTimeStamps(self.start, self.end, timedelta(hours=1))
        )
        self.assertEqual(len(data), 23)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetNinjaPvApi(self):
        try:
            metadata, data = getNinjaPvApi(
                52.5170,
                13.3889,
                constructTimeStamps(self.start, self.end, timedelta(hours=1)),
            )
            self.assertEqual(len(data), 24)
        except NetworkException:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
