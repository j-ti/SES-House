import unittest
from datetime import datetime, timedelta

from data import (
    getPriceData,
    getLoadsData,
    getNinjaPvApi,
    getNinja,
    NetworkException,
    getPecanstreetData,
)
from util import constructTimeStamps


class LoadsTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)
        self.dataFile = "./sample/pecan-home86-grid-201401010000_201402010000-15m.csv"

    def testGetLoadsData(self):
        stepsize = timedelta(minutes=15)
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
        self.assertAlmostEqual(loads[0], 2.09075)

    def testGetLoadsDataOversample(self):
        stepsize = timedelta(minutes=1)
        loads = getLoadsData(
            self.dataFile, constructTimeStamps(self.start, self.end, stepsize)
        )
        self.assertEqual(len(loads), 22 * 60 + 1)
        self.assertEqual(loads[0], 2.444)
        for index in range(14):
            self.assertEqual(loads[index], loads[index + 1])


class PecanstreetTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2018, 11, 21, 16, 0, 0)
        self.end = datetime(2018, 11, 21, 19, 0, 0)
        self.dataFile = (
            "./sample/austin_15minute_data_sample.csv"  # (local_15min), localminute
        )
        self.dataid = 661
        self.timeindex = "local_15min"
        self.column = "grid"

    def testGetLoadsData(self):
        stepsize = timedelta(minutes=15)
        loads = getPecanstreetData(
            self.dataFile,
            self.dataid,
            self.column,
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(loads)
        )
        [self.assertGreaterEqual(load, 0) for load in loads]

    def testGetLoadsDataDownsample(self):
        stepsize = timedelta(hours=2)
        loads = getPecanstreetData(
            self.dataFile,
            self.dataid,
            self.column,
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(len(loads), 2)
        self.assertAlmostEqual(loads[0], 1.064125)
        self.assertAlmostEqual(loads[1], 1.206375)

    def testGetLoadsDataOversample(self):
        stepsize = timedelta(minutes=1)
        loads = getPecanstreetData(
            self.dataFile,
            self.dataid,
            self.column,
            constructTimeStamps(self.start, self.end, stepsize),
        )
        self.assertEqual(len(loads), 3 * 60 + 1)
        self.assertEqual(loads[0], 0.833)
        for index in range(14):
            self.assertEqual(loads[index], loads[index + 1])


class PriceTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2014, 1, 1, 0, 0, 0)
        self.end = datetime(2014, 1, 1, 22, 0, 0)
        self.offset = 5 * 365 + 1  # 5 years difference
        self.constantPrice = 0.14

    def testGetPriceDataDownsample(self):
        stepsize = timedelta(hours=2)
        prices = getPriceData(
            "./sample/pecan-iso_neiso-day_ahead_lmp_avg-201901010000-201902010000.csv",
            constructTimeStamps(self.start, self.end, stepsize),
            timedelta(days=self.offset),
            self.constantPrice,
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(prices)
        )
        [self.assertGreaterEqual(price_n, 0) for price_n in prices]

        self.assertAlmostEqual(prices[0], (25.308 + 20.291) / 1000 + self.constantPrice)
        self.assertAlmostEqual(prices[-1], (24.2 + 23.417) / 1000 + self.constantPrice)

    def testGetPriceDataOversample(self):
        stepsize = timedelta(minutes=1)
        prices = getPriceData(
            "./sample/pecan-iso_neiso-day_ahead_lmp_avg-201901010000-201902010000.csv",
            constructTimeStamps(self.start, self.end, stepsize),
            timedelta(days=self.offset),
            self.constantPrice,
        )
        self.assertEqual(
            len(constructTimeStamps(self.start, self.end, stepsize)), len(prices)
        )
        [self.assertGreaterEqual(price_n, 0) for price_n in prices]

        self.assertAlmostEqual(prices[0], (25.308 / 60) / 1000 + self.constantPrice)
        self.assertAlmostEqual(prices[59], (25.308 / 60) / 1000 + self.constantPrice)
        self.assertAlmostEqual(prices[-1], 24.2 / 60 / 1000 + self.constantPrice)


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
