import unittest
from datetime import datetime, timedelta

from renew_ninja import getSamplePvApi, getSamplePv, getSampleWind, NetworkException

from util import constructTimeStamps


class Test(unittest.TestCase):
    def testGetSampleWind(self):
        data = getSampleWind(
            "./sample/ninja_wind_52.5170_13.3889_corrected.csv",
            constructTimeStamps(
                datetime(2014, 1, 1, 0, 0, 0),
                datetime(2014, 1, 1, 23, 00, 00),
                timedelta(hours=1),
            ),
        )
        self.assertEqual(len(data), 24)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetSamplePv(self):
        data = getSamplePv(
            "./sample/ninja_pv_52.5170_13.3889_corrected.csv",
            constructTimeStamps(
                datetime(2014, 1, 1, 0, 0, 0),
                datetime(2014, 1, 1, 23, 00, 00),
                timedelta(hours=1),
            ),
        )
        self.assertEqual(len(data), 24)
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
