import unittest
from datetime import datetime

from RenewNinja import getSamplePvApi, getSamplePv


class Test(unittest.TestCase):
    def testGetSamplePv(self):
        data = getSamplePv(
            datetime(2014, 1, 1, 0, 0, 0), datetime(2014, 1, 1, 23, 59, 59)
        )
        self.assertEqual(len(data), 24)
        for electricity in data:
            self.assertGreaterEqual(electricity, 0)

    def testGetSamplePvApi(self):
        metadata, data = getSamplePvApi(datetime(2014, 1, 1), datetime(2014, 1, 1))
        self.assertEqual(len(data), 24)


if __name__ == "__main__":
    unittest.main()
