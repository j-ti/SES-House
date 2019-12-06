import unittest
from datetime import datetime

from RenewNinja import getSamplePvApi


class Test(unittest.TestCase):
    def testGetSamplePvApi(self):
        metadata, data = getSamplePvApi(datetime(2014, 1, 1), datetime(2014, 1, 1))
        self.assertEqual(len(data), 24)


if __name__ == "__main__":
    unittest.main()
