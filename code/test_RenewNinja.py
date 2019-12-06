import unittest
from datetime import datetime, timedelta

from RenewNinja import getSamplePvApi


class Test(unittest.TestCase):
    def testGetSamplePvApi(self):
        beg = str(datetime.today() - timedelta(days=1) - timedelta(days=365)).split(" ")[0]
        end = str(datetime.today() - timedelta(days=365)).split(" ")[0]

        end = beg  # to only have 1 day
        metadata, data = getSamplePvApi(beg, end)
        self.assertEqual(len(data), 24)


if __name__ == "__main__":
    unittest.main()
