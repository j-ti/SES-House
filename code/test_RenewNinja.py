import unittest

from datetime import datetime, timedelta

from RenewNinja import getSamplePvApi


class Test(unittest.TestCase):
    def testGetSamplePvApi(self):
        metadata, data = getSamplePvApi(
            datetime.today() - timedelta(days=1), datetime.today()
        )
        self.assertEqual(len(data), 24)


if __name__ == "__main__":
    unittest.main()
