from datetime import datetime

from util import constructTimeStamps


class ForecastConfig:
    def __init__(self):
        self.SEED = 15

        self.BEGIN = "2019-05-01 00:00:00"
        self.END = "2019-10-31 23:45:00"
        self.STEPSIZE = "00:30:00"
        self.TIMESTAMPS = constructTimeStamps(
            datetime.strptime(self.BEGIN, "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(self.END, "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(self.STEPSIZE, "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )
        self.OUTPUT_FOLDER = ""
        self.TRAIN_FRACTION = 0.6
        self.VALIDATION_FRACTION = (1 - self.TRAIN_FRACTION) / 2
