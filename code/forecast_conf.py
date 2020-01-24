class ForecastConfig:
    def __init__(self):
        self.SEED = 23

        self.BEGIN = "2019-05-01 00:00:00"
        self.END = "2019-10-31 23:45:00"
        self.STEPSIZE = "00:15:00"

        self.TRAIN_FRACTION = 0.6
        self.VALIDATION_FRACTION = (1 - self.TRAINPART) / 2
