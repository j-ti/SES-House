class Config:
    def __init__(self):
        self.SEED = 3
        self.EPOCHS = 20
        self.BATCH_SIZE = 20

        self.LOOK_BACK = 48  # we have a 5 point history in our input
        self.PART = 0.8  # we train on part of the set

        self.BEGIN = "2019-05-01 00:00:00"
        self.END = "2019-05-31 00:00:00"
        self.STEPSIZE = "00:15:00"
        self.LAYERS = 256
        self.INPUT_SHAPE = (1, self.LOOK_BACK)
        self.ACTIVATION_FUNCTION = "tanh"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"

        self.DATA_FILE = "./data/15minute_data_newyork.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecasting/load/"
