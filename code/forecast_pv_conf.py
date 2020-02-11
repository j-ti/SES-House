class ForecastPvConfig:
    def __init__(self, conf):
        self.DATA_FILE = "./data/15minute_data_newyork.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecast/pv/ts30_out48_lb24_bs15"
        self.BEGIN = conf.BEGIN
        self.END = conf.END
        self.STEP_SIZE = "01:00:00"
        self.TIME_PER_DAY = 24

        self.LOAD_MODEL = True
        self.MODEL_ID_SAMPLES = [
            "ts60_out24_lb24_bs15",
            "ts30_out48_lb24_bs15",
            "ts15_out96_lb24_bs15",
        ]
        self.MODEL_ID = self.MODEL_ID_SAMPLES[0]
        self.OUTPUT_FOLDER = "./output/forecast/pv/" + self.MODEL_ID + "/"
        self.MODEL_FILE = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".json"
        self.MODEL_FILE_H5 = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".h5"

        self.EPOCHS = 30
        self.LOOK_BACK = 24
        self.BATCH_SIZE = 15
        self.DROPOUT = [0.3]
        self.DENSE = 1
        self.NEURONS = [150]
        self.PATIENCE = 5
        self.MIN_DELTA = 0.0001
        self.OUTPUT_SIZE = 24
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
        self.LEARNING_RATE = 0.001

        self.NB_PLOT = 4
