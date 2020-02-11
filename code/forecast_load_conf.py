from forecast_conf import ForecastConfig


class ForecastLoadConfig:
    def __init__(self):
        self.BEGIN = ForecastConfig().BEGIN
        self.END = ForecastConfig().END
        self.STEPSIZE = ForecastConfig().STEPSIZE
        self.TIMESTAMPS = ForecastConfig().TIMESTAMPS

        self.DATA_FILE = "./data/15minute_data_newyork_1222.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        # self.OUTPUT_FOLDER = "./output/forecast/load/60min/out24/bs50/lb20/"

        self.LOAD_MODEL = False
        self.NB_PLOT = 4

        self.MODEL_ID = "ts30_out48_lb48_bs100_1"
        self.OUTPUT_FOLDER = "./output/forecast/load/" + self.MODEL_ID + "/"
        self.MODEL_FILE = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".json"
        self.MODEL_FILE_H5 = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".h5"

        self.APPLIANCES = ["car1", "heater1", "waterheater1", "drye1"]
        self.EPOCHS = 50
        self.LOOK_BACK = 48
        self.OUTPUT_SIZE = 48
        self.BATCH_SIZE = 50
        self.PATIENCE = 10
        self.MIN_DELTA = 0.00001
        self.DROPOUT = [0.01, 0.01]
        self.NEURONS = [256, 256]
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
        self.LEARNING_RATE = 0.0001
