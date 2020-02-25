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

        self.LOAD_MODEL = True
        self.NB_PLOT = 4

        self.OUTPUT_SIZE = 24
        self.LOOK_BACK = 24
        self.BATCH_SIZE = 10
        self.DROPOUT = [0.3, 0.2, 0.1]
        self.NEURONS = [256, 256, 256]
        self.LEARNING_RATE = 0.0003

        self.MODEL_ID = "ts30_out{}_lb{}_bs{}_lay{}_do1{}_neu1{}_lr{}".format(
            self.OUTPUT_SIZE,
            self.LOOK_BACK,
            self.BATCH_SIZE,
            len(self.NEURONS),
            self.DROPOUT[0],
            self.NEURONS[0],
            self.LEARNING_RATE,
        )
        self.OUTPUT_FOLDER = "./output/forecast/load/" + self.MODEL_ID + "/"
        self.MODEL_FILE = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".json"
        self.MODEL_FILE_H5 = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".h5"
        self.MODEL_FILE_SC = self.OUTPUT_FOLDER + "model_" + self.MODEL_ID + ".save"

        self.APPLIANCES = ["heater1", "waterheater1", "drye1"]
        self.EPOCHS = 100
        self.PATIENCE = 10
        self.MIN_DELTA = 0.00001
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
