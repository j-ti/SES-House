class ForecastLoadConfig:
    def __init__(self):
        self.DATA_FILE = "./data/15minute_data_newyork_1222.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecast/load/60min/out24/bs50/lb20/"

        self.LOAD_MODEL = False
        self.NB_PLOT = 4
        self.OUTPUT_FOLDER = "./output/forecast/load/60min/out24/bs50/lb20/"
        self.MODEL_FILE = "./output/forecast/load/60min/out24/lb20/bs50/model.json"
        self.MODEL_FILE_H5 = "./output/forecast/load/60min/out24/lb20/bs50/model.h5"

        self.APPLIANCES = ["car1", "heater1", "waterheater1", "drye1"]
        self.EPOCHS = 50
        self.LOOK_BACK = 20
        self.OUTPUT_SIZE = 24
        self.BATCH_SIZE = 50
        self.PATIENCE = 10
        self.MIN_DELTA = 0.00001
        self.DROPOUT = [0.01, 0.01]
        self.NEURONS = [256, 256]
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
