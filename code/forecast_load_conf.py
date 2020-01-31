class ForecastLoadConfig:
    def __init__(self):
        self.DATA_FILE = "./data/15minute_data_newyork.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecast/load/"

        self.LOAD_MODEL = False
        self.MODEL_FILE = "./output/forecast/load/model.json"

        self.EPOCHS = 30
        self.LOOK_BACK = 15
        self.OUTPUT_SIZE = 24
        self.BATCH_SIZE = 30
        self.PATIENCE = 5
        self.MIN_DELTA = 0.0001
        self.DROPOUT = 0.1
        self.NEURONS = [512]
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
