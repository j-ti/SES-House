class ForecastPvConfig:
    def __init__(self):
        self.DATA_FILE = "./data/15minute_data_newyork.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecast/pv/"
        self.DATE_START = "2019-05-01 00:00:00"
        self.DATE_END = "2019-06-01 00:00:00"

        self.LOAD_MODEL = True
        self.MODEL_FILE = "./output/forecast/pv/model.json"
        self.MODEL_FILE_H5 = "./output/forecast/pv/model.h5"

        # TODO : for multiple layer of same type : put inside an array
        self.EPOCHS = 30
        self.LOOK_BACK = 15
        self.BATCH_SIZE = 30
        self.DROPOUT = 0.1
        self.DENSE = 1
        self.NEURONS = [128]
        self.PATIENCE = 5
        self.MIN_DELTA = 0.0001
        self.OUTPUT_SIZE = 96
        self.ACTIVATION_FUNCTION = "relu"
        self.LOSS_FUNCTION = "mean_squared_error"
        self.OPTIMIZE_FUNCTION = "adam"
