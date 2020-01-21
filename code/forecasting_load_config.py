class Config:
    def __init__(self):
        self.LOOK_BACK = 15

        self.SEED = 23

        self.BEGIN = "2019-05-01 00:00:00"
        self.END = "2019-10-31 23:45:00"
        self.STEPSIZE = "00:15:00"

        self.TRAINPART = 0.6
        self.VALIDATIONPART = (1 - self.TRAINPART) / 2

        self.DATA_FILE = "./data/15minute_data_newyork.csv"
        self.TIME_HEADER = "local_15min"
        self.DATAID = 1222
        self.OUTPUT_FOLDER = "./output/forecasting/load/tryLate"
        # self.OUTPUT_FOLDER = "./output/forecasting/load/try_17_00"

        self.LOAD = False


INIT_MODEL_CONFIG = {
    "epochs": 30,
    "batch_size": 30,
    "dropout": 0.1,
    "dense": 1,
    "neurons": 8,
    "activation_function": "linear",
    "loss_function": "mean_squared_error",
    "optimize_function": "adam",
    "next": "epochs",
}


modelOptimizationConfig = {
    # "batch_size": [1, 10, 20, 30, 40, 50],
    # "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "neurons": [1, 20, 60, 100, 200],
    "activation_function": ["linear", "tanh", "elu", "softmax", "selu"],
}