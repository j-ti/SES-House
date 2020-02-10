from datetime import timedelta

import numpy as np
import pandas as pd
from keras import metrics, optimizers
from keras.callbacks import EarlyStopping
from keras.engine.saving import model_from_json
from keras.layers import LSTM, Dropout, Activation
from keras.layers.core import Dense
from keras.models import Sequential

from util import getStepsize


def get_timestamps_per_day(config):
    timestamps_per_day = timedelta(hours=24) / getStepsize(config.TIMESTAMPS)
    assert timestamps_per_day.is_integer()
    return int(timestamps_per_day)


def get_split_indexes(config):
    diff = config.TIMESTAMPS[-1] - config.TIMESTAMPS[0]
    timestamps_per_day = get_timestamps_per_day(config)
    end_train = timestamps_per_day * int(diff.days * config.TRAIN_FRACTION)
    end_validation = end_train + timestamps_per_day * int(
        diff.days * config.VALIDATION_FRACTION
    )
    return end_train, end_validation


def splitData(config, loadsData):
    endTrain, endValidation = get_split_indexes(config)
    return (
        loadsData[:endTrain],
        loadsData[endTrain:endValidation],
        loadsData[endValidation:],
    )


def addMinutes(data):
    minutes = pd.Series(
        [(i.hour * 60 + i.minute) for i in data.index], index=data.index
    )
    return pd.concat([data, minutes], axis=1)


def add0(data):
    zeros = pd.Series(
        [0 for i in data.index], index=data.index
    )
    return pd.concat([data, zeros], axis=1)


def add_day_of_week(data):
    days_of_week = pd.Series([i.weekday() for i in data.index], index=data.index)
    return pd.concat([data, days_of_week], axis=1)


def add_weekend(data):
    is_weekend = pd.Series(
        [1 if i.weekday() in [5, 6] else 0 for i in data.index], index=data.index
    )
    return pd.concat([data, is_weekend], axis=1)


def addMonthOfYear(data, timestamps):
    months = pd.Series(
        [timestamps[i].month for i in range(len(timestamps))], index=data.index
    )
    return pd.concat([data, months], axis=1)


# we assume that data is either train, test or validation and is shape (nbPts, nbFeatures)
def buildSet(data, look_back, nbOutput):
    x = np.empty((len(data) - look_back - nbOutput, look_back, data.shape[1]))
    y = np.empty((len(data) - look_back - nbOutput, nbOutput))
    for i in range(len(data) - look_back - nbOutput):
        x[i] = data[i : i + look_back, :]
        y[i] = data[i + look_back : i + look_back + nbOutput, 0]
    return x, y


def buildModel(config, trainXShape):
    assert len(config.DROPOUT) == len(config.NEURONS)

    model = Sequential()

    model.add(
        LSTM(
            config.NEURONS[0],
            return_sequences=len(config.NEURONS) != 1,
            input_shape=(trainXShape[1], trainXShape[2]),
        )
    )
    model.add(Dropout(config.DROPOUT[0]))

    for idx in range(1, len(config.NEURONS) - 1):
        model.add(LSTM(config.NEURONS[idx], return_sequences=True))
        model.add(Dropout(config.DROPOUT[idx]))

    if len(config.NEURONS) > 1:
        model.add(LSTM(config.NEURONS[-1]))
        model.add(Dropout(config.DROPOUT[-1]))

    model.add(Dense(config.OUTPUT_SIZE))
    model.add(Activation(config.ACTIVATION_FUNCTION))

    if config.OPTIMIZE_FUNCTION == "adam":
        optimizer = optimizers.Adam(lr=config.LEARNING_RATE)
    else:
        optimizer = config.OPTIMIZE_FUNCTION
    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=optimizer,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return model


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    return model.evaluate(testx, testy, verbose=0)


def saveModel(config, model):
    model_json = model.to_json()
    with open(config.MODEL_FILE, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(config.MODEL_FILE_H5)


def loadModel(config):
    json_file = open(config.MODEL_FILE, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.MODEL_FILE_H5)

    # evaluate loaded model
    loaded_model.compile(
        loss=config.LOSS_FUNCTION, optimizer=config.OPTIMIZE_FUNCTION, metrics=["accuracy"]
    )
    return loaded_model


def train(config, model, trainX, trainY, validationX, validationY):
    earlyStopCallback = EarlyStopping(
        monitor="val_loss",
        min_delta=config.MIN_DELTA,
        patience=config.PATIENCE,
        mode="min",
        restore_best_weights=True,
    )

    history = model.fit(
        trainX,
        trainY,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(validationX, validationY),
        callbacks=[earlyStopCallback],
        verbose=2,
    )
    return history


# first value must be an array with 5 pages
def forecast(model, nbToPredict, firstValue):
    pred = []
    val = firstValue
    for i in range(nbToPredict):
        pred.append(model.predict(val))
        val.pop(0)
        val.append(pred[-1])
    return pred
