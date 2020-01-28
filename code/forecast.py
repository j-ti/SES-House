import pandas as pd
from keras.callbacks import EarlyStopping
from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation
from keras import metrics
from keras.layers.core import Dense
from plot_forecast import *


def splitData(config, loadsData):
    diff = loadsData.index[-1] - loadsData.index[0]
    endTrain = 96 * int(diff.days * config.TRAIN_FRACTION)
    endValidation = endTrain + 96 * int(diff.days * config.VALIDATION_FRACTION)
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


# we assume that data is either train, test or validation and is shape (nbPts, nbFeatures)
def buildSet(data, look_back, nbOutput):
    x = np.empty((len(data) - look_back - nbOutput, look_back, data.shape[1]))
    y = np.empty((len(data) - look_back - nbOutput, nbOutput))
    for i in range(len(data) - look_back - nbOutput):
        x[i] = data[i: i + look_back, :]
        y[i] = data[i + look_back: i + look_back + nbOutput, 0]
    return x, y


def buildModel(config, trainXShape):
    model = Sequential()

    model.add(
        LSTM(
            config.NEURONS[0],
            return_sequences=len(config.NEURONS) != 1,
            input_shape=(trainXShape[1], trainXShape[2]),
        )
    )

    for neuron in config.NEURONS[1:-1]:
        model.add(LSTM(config.NEURONS, return_sequences=True))

    if len(config.NEURONS) > 1:
        model.add(LSTM(config.NEURONS[-1]))

    model.add(Dropout(config.DROPOUT))
    model.add(Dense(config.OUTPUT_SIZE))
    model.add(Activation(config.ACTIVATION_FUNCTION))
    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=config.OPTIMIZE_FUNCTION,
        metrics=[metrics.mape, metrics.mae, metrics.mse],
    )
    return model


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    return model.evaluate(testx, testy, verbose=0)


def saveModel(config, model):
    model_json = model.to_json()
    with open(config.OUTPUT_FOLDER + "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(config.OUTPUT_FOLDER + "model.h5")


def loadModel():
    # load json and create model
    json_file = open(outputFolder + "model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(outputFolder + "model.h5")

    # evaluate loaded model
    loaded_model.compile(
        loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
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
