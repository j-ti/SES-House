from datetime import datetime

from data import getPecanstreetData
from keras.engine.saving import model_from_json
from plot_forecast import *
from util import constructTimeStamps
from util import makeShift

# fixing the random seed to have a better reproducibility
seed = 3
np.random.seed(seed)
# param
look_back = 10  # we have a 5 point history in our input
part = 0.6  # we train on part of the set
nbOut = 2
config = ""
nbFeatures = 1


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
    minutes = pd.Series([(i.hour * 60 + i.minute) for i in data.index], index=data.index)
    return pd.concat([data, minutes], axis=1)


# we assume that data is either train, test or validation and is shape (nbPts, nbFeatures)
def buildSet(data, look_back, nbOutput, nbFeatures):
    X = makeShift(data, look_back, nbFeatures)
    col = []
    for i in range(len(data) - look_back):
        col.append(data[look_back + i: i + look_back + nbOutput])
    Y = np.array(col)
    return X, Y


# WARNING ! depending on if we load the model or if we build it, the return value of evaluate change
# I still don't know why
def evalModel(model, testx, testy):
    ret = model.evaluate(testx, testy, verbose=0)
    print(ret)
    return ret


def saveModel(model):
    model_json = model.to_json()
    with open(outputFolder + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(outputFolder + "model.h5")
    print("Saved model to disk")


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
    history = model.fit(
        trainX,
        trainY,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(validationX, validationY),
        verbose=2,
    )
    return model, history


# first value must be an array with 5 pages
def forecast(model, nbToPredict, firstValue):
    pred = []
    val = firstValue
    for i in range(nbToPredict):
        pred.append(model.predict(val))
        val.pop(0)
        val.append(pred[-1])
    return pred
