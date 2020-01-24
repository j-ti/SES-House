from forecasting import splitData, buildSet, evalModel, loadModel, dataImport, saveModel, train
from keras import Sequential, metrics
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np
import os
import sys
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from forecast_pv_conf import ForecastPvConfig
from plot_forecast import plotHistory, plotPrediction, plot100first, plotEcart, plotInput


def buildModel(trainX, trainY, valX, valY, config, nbFeatures):
    model = Sequential()
    model.add(LSTM(256, input_shape=(config.LOOK_BACK, nbFeatures)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    # model.add(Dense(3))
    # model.add(Dense(1))
    model.add(Activation("relu"))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[metrics.mae, metrics.mape, metrics.mse],
    )

    # training it
    model, history = train(config, model, trainX, trainY, valX, valY)
    saveModel(model)
    return model, history


def forecasting(config):
    # import data
    df, timestamps = dataImport()
    df_train, df_validation, df_test = splitData(config, df)

    nbFeatures = 1

    # here we have numpy array
    trainX, trainY = buildSet(df_train, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)
    validationX, validationY = buildSet(df_validation, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)
    testX, testY = buildSet(df_test, config.LOOK_BACK, config.OUTPUT_SIZE, nbFeatures)

    plotInput(df, timestamps)

    history = None
    if config.LOAD_MODEL:
        model = loadModel()
    else:
        model, history = buildModel(trainX, trainY, validationX, validationY)

    evalModel(model, testX, testY)

    # plotting
    testPrediction = model.predict(testX)
    trainPrediction = model.predict(trainX)

    if history is not None:
        plotHistory(history)

    plotPrediction(
        trainY, trainPrediction, testY, testPrediction, timestamps
    )
    plot100first(trainY, trainPrediction)
    plot100first(testY, testPrediction)
    plotEcart(
        np.array(trainY),
        np.array(trainPrediction),
        np.array(testY),
        np.array(testPrediction),
        timestamps,
    )
    print(
        "training\tMSE :\t{}".format(
            mean_squared_error(np.array(trainY), np.array(trainPrediction))
        )
    )
    print(
        "testing\t\tMSE :\t{}".format(
            mean_squared_error(np.array(testY), np.array(testPrediction))
        )
    )

    print(
        "training\tMAE :\t{}".format(
            mean_absolute_error(np.array(trainY), np.array(trainPrediction))
        )
    )
    print(
        "testing\t\tMAE :\t{}".format(
            mean_absolute_error(np.array(testY), np.array(testPrediction))
        )
    )

    print(
        "training\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(trainY), np.array(trainPrediction)
            )
        )
    )
    print(
        "testing\t\tMAPE :\t{} %".format(
            mean_absolute_percentage_error(
                np.array(testY), np.array(testPrediction)
            )
        )
    )


# if argv = 1, then we rebuild the model
def main(argv):
    config = ForecastPvConfig()
    global outputFolder
    outputFolder = "output/" + "modelKeras" + "/"
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)

    forecasting(config)


if __name__ == "__main__":
    main(sys.argv)