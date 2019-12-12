import pandas as pd

from util import getStepsize


def getPriceData(filePath, timestamps):
    with open(filePath, "r", encoding="utf-8") as sampleFile:
        data = pd.read_csv(sampleFile, parse_dates=[0], index_col=0)
        data = data.loc[timestamps[0] : timestamps[-1]]
        origStepsize = getStepsize(data.index)
        wantedStepsize = getStepsize(timestamps)
        if origStepsize > wantedStepsize:
            data = data.resample(wantedStepsize).ffill()
        elif origStepsize < wantedStepsize:
            data = data.resample(wantedStepsize).mean()
        assert data.shape[1] <= 2
        return data.iloc[:, 0]
