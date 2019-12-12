import pandas as pd

from util import getStepsize


def getPriceData(filePath, timestamps):
    with open(filePath, "r", encoding="utf-8") as sampleFile:
        data = pd.read_csv(sampleFile, parse_dates=["DateTime"])
        data = data.loc[(data["DateTime"] >= timestamps[0]) & (data["DateTime"] <= timestamps[-1])]
        data = data.set_index(data["DateTime"])
        data = data.resample(getStepsize(timestamps)).sum()
        assert data.shape[1] <= 2
        return data.iloc[:, 0]
