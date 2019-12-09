import datetime

import pandas as pd


def getPriceData(filePath, start, end, stepsize=datetime.timedelta(hours=1)):
    with open(filePath, "r", encoding="utf-8") as sampleFile:
        data = pd.read_csv(sampleFile, parse_dates=["DateTime"])
        data = data.loc[(data["DateTime"] >= start) & (data["DateTime"] <= end)]
        data = data.set_index(data["DateTime"])
        data = data.resample(stepsize).sum()
        assert data.shape[1] <= 2
        return data.iloc[:, 0]
