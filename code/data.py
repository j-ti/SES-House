import json
import numpy as np
import pandas as pd
import requests

from datetime import timedelta

from util import getStepsize


FROM_MEGAWATTHOURS_TO_KILOWATTHOURS = 1000


class NetworkException(Exception):
    pass


def getNinja(filePath, timestamps):
    with open(filePath, "r", encoding="utf-8") as dataFile:
        [dataFile.readline() for i in range(3)]
        data = pd.read_csv(
            dataFile, parse_dates=["time", "local_time"], index_col="local_time"
        )
        data = data.loc[timestamps[0] : timestamps[-1] + getStepsize(timestamps)]
        origStepsize = getStepsize(data.index)
        wantedStepsize = getStepsize(timestamps)
        if origStepsize > wantedStepsize:
            assert (origStepsize / wantedStepsize).is_integer()
            data = data.resample(wantedStepsize).ffill()
        elif origStepsize < wantedStepsize:
            data = _dropUnfittingValuesAtEndForDownSampling(
                origStepsize, wantedStepsize, timestamps, data
            )
            data = data.resample(wantedStepsize).mean()
        data = data.loc[timestamps[0] : timestamps[-1]]
        return data["electricity"]


def getNinjaPvApi(lat, long, timestamps):
    renewNinja = RenewNinja()
    return renewNinja.getPvData(
        lat, long, str(timestamps[0].date()), str(timestamps[-1].date())
    )


def getNinjaWindApi(lat, long, timestamps):
    renewNinja = RenewNinja()
    return renewNinja.getWindData(
        lat, long, str(timestamps[0].date()), str(timestamps[-1].date())
    )


class RenewNinja:
    """
        Class to query https://www.renewables.ninja/ API to get pv and wind
        output power based on their dataset + simulation

        ...

        Attributes
        ----------
        token : API token to access renewables ninja API
        api_base : str
            url of the api of renewables ninja
        s : requests.sessions.Session
            Object to request (package requests)

        Methods
        -------
        getPvData(self, lat, long, date_from, date_to, dataset = 'merra2', cap = 1.0, sys_loss = 0.1, track = 0, tilt = 35, azim = 180)
            get the data of the pv from renewables ninja
        """

    def __init__(self):
        self.token = "732eaff288b11d478c42381c75173e8e17355fdb"
        self.api_base = "https://www.renewables.ninja/api/"
        self.s = requests.session()
        self.s.headers = {"Authorization": "Token " + self.token}

    def __del__(self):
        self.s.close()

    def getPvData(
        self,
        lat,
        long,
        date_from,
        date_to,
        dataset="merra2",
        cap=1.0,
        sys_loss=0.1,
        track=0,
        tilt=35,
        azim=180,
    ):
        """Request PV power value

            Parameters
            ----------
            lat : float
                latitude of the pv
            long : float
                Longitude of the pv
            date_from  : str
                format : year-month-day. Starting date of the requested data
            date_to  : str
                format : year-month-day. Ending date of the requested data
            dataset : str, optional
                name of the dataset
            cap : float, optional
                capacity of the pv
            sys_loss : float, optional
                system loss of the pv
            track : bool, optional
                presence of a tracking system
            tilt : int, optional
            azim : int, optional

            Returns
            -------
            tuple
                0 : metadata (dict)
                1 : data (pandas Dataframe)
            """

        url = self.api_base + "data/pv"
        args = {
            "lat": lat,
            "lon": long,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": dataset,
            "capacity": cap,
            "system_loss": sys_loss,
            "tracking": track,
            "tilt": tilt,
            "azim": azim,
            "format": "json",
        }
        r = self.s.get(url, params=args)
        if r.status_code != 200:
            print(r.text)
            raise NetworkException()

        # Parse JSON to get a pandas.DataFrame of data and dict of metadata
        parsed_response = json.loads(r.text)

        data = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
        metadata = parsed_response["metadata"]
        return metadata, data

    def getWindData(
        self,
        lat,
        long,
        date_from,
        date_to,
        cap=1.0,
        height=100,
        turbine="Vestas V80 2000",
    ):
        """Request wind power value

            Parameters
            ----------
            lat : float
                latitude of the windmill
            long : float
                Longitude of the windmill
            date_from  : str
                format : year-month-day. Starting date of the requested data
            date_to  : str
                format : year-month-day. Ending date of the requested data
            cap : float, optional
                capacity of the windmill
            height : int, optional
                height of the windmill
            turbine : str, optional
                type of the turbine


            Returns
            -------
            tuple
                0 : metadata (dict)
                1 : data (pandas Dataframe)
            """

        url = self.api_base + "data/wind"
        args = {
            "lat": lat,
            "lon": long,
            "date_from": date_from,
            "date_to": date_to,
            "capacity": cap,
            "height": height,
            "turbine": turbine,
            "format": "json",
        }
        r = self.s.get(url, params=args)
        if r.status_code != 200:
            print(r.text)
            raise NetworkException()

        # Parse JSON to get a pandas.DataFrame of data and dict of metadata
        parsed_response = json.loads(r.text)

        data = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
        metadata = parsed_response["metadata"]
        return metadata, data


def resampleData(data, timestamps, offset=timedelta(days=0), positive=True):
    origStepsize = getStepsize(data.index)
    wantedStepsize = getStepsize(timestamps)
    if origStepsize > wantedStepsize:
        assert (origStepsize / wantedStepsize).is_integer()
        data = data.resample(wantedStepsize).ffill()
    elif origStepsize < wantedStepsize:
        data = _dropUnfittingValuesAtEndForDownSampling(
            origStepsize, wantedStepsize, timestamps, data
        )
        data = data.resample(wantedStepsize).mean()
    data = data.loc[timestamps[0] + offset : timestamps[-1] + offset]
    assert data.shape[1] <= 3
    if data.shape[1] == 3:
        not_numbers = np.isnan(data)
        data.iloc[not_numbers] = 0
        dataOut = data.iloc[:, 0] + data.iloc[:, 1] + data.iloc[:, 2]
    else:
        dataOut = data.iloc[:, 0]
        if positive:
            dataOut[dataOut <= 0] = 0
    for value in dataOut:
        if positive:
            assert value >= 0
    return dataOut


def getLoadsData(filePath, timestamps):
    with open(filePath, "r", encoding="utf-8") as dataFile:
        data = pd.read_csv(
            dataFile,
            parse_dates=["DateTime"],
            index_col="DateTime",
            sep=";",
            decimal=",",
        )
        data = data.loc[timestamps[0] : timestamps[-1] + getStepsize(timestamps)]
        return resampleData(data, timestamps)


def dateparserWithoutUTC(x):
    d, h = x.split(" ")[0], x.split(" ")[1].split("-")[0]
    return pd.datetime.strptime(d + " " + h, "20%y-%m-%d %H:%M:%S")


def getPecanstreetData(
    filePath, timeHeader, dataid, column, timestamps, offset=timedelta(days=0)
):
    with open(filePath, "r", encoding="utf-8") as dataFile:
        # TODO: read more rows or split dataid into files
        data = pd.read_csv(
            dataFile,
            parse_dates=[timeHeader],
            date_parser=dateparserWithoutUTC,
            nrows=10000,
        )
        data = data[data["dataid"] == int(dataid)]
        pd.to_datetime(data[timeHeader])
        data = data.set_index(timeHeader)
        data = data.sort_index()
        if column == "grid":
            data = data.loc[:, [column, "solar", "solar2"]]
        else:
            data = data.loc[:, [column]]
        stepsize = getStepsize(timestamps)
        if stepsize < timedelta(minutes=15):
            stepsize = timedelta(hours=0)

        data = data.loc[timestamps[0] + offset : timestamps[-1] + offset + stepsize]
        return resampleData(data, timestamps, offset)


def splitPecanstreetData(filePath, timeHeader):
    with open(filePath, "r", encoding="utf-8") as dataFile:
        data = pd.read_csv(dataFile, parse_dates=[timeHeader],)
        current = 0
        # TODO add for loop and store into new csv files
        dataid = data["dataid"][current]
        data = data[data["dataid"] == dataid]

        return data


def getPriceData(filePath, timestamps, offset, constantPrice):
    with open(filePath, "r", encoding="utf-8") as dataFile:
        data = pd.read_csv(
            dataFile,
            parse_dates=["DateTime"],
            index_col="DateTime",
            sep=";",
            decimal=",",
        )
        data = data.loc[
            timestamps[0] + offset : timestamps[-1] + offset + getStepsize(timestamps)
        ]
        origStepsize = getStepsize(data.index)
        assert origStepsize == timedelta(hours=1)
        wantedStepsize = getStepsize(timestamps)
        if origStepsize > wantedStepsize:
            assert (origStepsize / wantedStepsize).is_integer()
            data = data.resample(wantedStepsize).asfreq()
            _applyOppositeOfResampleSum(data, timestamps, origStepsize / wantedStepsize)
        elif origStepsize < wantedStepsize:
            data = _dropUnfittingValuesAtEndForDownSampling(
                origStepsize, wantedStepsize, timestamps, data
            )
            data = data.resample(wantedStepsize).sum()
        assert data.shape[1] <= 2

        data = data.loc[timestamps[0] + offset : timestamps[-1] + offset]
        return data.iloc[:, 0] / FROM_MEGAWATTHOURS_TO_KILOWATTHOURS + constantPrice


def _applyOppositeOfResampleSum(data, timestamps, relation):
    for index in range(len(timestamps)):
        if np.isnan(data.iloc[index, 0]):
            data.iloc[index, 0] = newValue  # noqa F821
        else:
            newValue = data.iloc[index, 0] / relation
            data.iloc[index, 0] = newValue


def _dropUnfittingValuesAtEndForDownSampling(
    origStepsize, wantedStepsize, timestamps, data
):
    relation = _computeIntRelation(wantedStepsize, origStepsize)
    if data.size % relation != 0:
        data = data[: -(data.size % relation)]
    return data


def _computeIntRelation(stepsize1, stepsize2):
    relation = stepsize1 / stepsize2
    assert relation.is_integer(), "1 stepsize should be a multiple of the other."
    return int(relation)
