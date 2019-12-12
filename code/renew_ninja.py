import json

import pandas as pd
import requests


class NetworkException(Exception):
    pass


def getSampleWind(file, timestamps):
    return _getSample(file, timestamps)


def getSamplePv(file, timestamps):
    return _getSample(file, timestamps)


def _getSample(filePath, timestamps):
    with open(filePath, "r", encoding="utf-8") as sampleFile:
        [sampleFile.readline() for i in range(3)]
        data = pd.read_csv(sampleFile, parse_dates=["time", "local_time"])
        data = data.loc[
            (data["time"] >= timestamps[0]) & (data["time"] <= timestamps[-1])
        ]
        return data["electricity"]


def getSamplePvApi(lat, long, timestamps):
    # TODO include stepsize into getDataPv
    renewNinja = RenewNinja()
    return renewNinja.getDataPv(
        lat, long, str(timestamps[0].date()), str(timestamps[-1].date())
    )


def getSampleWindApi(lat, long, timestamps):
    # TODO include stepsize into getDataPv
    renewNinja = RenewNinja()
    return renewNinja.getDataWind(
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
        getDataPv(self, lat, long, date_from, date_to, dataset = 'merra2', cap = 1.0, sys_loss = 0.1, track = 0, tilt = 35, azim = 180)
            get the data of the pv from renewables ninja
        """

    def __init__(self):
        self.token = "732eaff288b11d478c42381c75173e8e17355fdb"
        self.api_base = "https://www.renewables.ninja/api/"
        self.s = requests.session()
        self.s.headers = {"Authorization": "Token " + self.token}

    def __del__(self):
        self.s.close()

    def getDataPv(
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

    def getDataWind(
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
