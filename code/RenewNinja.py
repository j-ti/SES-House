import datetime
import json

import pandas as pd
import requests


def getSamplePvApi(start, end, stepsize=datetime.timedelta(hours=1)):
    # TODO include stepsize into getDataPv
    renewNinja = RenewNinja()
    return renewNinja.getDataPv(52.5170, 13.3889, start, end)


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
        dek

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

        # Parse JSON to get a pandas.DataFrame of data and dict of metadata
        parsed_response = json.loads(r.text)

        data = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
        metadata = parsed_response["metadata"]
        return metadata, data
