#!/usr/bin/env python3.7

import configparser
from datetime import datetime, timedelta
from enum import Enum
import sys

from util import constructTimeStamps, getStepsize, getTimeIndexRange

from data import getNinja, getNinjaPvApi, getNinjaWindApi, getPriceData, getLoadsData

import gurobipy as gp
from gurobipy import QuadExpr
from gurobipy import GRB
from gurobipy import LinExpr


class Goal(Enum):
    MINIMIZE_COST = "MINIMIZE_COST"
    GREEN_HOUSE = "GREEN_HOUSE"
    GRID_INDEPENDENCE = "GRID_INDEPENDENCE"


class Configure:
    def __init__(self, config):
        # Global
        self.goal = Goal(config["GLOBAL"]["goal"])
        self.loc_flag = "yes" == config["GLOBAL"]["loc"]
        self.loc_lat = float(config["GLOBAL"]["lat"])
        self.loc_lon = float(config["GLOBAL"]["lon"])

        # Battery init (to be moved to a initialization file)
        self.SOC_bat_min = float(config["BAT"]["SOC_bat_min"])
        self.SOC_bat_init = float(config["BAT"]["SOC_bat_init"])
        self.SOC_bat_max = float(config["BAT"]["SOC_bat_max"])
        self.E_bat_max = float(config["BAT"]["E_bat_max"])
        self.eta_bat = float(config["BAT"]["eta_bat"])
        self.P_bat_max = float(config["BAT"]["P_bat_max"])

        # EV init
        self.SOC_ev_min = float(config["EV"]["SOC_ev_min"])
        self.SOC_ev_init = float(config["EV"]["SOC_ev_init"])
        self.SOC_ev_max = float(config["EV"]["SOC_ev_max"])
        self.P_ev_max = float(config["EV"]["P_ev_max"])
        self.E_ev_max = float(config["EV"]["E_ev_max"])
        self.eta_ev = float(config["EV"]["eta_ev"])
        self.t_a_ev = datetime.strptime(config["EV"]["t_a_ev"], "20%y-%m-%d %H:%M:%S")
        self.t_b_ev = datetime.strptime(config["EV"]["t_b_ev"], "20%y-%m-%d %H:%M:%S")
        self.t_goal_ev = datetime.strptime(
            config["EV"]["t_goal_ev"], "20%y-%m-%d %H:%M:%S"
        )

        # Time frame of optimization
        self.timestamps = constructTimeStamps(
            datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["stepsize"], "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )

        # Generators
        self.P_dg_max = float(config["DIESEL"]["P_dg_max"])
        self.dieselQuadraticCof = float(config["DIESEL"]["a_dg"])
        self.dieselLinearCof = float(config["DIESEL"]["b_dg"])
        self.dieselConstantCof = float(config["DIESEL"]["c_dg"])
        self.dieselFuelPrice = float(config["DIESEL"]["c_gen"])
        self.startUpCost = float(config["DIESEL"]["StartUpCost"])
        self.dieselLeastRunHour = int(config["DIESEL"]["LeastRunningTime"])
        self.startUpHour = int(config["DIESEL"]["StartUpTime"])
        self.shutDownHour = int(config["DIESEL"]["ShutDownTime"])
        self.deltaShutDown = (
            self.P_dg_max
            / self.shutDownHour
            * (getStepsize(self.timestamps).total_seconds() / 3600)
        )
        self.deltaStartUp = (
            self.P_dg_max
            / self.startUpHour
            * (getStepsize(self.timestamps).total_seconds() / 3600)
        )
        self.pvFile = config["PV"]["file"]
        self.windFile = config["WIND"]["file"]
        self.loadsFile = config["LOADS"]["file"]
        self.costFileGrid = config["COST"]["file_grid"]
        self.co2Grid = config["CO2"]["grid_CO2"]
        self.co2Diesel = config["CO2"]["diesel_CO2"]


def runSimpleModel(ini):

    model = gp.Model("simple-model")

    pvVars = setUpPV(model, ini)
    windVars = setUpWind(model, ini)
    batteryPowerVars = setUpBattery(model, ini)
    evPowerVars = setUpEv(model, ini)
    fixedLoadVars = setUpFixedLoads(model, ini)
    [dieselGeneratorsVars, dieselStatusVars] = setUpDiesel(model, ini)
    gridVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="gridPowers",
    )

    model.addConstrs(
        (
            gridVars.sum(i, "*")
            + pvVars.sum(i, "*")
            + windVars.sum(i, "*")
            + dieselGeneratorsVars.sum(i, "*")
            + batteryPowerVars.sum(i, "*")
            + evPowerVars.sum(i, "*")
            == fixedLoadVars.sum(i, "*")
            for i in range(len(ini.timestamps))
        ),
        "power balance",
    )

    setObjective(model, ini, dieselGeneratorsVars, dieselStatusVars, gridVars)

    model.optimize()

    file = (
        "./results/"
        + str(datetime.now()).replace(" ", "_").replace(":", "-")
        + "_res.sol"
    )
    model.write(file)

    printResults(model, ini)


def setObjective(model, ini, dieselGeneratorsVars, dieselStatusVars, gridVars):
    if ini.goal is Goal.MINIMIZE_COST:
        prices = getPriceData(
            ini.costFileGrid, ini.timestamps, timedelta(days=365 * 5 + 1)
        )

        dieselObjExp = QuadExpr()
        for index in range(len(ini.timestamps)):
            dieselObjExp.add(
                dieselGeneratorsVars[index, 0]
                * dieselGeneratorsVars[index, 0]
                * ini.dieselQuadraticCof
                * ini.dieselFuelPrice
            )
            dieselObjExp.add(
                dieselGeneratorsVars[index, 0]
                * ini.dieselLinearCof
                * ini.dieselFuelPrice
            )
            dieselObjExp.add(ini.dieselConstantCof)
            dieselObjExp.add(
                ini.startUpCost * dieselStatusVars[index, 2] / ini.startUpHour
            )

        model.setObjective(
            dieselObjExp
            + sum([gridVars[index, 0] * price for index, price in enumerate(prices)]),
            0,
        )
    elif ini.goal is Goal.GREEN_HOUSE:
        # TODO adapt to new diesel model
        model.setObjective(
            ini.co2Diesel * gp.quicksum(dieselGeneratorsVars)
            + ini.co2Grid * gp.quicksum(gridVars),
            0,
        )
    elif ini.goal is Goal.GRID_INDEPENDENCE:
        model.setObjective(
            sum(
                [
                    gridVars[index, 0] * gridVars[index, 0]
                    for index in range(len(ini.timestamps))
                ]
            ),
            0,
        )


def setUpDiesel(model, ini):
    dieselGeneratorsVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=0.0,
        ub=ini.P_dg_max,
        vtype=GRB.CONTINUOUS,
        name="dieselGenerators",
    )

    dieselStatusVars = model.addVars(
        len(ini.timestamps),
        4,  # the first column: diesel not in work/second: diesel start up, third: diesel shut down/fourth: diesel is working
        vtype=GRB.BINARY,
        name="dieselStatus",  # startup/shutdown/keep constant
    )

    model.addConstrs(
        (dieselStatusVars.sum(i, "*") == 1 for i in range(len(ini.timestamps)))
    )

    for i in range(len(ini.timestamps) - 1):
        model.addConstr(
            (
                (dieselStatusVars[i, 1] == 1)
                >> (
                    dieselGeneratorsVars[i + 1, 0]
                    == dieselGeneratorsVars[i, 0] + ini.deltaStartUp
                )
            ),
            "diesel generator power change considering Startup/Shutdown",
        )
        model.addConstr(
            (
                (dieselStatusVars[i, 2] == 1)
                >> (
                    dieselGeneratorsVars[i + 1, 0]
                    == dieselGeneratorsVars[i, 0] - ini.deltaShutDown
                )
            ),
            "diesel generator power change considering Startup/Shutdown",
        )

    model.addConstrs(
        (
            (dieselStatusVars[index, 0] == 1) >> (dieselStatusVars[index + 1, 3] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Not Working -> Working IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 0] == 1) >> (dieselStatusVars[index + 1, 2] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Not Working -> Shutdown IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 1] == 1) >> (dieselStatusVars[index + 1, 2] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Startup -> Shutdown IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 1] == 1) >> (dieselStatusVars[index + 1, 3] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Startup -> Not working IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 2] == 1) >> (dieselStatusVars[index + 1, 3] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Shutdown -> working IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 3] == 1) >> (dieselStatusVars[index + 1, 0] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Working -> Not working IMPOSSIBLE",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 3] == 1) >> (dieselStatusVars[index + 1, 1] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Working -> Startup IMPOSSIBLE",
    )

    # TODO: to be changed, if timestep not equals 1hour
    model.addConstrs(
        (
            (dieselStatusVars[index + 1, 1] == 1)
            >> (dieselStatusVars[index + 2, 1] == 1 - dieselStatusVars[index, 1])
            for index in range(len(ini.timestamps) - 2)  # 0 1 1/0 0 0/1 1 0
        ),
        "StartUp Constraint",
    )

    model.addConstrs(
        (
            (dieselStatusVars[index + 1, 2] == 1)
            >> (dieselStatusVars[index + 2, 2] == 1 - dieselStatusVars[index, 2])
            for index in range(len(ini.timestamps) - 2)  # 0 1 1/0 0 0/1 1 0
        ),
        "ShutDown Constraint",
    )

    model.addConstrs(
        (
            (dieselStatusVars[index + 1, 3] == 1)
            >> (dieselStatusVars[index + d, 3] == 1 - dieselStatusVars[index, 3])
            for index in range(len(ini.timestamps) - (ini.dieselLeastRunHour + 1))
            for d in range(2, ini.dieselLeastRunHour + 1)
        ),
        "Least Running Time",
    )
    model.addConstr(
        ((dieselStatusVars[0, 0] == 1)), "Diesel Generator status initialization"
    )
    model.addConstr(
        ((dieselGeneratorsVars[0, 0] == 0)), "Diesel Generator power initialization "
    )

    return [dieselGeneratorsVars, dieselStatusVars]


def setUpPV(model, ini):
    pvVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="PVPowers"
    )

    if ini.loc_flag:
        print("PV data: use location")
        metadata, pvPowerValues = getNinjaPvApi(
            ini.loc_lat, ini.loc_lon, ini.timestamps
        )
        pvPowerValues = pvPowerValues.values
    else:
        print("PV data: use sample files")
        pvPowerValues = getNinja(ini.pvFile, ini.timestamps)
    assert len(pvPowerValues) == len(ini.timestamps)
    model.addConstrs(
        (pvVars[i, 0] == pvPowerValues[i] for i in range(len(ini.timestamps))),
        "1st pv panel generation",
    )

    return pvVars


def setUpFixedLoads(model, ini):
    fixedLoadVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="fixedLoads"
    )

    loadValues = getLoadsData(ini.loadsFile, ini.timestamps)
    assert len(loadValues) == len(ini.timestamps)
    model.addConstrs(
        (fixedLoadVars[i, 0] == loadValues[i] for i in range(len(ini.timestamps))),
        "power of fixed loads",
    )

    return fixedLoadVars


def setUpWind(model, ini):
    windVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="windPowers"
    )

    if ini.loc_flag:
        print("Wind data: use location")
        metadata, windPowerValues = getNinjaWindApi(
            ini.loc_lat, ini.loc_lon, ini.timestamps
        )
        windPowerValues = windPowerValues.values
    else:
        print("Wind data: use sample files")
        windPowerValues = getNinja(ini.windFile, ini.timestamps)

    assert len(windPowerValues) == len(ini.timestamps)
    model.addConstrs(
        (windVars[i, 0] == windPowerValues[i] for i in range(len(ini.timestamps))),
        "1st wind panel generation",
    )

    return windVars


def setUpBattery(model, ini):
    batteryPowerVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=-ini.P_bat_max,
        ub=ini.P_bat_max,
        vtype=GRB.CONTINUOUS,
        name="batPowers",
    )
    batteryEnergyVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=ini.SOC_bat_min * ini.E_bat_max,
        ub=ini.SOC_bat_max * ini.E_bat_max,
        vtype=GRB.CONTINUOUS,
        name="batEnergys",
    )

    model.addConstrs(
        (
            batteryEnergyVars[i + 1, 0]
            == batteryEnergyVars[i, 0]
            - ini.eta_bat
            * batteryPowerVars[i, 0]
            * getStepsize(ini.timestamps).total_seconds()
            / 3600  # stepsize: 1 hour
            for i in range(len(ini.timestamps) - 1)
        ),
        "battery charging",
    )

    model.addConstr(
        (batteryEnergyVars[0, 0] == ini.SOC_bat_init * ini.E_bat_max), "battery init"
    )

    # TODO: to be changed, if multiple days are considered
    model.addConstr(
        (
            batteryEnergyVars[len(ini.timestamps) - 1, 0]
            == ini.SOC_bat_init * ini.E_bat_max
        ),
        "battery end-of-day value",
    )

    return batteryPowerVars


def setUpEv(model, ini):
    evPowerVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=-ini.P_ev_max,
        ub=ini.P_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evPowers",
    )
    evEnergyVars = model.addVars(
        len(ini.timestamps),
        1,
        lb=ini.SOC_ev_min * ini.E_ev_max,
        ub=ini.SOC_ev_max * ini.E_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evEnergys",
    )

    model.addConstrs(
        (
            evPowerVars[i, 0] == 0
            for i in getTimeIndexRange(ini.timestamps, ini.t_a_ev, ini.t_b_ev)
        ),
        "ev charging",
    )
    model.addConstrs(
        (
            evEnergyVars[i + 1, 0]
            == evEnergyVars[i, 0]
            - ini.eta_ev
            * evPowerVars[i, 0]
            * getStepsize(ini.timestamps).total_seconds()
            / 3600  # stepsize: 1 hour
            for i in getTimeIndexRange(ini.timestamps, ini.timestamps[0], ini.t_a_ev)
            + getTimeIndexRange(ini.timestamps, ini.t_b_ev, ini.timestamps[-1])
        ),
        "ev charging",
    )

    model.addConstrs(
        (
            evEnergyVars[i, 0] >= 0.7 * ini.E_ev_max
            for i in range(
                int((ini.t_goal_ev - ini.timestamps[0]).total_seconds() / 3600),
                int((ini.t_b_ev - ini.timestamps[0]).total_seconds() / 3600) + 1,
            )
        ),
        "ev init",
    )
    # model.addConstr(
    #     (
    #         evEnergyVars[ini.timestamps.index(ini.t_b_ev)-2, 0] == 0.1 * ini.E_ev_max
    #     ),
    #     "ev after work",
    # )
    # TODO: to be changed, if multiple days are considered
    model.addConstr(
        (
            evEnergyVars[len(ini.timestamps) - 1, 0]
            - ini.eta_ev
            * evPowerVars[len(ini.timestamps) - 1, 0]
            * getStepsize(ini.timestamps).total_seconds()
            / 3600
            == evEnergyVars[0, 0]  # ini.SOC_ev_init * ini.E_ev_max
        ),
        "ev end-of-day value",
    )

    return evPowerVars


def printResults(model, ini):
    for v in model.getVars():
        print("%s %g" % (v.varName, v.x))

    print("Value of objective %s is %s" % (ini.goal, model.ObjVal))


def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[1])
    runSimpleModel(Configure(config))


if __name__ == "__main__":
    main(sys.argv)
