#!/usr/bin/env python3.7

import configparser
from datetime import datetime
from enum import Enum
import math
import sys
import os
from shutil import copyfile

from util import constructTimeStamps, getStepsize, getTimeIndexRangeDaily, diffIndexList

from data import (
    getNinja,
    getNinjaPvApi,
    getNinjaWindApi,
    getPriceData,
    getLoadsData,
    getPecanstreetData,
)
from plotting import plotting

import gurobipy as gp

from gurobipy import QuadExpr
from gurobipy import GRB


outputFolder = ""


class Goal(Enum):
    MINIMIZE_COST = "MINIMIZE_COST"
    GREEN_HOUSE = "GREEN_HOUSE"
    GREEN_HOUSE_QUADRATIC = "GREEN_HOUSE_QUADRATIC"
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
        self.t_a_ev = datetime.strptime(config["EV"]["t_a_ev"], "%H:%M:%S")
        self.t_b_ev = datetime.strptime(config["EV"]["t_b_ev"], "%H:%M:%S")
        self.t_goal_ev = datetime.strptime(config["EV"]["t_goal_ev"], "%H:%M:%S")

        # Time frame of optimization
        self.timestamps = constructTimeStamps(
            datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["stepsize"], "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )
        self.stepsize = getStepsize(self.timestamps)
        self.stepsizeHour = self.stepsize.total_seconds() / 3600

        # Generators
        self.P_dg_max = float(config["DIESEL"]["P_dg_max"])
        self.P_dg_min = float(config["DIESEL"]["P_dg_min"])
        self.dieselQuadraticCof = float(config["DIESEL"]["a_dg"])
        self.dieselLinearCof = float(config["DIESEL"]["b_dg"])
        self.dieselConstantCof = float(config["DIESEL"]["c_dg"])
        self.dieselFuelPrice = float(config["DIESEL"]["c_gen"])
        self.startUpCost = float(config["DIESEL"]["StartUpCost"])
        self.dieselLeastRunHour = datetime.strptime(
            config["DIESEL"]["LeastRunningTime"], "%H:%M:%S"
        ).hour
        self.dieselLeastPauseHour = datetime.strptime(
            config["DIESEL"]["LeastPauseTime"], "%H:%M:%S"
        ).hour
        self.dieselLeastRunTimestepNumber = int(
            math.ceil(self.dieselLeastRunHour / self.stepsizeHour)
        )
        self.dieselLeastPauseTimestepNumber = int(
            math.ceil(self.dieselLeastPauseHour / self.stepsizeHour)
        )
        self.startUpHour = datetime.strptime(
            config["DIESEL"]["StartUpTime"], "%H:%M:%S"
        ).hour
        self.shutDownHour = datetime.strptime(
            config["DIESEL"]["ShutDownTime"], "%H:%M:%S"
        ).hour
        self.shutDownTimestepNumber = int(
            math.ceil(self.shutDownHour / self.stepsizeHour)
        )
        self.startUpTimestepNumber = int(
            math.ceil(self.startUpHour / self.stepsizeHour)
        )
        self.deltaShutDown = self.P_dg_min / self.shutDownHour * self.stepsizeHour
        self.deltaStartUp = self.P_dg_min / self.startUpHour * self.stepsizeHour

        self.pvFile = config["PV"]["file"]
        self.windFile = config["WIND"]["file"]
        self.loadsFile = config["LOADS"]["file"]
        self.dataFile = config["DATA_PS"]["file"]
        self.dataPSLoads = "yes" == config["DATA_PS"]["loads"]
        self.dataPSPv = "yes" == config["DATA_PS"]["pv"]
        self.timeHeader = config["DATA_PS"]["timeHeader"]
        self.dataid = config["DATA_PS"]["dataid"]
        self.dataStart = datetime.strptime(
            config["DATA_PS"]["dataStart"], "20%y-%m-%d %H:%M:%S"
        )
        self.dataDelta = self.dataStart - datetime.strptime(
            config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"
        )
        self.costFileGrid = config["COST"]["file_grid"]
        self.constantPrice = float(config["COST"]["constant_price"])
        self.priceDataStart = datetime.strptime(
            config["COST"]["priceDataStart"], "20%y-%m-%d %H:%M:%S"
        )
        self.priceDataDelta = self.dataStart - datetime.strptime(
            config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"
        )
        self.co2Grid = float(config["CO2"]["grid_CO2"])
        self.co2Diesel = float(config["CO2"]["diesel_CO2"])


def runSimpleModel(ini):
    model = gp.Model("simple-model")

    pvVars = setUpPV(model, ini)
    windVars = setUpWind(model, ini)
    batteryPowerVars = setUpBattery(model, ini)
    evPowerVars = setUpEv(model, ini)
    fixedLoadVars = setUpFixedLoads(model, ini)
    dieselGeneratorsVars, dieselStatusVars = setUpDiesel(model, ini)
    fromGridVars, toGridVars = setUpGrid(model, ini)
    gridPrices = getPriceData(
        ini.costFileGrid, ini.timestamps, ini.priceDataDelta, ini.constantPrice
    )

    model.addConstrs(
        (
            fromGridVars.sum(i, "*")
            + pvVars.sum(i, "*")
            + windVars.sum(i, "*")
            + dieselGeneratorsVars.sum(i, "*")
            + batteryPowerVars.sum(i, "*")
            + evPowerVars.sum(i, "*")
            == fixedLoadVars.sum(i, "*") + toGridVars.sum(i, "*")
            for i in range(len(ini.timestamps))
        ),
        "power balance",
    )

    setObjective(
        model,
        ini,
        dieselGeneratorsVars,
        dieselStatusVars,
        fromGridVars,
        toGridVars,
        gridPrices,
    )

    model.optimize()
    model.write(outputFolder + "/res.sol")

    printResults(model, ini)
    printObjectiveResults(
        ini,
        fromGridVars,
        toGridVars,
        gridPrices,
        dieselGeneratorsVars,
        dieselStatusVars,
    )
    plotResults(model, ini, gridPrices)


def calcGreenhouseObjective(ini, fromGridVars, dieselGeneratorsVars):
    return ini.co2Diesel * gp.quicksum(
        dieselGeneratorsVars
    ) + ini.co2Grid * gp.quicksum(fromGridVars)


def calcDieselMinCostObjective(ini, dieselGeneratorsVars, dieselStatusVars):
    dieselObjExp = QuadExpr()
    for index in range(len(ini.timestamps)):
        dieselObjExp.add(
            dieselGeneratorsVars[index, 0]
            * dieselGeneratorsVars[index, 0]
            * ini.dieselQuadraticCof
            * ini.dieselFuelPrice
            * ini.stepsizeHour
        )
        dieselObjExp.add(
            dieselGeneratorsVars[index, 0]
            * ini.dieselLinearCof
            * ini.dieselFuelPrice
            * ini.stepsizeHour
        )
        dieselObjExp.add(
            ini.dieselConstantCof
            * (1 - dieselStatusVars[index, 0])
            * ini.dieselFuelPrice
            * ini.stepsizeHour
        )
        dieselObjExp.add(
            ini.startUpCost * dieselStatusVars[index, 2] / ini.startUpTimestepNumber
        )
    return dieselObjExp


def calcGridMinCostObjective(ini, fromGridVars, toGridVars, prices):
    return sum(
        [
            (fromGridVars[index, 0] - toGridVars[index, 0]) * price
            for index, price in enumerate(prices)
        ]
    )


def calcMinCostObjective(
    ini, fromGridVars, toGridVars, prices, dieselGeneratorsVars, dieselStatusVars
):
    dieselObjExp = calcDieselMinCostObjective(
        ini, dieselGeneratorsVars, dieselStatusVars
    )
    return dieselObjExp + calcGridMinCostObjective(
        ini, fromGridVars, toGridVars, prices
    )


def calcGreenhouseQuadraticObjective(ini, fromGridVars, dieselGeneratorsVars):
    return ini.co2Diesel * sum(
        [
            dieselGeneratorsVars[index, 0] * dieselGeneratorsVars[index, 0]
            for index in range(len(ini.timestamps))
        ]
    ) + ini.co2Grid * sum(
        [
            fromGridVars[index, 0] * fromGridVars[index, 0]
            for index in range(len(ini.timestamps))
        ]
    )


def calcGridIndependenceObjective(ini, fromGridVars, toGridVars):
    return gp.quicksum(fromGridVars) + gp.quicksum(toGridVars)


def setObjective(
    model, ini, dieselGeneratorsVars, dieselStatusVars, fromGridVars, toGridVars, prices
):
    if ini.goal is Goal.MINIMIZE_COST:
        model.setObjective(
            calcMinCostObjective(
                ini,
                fromGridVars,
                toGridVars,
                prices,
                dieselGeneratorsVars,
                dieselStatusVars,
            ),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GREEN_HOUSE:
        model.setObjective(
            calcGreenhouseObjective(ini, fromGridVars, dieselGeneratorsVars),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GREEN_HOUSE_QUADRATIC:
        model.setObjective(
            calcGreenhouseQuadraticObjective(ini, fromGridVars, dieselGeneratorsVars),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GRID_INDEPENDENCE:
        model.setObjective(
            calcGridIndependenceObjective(ini, fromGridVars, toGridVars), GRB.MINIMIZE
        )


def setUpGrid(model, ini):
    fromGridVars = model.addVars(
        len(ini.timestamps), 1, vtype=GRB.CONTINUOUS, name="fromGridPowers"
    )
    toGridVars = model.addVars(
        len(ini.timestamps), 1, vtype=GRB.CONTINUOUS, name="toGridPowers"
    )

    model.addConstrs(
        (
            fromGridVars[index, 0] * toGridVars[index, 0] == 0
            for index in range(len(ini.timestamps))
        )
    )

    return fromGridVars, toGridVars


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
        4,  # the first column: diesel is not committed/second: diesel at start up, third: diesel at shut down/fourth: diesel is committed
        vtype=GRB.BINARY,
        name="dieselStatus",
    )

    model.addConstrs(
        (dieselStatusVars.sum(i, "*") == 1 for i in range(len(ini.timestamps)))
    )

    model.addConstrs(
        (
            (dieselStatusVars[index, 3] == 1)
            >> (dieselGeneratorsVars[index, 0] >= ini.P_dg_min)
            for index in range(len(ini.timestamps))
        ),
        "Power generation when diesel generator is turned on",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index, 0] == 1) >> (dieselGeneratorsVars[index, 0] == 0)
            for index in range(len(ini.timestamps))
        ),
        "No power generation when diesel generator is turned off",
    )

    for index in range(len(ini.timestamps) - 1):
        model.addConstr(
            (
                (dieselStatusVars[index, 1] == 1)
                >> (
                    dieselGeneratorsVars[index + 1, 0]
                    == dieselGeneratorsVars[index, 0] + ini.deltaStartUp
                )
            ),
            "diesel generator power increase during Startup",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 2] == 1)
                >> (
                    dieselGeneratorsVars[index + 1, 0]
                    == dieselGeneratorsVars[index, 0] - ini.deltaShutDown
                )
            ),
            "diesel generator power decrease during Shutdown",
        )

        model.addConstr(
            (
                (dieselStatusVars[index, 0] == 1)
                >> (dieselStatusVars[index + 1, 3] == 0)
            ),
            "Not Working -> Working IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 0] == 1)
                >> (dieselStatusVars[index + 1, 2] == 0)
            ),
            "Not Working -> Shutdown IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 1] == 1)
                >> (dieselStatusVars[index + 1, 2] == 0)
            ),
            "Startup -> Shutdown IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 1] == 1)
                >> (dieselStatusVars[index + 1, 0] == 0)
            ),
            "Startup -> Not working IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 2] == 1)
                >> (dieselStatusVars[index + 1, 3] == 0)
            ),
            "Shutdown -> working IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 2] == 1)
                >> (dieselStatusVars[index + 1, 1] == 0)
            ),
            "Shutdown -> startup IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 3] == 1)
                >> (dieselStatusVars[index + 1, 0] == 0)
            ),
            "Working -> Not working IMPOSSIBLE",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 3] == 1)
                >> (dieselStatusVars[index + 1, 1] == 0)
            ),
            "Working -> Startup IMPOSSIBLE",
        )

    # e. g.: d_min = 5
    # index:    1 2 3 4 5 6 7 8 9 10 11 12 13
    # s_3:      0 0 0 0 1 1 1 1 1  0  0  0  0
    # sum:      1 2 3 4 5 4 3 2 1  0  0  0  0
    # d*s_3':   0 0 0 0 5 0 0 0 0 -5  0  0  0
    model.addConstrs(
        (
            sum(
                dieselStatusVars[index2, 3]
                for index2 in range(index, index + ini.dieselLeastRunTimestepNumber)
            )
            >= (ini.dieselLeastRunTimestepNumber)
            * (dieselStatusVars[index, 3] - dieselStatusVars[index - 1, 3])
            for index in range(
                1, len(ini.timestamps) - (ini.dieselLeastRunTimestepNumber)
            )
        ),
        "Least Running time",
    )
    model.addConstrs(
        (
            sum(
                dieselStatusVars[index2, 0]
                for index2 in range(index, index + ini.dieselLeastPauseTimestepNumber)
            )
            >= (ini.dieselLeastPauseTimestepNumber)
            * (dieselStatusVars[index, 0] - dieselStatusVars[index - 1, 0])
            for index in range(
                1, len(ini.timestamps) - (ini.dieselLeastPauseTimestepNumber)
            )
        ),
        "Least Pause time",
    )

    # TODO: this constraint should not be necessary, but there is a bug sometimes change faster than ini.deltaShutDown or deltaStartUp
    model.addConstrs(
        (
            (dieselStatusVars[index + 1, 1] == 1)
            >> (
                dieselStatusVars[index + ini.startUpTimestepNumber, 1]
                == 1 - dieselStatusVars[index, 1]
            )
            for index in range(
                len(ini.timestamps) - ini.startUpTimestepNumber
            )  # 0 1 1/0 0 0/1 1 0
        ),
        "StartUp Constraint",
    )
    model.addConstrs(
        (
            (dieselStatusVars[index + 1, 2] == 1)
            >> (
                dieselStatusVars[index + ini.shutDownTimestepNumber, 2]
                == 1 - dieselStatusVars[index, 2]
            )
            for index in range(
                len(ini.timestamps) - ini.shutDownTimestepNumber
            )  # 0 1 1/0 0 0/1 1 0
        ),
        "ShutDown Constraint",
    )
    model.addConstrs(
        (
            dieselStatusVars[index, 1] == 0
            for index in range(
                len(ini.timestamps)
                - (ini.dieselLeastRunTimestepNumber + ini.shutDownTimestepNumber),
                len(ini.timestamps),
            )
        ),
        "do not startup if not enough time before end of simulation",
    )
    model.addConstr(
        ((dieselStatusVars[0, 0] == 1)), "diesel generator is not committed at start"
    )
    model.addConstr(
        ((dieselStatusVars[len(ini.timestamps) - 1, 0] == 1)),
        "Diesel Generator status in the end of simulation",
    )

    return dieselGeneratorsVars, dieselStatusVars


def setUpPV(model, ini):
    pvVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="PVPowers"
    )

    if ini.loc_flag:
        print("PV data: use location and query from renewables.ninja API")
        metadata, pvPowerValues = getNinjaPvApi(
            ini.loc_lat, ini.loc_lon, ini.timestamps
        )
        pvPowerValues = pvPowerValues.values
    else:
        print("PV data: use sample files")
        if ini.dataPSPv:
            print("PV data: use Pecanstreet dataset with dataid:", (ini.dataid))
            pvPowerValues = getPecanstreetData(
                ini.dataFile,
                ini.timeHeader,
                ini.dataid,
                "solar",
                ini.timestamps,
                ini.dataDelta,
            )
        else:
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
    if ini.dataPSLoads:
        loadValues = getPecanstreetData(
            ini.dataFile,
            ini.timeHeader,
            ini.dataid,
            "grid",
            ini.timestamps,
            ini.dataDelta,
        )
    else:
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
        len(ini.timestamps) + 1,
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
            * ini.stepsizeHour  # E in kW per hour
            for i in range(len(ini.timestamps))
        ),
        "battery charging",
    )

    model.addConstr(
        (batteryEnergyVars[0, 0] == ini.SOC_bat_init * ini.E_bat_max), "battery init"
    )

    # TODO: to be changed, if multiple days are considered
    model.addConstr(
        (batteryEnergyVars[len(ini.timestamps), 0] == ini.SOC_bat_init * ini.E_bat_max),
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
        len(ini.timestamps) + 1,
        1,
        lb=ini.SOC_ev_min * ini.E_ev_max,
        ub=ini.SOC_ev_max * ini.E_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evEnergys",
    )
    evNotChargableIndices = getTimeIndexRangeDaily(
        ini.timestamps, ini.t_a_ev, ini.t_b_ev, varB=0
    )
    model.addConstrs((evPowerVars[i, 0] == 0 for i in evNotChargableIndices), "ev gone")
    allIndices = range(len(ini.timestamps))
    evChargableIndices = diffIndexList(allIndices, evNotChargableIndices)
    model.addConstrs(
        (
            evEnergyVars[i + 1, 0]
            == evEnergyVars[i, 0]
            - ini.eta_ev * evPowerVars[i, 0] * ini.stepsizeHour  # E in kW per hour
            for i in evChargableIndices
        ),
        "ev charging",
    )

    model.addConstrs(
        (
            evEnergyVars[i, 0] >= 0.7 * ini.E_ev_max
            for i in getTimeIndexRangeDaily(ini.timestamps, ini.t_goal_ev, ini.t_a_ev)
        ),
        "ev charging goal",
    )

    evEnergyWhileGone = getTimeIndexRangeDaily(
        ini.timestamps, ini.t_a_ev, ini.t_b_ev, varA=1, varB=1
    )
    model.addConstrs(
        (evEnergyVars[i, 0] == 0.1 * ini.E_ev_max for i in evEnergyWhileGone),
        "ev after work",
    )
    model.addConstr(
        (evEnergyVars[len(ini.timestamps), 0] == evEnergyVars[0, 0]),
        "ev end-start energy are equal",
    )
    # print(evNotChargableIndices)
    # print(evChargableIndices)
    # print(evEnergyWhileGone)

    return evPowerVars


def printObjectiveResults(
    ini, fromGridVars, toGridVars, gridPrices, dieselGeneratorsVars, dieselStatusVars
):
    print(
        "MINIMIZE_COST goal: %.2f"
        % calcMinCostObjective(
            ini,
            fromGridVars,
            toGridVars,
            gridPrices,
            dieselGeneratorsVars,
            dieselStatusVars,
        ).getValue()
    )
    print(
        "GREEN_HOUSE goal: %.1f"
        % calcGreenhouseObjective(ini, fromGridVars, dieselGeneratorsVars).getValue()
    )
    print(
        "GREEN_HOUSE_QUADRATIC goal: %.1f"
        % calcGreenhouseQuadraticObjective(
            ini, fromGridVars, dieselGeneratorsVars
        ).getValue()
    )
    print(
        "GRID_INDEPENDENCE goal: %.1f"
        % calcGridIndependenceObjective(ini, fromGridVars, toGridVars).getValue()
    )


def printResults(model, ini):
    for v in model.getVars():
        print("%s %g" % (v.varName, v.x))

    print("Value of objective %s is %s" % (ini.goal, model.ObjVal))


def plotResults(model, ini, gridPrices):
    varN = []
    varX = []
    for v in model.getVars():
        varN.append(v.varName)
        varX.append(v.x)
    plotting(varN, varX, gridPrices, outputFolder, ini.timestamps)


def copyConfigFile(filepath, outputFolder):
    copyfile(filepath, os.path.join(outputFolder, "conf.ini"))


def main(argv):
    global outputFolder
    outputFolder = (
        "output/"
        + str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
        + "/"
    )
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)
    copyConfigFile(argv[1], outputFolder)
    config = configparser.ConfigParser()
    config.read(argv[1])
    runSimpleModel(Configure(config))


if __name__ == "__main__":
    main(sys.argv)
