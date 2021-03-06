#!/usr/bin/env python3.7
import configparser
import math
import os
import sys
from datetime import datetime
from enum import Enum
from itertools import product
from shutil import copyfile

import gurobipy as gp
import numpy as np
import pandas as pd
from data import (
    getNinja,
    getNinjaPvApi,
    getNinjaWindApi,
    getPriceData,
    getLoadsData,
    getPecanstreetData,
    getPredictedPVValue,
    getPredictedLoadValue,
    resampleData,
)
from gurobipy import QuadExpr, GRB
from plot_gurobi import plotting
from util import constructTimeStamps, getStepsize, getTimeIndexRangeDaily, diffIndexList

outputFolder = ""
caseID = ""
debugPlots = False


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
        self.loadResFlag = "yes" == config["GLOBAL"]["loadResFlag"]
        self.overwrite = "yes" == config["GLOBAL"]["overwrite"]
        self.calcAllFlag = "yes" == config["GLOBAL"]["calcAllFlag"]

        # Battery init (to be moved to a initialization file)
        self.SOC_bat_min = float(config["BAT"]["SOC_bat_min"])
        self.SOC_bat_init = float(config["BAT"]["SOC_bat_init"])
        self.SOC_bat_max = float(config["BAT"]["SOC_bat_max"])
        self.E_bat_max = float(config["BAT"]["E_bat_max"])
        self.eta_bat = float(config["BAT"]["eta_bat"])
        self.P_bat_max = float(config["BAT"]["P_bat_max"])
        self.ChargeConvertLoss = float(config["BAT"]["ConvertLoss"])

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

        # verify we have enough day to build the set for the prediction
        assert (
            datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S")
            - datetime.strptime(config["TIME"]["startPred"], "20%y-%m-%d %H:%M:%S")
        ).days >= 1, "a delay of at least 1 day is needed to predict"
        # Time frame of optimization
        self.timestamps = constructTimeStamps(
            datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["stepsize"], "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )
        self.timestampsPredPV = constructTimeStamps(
            datetime.strptime(config["TIME"]["startPred"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["stepsizePredPV"], "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )
        self.timestampsPredLoad = constructTimeStamps(
            datetime.strptime(config["TIME"]["startPred"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S"),
            datetime.strptime(config["TIME"]["stepsizePredLoad"], "%H:%M:%S")
            - datetime.strptime("00:00:00", "%H:%M:%S"),
        )
        self.stepsize = getStepsize(self.timestamps)
        self.stepsizeHour = self.stepsize.total_seconds() / 3600
        self.stepsizeMinute = self.stepsize.total_seconds() / 60
        # we add +1 because we are between 00:00 and 23:45 so < 1 day
        self.nbDay = (
            datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S")
            - datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S")
        ).days + 1

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
        self.pvPdct = "yes" == config["PV"]["usePredicted"]
        self.showErr = "yes" == config["GLOBAL"]["showErr"]
        self.pvScale = float(config["PV"]["scale"])
        self.windFile = config["WIND"]["file"]
        self.windScale = float(config["WIND"]["scale"])
        self.windStart = datetime.strptime(
            config["WIND"]["windStart"], "20%y-%m-%d %H:%M:%S"
        )
        self.windDelta = self.windStart - datetime.strptime(
            config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"
        )
        self.pvStart = datetime.strptime(config["PV"]["pvStart"], "20%y-%m-%d %H:%M:%S")
        self.pvDelta = self.pvStart - datetime.strptime(
            config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"
        )
        self.loadsFile = config["LOADS"]["file"]
        self.loadsPdct = "yes" == config["LOADS"]["usePredicted"]
        self.loadsScale = float(config["LOADS"]["scale"])
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
        self.priceDataDelta = self.priceDataStart - datetime.strptime(
            config["TIME"]["start"], "20%y-%m-%d %H:%M:%S"
        )
        self.co2Grid = float(config["CO2"]["grid_CO2"])
        self.co2Diesel = float(config["CO2"]["diesel_CO2"])


def runSimpleModel(ini):
    model = gp.Model("simple-model")

    pvVars, pvVarsErr = setUpPV(model, ini)
    windVars = setUpWind(model, ini)
    batteryPowerVars = setUpBattery(model, ini)
    evPowerVars = setUpEv(model, ini)
    fixedLoadVars, loadsVarsErr = setUpFixedLoads(model, ini)
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
        batteryPowerVars,
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
        batteryPowerVars,
    )
    plotResults(model, ini, gridPrices)

    objres = getObjectiveResults(
        ini,
        fromGridVars,
        toGridVars,
        gridPrices,
        dieselGeneratorsVars,
        dieselStatusVars,
        batteryPowerVars,
    )
    recourse = printRecourseAction(model, ini, gridPrices, objres)
    objres.extend(recourse)

    return objres, gridPrices, extractResults(model)


def calcDieselCost(ini, dieselGeneratorsVars, dieselStatusVars):
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
            ini.startUpCost * dieselStatusVars[index, 1] / ini.startUpTimestepNumber
        )
    return dieselObjExp


def calcBatChargeLoss(ini, batteryPowerVars):
    return gp.quicksum(
        ini.ChargeConvertLoss
        * (batteryPowerVars[i, 0])
        * (batteryPowerVars[i, 0])
        * ini.stepsizeHour
        for i in range(len(ini.timestamps))
    )


def calcErrObjClassic(model, ini, objValue):
    varName = []
    varVal = []
    for v in model.getVars():
        varName.append(v.varName)
        varVal.append(v.x)

    dico = {"PVPowers": [], "fixedLoads": []}
    for i in range(len(varName)):
        for val in dico.keys():
            if val in varName[i]:
                dico[val].append(varVal[i])
                break

    pvPowerReal = (
        getPecanstreetData(
            ini.dataFile,
            ini.timeHeader,
            ini.dataid,
            "solar",
            ini.timestamps,
            ini.dataDelta,
        )
        * ini.pvScale
    )

    pvPowerErr = np.array(dico["PVPowers"]) - np.array(pvPowerReal)

    fixedLoadPowerReal = (
        getPecanstreetData(
            ini.dataFile,
            ini.timeHeader,
            ini.dataid,
            "grid",
            ini.timestamps,
            ini.dataDelta,
        )
        * ini.loadsScale
    )
    fixedLoadPowerErr = np.array(fixedLoadPowerReal) - np.array(dico["fixedLoads"])

    if debugPlots:
        import matplotlib.pyplot as plt

        print(pvPowerReal)
        plt.figure()
        plt.plot(np.array(pvPowerReal), label="PV real", ls="--", color="orange")
        plt.plot(np.array(dico["PVPowers"]), label="PV", color="orange")
        plt.plot(np.array(fixedLoadPowerReal), label="Loads real", ls="--", color="red")
        plt.plot(np.array(dico["fixedLoads"]), label="Loads", color="red")
        plt.show()

    return np.sum(
        [
            (fixedLoadPowerErr[index] + pvPowerErr[index]) * objValue[index]
            for index in range(len(objValue))
        ]
    )


def calcGridCost(ini, fromGridVars, toGridVars, prices):
    return sum(
        [
            (fromGridVars[index, 0] - toGridVars[index, 0]) * price
            for index, price in enumerate(prices)
        ]
    )


def calcMinCostObjective(
    ini,
    fromGridVars,
    toGridVars,
    prices,
    dieselGeneratorsVars,
    dieselStatusVars,
    batteryPowerVars,
    type,
):
    dieselObjExp = calcDieselCost(ini, dieselGeneratorsVars, dieselStatusVars)
    gridCostObjExp = calcGridCost(ini, fromGridVars, toGridVars, prices)
    batCostObjExp = calcBatChargeLoss(ini, batteryPowerVars)
    if type == "Virtual":
        return dieselObjExp + gridCostObjExp + batCostObjExp
    elif type == "True":
        return dieselObjExp + gridCostObjExp


def calcGreenhouseObjective(
    ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, type
):
    dieselGreenhouse = (
        ini.co2Diesel * gp.quicksum(dieselGeneratorsVars) * ini.stepsizeHour
    )
    gridGreenhouse = ini.co2Grid * gp.quicksum(fromGridVars) * ini.stepsizeHour
    if type == "Virtual":
        return (
            dieselGreenhouse + gridGreenhouse + calcBatChargeLoss(ini, batteryPowerVars)
        )
    elif type == "True":
        return dieselGreenhouse + gridGreenhouse


def calcGreenhouseQuadraticObjective(
    ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, type
):
    dieselGreenhouseQuadratic = ini.co2Diesel * sum(
        [
            dieselGeneratorsVars[index, 0] * dieselGeneratorsVars[index, 0]
            for index in range(len(ini.timestamps))
        ]
    )

    gridGreenhouseQuadratic = ini.co2Grid * sum(
        [
            fromGridVars[index, 0] * fromGridVars[index, 0]
            for index in range(len(ini.timestamps))
        ]
    )
    if type == "Virtual":
        return (
            dieselGreenhouseQuadratic
            + gridGreenhouseQuadratic
            + calcBatChargeLoss(ini, batteryPowerVars)
        )
    elif type == "True":
        return dieselGreenhouseQuadratic + gridGreenhouseQuadratic


def calcGridIndependenceObjective(
    ini, fromGridVars, toGridVars, batteryPowerVars, type
):
    if type == "Virtual":
        return (
            gp.quicksum(fromGridVars)
            + gp.quicksum(toGridVars)
            + calcBatChargeLoss(ini, batteryPowerVars)
        )
    elif type == "True":
        return gp.quicksum(fromGridVars) + gp.quicksum(toGridVars)


def setObjective(
    model,
    ini,
    dieselGeneratorsVars,
    dieselStatusVars,
    fromGridVars,
    toGridVars,
    batteryPowerVars,
    prices,
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
                batteryPowerVars,
                "Virtual",
            ),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GREEN_HOUSE:
        model.setObjective(
            calcGreenhouseObjective(
                ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "Virtual"
            ),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GREEN_HOUSE_QUADRATIC:
        model.setObjective(
            calcGreenhouseQuadraticObjective(
                ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "Virtual"
            ),
            GRB.MINIMIZE,
        )
    elif ini.goal is Goal.GRID_INDEPENDENCE:
        model.setObjective(
            calcGridIndependenceObjective(
                ini, fromGridVars, toGridVars, batteryPowerVars, "Virtual"
            ),
            GRB.MINIMIZE,
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
        4,
        # the first column: diesel is not committed/second: diesel at start up, third: diesel at shut down/fourth: diesel is committed
        vtype=GRB.BINARY,
        name="dieselStatus",
    )

    dieselPowerRamp = model.addVars(
        len(ini.timestamps),
        1,
        lb=-ini.deltaShutDown,
        ub=ini.deltaStartUp,
        vtype=GRB.CONTINUOUS,
        name="dieselPowerRamp",
    )

    dieselStartupRamp = model.addVars(
        len(ini.timestamps),
        1,
        lb=0,
        ub=ini.deltaStartUp,
        vtype=GRB.CONTINUOUS,
        name="dieselStartupRamp",
    )

    dieselShutdownRamp = model.addVars(
        len(ini.timestamps),
        1,
        lb=0,
        ub=ini.deltaShutDown,
        vtype=GRB.CONTINUOUS,
        name="dieselShutdownRamp",
    )

    model.addConstrs(
        (dieselStatusVars.sum(i, "*") == 1 for i in range(len(ini.timestamps)))
    )

    model.addConstrs(
        (
            (dieselStatusVars[index, 3] == 1)
            >> (dieselGeneratorsVars[index + 1, 0] >= ini.P_dg_min)
            for index in range(len(ini.timestamps) - 1)
        ),
        "Power generation when diesel generator is turned on",
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
            (dieselStatusVars[index, 0] == 1)
            >> (dieselGeneratorsVars[index + 1, 0] == 0)
            for index in range(len(ini.timestamps) - 1)
        ),
        "No power generation when diesel generator is turned off",
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
                    == dieselGeneratorsVars[index, 0] + dieselStartupRamp[index, 0]
                )
            ),
            "diesel generator power increase during Startup",
        )
        model.addConstr(
            (
                (dieselStatusVars[index, 2] == 1)
                >> (
                    dieselGeneratorsVars[index + 1, 0]
                    == dieselGeneratorsVars[index, 0] - dieselShutdownRamp[index, 0]
                )
            ),
            "diesel generator power decrease during Shutdown",
        )

        model.addConstr(
            (
                (dieselStatusVars[index, 3] == 1)
                >> (
                    dieselGeneratorsVars[index + 1, 0]
                    == dieselGeneratorsVars[index, 0] + dieselPowerRamp[index, 0]
                )
            ),
            "ramp limit during fully committed",
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
            >= ini.dieselLeastPauseTimestepNumber
            * (dieselStatusVars[index, 3] - dieselStatusVars[index - 1, 3])
            for index in range(
                1, len(ini.timestamps) - ini.dieselLeastRunTimestepNumber
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
            >= ini.dieselLeastPauseTimestepNumber
            * (dieselStatusVars[index, 0] - dieselStatusVars[index - 1, 0])
            for index in range(
                1, len(ini.timestamps) - ini.dieselLeastPauseTimestepNumber
            )
        ),
        "Least Pause time",
    )

    model.addConstr(
        (dieselStatusVars[0, 3] == 0),
        "diesel generator is not in status committed at start",
    )
    model.addConstr(
        (dieselStatusVars[0, 2] == 0), "diesel generator is not in shutdown at start"
    )
    model.addConstr(
        (dieselGeneratorsVars[0, 0] == 0),
        "In any case at start the generation has to be 0",
    )

    model.addConstr(
        (dieselStatusVars[len(ini.timestamps) - 1, 0] == 1),
        "Diesel Generator status in the end of simulation",
    )

    return dieselGeneratorsVars, dieselStatusVars


def setUpPV(model, ini):
    if ini.pvPdct:
        pvPowerValuesReal = (
            getPecanstreetData(
                ini.dataFile,
                ini.timeHeader,
                ini.dataid,
                "solar",
                ini.timestampsPredPV,
                ini.dataDelta,
            )
            * ini.pvScale
        )
        print("PV data: use predicted values")
        pvPowerValues, lookback, out = getPredictedPVValue(
            pvPowerValuesReal, ini.timestampsPredPV, ini.dataDelta
        )
        # check if it works TODO
        # pvPowerValues = pvPowerValues[lookback % out]
        # data = pd.DataFrame(pvPowerValues, index=ini.timestampsPredPV[-len(pvPowerValues):])
        # pvPowerValues = resampleData(data, ini.timestamps)
        pvPowerValuesConcat = []
        for i in range(0, ini.nbDay):
            pvPowerValuesConcat.extend(pvPowerValues[(out - lookback) + (i * out)])
        data = pd.DataFrame(
            pvPowerValuesConcat, index=ini.timestampsPredPV[-len(pvPowerValuesConcat) :]
        )
        pvPowerValues = np.array(resampleData(data, ini.timestamps))
        dataReal = pd.DataFrame(pvPowerValuesReal, index=ini.timestampsPredPV)
        dataReal = dataReal.loc[ini.timestampsPredPV[-out:]]
        pvPowerValuesReal = resampleData(dataReal, ini.timestamps)
    else:
        if ini.loc_flag:
            print("PV data: use location and query from renewables.ninja API")
            metadata, pvPowerValues = getNinjaPvApi(
                ini.loc_lat, ini.loc_lon, ini.timestamps
            )
            pvPowerValues = pvPowerValues.values * ini.pvScale
        else:
            print("PV data: use sample files")
            if ini.dataPSPv:
                print("PV data: use Pecanstreet dataset with dataid:", ini.dataid)
                pvPowerValues = (
                    getPecanstreetData(
                        ini.dataFile,
                        ini.timeHeader,
                        ini.dataid,
                        "solar",
                        ini.timestamps,
                        ini.dataDelta,
                    )
                    * ini.pvScale
                )
            else:
                pvPowerValues = getNinja(ini.pvFile, ini.timestamps) * ini.pvScale
        pvPowerValuesReal = pvPowerValues
    assert len(pvPowerValues) == len(ini.timestamps)

    pvPowerValues = np.abs(pvPowerValues)

    assert all(i >= 0 for i in pvPowerValues)

    pvVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="PVPowers"
    )
    model.addConstrs(
        (pvVars[i, 0] == pvPowerValues[i] for i in range(len(ini.timestamps))),
        "1st pv panel generation",
    )

    errPvValues = [r - p for r, p in zip(pvPowerValuesReal, pvPowerValues)]
    # errPvVars = model.addVars(
    #     len(ini.timestamps), 1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="errPv"
    # )
    # model.addConstrs(
    #     (errPvVars[i, 0] == - errPvValues[i] for i in range(len(ini.timestamps))),
    #     "pv power error",
    # )

    return pvVars, errPvValues  # errPvVars


def setUpFixedLoads(model, ini):
    if ini.loadsPdct:
        loadValuesReal = (
            getPecanstreetData(
                ini.dataFile,
                ini.timeHeader,
                ini.dataid,
                "grid",
                ini.timestampsPredLoad,
                ini.dataDelta,
            )
            * ini.loadsScale
        )
        print("Load data: use predicted values")
        loadValues, lookback, out = getPredictedLoadValue(
            loadValuesReal, ini.timestampsPredLoad, ini.dataDelta
        )
        # the 0 index is not the value at midnight
        loadValuesConcat = []
        for i in range(0, ini.nbDay):
            loadValuesConcat.extend(loadValues[(lookback % out) + (i * out)])
        data = pd.DataFrame(
            loadValuesConcat, index=ini.timestampsPredLoad[-len(loadValuesConcat) :]
        )
        loadValues = np.array(resampleData(data, ini.timestamps))

    else:
        if ini.dataPSLoads:
            loadValues = (
                getPecanstreetData(
                    ini.dataFile,
                    ini.timeHeader,
                    ini.dataid,
                    "grid",
                    ini.timestamps,
                    ini.dataDelta,
                )
                * ini.loadsScale
            )
        else:
            loadValues = getLoadsData(ini.loadsFile, ini.timestamps) * ini.loadsScale
        loadValuesReal = loadValues

    assert len(loadValues) == len(ini.timestamps)
    assert all(i >= 0 for i in loadValues)

    fixedLoadVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="fixedLoads"
    )
    model.addConstrs(
        (fixedLoadVars[i, 0] == loadValues[i] for i in range(len(ini.timestamps))),
        "power of fixed loads",
    )

    errLoadValues = [
        r - p for r, p in zip(loadValuesReal[-len(loadValues) :], loadValues)
    ]
    # errLoadVars = model.addVars(
    #     len(ini.timestamps), 1, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="errLoads"
    # )
    # model.addConstrs(
    #     (errLoadVars[i, 0] == errLoadValues[i] for i in range(len(ini.timestamps))),
    #     "power of fixed loads",
    # )

    return fixedLoadVars, errLoadValues  # errLoadVars


def setUpWind(model, ini):
    windVars = model.addVars(
        len(ini.timestamps), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="windPowers"
    )

    if ini.loc_flag:
        print("Wind data: use location")
        metadata, windPowerValues = getNinjaWindApi(
            ini.loc_lat, ini.loc_lon, ini.timestamps
        )
        windPowerValues = windPowerValues.values * ini.windScale
    else:
        print("Wind data: use sample files")
        windPowerValues = (
            getNinja(ini.windFile, ini.timestamps, ini.windDelta) * ini.windScale
        )

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

    model.addConstr(
        (batteryEnergyVars[len(ini.timestamps), 0] == ini.SOC_bat_init * ini.E_bat_max),
        "battery start-end energy equality",
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


def getObjectiveResults(
    ini,
    fromGridVars,
    toGridVars,
    gridPrices,
    dieselGeneratorsVars,
    dieselStatusVars,
    batteryPowerVars,
):
    return [
        calcMinCostObjective(
            ini,
            fromGridVars,
            toGridVars,
            gridPrices,
            dieselGeneratorsVars,
            dieselStatusVars,
            batteryPowerVars,
            "True",
        ).getValue(),
        calcGreenhouseObjective(
            ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "True"
        ).getValue(),
        calcGreenhouseQuadraticObjective(
            ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "True"
        ).getValue(),
        calcGridIndependenceObjective(
            ini, fromGridVars, toGridVars, batteryPowerVars, "True"
        ).getValue(),
    ]


def printObjectiveResults(
    ini,
    fromGridVars,
    toGridVars,
    gridPrices,
    dieselGeneratorsVars,
    dieselStatusVars,
    batteryPowerVars,
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
            batteryPowerVars,
            "True",
        ).getValue()
    )
    print(
        "GREEN_HOUSE goal: %.1f"
        % calcGreenhouseObjective(
            ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "True"
        ).getValue()
    )
    print(
        "GREEN_HOUSE_QUADRATIC goal: %.1f"
        % calcGreenhouseQuadraticObjective(
            ini, fromGridVars, dieselGeneratorsVars, batteryPowerVars, "True"
        ).getValue()
    )
    print(
        "GRID_INDEPENDENCE goal: %.1f"
        % calcGridIndependenceObjective(
            ini, fromGridVars, toGridVars, batteryPowerVars, "True"
        ).getValue()
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

    plotting(varN, varX, gridPrices, outputFolder, ini, [False] * 11, caseID)


def extractResults(model):
    varN = []
    varX = []
    for v in model.getVars():
        varN.append(v.varName)
        varX.append(v.x)

    return varN, varX


def printRecourseAction(model, ini, gridPrices, objres):
    ret = []
    rec = calcErrObjClassic(model, ini, gridPrices)
    ret.append(rec)
    print("recourse action : Error cost : % 2.2f $" % rec)
    diff = rec + objres[0]
    ret.append(diff)
    print("recourse action : final cost : % 2.2f $" % diff)
    rec = calcErrObjClassic(model, ini, [ini.co2Grid] * len(gridPrices))
    ret.append(rec)
    print("recourse action : Error CO2 : % 2.2f g" % rec)
    diff = rec + objres[1]
    ret.append(diff)
    print("recourse action : final CO2 : % 2.2f g" % diff)
    rec = calcErrObjClassic(model, ini, [1] * len(gridPrices))
    ret.append(rec)
    print("recourse action : Error Taken from the grid : % 2.2f kWh" % rec)
    diff = rec + objres[3]
    ret.append(diff)
    print("recourse action : final Taken from the grid : % 2.2f kWh" % diff)
    return ret


def copyConfigFile(filepath, outputFolder):
    copyfile(filepath, os.path.join(outputFolder, "conf.ini"))


def getCaseID(g, be, ee, l):
    gs = g.value.split("_")[-1]
    caseID = "{}_BE{}_EE{}_L{}".format(gs, be, ee, l)
    return caseID


def main(argv):
    global caseID
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
    ini = Configure(config)

    baseOutputFolder = outputFolder

    goalsRange = [
        Goal(x) for x in ["MINIMIZE_COST", "GREEN_HOUSE", "GRID_INDEPENDENCE"]
    ]
    batRangeEmax = [(0, 0), (20, 0), (0, 20), (20, 20)]
    # batRangeEmax = [(0, 20), (10, 20), (20, 20), (30, 20)] # vary stat. bat.
    # batRangePmax = [3, 5, 7]
    loadRangeScale = [0, 1, 4]

    idx = pd.DataFrame(
        list(
            [
                [x[0][0], x[0][1], x[1]]
                for x in list(
                    product(product(goalsRange, batRangeEmax), loadRangeScale)
                )
            ]
        ),
        columns=["goals", "E_bat_mas", "loadsScale"],
    )
    casesArr = np.array(idx).reshape(3, 4, 3, 3)

    if ini.loadResFlag:
        # try to load previous results
        try:
            resultsGoals = np.load("resultsGoal.npy", allow_pickle=True)
            print("Loaded Results")
        except IOError:
            print("No resultsGoal.npy found. Continue without loading results.")
            resultsGoals = np.full(
                (len(goalsRange), len(batRangeEmax), len(loadRangeScale), 13), None
            )
    else:
        resultsGoals = np.full(
            (len(goalsRange), len(batRangeEmax), len(loadRangeScale), 13), None
        )

    # Only update the results selected in update Results with a True value
    updateResults = np.full(
        (len(goalsRange), len(batRangeEmax), len(loadRangeScale)), False
    )
    # baseline
    updateResults[0, 0, 1] = True
    # sensibility analysis
    updateResults[:, :, 1] = True

    for ig, g in enumerate(goalsRange):
        ini.goal = g
        for ibe, (be, ee) in enumerate(batRangeEmax):
            ini.E_bat_max = be
            ini.E_ev_max = ee
            for il, l in enumerate(loadRangeScale):
                ini.loadsScale = l
                if (
                    updateResults[ig, ibe, il]
                    and (ini.overwrite or resultsGoals[ig, ibe, il, 0] is None)
                ) or ini.calcAllFlag:
                    caseID = getCaseID(g, be, ee, l)
                    print(
                        "------------------ {}{} ------------------".format(
                            baseOutputFolder, caseID
                        )
                    )
                    outputFolder = "{}{}/".format(baseOutputFolder, caseID)
                    os.makedirs(outputFolder)
                    objectiveResults, gridPrices, [varN, varX] = runSimpleModel(ini)
                    resultsGoals[ig, ibe, il, 0:10] = np.array(objectiveResults)[:]
                    resultsGoals[ig, ibe, il, 10] = gridPrices
                    resultsGoals[ig, ibe, il, 11] = varN
                    resultsGoals[ig, ibe, il, 12] = varX

    np.save(os.path.join(baseOutputFolder, "resultsGoal.npy"), resultsGoals)
    dfResults = pd.DataFrame(
        resultsGoals.reshape(36, 13),
        index=pd.MultiIndex.from_frame(idx),
        columns=[
            "COST",
            "GGE",
            "GGEsq",
            "GRID_INDEPENDENCE",
            "ERROR_COST",
            "FINAL_COST",
            "ERROR_CO2",
            "FINAL_C02",
            "ERROR_IND",
            "FINAL_IND",
            "gridPrices",
            "varN",
            "varX",
        ],
    ).round(3)
    # with pd.option_context("display.max_rows", 99, "display.max_columns", 12):
    #     dfResults = pd.DataFrame(
    #         resultsGoals.reshape(27, 10),
    #         index=pd.MultiIndex.from_frame(idx),
    #         columns=[
    #             "COST",
    #             "GGE",
    #             "GGEsq",
    #             "GRID_INDEPENDENCE",
    #             "ERROR_COST",
    #             "FINAL_COST",
    #             "ERROR_CO2",
    #             "FINAL_C02",
    #             "ERROR_IND",
    #             "FINAL_IND",
    #         ],
    #     )
    dfResults = dfResults.round(3)
    with pd.option_context("display.max_rows", 99, "display.max_columns", 12):
        print(dfResults)

    dfResults.to_csv(baseOutputFolder + "resultDataframe.csv")

    np.save(os.path.join(baseOutputFolder, "resultsGoal.npy"), resultsGoals)
    # dfResults = pd.DataFrame(resultsGoals.reshape(27, 7), index=pd.MultiIndex.from_frame(idx),
    #                         columns=["COST", "GGE", "GGEsq", "GRID_INDEPENDENCE", "gridPrices", "varN", "varX"])

    # Experimental AREA ------------------------------------------------------------------------
    print(dfResults)
    plotList = [False] * 11
    plotList[5] = True
    print(casesArr[0, 1, 1])
    # Plot goals over varying parameter
    # import matplotlib.pyplot as plt
    # from plot_parameter_variation import plot_parameter_variation
    # plot_parameter_variation(resultsGoals, goalsRange, batRangeEmax, loadRangeScale, baseOutputFolder)
    print(dfResults.loc[casesArr[0, 1, 1]])
    print(dfResults.loc[(Goal("MINIMIZE_COST"), (20, 20), 1)])

    # from plot_gurobi import plotInteractive
    # plotInteractive(dfResults,outputFolder,ini)

    # sel = (Goal("MINIMIZE_COST"), (20, 0), 1)
    # caseID = getCaseID(sel[0], sel[1], sel[2])
    # plotting(
    #     dfResults.loc[sel]["varN"],
    #     dfResults.loc[sel]["varX"],
    #     dfResults.loc[sel]["gridPrices"],
    #     outputFolder,
    #     ini,
    #     plotList,
    #     caseID,
    # )
    #
    # plotting(
    #     dfResults["varN"].values[c],
    #     dfResults["varX"].values[c],
    #     dfResults["gridPrices"].values[c],
    #     outputFolder,
    #     ini,
    #     plotList,
    #     caseID,
    # )
    # plotting_additive_all_powers_sym(resultsDf, outputFolder, time, tick, "bar", plotList[5])
    # work in Progress: plot_parameter_variation()


if __name__ == "__main__":
    main(sys.argv)
