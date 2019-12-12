#!/usr/bin/env python3.7

import configparser
from datetime import datetime
import sys

from util import constructTimeStamps

from load_loads import getLoadsData
from load_price import getPriceData

from RenewNinja import getSamplePv, getSampleWind, getSamplePvApi, getSampleWindApi

import gurobipy as gp
from gurobipy import GRB


class Configure:
    def __init__(self, config):
        # Global
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
        self.start = datetime.strptime(config["TIME"]["start"], "20%y-%m-%d %H:%M:%S")
        self.end = datetime.strptime(config["TIME"]["end"], "20%y-%m-%d %H:%M:%S")
        self.stepsize = datetime.strptime(
            config["TIME"]["stepsize"], "%H:%M:%S"
        ) - datetime.strptime("00:00:00", "%H:%M:%S")
        self.times = constructTimeStamps(self.start, self.end, self.stepsize)

        # Generators
        self.P_dg_max = float(config["DIESEL"]["P_dg_max"])
        self.CostDiesel = float(config["DIESEL"]["CostDiesel"])
        self.pvFile = config["PV"]["file"]
        self.windFile = config["WIND"]["file"]
        self.loadsFile = config["LOADS"]["file"]
        self.costFileGrid = config["COST"]["file_grid"]


def runSimpleModel(ini):
    model = gp.Model("simple-model")

    pvVars = setUpPV(model, ini)
    windVars = setUpWind(model, ini)
    batteryPowerVars = setUpBattery(model, ini)
    evPowerVars = setUpEv(model, ini)
    fixedLoadVars = setUpFixedLoads(model, ini)

    gridVars = model.addVars(
        len(ini.times),
        1,
        obj=getPriceData(ini.costFileGrid, ini.start, ini.end, stepsize=ini.stepsize),
        vtype=GRB.CONTINUOUS,
        name="gridPowers",
    )
    dieselGeneratorsVars = model.addVars(
        len(ini.times),
        1,
        lb=0.0,
        ub=ini.P_dg_max,
        vtype=GRB.CONTINUOUS,
        name="dieselGenerators",
    )

    model.addConstrs(
        (
            gridVars.sum([i, "*"])
            + pvVars.sum([i, "*"])
            + windVars.sum([i, "*"])
            + dieselGeneratorsVars.sum([i, "*"])
            + batteryPowerVars.sum([i, "*"])
            + evPowerVars.sum([i, "*"])
            == fixedLoadVars.sum([i, "*"])
            for i in range(len(ini.times))
        ),
        "power balance",
    )

    model.setObjective(
        ini.CostDiesel * gp.quicksum(dieselGeneratorsVars) + gp.quicksum(gridVars),
        GRB.MINIMIZE,
    )

    model.optimize()

    printResults(model)


def setUpPV(model, ini):
    pvVars = model.addVars(
        len(ini.times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="pvPowers"
    )

    if ini.loc_flag:
        print("PV data: use location")
        metadata, pvPowerValues = getSamplePvApi(
            ini.loc_lat, ini.loc_lon, ini.start, ini.end, ini.stepsize
        )
        pvPowerValues = pvPowerValues.values
    else:
        print("PV data: use sample files")
        pvPowerValues = getSamplePv(ini.pvFile, ini.start, ini.end, ini.stepsize)
    assert len(pvPowerValues) == len(ini.times)
    model.addConstrs(
        (pvVars[i, 0] == pvPowerValues[i] for i in range(len(ini.times))),
        "1st pv panel generation",
    )

    return pvVars


def setUpFixedLoads(model, ini):
    fixedLoadVars = model.addVars(
        len(ini.times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="fixedLoads"
    )

    loadValues = getLoadsData(ini.loadsFile, ini.start, ini.end, ini.stepsize)
    assert len(loadValues) == len(ini.times)
    model.addConstrs(
        (fixedLoadVars[i, 0] == loadValues[i] for i in range(len(ini.times))),
        "power of fixed loads",
    )

    return fixedLoadVars


def setUpWind(model, ini):
    windVars = model.addVars(
        len(ini.times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="windPowers"
    )

    if ini.loc_flag:
        print("Wind data: use location")
        metadata, windPowerValues = getSampleWindApi(
            ini.loc_lat, ini.loc_lon, ini.start, ini.end, ini.stepsize
        )
        windPowerValues = windPowerValues.values
    else:
        print("Wind data: use sample files")
        windPowerValues = getSampleWind(
            ini.windFile, ini.start, ini.end, ini.stepsize
        )
    assert len(windPowerValues) == len(ini.times)
    model.addConstrs(
        (windVars[i, 0] == windPowerValues[i] for i in range(len(ini.times))),
        "1st wind panel generation",
    )

    return windVars


def setUpBattery(model, ini):
    batteryPowerVars = model.addVars(
        len(ini.times),
        1,
        lb=-ini.P_bat_max,
        ub=ini.P_bat_max,
        vtype=GRB.CONTINUOUS,
        name="batPowers",
    )
    batteryEnergyVars = model.addVars(
        len(ini.times),
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
            * ini.stepsize.total_seconds()
            / 3600  # stepsize: 1 hour
            for i in range(len(ini.times) - 1)
        ),
        "battery charging",
    )

    model.addConstr(
        (batteryEnergyVars[0, 0] == ini.SOC_bat_init * ini.E_bat_max), "battery init"
    )

    return batteryPowerVars


def setUpEv(model, ini):
    evPowerVars = model.addVars(
        len(ini.times),
        1,
        lb=-ini.P_ev_max,
        ub=ini.P_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evPowers",
    )
    evEnergyVars = model.addVars(
        len(ini.times),
        1,
        lb=ini.SOC_ev_min * ini.E_ev_max,
        ub=ini.SOC_ev_max * ini.E_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evEnergys",
    )

    model.addConstrs(
        (
            evEnergyVars[i + 1, 0]
            == evEnergyVars[i, 0]
            - ini.eta_ev
            * evPowerVars[i, 0]
            * ini.stepsize.total_seconds()
            / 3600  # stepsize: 1 hour
            for i in range(len(ini.times) - 1)
        ),
        "ev charging",
    )

    model.addConstrs(
        (
            evEnergyVars[i, 0] >= 0.7 * ini.E_ev_max
            for i in range(
                int((ini.t_goal_ev - ini.start).total_seconds() / 3600),
                int((ini.t_b_ev - ini.start).total_seconds() / 3600) + 1,
            )
        ),
        "ev init",
    )

    return evPowerVars


def printResults(model):
    for v in model.getVars():
        print("%s %g" % (v.varName, v.x))

    print("Obj: %g" % model.objVal)


def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[1])
    runSimpleModel(Configure(config))


if __name__ == "__main__":
    main(sys.argv)
