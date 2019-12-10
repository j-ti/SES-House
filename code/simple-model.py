#!/usr/bin/env python3.7

import configparser
from datetime import datetime, timedelta
import sys

from util import constructTimeStamps

from load_loads import getLoadsData
from load_price import getPriceData

from RenewNinja import getSamplePv, getSampleWind

import gurobipy as gp
from gurobipy import GRB


def runSimpleModel(config):
    # Initialization
    # Battery init (to be moved to a initialization file)
    SOC_bat_min = float(config['BAT']['SOC_bat_min'])
    SOC_bat_init = float(config['BAT']['SOC_bat_init'])
    SOC_bat_max = float(config['BAT']['SOC_bat_max'])
    E_bat_max = float(config['BAT']['E_bat_max'])
    eta_bat = float(config['BAT']['eta_bat'])
    P_bat_max = float(config['BAT']['P_bat_max'])

    # EV init
    SOC_ev_min = float(config['EV']['SOC_ev_min'])
    SOC_ev_init = float(config['EV']['SOC_ev_init'])
    SOC_ev_max = float(config['EV']['SOC_ev_max'])
    P_ev_max = float(config['EV']['P_ev_max'])
    E_ev_max = float(config['EV']['E_ev_max'])
    eta_ev = float(config['EV']['eta_ev'])

    t_a_ev = datetime.fromisoformat(config['EV']['t_a_ev'])
    t_b_ev = datetime.fromisoformat(config['EV']['t_b_ev'])
    t_goal_ev = datetime.fromisoformat(config['EV']['t_goal_ev'])


    start = datetime.fromisoformat(config['TIME']['start'])
    end = datetime.fromisoformat(config['TIME']['end'])
    stepsize = timedelta(hours=1)

    times = constructTimeStamps(start, end, stepsize)
    batPowersTest = {"bat1": [1, -2, 1]}

    CostDiesel = 0.2
    CostGrid = getPriceData(
        "./sample/pecan-iso_neiso-day_ahead_lmp_avg-20190602.csv",
        start,
        end,
        stepsize=stepsize,
    )
    m = gp.Model("simple-model")

    pvVars = m.addVars(
        len(times), len(config['PV'].items()), lb=0.0, vtype=GRB.CONTINUOUS, name="pvPowers"
    )
    fixedLoadVars = m.addVars(
        len(times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="fixedLoads"
    )
    windVars = m.addVars(
        len(times), len(config['WIND'].items()), lb=0.0, vtype=GRB.CONTINUOUS, name="windPowers"
    )
    gridVars = m.addVars(len(times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="gridPowers")
    dieselGeneratorsVars = m.addVars(
        len(times),
        1,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="dieselGenerators",
    )
    batteryPowerVars = m.addVars(
        len(times), 1, lb=-P_bat_max, ub=P_bat_max, vtype=GRB.CONTINUOUS, name="batPowers"
    )
    evPowerVars = m.addVars(
        len(times), 1, lb=-P_ev_max, ub=P_ev_max, vtype=GRB.CONTINUOUS, name="evPowers"
    )

    batteryEnergyVars = m.addVars(
        len(times),
        1,
        lb=SOC_bat_min * E_bat_max,
        ub=SOC_bat_max * E_bat_max,
        vtype=GRB.CONTINUOUS,
        name="batEnergys",
    )
    evEnergyVars = m.addVars(
        len(times),
        1,
        lb=SOC_ev_min * E_ev_max,
        ub=SOC_ev_max * E_ev_max,
        vtype=GRB.CONTINUOUS,
        name="evEnergys",
    )


    m.setObjective(
        CostDiesel * gp.quicksum(dieselGeneratorsVars) + gp.quicksum(gridVars), GRB.MINIMIZE
    )

    # A battery charges when the 'batteryPowerVars' value is negative and discharging otherwise
    m.addConstrs(
        (
            batteryEnergyVars[i + 1, 0]
            == batteryEnergyVars[i, 0]
            - eta_bat
            * batteryPowerVars[i, 0]
            * stepsize.total_seconds()
            / 3600  # stepsize: 1 hour
            for i in range(len(times) - 1)
        ),
        "battery charging",
    )
    m.addConstrs(
        (
            evEnergyVars[i + 1, 0]
            == evEnergyVars[i, 0]
            - eta_ev
            * evPowerVars[i, 0]
            * stepsize.total_seconds()
            / 3600  # stepsize: 1 hour
            for i in range(len(times) - 1)
        ),
        "ev charging",
    )

    m.addConstr((batteryEnergyVars[0, 0] == SOC_bat_init * E_bat_max), "battery init")
    m.addConstrs(
        (
            evPowerVars[i, 0] == 0
            for i in range(int((t_a_ev - start).total_seconds() / 3600))
        ),
        "ev no charging before t_a",
    )
    m.addConstrs(
        (
            evPowerVars[i, 0] == 0
            for i in range(
                int((t_b_ev - start).total_seconds() / 3600),
                int((end - t_b_ev).total_seconds() / 3600),
            )
        ),
        "ev no charging before t_a",
    )

    # m.addConstr((evEnergyVars[(t_a_ev-start).total_seconds()/3600, 0] == SOC_ev_init * E_ev_max), "ev init")

    m.addConstrs(
        (
            gridVars.sum([i, "*"])
            + pvVars.sum([i, "*"])
            + windVars.sum([i, "*"])
            + dieselGeneratorsVars.sum([i, "*"])
            # +batteryEnergyVars.sum([i,"*"])
            # +evPowerVars.sum([i,"*"])
            == fixedLoadVars.sum([i, "*"])
            for i in range(len(times))
        ),
        "power balance",
    )

    # Generators with fixed values
    pvPowerValues = getSamplePv(start, end, stepsize)
    assert len(pvPowerValues) == len(times)
    m.addConstrs(
        (pvVars[i, 0] == pvPowerValues[i] for i in range(len(times))),
        "1st pv panel generation",
    )

    windPowerValues = getSampleWind(start, end, stepsize)
    assert len(windPowerValues) == len(times)
    m.addConstrs(
        (windVars[i, 0] == windPowerValues[i] for i in range(len(times))),
        "1st wind panel generation",
    )

    m.addConstrs(
        (pvVars[i, 1] == 0 for i in range(len(times))), "2nd pv panel is turned off"
    )

    loadValues = getLoadsData(
        "./sample/pecan-home-grid_solar-manipulated.csv", start, end, stepsize
    )
    assert len(loadValues) == len(times)
    m.addConstrs(
        (fixedLoadVars[i, 0] == loadValues[i] for i in range(len(times))),
        "power of fixed loads",
    )

    m.optimize()

    for v in m.getVars():
        print("%s %g" % (v.varName, v.x))
    9
    print("Obj: %g" % m.objVal)


def main(argv):
    config = configparser.ConfigParser()
    config.read(argv[1])
    runSimpleModel(config)


if __name__ == "__main__":
    main(sys.argv)
