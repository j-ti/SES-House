#!/usr/bin/env python3.7

from datetime import datetime, timedelta
from util import constructTimeStamps

from load_loads import getLoadsData
from RenewNinja import getSamplePv, getSampleWind

import gurobipy as gp
from gurobipy import GRB


# Initialization
# Battery init (to be moved to a initialization file)
SOCmin = 0.1
SOCinit = 0.5
SOCmax = 0.9
E_bat_max = 10  # 1 kWh
eta = 0.9  # charging efficientcy 90%
P_bat_max = 3

pvGenerators = ["pv1", "pv2"]
dieselGenerators = ["dg1"]
windGenerators = ["wind1"]
generators = ["grid"] + pvGenerators + dieselGenerators + windGenerators

electricVehicles = ["ev1"]
nonEVBatteries = ["bat1"]
batteries = electricVehicles + nonEVBatteries
shiftableLoads = ["shift1", "shift2"]
interrupableLoads = ["interrupt1"]
loads = ["uncontrollable"] + shiftableLoads + interrupableLoads

start = datetime(2014, 1, 1, 0, 0, 0)
end = datetime(2014, 1, 1, 23, 59, 59)
stepsize = timedelta(hours=1)
times = constructTimeStamps(start, end, stepsize)
batPowersTest = {"bat1": [1, -2, 1]}

m = gp.Model("simple-model")

pvVars = m.addVars(
    len(times), len(pvGenerators), lb=0.0, vtype=GRB.CONTINUOUS, name="pvPowers"
)
fixedLoadVars = m.addVars(
    len(times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="fixedLoads"
)
windVars = m.addVars(
    len(times), len(windGenerators), lb=0.0, vtype=GRB.CONTINUOUS, name="windPowers"
)
gridVars = m.addVars(len(times), 1, lb=0.0, vtype=GRB.CONTINUOUS, name="gridPowers")
dieselGeneratorsVars = m.addVars(
    len(times),
    len(dieselGenerators),
    lb=0.0,
    vtype=GRB.CONTINUOUS,
    name="dieselGenerators",
)
batteryPowerVars = m.addVars(
    len(times), 1, lb=-P_bat_max, ub=P_bat_max, vtype=GRB.CONTINUOUS, name="batPowers"
)
batteryEnergyVars = m.addVars(
    len(times),
    1,
    lb=SOCmin * E_bat_max,
    ub=SOCmax * E_bat_max,
    vtype=GRB.CONTINUOUS,
    name="batEnergys",
)

m.setObjective(gp.quicksum(dieselGeneratorsVars) + gp.quicksum(gridVars), GRB.MINIMIZE)

# A battery charges when the 'batteryPowerVars' value is negative and discharging otherwise
m.addConstrs(
    (
        batteryEnergyVars[i + 1, 0]
        == batteryEnergyVars[i, 0]
        - eta * batteryPowerVars[i, 0] * 1  # stepsize: 1 hour
        for i in range(len(times) - 1)
    ),
    "battery charging",
)

m.addConstr((batteryEnergyVars[0, 0] == SOCinit * E_bat_max), "battery init")

m.addConstrs(
    (
        gridVars.sum([i, "*"])
        + pvVars.sum([i, "*"])
        + windVars.sum([i, "*"])
        + dieselGeneratorsVars.sum([i, "*"])
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
