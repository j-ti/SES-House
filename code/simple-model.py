#!/usr/bin/env python3.7

from datetime import datetime, timedelta
from util import constructTimeStamps

from RenewNinja import getSamplePv, getSampleWind

import gurobipy as gp
from gurobipy import GRB

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

m.setObjective(gp.quicksum(dieselGeneratorsVars) + gp.quicksum(gridVars), GRB.MINIMIZE)

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
    "1st pv panel generation",
)

m.addConstrs(
    (pvVars[i, 1] == 0 for i in range(len(times))), "2nd pv panel is turned off"
)

m.optimize()

for v in m.getVars():
    print("%s %g" % (v.varName, v.x))

print("Obj: %g" % m.objVal)
