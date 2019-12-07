#!/usr/bin/env python3.7

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

pvPowers = {"pv1": [1, 3, 0], "pv2": [0, 0, 2]}
windPowers = {"wind1": [1, 3, 0], "wind2": [0, 0, 2]}
times = [1575375592443, 1575375592449, 1575375593949]

assert len(pvPowers) == len(pvGenerators)
for value in pvPowers.values():
    assert len(value) == len(times)


# Create a new model
m = gp.Model("simple")

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

m.optimize()

for v in m.getVars():
    print("%s %g" % (v.varName, v.x))

print("Obj: %g" % m.objVal)