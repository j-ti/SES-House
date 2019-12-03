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

# Create variables
gridVars = m.addVars(len(times), 1, vtype=GRB.CONTINUOUS, name="gridPowers")
dieselGeneratorsVars = m.addVars(
    len(times), 1, vtype=GRB.CONTINUOUS, name="dieselGenerators"
)
z = m.addVar(vtype=GRB.BINARY, name="z")

# Set objective
m.setObjective(gp.quicksum(dieselGeneratorsVars), GRB.MINIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
m.addConstrs(
    (dieselGeneratorsVars[i, 0] >= 0 for i in range(len(times))),
    "diesel generator positive",
)

# Optimize model
m.optimize()

for v in m.getVars():
    print("%s %g" % (v.varName, v.x))

print("Obj: %g" % m.objVal)
