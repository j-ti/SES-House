#!/usr/bin/env python3.7

import gurobipy as gp
from gurobipy import GRB

## Initialization
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

pvPowers = {"pv1": [1, 3, 0], "pv2": [0, 0, 2]}
windPowers = {"wind1": [1, 3, 0], "wind2": [0, 0, 2]}
batPowersTest = {"bat1": [1, -2, 1]}
times = [1575375592443, 1575375592449, 1575375593949]
step = 1 # one time step is one hour

assert len(pvPowers) == len(pvGenerators)
for value in pvPowers.values():
    assert len(value) == len(times)

# Create a new model
m = gp.Model("simple")

pvVars = m.addVars(len(times), 1, vtype=GRB.CONTINUOUS, name="pvPowers")
fixedLoadVars = m.addVars(len(times), 1, vtype=GRB.CONTINUOUS, name="fixedLoads")
windVars = m.addVars(len(times), 1, vtype=GRB.CONTINUOUS, name="windPowers")
gridVars = m.addVars(len(times), 1, vtype=GRB.CONTINUOUS, name="gridPowers")
dieselGeneratorsVars = m.addVars(
    len(times), 1, vtype=GRB.CONTINUOUS, name="dieselGenerators"
)
batteryPowerVars = m.addVars(
    len(times), 1, lb=-P_bat_max, ub=P_bat_max, vtype=GRB.CONTINUOUS, name="batPowers"
)
batteryEnergyVars = m.addVars(
    len(times), 1, lb=SOCmin*E_bat_max, ub=SOCmax*E_bat_max, vtype=GRB.CONTINUOUS, name="batEnergys"
)

m.setObjective(gp.quicksum(dieselGeneratorsVars) + gp.quicksum(gridVars), GRB.MINIMIZE)

# Manually specify the (dis-)charging power (has to be optimized by the price later on)
m.addConstrs(
    (batteryPowerVars[i, 0] == batPowersTest["bat1"][i] for i in range(len(times))),
    "battery power",
)

# A battery charges when the 'batteryPowerVars' value is negative and discharging otherwise
m.addConstrs(
    (
        batteryEnergyVars[i + 1, 0]
        == batteryEnergyVars[i, 0] - eta * batteryPowerVars[i, 0] * step
        for i in range(len(times) - 1)
    ),
    "battery charging",
)

m.addConstr(
    (batteryEnergyVars[0, 0] == SOCinit * E_bat_max), "battery init",
)

m.addConstrs(
    (gridVars[i, 0] >= 0 for i in range(len(times))), "grid positive",
)

m.addConstrs(
    (pvVars[i, 0] >= 0 for i in range(len(times))), "pv positive",
)

m.addConstrs(
    (windVars[i, 0] >= 0 for i in range(len(times))), "wind positive",
)

m.addConstrs(
    (fixedLoadVars[i, 0] >= 0 for i in range(len(times))), "fixed loads positive",
)

m.addConstrs(
    (dieselGeneratorsVars[i, 0] >= 0 for i in range(len(times))),
    "diesel generator positive",
)

m.addConstrs(
    (
        gridVars[i, 0]
        + pvVars[i, 0]
        + windVars[i, 0]
        + dieselGeneratorsVars[i, 0]
        + batteryPowerVars[i, 0]
        - fixedLoadVars[i, 0]
        == 0
        for i in range(len(times))
    ),
    "power balance",
)

m.optimize()

for v in m.getVars():
    print("%s %g" % (v.varName, v.x))
9
print("Obj: %g" % m.objVal)
