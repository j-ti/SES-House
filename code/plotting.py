import matplotlib.pyplot as plt
import numpy as np

colorDico = {
    "PVPowers": "orange",
    "windPowers": "blue",
    "batPowers": "green",
    "batEnergys": "green",
    "evPowers": "lime",
    "evEnergys": "lime",
    "batPowersNeg": "springgreen",
    "evPowersNeg": "lawngreen",
    "fixedLoads": "red",
    "fromGridPowers": "black",
    "toGridPowers": "cyan",
    "dieselGenerators": "silver",
    "gridPrice": "gold"
}


def plotting(varName, varVal, gridPrices, outputFolder):
    dico = {
        "PVPowers": [],
        "windPowers": [],
        "batPowers": [],
        "batEnergys": [],
        "evPowers": [],
        "evEnergys": [],
        "fixedLoads": [],
        "fromGridPowers": [],
        "toGridPowers": [],
        "dieselGenerators": [],
    }

    for i in range(len(varName)):
        for val in dico.keys():
            if val in varName[i]:
                dico[val].append(varVal[i])
                break

    plotting_powers(dico, outputFolder)
    plotting_energys(dico, outputFolder)
    plotting_all_powers(dico, outputFolder)
    plotting_in_out_price(dico, outputFolder, gridPrices)
    plotting_pie_gen_pow(dico, outputFolder)
    plotting_bar_in_out(dico, outputFolder)
    plotting_bar_all_powers(dico, outputFolder)


# Plotting PV power, wind power and fixed loads power.
def plotting_powers(dico, outputFolder):
    plt.plot(dico["PVPowers"], label="pv", color=colorDico["PVPowers"])
    plt.plot(dico["windPowers"], label="wind", color=colorDico["windPowers"])
    plt.plot(dico["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"])
    plt.xlabel("Hour")
    plt.ylabel("Output power - kW")
    plt.legend(loc="upper left")
    plt.savefig(outputFolder + "/pv_wind-power.png")
    plt.show()


# Plotting EV and batteries energies
def plotting_energys(dico, outputFolder):
    plt.plot(dico["batEnergys"], label="Battery Energy", color=colorDico["batEnergys"])
    plt.plot(dico["evEnergys"], label="EV Energy", color=colorDico["evEnergys"])
    plt.legend(loc="upper right", prop={"size": 8})
    plt.xlabel("Hour")
    plt.ylabel("Energy (kWh)")
    plt.savefig(outputFolder + "/bat_ev-energy.png")
    plt.show()


# Plotting all the powers from our system inside one graph
def plotting_all_powers(dico, outputFolder):
    plt.plot(dico["batPowers"], label="Battery Power", color=colorDico["batPowers"])
    plt.plot(dico["evPowers"], label="EV Power", color=colorDico["evPowers"])
    plt.plot(dico["windPowers"], label="Wind Power", color=colorDico["windPowers"])
    plt.plot(dico["PVPowers"], label="PV Power", color=colorDico["PVPowers"])
    plt.plot(dico["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"])
    plt.plot(dico["fromGridPowers"], label="Grid Power In", color=colorDico["fromGridPowers"])
    plt.plot(dico["toGridPowers"], label="Grid Power Out", color=colorDico["toGridPowers"])
    plt.plot(dico["dieselGenerators"], label="Diesel Power", color=colorDico["dieselGenerators"])
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper right")
    plt.savefig(outputFolder + "/power-balance.png")
    plt.show()


# Plotting the evolution of the power in and out on the grid and the evolution of prices
def plotting_in_out_price(dico, outputFolder, gridPrices):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(gridPrices)), gridPrices, label="Grid Price", color=colorDico["gridPrice"])
    ax2.plot(dico["fromGridPowers"], label="Grid Power In", color=colorDico["fromGridPowers"])
    ax2.plot(dico["toGridPowers"], label="Grid Power Out", color=colorDico["toGridPowers"])
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price - $ / kWh")
    ax2.set_ylabel("Power (kW)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.savefig(outputFolder + "grid_in-out_price.png")
    plt.show()


# Pie plot with the repartion of the power generated inside our system
def plotting_pie_gen_pow(dico, outputFolder):
    plot1 = [np.sum(dico["PVPowers"]), np.sum(dico["windPowers"]), np.sum(dico["fromGridPowers"]),
             np.sum(dico["dieselGenerators"])]
    plt.pie(plot1, labels=["PVPowers", "windPowers", "fromGridPowers", "dieselGenerators"]
            , colors=[colorDico["PVPowers"], colorDico["windPowers"], colorDico["fromGridPowers"],
                      colorDico["dieselGenerators"]])
    plt.title("Repartion of power given by the grid, pv, wind and diesel")
    plt.savefig(outputFolder + "/power_generated_pie.png")
    plt.show()


# Bar plot power in out by the grid
def plotting_bar_in_out(dico, outputFolder):
    plt.bar(0, np.sum(dico["fromGridPowers"]), color=colorDico["fromGridPowers"])
    plt.bar(1, np.sum(dico["toGridPowers"]), color=colorDico["toGridPowers"])
    plt.xticks([0, 1], ["From Grid", "To Grid"])
    plt.ylabel("Power (kW)")
    plt.title("Power from the grid in and out")
    plt.savefig(outputFolder + "/grid_in-out_bar.png")
    plt.show()


# Bar plot of all the power
# small bar is "kind of generator" or "kind of load" (for battery / EV discharging // charging and selling
# the diff is due to the EV leaving (energy drop so power discharge is high but not in the system)
def plotting_bar_all_powers(dico, outputFolder):
    batPosPow = abs(np.sum([foo for foo in dico["batPowers"] if foo >= 0]))
    batNegPow = abs(np.sum([foo for foo in dico["batPowers"] if foo < 0]))
    evPosPow = abs(np.sum([foo for foo in dico["evPowers"] if foo >= 0]))
    evNegPow = abs(np.sum([foo for foo in dico["evPowers"] if foo < 0]))
    pvPow = np.sum(dico["PVPowers"])
    windPow = np.sum(dico["windPowers"])
    fromGridPow = np.sum(dico["fromGridPowers"])
    toGridPow = np.sum(dico["toGridPowers"])
    dieselPow = np.sum(dico["dieselGenerators"])
    fixLoadPow = np.sum(dico["fixedLoads"])

    p1 = plt.bar(0, pvPow, 0.2, color=colorDico["PVPowers"])
    p2 = plt.bar(0, windPow, 0.2, bottom=pvPow, color=colorDico["windPowers"])
    p3 = plt.bar(0, fromGridPow, 0.2, bottom=pvPow + windPow, color=colorDico["fromGridPowers"])
    p4 = plt.bar(0, dieselPow, 0.1, bottom=pvPow + windPow + fromGridPow, color=colorDico["dieselGenerators"])
    p5 = plt.bar(0.05, batNegPow, 0.1, bottom=pvPow + windPow + fromGridPow + dieselPow,
                 color=colorDico["batPowersNeg"])
    p6 = plt.bar(0.05, evNegPow, 0.1, bottom=pvPow + windPow + fromGridPow + dieselPow + batNegPow,
                 color=colorDico["evPowersNeg"])

    p7 = plt.bar(0.7, fixLoadPow, 0.2, color=colorDico["fixedLoads"])
    p8 = plt.bar(0.7, batPosPow, 0.2, bottom=fixLoadPow, color=colorDico["batPowers"])
    p9 = plt.bar(0.7, evPosPow, 0.2, bottom=fixLoadPow + batPosPow, color=colorDico["evPowers"])
    p10 = plt.bar(0.75, toGridPow, 0.1, bottom=fixLoadPow + batPosPow + evPosPow, color=colorDico["toGridPowers"])

    plt.xticks([0, 0.7], ("Generator", "Loads"))
    plt.ylabel("Power (kW)")
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0]),
               ('PV', 'Wind', 'Buy from the Grid', 'Diesel', 'Battery discharging', 'EV discharging',
                'Fixed Loads', 'Battery charging', 'EV charging', 'Sell to the Grid'))
    plt.title("Power generated & Power needed")
    plt.savefig(outputFolder + "all-pow_bar.png")
    plt.show()
