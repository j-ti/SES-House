import matplotlib.pyplot as plt
import numpy as np


def plotting(varName, varVal, soc_min_val, gridPrices, outputFolder, pieChart=True):
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
    if pieChart:
        plotting_pieChart(dico, outputFolder, gridPrices)
    else:
        plotting_1day(dico, outputFolder, gridPrices)


def plotting_1day(dico, outputFolder, gridPrices):
    plt.plot(dico["PVPowers"], label="pv", color="orange")
    plt.plot(dico["windPowers"], label="wind", color="blue")
    plt.plot(dico["fixedLoads"], label="Loads Power", color="red")
    plt.xlabel("Hour")
    plt.ylabel("Output power - kW")
    plt.legend(loc="upper left")
    plt.savefig(outputFolder + "/pv_wind-power.png")
    plt.show()

    plt.plot(dico["batEnergys"], label="Battery Energy", color="green")
    plt.plot(dico["evEnergys"], label="EV Energy", color="lime")
    plt.legend(loc="upper right", prop={"size": 8})
    plt.xlabel("Hour")
    plt.ylabel("Energy (kWh)")
    plt.savefig(outputFolder + "/bat_ev-energy.png")
    plt.show()

    # plt.plot([sum(x) for x in zip(dico["evPowers"],dico["batPowers"])], label="Battery + EV Power")
    plt.plot(dico["batPowers"], label="Battery Power", color="green")
    plt.plot(dico["evPowers"], label="EV Power", color="lime")
    plt.plot(dico["windPowers"], label="Wind Power", color="blue")
    plt.plot(dico["PVPowers"], label="PV Power", color="orange")
    plt.plot(dico["fixedLoads"], label="Loads Power", color="red")
    plt.plot(dico["fromGridPowers"], label="Grid Power In", color="black")
    plt.plot(dico["toGridPowers"], label="Grid Power Out", color="cyan")
    plt.plot(dico["dieselGenerators"], label="Diesel Power", color="silver")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper right")
    plt.savefig(outputFolder + "/power-balance.png")
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(gridPrices)), gridPrices, label="Grid Price", color="gold")
    ax2.plot(dico["fromGridPowers"], label="Grid Power In", color="black")
    ax2.plot(dico["toGridPowers"], label="Grid Power Out", color="cyan")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price - $ / kWh")
    ax2.set_ylabel("Power (kW)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.savefig(outputFolder + "price - 1day.png")
    plt.show()


# PVPowers": [],
#         "windPowers": [],
#         "batPowers": [],
#         "batEnergys": [],
#         "evPowers": [],
#         "evEnergys": [],
#         "fixedLoads": [],
#         "gridPowers": [],
#         "dieselGenerators": [],

def plotting_pieChart(dico, outputFolder, gridPrices):
    plot1 = [np.sum(dico["PVPowers"]), np.sum(dico["windPowers"]), np.sum(dico["gridPowers"]),
             np.sum(dico["dieselGenerators"])]
    plt.pie(plot1, labels=["PVPowers", "windPowers", "gridPowers", "dieselGenerators"])
    plt.title("Power given by the grid, pv, wind and diesel")
    plt.savefig(outputFolder + "/power_output_gen.png")
    plt.show()

    plot2 = [np.sum(dico["batEnergys"]), np.sum(dico["evEnergys"])]
    plt.pie(plot2, labels=["batEnergys", "evEnergys"])
    plt.title("Energy inside the batteries")
    plt.savefig(outputFolder + "/energy_batteries.png")
    plt.show()

    p1 = plt.bar(0, np.sum(dico["PVPowers"]), 0.2)
    p2 = plt.bar(0, np.sum(dico["windPowers"]), 0.2, bottom=np.sum(dico["PVPowers"]))
    p3 = plt.bar(0, np.sum(dico["gridPowers"]), 0.2, bottom=np.sum(dico["windPowers"]) + np.sum(dico["PVPowers"]))
    p4 = plt.bar(0, np.sum(dico["dieselGenerators"]), 0.2,
                 bottom=np.sum(dico["windPowers"]) + np.sum(dico["PVPowers"]) + np.sum(dico["gridPowers"]))
    p5 = plt.bar(0.7, np.sum(dico["fixedLoads"]), 0.2)
    p6 = plt.bar(0.7, np.sum(dico["batPowers"]), 0.2, bottom=np.sum(dico["fixedLoads"]))
    p7 = plt.bar(0.7, np.sum(dico["evPowers"]), 0.2, bottom=np.sum(dico["fixedLoads"]) + np.sum(dico["batPowers"]))

    plt.xticks([0, 0.7], ("Generator", "Loads"))
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]),
               ('PV power', 'wind Power', 'grid Power', 'diesel Power', 'fixed Loads', 'battery Power', 'EV power'))
    plt.show()
