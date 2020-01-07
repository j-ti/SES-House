import matplotlib.pyplot as plt
import numpy as np


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


def plotting_powers(dico, outputFolder):
    # Plotting PV power, wind power and fixed loads power.
    plt.plot(dico["PVPowers"], label="pv", color="orange")
    plt.plot(dico["windPowers"], label="wind", color="blue")
    plt.plot(dico["fixedLoads"], label="Loads Power", color="red")
    plt.xlabel("Hour")
    plt.ylabel("Output power - kW")
    plt.legend(loc="upper left")
    plt.savefig(outputFolder + "/pv_wind-power.png")
    plt.show()


def plotting_energys(dico, outputFolder):
    # Plotting EV and batteries energies
    plt.plot(dico["batEnergys"], label="Battery Energy", color="green")
    plt.plot(dico["evEnergys"], label="EV Energy", color="lime")
    plt.legend(loc="upper right", prop={"size": 8})
    plt.xlabel("Hour")
    plt.ylabel("Energy (kWh)")
    plt.savefig(outputFolder + "/bat_ev-energy.png")
    plt.show()


def plotting_all_powers(dico, outputFolder):
    # Plotting all the powers from our system inside one graph
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


def plotting_in_out_price(dico, outputFolder, gridPrices):
    # Plotting the evolution of the power in and out on the grid and the evolution of prices
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


def plotting_pie_gen_pow(dico, outputFolder):
    # Pie plot with the repartion of the power generated inside our system
    plot1 = [np.sum(dico["PVPowers"]), np.sum(dico["windPowers"]), np.sum(dico["fromGridPowers"]),
             np.sum(dico["dieselGenerators"])]
    plt.pie(plot1, labels=["PVPowers", "windPowers", "fromGridPowers", "dieselGenerators"])
    plt.title("Power given by the grid, pv, wind and diesel")
    plt.savefig(outputFolder + "/power_output_gen.png")
    plt.show()


def plotting_bar_in_out(dico, outputFolder):
    # Bar plot
    plt.bar(0, np.sum(dico["fromGridPowers"]))
    plt.bar(1, np.sum(dico["toGridPowers"]))
    plt.xticks([0, 1], ["From Grid", "To Grid"])
    plt.ylabel("Power (kW)")
    plt.title("Proportion grid in and out")
    plt.savefig(outputFolder + "/grid_in-out.png")
    plt.show()


def plotting_bar_all_powers(dico, outputFolder):
    # bar chart of the production vs the loads
    p1 = plt.bar(0, np.sum(dico["PVPowers"]), 0.2)
    p2 = plt.bar(0, np.sum(dico["windPowers"]), 0.2, bottom=np.sum(dico["PVPowers"]))
    p3 = plt.bar(0, np.sum(dico["fromGridPowers"]), 0.2, bottom=np.sum(dico["windPowers"]) + np.sum(dico["PVPowers"]))
    p4 = plt.bar(0, np.sum(dico["dieselGenerators"]), 0.1,
                 bottom=np.sum(dico["windPowers"]) + np.sum(dico["PVPowers"]) + np.sum(dico["fromGridPowers"]))
    p5 = plt.bar(0.7, np.sum(dico["fixedLoads"]), 0.2)
    p6 = plt.bar(0.7, np.sum(dico["batPowers"]), 0.2, bottom=np.sum(dico["fixedLoads"]))
    p7 = plt.bar(0.7, np.sum(dico["evPowers"]), 0.2, bottom=np.sum(dico["fixedLoads"]) + np.sum(dico["batPowers"]))

    p8 = plt.bar(0.05, np.sum(dico["toGridPowers"]), 0.1,
                 np.sum(dico["windPowers"]) + np.sum(dico["PVPowers"]) + np.sum(dico["fromGridPowers"]) - np.sum(
                     dico["toGridPowers"]))

    plt.xticks([0, 0.7], ("Generator", "Loads"))
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0]),
               ('PV power', 'wind Power', 'from grid Power', 'diesel Power', 'fixed Loads', 'battery Power', 'EV power',
                'toGridPowers'))
    plt.show()
