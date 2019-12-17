import matplotlib.pyplot as plt


def plotting(varName, varVal, soc_min_val, gridPrices, outputFolder):
    dico = {
        "PVPowers": [],
        "windPowers": [],
        "batPowers": [],
        "batEnergys": [],
        "evPowers": [],
        "evEnergys": [],
        "fixedLoads": [],
        "gridPowers": [],
        "dieselGenerators": [],
    }

    for i in range(len(varName)):
        for val in dico.keys():
            if val in varName[i]:
                dico[val].append(varVal[i])
                break

    # for val in dico.keys():
    #     plt.plot(dico[val], label=val)
    # plt.legend()
    # plt.show()

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
    plt.plot(dico["gridPowers"], label="Grid Power", color="black")
    plt.plot(dico["dieselGenerators"], label="Diesel Power", color="silver")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper right")
    plt.savefig(outputFolder + "/power-balance.png")
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(gridPrices)), gridPrices, label="Grid Price", color="gold")
    ax2.plot(dico["gridPowers"], label="Grid Power", color="black")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price - $ / kWh")
    ax2.set_ylabel("Power (kW)")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.savefig(outputFolder + "price - 1day.png")
    plt.show()
