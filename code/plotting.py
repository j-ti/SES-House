import matplotlib.pyplot as plt


def plotting(varName, varVal, soc_min_val, outputFolder):
    dico = {
        "pvPowers": [],
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

    plt.plot(dico["batPowers"], label="Battery Power")
    plt.plot(dico["evPowers"], label="EV Power")
    plt.plot(dico["batEnergys"], label="Battery Energy")
    plt.plot(dico["evEnergys"], label="EV Energy")
    plt.plot(
        [0, max([len(i) for i in dico.values()])],
        [soc_min_val, soc_min_val],
        color="r",
        linestyle="--",
        linewidth=1,
        label="Minimal energy",
    )
    plt.legend(loc="upper right", prop={"size": 8})
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)\nEnergy (kWh)")
    plt.savefig(outputFolder + "/bat_ev.png")
    plt.show()

    plt.plot(dico["gridPowers"], label="gridPowers")
    plt.plot(dico["dieselGenerators"], label="dieselGenerators")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper right")
    plt.savefig(outputFolder + "/grid_diesel.png")
    plt.show()
