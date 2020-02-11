import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colorDico = {
    "PVPowers": "darkorange",
    "windPowers": "darkblue",
    "batPowers": "forestgreen",
    "batEnergys": "forestgreen",
    "evPowers": "lightseagreen",
    "evEnergys": "lightseagreen",
    "batPowersNeg": "forestgreen",
    "evPowersNeg": "lightseagreen",
    "fixedLoads": "brown",
    "fromGridPowers": "k",
    "toGridPowers": "slateblue",
    "dieselGenerators": "dimgray",
    "gridPrice": "goldenrod",
    "errPv": "yellow",
    "errLoads": "pink",
    "errPvNeg": "yellow",
    "errLoadsNeg": "pink",
}

labelDico = {
    "PVPowers": "PV",
    "windPowers": "Wind",
    "batPowers": "Bat. Discharging",
    "evPowers": "EV Discharging",
    "batPowersNeg": "Bat. Charging",
    "evPowersNeg": "EV Charging",
    "fixedLoads": "Loads",
    "fromGridPowers": "Grid In",
    "toGridPowers": "Grid Out",
    "dieselGenerators": "Diesel",
    "gridPrice": "Grid Price",
    "errPv": "Pred. Err. PV",
    "errLoads": "Pred. Err. Loads",
    "errPvNeg": "Pred. Err. PV",
    "errLoadsNeg": "Pred. Err. Loads",
}


def plotting(varName, varVal, gridPrices, outputFolder, ini, plotList):
    dico = {
        "PVPowers": [],
        "windPowers": [],
        "batPowers": [],
        "evPowers": [],
        "fixedLoads": [],
        "fromGridPowers": [],
        "toGridPowers": [],
        "dieselGenerators": [],
        #"errLoads": [],
        #"errPv": [],
    }

    dicoEnergy = {"batEnergys": [], "evEnergys": []}

    step = int(len(ini.timestamps) / 5)
    time = [
        ini.timestamps[i].strftime("%H:%M") for i in range(len(ini.timestamps))
    ][::step]
    tick = [i for i in range(len(ini.timestamps))][::step]

    for i in range(len(varName)):
        for val in dico.keys():
            if val in varName[i]:
                dico[val].append(varVal[i])
                break
        for val in dicoEnergy.keys():
            if val in varName[i]:
                dicoEnergy[val].append(varVal[i])
                break

    resultsDf = pd.DataFrame.from_dict(dict(dico), orient="columns")

    plt.style.use("bmh")
    plotting_powers(dico, outputFolder, time, tick, plotList[0])
    plotting_energys(
        dicoEnergy,
        ini.E_bat_max,
        ini.SOC_bat_min,
        ini.SOC_bat_max,
        outputFolder,
        time,
        tick,
        plotList[1],
    )
    plotting_all_powers(dico, outputFolder, time, tick, plotList[2])
    plotting_additive_all_powers(resultsDf, outputFolder, time, tick, "bar", plotList[3])
    plotting_additive_all_powers(resultsDf, outputFolder, time, tick, "area", plotList[4])
    plotting_additive_all_powers_sym(resultsDf, outputFolder, time, tick, "bar", plotList[5])
    plotting_additive_all_powers_sym(resultsDf, outputFolder, time, tick, "area", plotList[6])
    plotting_in_out_price(dico, outputFolder, gridPrices, time, tick, plotList[7])
    plotting_pie_gen_pow(dico, outputFolder, plotList[8])
    plotting_bar_in_out(dico, outputFolder, plotList[9])
    plotting_bar_all_powers(dico, outputFolder, plotList[10])


# Plotting PV power, wind power and fixed loads power.
def plotting_powers(dico, outputFolder, time, tick, showFlag=False):

    plt.plot(dico["PVPowers"], label="pv", color=colorDico["PVPowers"])
    plt.plot(dico["windPowers"], label="wind", color=colorDico["windPowers"])
    plt.plot(dico["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"])
    # plt.plot(dico["errPv"], label="pv", color=colorDico["errPv"])
    # plt.plot(dico["errLoads"], label="Loads Power", color=colorDico["errLoads"])
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Output power - kW")
    plt.legend(loc="upper left", ncol=3)
    plt.savefig(outputFolder + "/pv_wind-power.png")
    if showFlag:
        plt.show()
    plt.close()


# Plotting EV and batteries energies
def plotting_energys(
    dico, E_bat_max, SOC_bat_min, SOC_bat_max, outputFolder, time, tick, showFlag=False
):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(dico["batEnergys"], label="Battery Energy", color=colorDico["batEnergys"])
    ax1.plot(
        [0, len(dico["batEnergys"])],
        [E_bat_max * SOC_bat_max, E_bat_max * SOC_bat_max],
        ls="--",
        c="darkgray",
    )
    ax1.plot(
        [0, len(dico["batEnergys"])],
        [E_bat_max * SOC_bat_min, E_bat_max * SOC_bat_min],
        ls="--",
        c="darkgray",
    )
    ax2.plot(dico["evEnergys"], label="EV Energy", color=colorDico["evEnergys"])
    ax1.legend(loc="upper left", ncol=1, prop={"size": 8})
    ax2.legend(loc="upper center", ncol=1, prop={"size": 8})
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_xticks(tick)
    ax1.set_xticklabels(time)
    ax1.set_ylabel("Energy (kWh)")
    ax2.set_yticks([E_bat_max * SOC_bat_max, E_bat_max * SOC_bat_min])
    ax2.set_yticklabels(
        ["SOC" + str(SOC_bat_max * 100) + "%", "SOC" + str(SOC_bat_min * 100) + "%"]
    )

    plt.xlabel("Time")
    plt.savefig(outputFolder + "/bat_ev-energy.png")
    if showFlag:
        plt.show()
    plt.close()


# Plotting all the powers from our system inside one graph
def plotting_all_powers(dico, outputFolder, time, tick, showFlag=False):
    plt.plot(dico["batPowers"], label="Battery Power", color=colorDico["batPowers"])
    plt.plot(dico["evPowers"], label="EV Power", color=colorDico["evPowers"])
    plt.plot(dico["windPowers"], label="Wind Power", color=colorDico["windPowers"])
    plt.plot(dico["PVPowers"], label="PV Power", color=colorDico["PVPowers"])
    plt.plot(dico["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"])
    plt.plot(
        dico["fromGridPowers"], label="Grid Power In", color=colorDico["fromGridPowers"]
    )
    plt.plot(
        dico["toGridPowers"], label="Grid Power Out", color=colorDico["toGridPowers"]
    )
    plt.plot(
        dico["dieselGenerators"],
        label="Diesel Power",
        color=colorDico["dieselGenerators"],
    )

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper left", ncol=2)
    plt.savefig(outputFolder + "/power-balance.png")
    if showFlag:
        plt.show()
    plt.close()


# Area plotting of all the powers from our system (in and out) inside one graph with consumption (loads) as baseline
def plotting_additive_all_powers(
    resultsPd, outputFolder, time, tick, kindPlot="area", showFlag=False
):
    if kindPlot is "bar":
        style = "steps-mid"
        step = "mid"
        kwargs = {"width": 1.0}
    else:
        style = "default"
        step = None
        kwargs = {}

    # Devide in and out flows (esp. for batteries) and make them all positive
    negResults, resultsPd = resultsPd.clip(upper=0) * (-1), resultsPd.clip(lower=0)
    negResults.columns = [str(col) + "Neg" for col in negResults.columns]
    resultsPd[["batPowersNeg", "evPowersNeg"]] = negResults[
        ["batPowersNeg", "evPowersNeg"]#, "errPvNeg", "errLoadsNeg"]
    ]
    # selection list of series to be plotted as area-plot and in which order
    selOut = ["fixedLoads", "batPowersNeg", "evPowersNeg", "toGridPowers"]#, "errPvNeg", "errLoadsNeg"]
    selArea = [
        "PVPowers",
        "windPowers",
        "batPowers",
        "evPowers",
        "dieselGenerators",
        "fromGridPowers",
        #"errPv",
        #"errLoads",
    ]
    # Colorscheme with selection lists
    inColors = list(map(colorDico.get, selArea))
    inColors = ["pink" if c is None else c for c in inColors]
    outColors = list(map(colorDico.get, selOut))
    outColors = ["pink" if c is None else c for c in outColors]
    hatch = ["", "//", "--", "..","",""]

    # Plottting
    fig, ax = plt.subplots()
    resultsPd[selArea].plot(
        kind=kindPlot, linewidth=0, stacked=True, ax=ax, color=inColors, **kwargs
    )

    additiveOut = resultsPd[selOut].copy()
    additiveOut = additiveOut.cumsum(axis=1)

    for i in range(1, len(selOut)):
        plt.fill_between(
            range(len(additiveOut)),
            additiveOut[selOut[i - 1]],
            additiveOut[selOut[i]],
            facecolor="none",  # outColors[i],
            label=selOut[i],
            step=step,
            hatch=hatch[i],
            edgecolor=outColors[i],
            linewidth=2.0,
            ls="--",
            # alpha=0.3,
            zorder=2,
        )

    plt.plot(
        resultsPd["fixedLoads"],
        drawstyle=style,
        label="fixedLoads",
        color=colorDico["fixedLoads"],
    )

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    handles, labels = ax.get_legend_handles_labels()
    labelsList = list(map(labelDico.get, labels))
    labelsList = ["pink" if l is None else l for l in labelsList]
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    ax.legend(handles, labelsList, bbox_to_anchor=(1.5, 0.8), loc="upper right")
    plt.savefig(outputFolder + "/power-balance-" + kindPlot + ".png")
    if showFlag:
        plt.show()
    plt.close()


# Area plotting of all the powers from our system (in and out) inside one graph with consumption (loads) as baseline
def plotting_additive_all_powers_sym(
    resultsPd, outputFolder, time, tick, kindPlot="area", showFlag=False
):
    if kindPlot is "bar":
        style = "steps-mid"
        step = "mid"
        kwargs = {"width": 1.0}
    else:
        style = "default"
        step = None
        kwargs = {}

    # Devide in and out flows (esp. for batteries)
    # Selection list for in/out series in plotting order
    selOut = ["fixedLoads", "batPowersNeg", "evPowersNeg", "toGridPowers"]#, "errPvNeg", "errLoadsNeg"]
    selIn = [
        "dieselGenerators",
        "PVPowers",
        "windPowers",
        "batPowers",
        "evPowers",
        "fromGridPowers",
        # "errPv",
        # "errLoads",
    ]
    #resultsPd["errLoads"] *= -1
    negResults, resultsPd = resultsPd.clip(upper=0), resultsPd.clip(lower=0)
    negResults.columns = [str(col) + "Neg" for col in negResults.columns]
    resultsPd[["batPowersNeg", "evPowersNeg"]] = negResults[
        ["batPowersNeg", "evPowersNeg"]#, "errPvNeg", "errLoadsNeg"]
    ]
    # make loads and toGrid values negative
    resultsPd[["fixedLoads", "toGridPowers"]] *= -1

    # Colorscheme with selection lists
    inColors = list(map(colorDico.get, selIn))
    inColors = ["pink" if c is None else c for c in inColors]
    outColors = list(map(colorDico.get, selOut))
    outColors = ["pink" if c is None else c for c in outColors]
    hatch = ["", "//", "--", ".."]

    # Plottting
    fig, ax = plt.subplots()
    resultsPd[selIn].plot(
        kind=kindPlot, linewidth=0, stacked=True, ax=ax, color=inColors, **kwargs
    )

    plt.plot(-resultsPd["fixedLoads"], drawstyle=style, color=colorDico["fixedLoads"])

    resultsPd[selOut].plot(
        kind=kindPlot,
        stacked=True,
        linewidth=0,
        ax=ax,
        ls="--",
        color=outColors,
        **kwargs
    )
    ax.set_ylim(
        [resultsPd[selOut].sum(axis=1).min(), resultsPd[selIn].sum(axis=1).max()]
    )

    # Settings
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    handles, labels = ax.get_legend_handles_labels()
    labelsList = list(map(labelDico.get, labels))
    labelsList = ["pink" if l is None else l for l in labelsList]
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    ax.legend(handles, labelsList, bbox_to_anchor=(1.5, 0.8), loc="upper right")
    plt.savefig(outputFolder + "/power-balance-symmetric-" + kindPlot + ".png")
    if showFlag:
        plt.show()
    plt.close()


# Plotting the evolution of the power in and out on the grid and the evolution of prices
def plotting_in_out_price(dico, outputFolder, gridPrices, time, tick, showFlag=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(
        range(len(gridPrices)),
        gridPrices,
        label="Grid Price",
        color=colorDico["gridPrice"],
    )
    ax2.plot(
        dico["fromGridPowers"], label="Grid Power In", color=colorDico["fromGridPowers"]
    )
    ax2.plot(
        dico["toGridPowers"], label="Grid Power Out", color=colorDico["toGridPowers"]
    )
    ax1.set_xticks(tick)
    ax1.set_xticklabels(time, rotation=20)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price - $ / kWh")
    ax2.set_ylabel("Power (kW)")
    ax1.legend(loc="upper center", ncol=1)
    ax2.legend(loc="upper left", ncol=1)
    plt.savefig(outputFolder + "grid_in-out_price.png")
    if showFlag:
        plt.show()
    plt.close()


# Pie plot with the repartion of the power generated inside our system
def plotting_pie_gen_pow(dico, outputFolder, showFlag=False):
    plot1 = [
        np.sum(dico["PVPowers"]),
        np.sum(dico["windPowers"]),
        np.sum(dico["fromGridPowers"]),
        np.sum(dico["dieselGenerators"]),
    ]
    plt.pie(
        plot1,
        labels=["PVPowers", "windPowers", "fromGridPowers", "dieselGenerators"],
        colors=[
            colorDico["PVPowers"],
            colorDico["windPowers"],
            colorDico["fromGridPowers"],
            colorDico["dieselGenerators"],
        ],
    )
    plt.title("Repartion of power given by the grid, pv, wind and diesel")
    plt.savefig(outputFolder + "/power_generated_pie.png")
    if showFlag:
        plt.show()
    plt.close()


# Bar plot power in out by the grid
def plotting_bar_in_out(dico, outputFolder, showFlag=False):
    plt.bar(0, np.sum(dico["fromGridPowers"]), color=colorDico["fromGridPowers"])
    plt.bar(1, np.sum(dico["toGridPowers"]), color=colorDico["toGridPowers"])
    plt.xticks([0, 1], ["From Grid", "To Grid"])
    plt.ylabel("Power (kW)")
    plt.title("Power from the grid in and out")
    plt.savefig(outputFolder + "/grid_in-out_bar.png")
    if showFlag:
        plt.show()
    plt.close()


# Bar plot of all the power
# small bar is "kind of generator" or "kind of load" (for battery / EV discharging // charging and selling
# the diff is due to the EV leaving (energy drop so power discharge is high but not in the system)
def plotting_bar_all_powers(dico, outputFolder, showFlag=False):
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
    p3 = plt.bar(
        0, fromGridPow, 0.2, bottom=pvPow + windPow, color=colorDico["fromGridPowers"]
    )
    p4 = plt.bar(
        0.05,
        dieselPow,
        0.1,
        bottom=pvPow + windPow + fromGridPow,
        color=colorDico["dieselGenerators"],
    )
    p5 = plt.bar(
        0.05,
        batNegPow,
        0.1,
        hatch="//",
        bottom=pvPow + windPow + fromGridPow + dieselPow,
        color=colorDico["batPowersNeg"],
    )
    p6 = plt.bar(
        0.05,
        evNegPow,
        0.1,
        hatch="--",
        bottom=pvPow + windPow + fromGridPow + dieselPow + batNegPow,
        color=colorDico["evPowersNeg"],
    )

    p7 = plt.bar(0.7, fixLoadPow, 0.2, color=colorDico["fixedLoads"])
    p8 = plt.bar(0.7, batPosPow, 0.2, bottom=fixLoadPow, color=colorDico["batPowers"])
    p9 = plt.bar(
        0.7, evPosPow, 0.2, bottom=fixLoadPow + batPosPow, color=colorDico["evPowers"]
    )
    p10 = plt.bar(
        0.75,
        toGridPow,
        0.1,
        bottom=fixLoadPow + batPosPow + evPosPow,
        color=colorDico["toGridPowers"],
    )

    plt.xticks([0, 0.7], ("Generator", "Loads"))
    plt.ylabel("Power (kW)")
    plt.legend(
        (p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0]),
        (
            "PV",
            "Wind",
            "Buy from the Grid",
            "Diesel",
            "Battery discharging",
            "EV discharging",
            "Fixed Loads",
            "Battery charging",
            "EV charging",
            "Sell to the Grid",
        ),
    )
    plt.title("Power generated & Power needed")
    plt.savefig(outputFolder + "all-pow_bar.png")
    if showFlag:
        plt.show()
    plt.close()
