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
    "toGridPowers": "cadetblue",
    "dieselGenerators": "dimgray",
    "gridPrice": "goldenrod",
}

labelDico = {
    "PVPowers": "PV Power",
    "windPowers": "Wind Power",
    "batPowers": "Battery Power",
    "evPowers": "EV Power",
    "fixedLoads": "Loads Power",
    "fromGridPowers": "Grid Power In",
    "toGridPowers": "Grid Power Out",
    "dieselGenerators": "Diesel Power",
    "gridPrice": "Grid Price",
}


def plotting(varName, varVal, gridPrices, outputFolder, ini):
    dico = {
        "PVPowers": [],
        "windPowers": [],
        "batPowers": [],
        "evPowers": [],
        "fixedLoads": [],
        "fromGridPowers": [],
        "toGridPowers": [],
        "dieselGenerators": [],
    }

    dicoEnergy = {"batEnergys": [], "evEnergys": []}

    step = int(len(ini.timestamps) / 10)
    time = [
        ini.timestamps[i].strftime("%m-%d %H:%M") for i in range(len(ini.timestamps))
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
    plotting_powers(dico, outputFolder, time, tick)
    plotting_energys(
        dicoEnergy,
        ini.E_bat_max,
        ini.SOC_bat_min,
        ini.SOC_bat_max,
        outputFolder,
        time,
        tick,
    )
    plotting_all_powers(dico, outputFolder, time, tick)
    plotting_additive_all_powers(resultsDf, outputFolder, time, tick)
    plotting_additive_all_powers_sym(resultsDf, outputFolder, time, tick)
    plotting_in_out_price(dico, outputFolder, gridPrices, time, tick)
    plotting_pie_gen_pow(dico, outputFolder)
    plotting_bar_in_out(dico, outputFolder)
    plotting_bar_all_powers(dico, outputFolder)


# Plotting PV power, wind power and fixed loads power.
def plotting_powers(dico, outputFolder, time, tick):

    plt.plot(dico["PVPowers"], label="pv", color=colorDico["PVPowers"])
    plt.plot(dico["windPowers"], label="wind", color=colorDico["windPowers"])
    plt.plot(dico["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"])
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Output power - kW")
    plt.legend(loc="upper left", ncol=3)
    plt.savefig(outputFolder + "/pv_wind-power.png")
    plt.show()


# Plotting EV and batteries energies
def plotting_energys(
    dico, E_bat_max, SOC_bat_min, SOC_bat_max, outputFolder, time, tick
):
    plt.plot(dico["batEnergys"], label="Battery Energy", color=colorDico["batEnergys"])
    plt.plot(dico["evEnergys"], label="EV Energy", color=colorDico["evEnergys"])
    plt.plot(
        [0, len(dico["batEnergys"])], [E_bat_max, E_bat_max], ls="--", c="turquoise"
    )
    plt.fill_between(
        [0, len(dico["batEnergys"])],
        [E_bat_max, E_bat_max],
        hatch=".",
        color="turquoise",
    )
    plt.legend(loc="upper left", ncol=2, prop={"size": 8})
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.savefig(outputFolder + "/bat_ev-energy.png")
    plt.show()


# Plotting all the powers from our system inside one graph
def plotting_all_powers(dico, outputFolder, time, tick):
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
    plt.show()


# Area plotting of all the powers from our system (in and out) inside one graph with consumption (loads) as baseline
def plotting_additive_all_powers(resultsPd, outputFolder, time, tick):
    kindPlot = "area"  # 'bar'
    style = "default"  # 'steps-mid'
    step = None  # 'mid'

    # Devide in and out flows (esp. for batteries) and make them all positive
    negResults, resultsPd = resultsPd.clip(upper=0) * (-1), resultsPd.clip(lower=0)
    negResults.columns = [str(col) + "Neg" for col in negResults.columns]
    resultsPd[["batPowersNeg", "evPowersNeg"]] = negResults[
        ["batPowersNeg", "evPowersNeg"]
    ]
    # selection list of series to be plotted as area-plot and in which order
    selOut = ["fixedLoads", "batPowersNeg", "evPowersNeg", "toGridPowers"]
    selArea = [
        "PVPowers",
        "windPowers",
        "batPowers",
        "evPowers",
        "dieselGenerators",
        "fromGridPowers",
    ]
    # Colorscheme with selection lists
    inColors = list(map(colorDico.get, selArea))
    inColors = ["pink" if c is None else c for c in inColors]
    outColors = list(map(colorDico.get, selOut))
    outColors = ["pink" if c is None else c for c in outColors]
    hatch = ["", "//", "--", ".."]

    # Plottting
    fig, ax = plt.subplots()
    resultsPd[selArea].plot(
        kind=kindPlot, linewidth=0, stacked=True, ax=ax, color=inColors
    )

    additiveOut = resultsPd[selOut].copy()
    additiveOut = additiveOut.cumsum(axis=1)
    additiveOut[selOut[1:]].plot(
        kind="line", drawstyle=style, linewidth=2, ls="--", ax=ax, color=outColors[1:]
    )

    for i in range(1, len(selOut)):
        plt.fill_between(
            range(len(additiveOut)),
            additiveOut[selOut[i - 1]],
            additiveOut[selOut[i]],
            facecolor="none",
            step=step,
            hatch=hatch[i],
            edgecolor=colorDico[selOut[i]],
            linewidth=1.0,
        )

    plt.plot(
        resultsPd["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"]
    )

    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    plt.legend(bbox_to_anchor=(1.5, 0.8), loc="upper right")
    plt.savefig(outputFolder + "/power-balance2.png")
    plt.show()


# Area plotting of all the powers from our system (in and out) inside one graph with consumption (loads) as baseline
def plotting_additive_all_powers_sym(resultsPd, outputFolder, time, tick):
    kindPlot = "bar"  # 'bar'
    style = "steps-mid"
    step = None  # 'mid'

    # Devide in and out flows (esp. for batteries)
    # Selection list for in/out series in plotting order
    selOut = ["fixedLoads", "batPowersNeg", "evPowersNeg", "toGridPowers"]
    selIn = [
        "PVPowers",
        "windPowers",
        "batPowers",
        "evPowers",
        "dieselGenerators",
        "fromGridPowers",
    ]
    negResults, resultsPd = resultsPd.clip(upper=0), resultsPd.clip(lower=0)
    negResults.columns = [str(col) + "Neg" for col in negResults.columns]
    resultsPd[["batPowersNeg", "evPowersNeg"]] = negResults[
        ["batPowersNeg", "evPowersNeg"]
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
        kind=kindPlot, linewidth=0, stacked=True, ax=ax, color=inColors
    )

    plt.plot(
        -resultsPd["fixedLoads"], label="Loads Power", color=colorDico["fixedLoads"]
    )

    resultsPd[selOut].plot(
        kind=kindPlot, stacked=True, linewidth=0, ax=ax, ls="--", color=outColors
    )
    [resultsPd[selOut].sum(axis=1).min(), resultsPd[selIn].sum(axis=1).max()]
    ax.set_ylim(
        [resultsPd[selOut].sum(axis=1).min(), resultsPd[selIn].sum(axis=1).max()]
    )

    # Settings
    plt.xticks(tick, time, rotation=20)
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    plt.legend(bbox_to_anchor=(1.5, 0.8), loc="upper right")
    plt.savefig(outputFolder + "/power-balance3.png")
    plt.show()


# Plotting the evolution of the power in and out on the grid and the evolution of prices
def plotting_in_out_price(dico, outputFolder, gridPrices, time, tick):
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
    plt.show()


# Pie plot with the repartion of the power generated inside our system
def plotting_pie_gen_pow(dico, outputFolder):
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
    plt.show()
