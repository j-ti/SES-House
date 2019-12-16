import matplotlib.pyplot as plt

def plotting(varName, varVal, soc_min_val):
    dico = {'pvPowers': [],
            'windPowers': [],
            'batPowers': [],
            'batEnergys': [],
            'evPowers': [],
            'evEnergys': [],
            'fixedLoads': [],
            'gridPowers': [],
            'dieselGenerators': []
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

    plt.plot(dico["batPowers"], label="bat Powers")
    plt.plot(dico["batEnergys"], label="bat Energys")
    plt.plot(dico["evPowers"], label="ev Powers")
    plt.plot(dico["evEnergys"], label="ev Energys")
    plt.plot([0, max([len(i) for i in dico.values()])], [soc_min_val, soc_min_val], color='r', linestyle='--',
             linewidth=1, label='Minimal energy')
    plt.legend(loc = 'upper right', prop={'size': 8})
    plt.show()

    plt.plot(dico["gridPowers"], label="gridPowers")
    plt.plot(dico["dieselGenerators"], label="dieselGenerators")
    plt.legend(loc = 'upper right')
    plt.show()
