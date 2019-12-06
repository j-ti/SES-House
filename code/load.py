import pandas as pd
import numpy as np

file = r"load.csv"

# df_load = pd.read_csv(file)

# for simulation
time_step = 1
beg = 0
end = 100

time = np.arange(beg, end, time_step)
loads = np.arange(beg, end, time_step)
df_load = pd.DataFrame({"TimeStamp": time, "loads1": loads[:]})


def load(beg, end, step):
    return df_load[df_load["TimeStamp" > beg and "TimeStamp" < end]]


load(1, 2, 3)
