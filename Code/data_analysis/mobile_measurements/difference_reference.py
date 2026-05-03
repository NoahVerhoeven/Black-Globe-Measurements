import geopandas as gpd
import pandas as pd
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=11.5)

files = ["Average-Gray.CSV", "Average-Black.CSV", "Referentie.CSV", "Kestrel.CSV"]
paths = [f"Data\mobile_measurements\{file}" for file in files]

data_dict = {file: {"data": pd.read_csv(path)} for file, path in zip(files, paths)}

start = data_dict["Referentie.CSV"]["data"]["TotalSeconds"].iloc[0]

referentie_func = interp1d(data_dict["Referentie.CSV"]["data"]["TotalSeconds"], data_dict["Referentie.CSV"]["data"]["BlackGlobetemp"])

fig, axis = plt.subplots(1, 1, figsize=(11, 10))

for file in files:
    df = data_dict[file]["data"]
    t_arr = []
    diff_arr = []

    for t, value in zip(df["TotalSeconds"], df["BlackGlobetemp"]):
        if t >= start:
            t_arr.append(t)
            diff_arr.append(value - referentie_func(t))
    print(f"Mean difference between {file[:-4]} and reference: {np.mean(diff_arr)}")
    axis.plot(t_arr, diff_arr, label=file[:-4])

plt.legend()
plt.show()