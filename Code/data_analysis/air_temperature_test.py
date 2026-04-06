import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Times New Roman'

fig, axis = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)

data_dict = {
    "B1": pd.read_csv("Data\\air_and_humidity_test\\B1.CSV"),
    "B2": pd.read_csv("Data\\air_and_humidity_test\\B2.CSV"),
    "BP": pd.read_csv("Data\\air_and_humidity_test\\BP.CSV"),
    "G1": pd.read_csv("Data\\air_and_humidity_test\\G1.CSV"),
    "G2": pd.read_csv("Data\\air_and_humidity_test\\G2.CSV"),
    "GP": pd.read_csv("Data\\air_and_humidity_test\\GP.CSV"),
    "S1": pd.read_csv("Data\\air_and_humidity_test\\S1.CSV"),
    "S2": pd.read_csv("Data\\air_and_humidity_test\\S2.CSV")
}

for key in data_dict:
    df = data_dict[key]
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    # df_filtered = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)].copy()
    df_filtered = df.copy()
    # df_filtered["time_sec"] = df_filtered["hour"]*3600 + df_filtered["minute"]*60 + df_filtered["second"]
    # df_filtered["time_sec"] = df_filtered["time_sec"] / 60 - 1395.9
    df_filtered.plot(y="Davistemp", ax=axis[0], label=key, alpha=0.75, lw=2)
    df_filtered.plot(y="Davishumidity", ax=axis[1], label=key, alpha=0.75, marker="o", markersize=2, lw=0)

axis[0].set_xticks(np.linspace(0, 1331, 7), np.int16(np.linspace(0, 1331, 7) * 6 / 60))
axis[0].set_xlabel("Time (min)")
axis[0].set_ylabel("Davis Temperature (°C)")
axis[0].set_title("Air Temperature Test", fontsize=14)
axis[0].legend()
axis[0].grid()


axis[1].set_xticks(np.linspace(0, 1331, 7), np.int16(np.linspace(0, 1331, 7) * 6 / 60))
axis[1].set_xlabel("Time (min)")
axis[1].set_ylabel("Davis Humidity (%)")
axis[1].set_title("Humidity Test", fontsize=14)
legend = axis[1].legend()
legend.remove()
axis[1].grid()

fig.suptitle("Air Temperature and Humidity Test (Indoor)", fontsize=17, fontweight="bold")

plt.show()