import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Times New Roman'

fig, axis = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)

data_dict = {
    # "B1": pd.read_csv("Data\\globe_temperature_test\\B1.CSV"),
    "B2": pd.read_csv("Data\\globe_temperature_test\\B2.CSV"),
    "BP": pd.read_csv("Data\\globe_temperature_test\\BP.CSV"),
    "G1": pd.read_csv("Data\\globe_temperature_test\\G1.CSV"),
    "G2": pd.read_csv("Data\\globe_temperature_test\\G2.CSV"),
    "GP": pd.read_csv("Data\\globe_temperature_test\\GP.CSV"),
    "S1": pd.read_csv("Data\\globe_temperature_test\\S1.CSV"),
    "S2": pd.read_csv("Data\\globe_temperature_test\\S2.CSV"),
    "Reference": pd.read_csv("Data\\globe_temperature_test\\Reference.CSV")
}

start_time = pd.Timestamp(2026, 4, 8, 14, 0, 0)
end_time = pd.Timestamp(2026, 4, 8, 20, 0, 0)


for key in data_dict:
    df = data_dict[key]
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df_filtered = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)].copy()
    df_filtered = df.copy()
    # df_filtered["time_sec"] = df_filtered["hour"]*3600 + df_filtered["minute"]*60 + df_filtered["second"]
    # df_filtered["time_sec"] = df_filtered["time_sec"] / 60 - 1395.9
    df_filtered.plot(x="datetime", y="BlackGlobetemp", ax=axis[0], label=key, alpha=0.75, lw=2)
    df_filtered.plot(x="datetime", y="Davistemp", ax=axis[1], label=key, alpha=0.75, markersize=2, lw=2)

# axis[0].set_xticks(np.linspace(0, 1331, 7), np.int16(np.linspace(0, 1331, 7) * 6 / 60))
axis[0].set_xlabel("Time (min)")
axis[0].set_ylabel("Black Globe Temperature (°C)")
axis[0].set_title("Black Globes", fontsize=14)
axis[0].legend()
axis[0].grid()


# axis[1].set_xticks(np.linspace(0, 1331, 7), np.int16(np.linspace(0, 1331, 7) * 6 / 60))
axis[1].set_xlabel("Time (min)")
axis[1].set_ylabel("Davis Temperature (°C)")
axis[1].set_title("Air Temperature", fontsize=14)
legend = axis[1].legend()
legend.remove()
axis[1].grid()

fig.suptitle("Black Globe Test (All Balls)", fontsize=17, fontweight="bold")

plt.show()