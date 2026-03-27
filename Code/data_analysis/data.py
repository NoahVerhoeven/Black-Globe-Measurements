import pandas as pd
import matplotlib.pyplot as plt

fig, ax  = plt.subplots(1, 1, figsize=(8,7))

df = pd.read_csv("Data\First-measurements\B1.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=ax, label="B1 Globe Temperature",color="red")
df.plot(x="time_sec", y="Davistemp", ax=ax, label="B1 Air Temperature", color="red", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\B2.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=ax, label="B2", color="green")
df.plot(x="time_sec", y="Davistemp", ax=ax, label="B2 Air Temperature", color="green", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\BP.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=ax, label="BP", color="blue")
df.plot(x="time_sec", y="Davistemp", ax=ax, label="BP Air Temperature", color="blue", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\S1.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=ax, label="S1", color="black")
df.plot(x="time_sec", y="Davistemp", ax=ax, label="S1 Air Temperature", color="black", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\S2.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=ax, label="S2", color="purple")
df.plot(x="time_sec", y="Davistemp", ax=ax, label="S2 Davis Temperature", color="purple", linestyle="--", alpha=0.8)

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Globe Temperature")
plt.show()