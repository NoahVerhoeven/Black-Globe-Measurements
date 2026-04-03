import pandas as pd
import matplotlib.pyplot as plt

fig, axis  = plt.subplots(1, 2, figsize=(8,7))

df = pd.read_csv("Data\First-measurements\B1.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="B1",color="red")
df.plot(x="time_sec", y="Davistemp", ax=axis[1], label="B1", color="red", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\B2.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="B2", color="green")
df.plot(x="time_sec", y="Davistemp", ax=axis[1], label="B2", color="green", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\BP.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="BP", color="blue")
df.plot(x="time_sec", y="Davistemp", ax=axis[1], label="BP", color="blue", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\S1.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="S1", color="black")
df.plot(x="time_sec", y="Davistemp", ax=axis[1], label="S1", color="black", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\S2.CSV")
df["time_sec"] = df["hour"]*3600 + df["minute"]*60 + df["second"]
df["time_sec"] = df["time_sec"] / 60
df.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="S2", color="purple")
df.plot(x="time_sec", y="Davistemp", ax=axis[1], label="S2", color="purple", linestyle="--", alpha=0.8)

df = pd.read_csv("Data\First-measurements\Copper.CSV")

# Create a datetime column for filtering
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])

# Define your start and end times
start_time = pd.Timestamp(2026, 3, 26, 13, 1, 17)
end_time = pd.Timestamp(2026, 3, 26, 13, 49, 46)

# Filter the DataFrame
df_filtered = df[(df["datetime"] >= start_time) & (df["datetime"] <= end_time)].copy()

# Calculate time_sec (in minutes) for the filtered data
df_filtered["time_sec"] = df_filtered["hour"]*3600 + df_filtered["minute"]*60 + df_filtered["second"]
df_filtered["time_sec"] = df_filtered["time_sec"] / 60

# Plot
df_filtered.plot(x="time_sec", y="BlackGlobetemp", ax=axis[0], label="Reference", color="orange")
df_filtered.plot(x="time_sec", y="Davistemp", ax=axis[1], label="Reference", color="orange", linestyle="--")

axis[0].set_xlabel("Time (minutes)")
axis[0].set_ylabel("Globe Temperature")
axis[0].legend()
axis[0].grid()

axis[1].set_xlabel("Time (minutes)")
axis[1].set_ylabel("Davis Temperature")
axis[1].legend()
axis[1].grid()

plt.show()