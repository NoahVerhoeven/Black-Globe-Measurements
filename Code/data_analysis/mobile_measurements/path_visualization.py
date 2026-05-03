import geopandas as gpd
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=11.5)

# files = ["Grijs-1.CSV", "Grijs-GP.CSV", "Zwart-BP.CSV", "Zwart-2.CSV", "Referentie.CSV", "Kestrel.CSV"]
files = ["Average-Gray.CSV", "Average-Black.CSV", "Referentie.CSV", "Kestrel.CSV"]
paths = [f"Data\mobile_measurements\{file}" for file in files]

data_dict = {file: {"data": pd.read_csv(path)} for file, path in zip(files, paths)}

# for file in files[:5]:
#     df = data_dict[file]["data"]
#     df["datetime"] = pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=2)
#     data_dict[file]["data"] = df

# start_times = [data_dict[file]["data"]["datetime"].iloc[0] for file in files]
# start_time = min(start_times[:4])

# for file in files:
#     df = data_dict[file]["data"]

#     df["diff"] = pd.to_datetime(df["datetime"]) - pd.to_datetime(start_time)
#     df["TotalSeconds"] = df["diff"].dt.total_seconds()

#     df.drop(columns=["diff"], inplace=True)

#     data_dict[file]["data"] = df

#     if df["TotalSeconds"].iloc[0] == 0:
#         coords_func = make_interp_spline(df["TotalSeconds"], df[["latitude", "longitude"]], k=1)

# df = data_dict["Kestrel.CSV"]["data"]
# df["latitude"], df["longitude"] = coords_func(df["TotalSeconds"]).T
# data_dict["Kestrel.CSV"]["data"] = df

# df = data_dict["Referentie.CSV"]["data"]
# df["latitude"], df["longitude"] = coords_func(df["TotalSeconds"]).T
# data_dict["Referentie.CSV"]["data"] = df

# for file in files:
#     data_dict[file]["data"].to_csv(f"{file}", index=False)

fig, axis = plt.subplots(2, 2, figsize=(11, 10))

quant = "Davistemp"

for ax, file in zip(axis.flatten(), files):
    ax.set_title(file[:-4])

    df = data_dict[file]["data"]
    gdf = gpd.GeoDataFrame(df)
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude), crs="EPSG:4326"
    )
    gdf.to_crs(epsg=3857, inplace=True)

    gdf.plot(
    column=quant,
    cmap="bwr",
    legend=True,
    ax=ax
    )
    ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik,
    )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    cbar_ax = ax.get_figure().axes[-1]

    # Set label and font size
    cbar_ax.set_ylabel(quant)
    # cbar_ax.tick_params(labelsize=12)

fig.suptitle(f"{quant}", fontsize=15)
fig.tight_layout(pad=1.5)
fig.savefig(f"{quant}" + f"-Average.png", dpi=350)
plt.show()