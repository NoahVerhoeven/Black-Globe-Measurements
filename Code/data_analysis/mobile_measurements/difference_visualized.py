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

fig, axis = plt.subplots(2, 2, figsize=(11, 10))

quant = "BlackGlobetemp"

for ax, file in zip(axis.flatten(), files):
    ax.set_title(file[:-4])

    df = data_dict[file]["data"]

    t_arr = []
    diff_arr = []
    lat_arr = []
    lon_arr = []

    for t, value, lat, lon in zip(df["TotalSeconds"], df[quant], df["latitude"], df["longitude"]):
        if t >= start:
            t_arr.append(t)
            diff_arr.append(value - referentie_func(t))
            lat_arr.append(lat)
            lon_arr.append(lon)

    print(f"Mean difference between {file[:-4]} and reference: {np.mean(diff_arr)}")

    df = pd.DataFrame({
        "TotalSeconds": t_arr,
        quant: diff_arr,
        "latitude": lat_arr,
        "longitude": lon_arr
    })

    gdf = gpd.GeoDataFrame(df)
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude), crs="EPSG:4326"
    )
    gdf.to_crs(epsg=3857, inplace=True)

    gdf.plot(
    column=quant,
    cmap="viridis",
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
    cbar_ax.set_ylabel("Diff")
    # cbar_ax.tick_params(labelsize=12)

fig.suptitle(f"{quant}", fontsize=15)
fig.tight_layout(pad=1.5)
fig.savefig(f"{quant}" + f"-Average.png", dpi=350)
plt.show()
