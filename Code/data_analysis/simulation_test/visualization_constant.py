import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import skew, describe

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=10)

path = "Data\simulation_test\constant"

df = pd.read_csv(f"{path}\Constant_{0}.CSV")

MRT = df["MRT"]
t_eval = np.linspace(0, 25 * 60, 25 * 9)

left = [
    ["histoinverse"],
    ["histofinite"],
    ["histosmoothed"],
]
right = [
    ["inverse"],
    ["finite"],
    ["smoothed"]
]

fig, axis = plt.subplot_mosaic(
        [[left, right]], figsize=(16, 8),
        layout="constrained",
        width_ratios=[1, 2]
    )

inverse_delta = np.array([])
finite_delta = np.array([])
smoothed_delta = np.array([])

inverse_patch = mpatches.Patch(color="red", label='Inverse Smoothing Operation')  
finite_patch = mpatches.Patch(color="green", label='Finite Difference Approximation')  
smoothed_patch = mpatches.Patch(color="blue", label='Uncorrected B-Spline')
line = Line2D([0], [0], label='MRT', color='black', lw=2.5, linestyle="--")

handles = [inverse_patch, finite_patch, smoothed_patch, line]

true_average = np.mean(MRT)
inverse_tot = np.array([])
finite_tot = np.array([])
smoothed_tot = np.array([])

for i in range(100):
    df = pd.read_csv(f"{path}\Constant_{i}.CSV")

    inverse_smoothed_MRT = df["InverseSmoothedMRT"]
    finite_diff_MRT = df["FiniteDiffMRT"]
    smoothed_estimated_MRT = df["SmoothedEstimatedMRT"]

    inverse_tot = np.concatenate((inverse_tot, inverse_smoothed_MRT))
    finite_tot= np.concatenate((finite_tot, finite_diff_MRT))
    smoothed_tot= np.concatenate((smoothed_tot, smoothed_estimated_MRT))

    inverse_delta = np.concatenate((inverse_delta, np.absolute(MRT - inverse_smoothed_MRT)), axis=None)
    finite_delta = np.concatenate((finite_delta, np.absolute(MRT - finite_diff_MRT)), axis=None)
    smoothed_delta = np.concatenate((smoothed_delta, np.absolute(MRT - smoothed_estimated_MRT)), axis=None)

    # axis["inverse"].plot(t_eval / 60, inverse_smoothed_MRT, color="red", alpha=0.2, lw=2.5)
    # axis["finite"].plot(t_eval / 60, finite_diff_MRT, color="green", alpha=0.2, lw=2.5)
    # axis["smoothed"].plot(t_eval / 60, smoothed_estimated_MRT, color="blue", alpha=0.2, lw=2.5)

# for place, title in zip(right, ["(a) Recovered MRT From Inverse Smoothing Operations", '(b) Recovered MRT From Finite Difference Approximations', '(c) Recovered MRT From Uncorrected B-Splines']):
#     place = place[0]
    
#     axis[place].sharex(axis["smoothed"])
#     axis[place].plot(t_eval / 60, df["MRT"], color="black", lw=2.5, linestyle="--")
#     axis[place].grid(axis="both", linestyle='--', color="lightgrey")
#     axis[place].set_ylabel(r"Temperature (K)")
#     axis[place].set_title(title)
#     axis[place].set_xlim(0, 25)

#     if place == "smoothed":
#         axis[place].set_xlabel(r"Time (min)")

# axis["histoinverse"].hist(inverse_delta, bins=14, color="red", alpha=1, log=True)
# axis["histofinite"].hist(finite_delta, bins=14, color="green", alpha=1, log=True)
# axis["histosmoothed"].hist(smoothed_delta, bins=14, color="blue", alpha=1, log=True)

# for place, title in zip(left, ["(d) Absolute Error Distribution for Inverse Smoothing Operation", "(e) Absolute Error Distribution for Finite Difference Approximation", "(f) Absolute Error Distribution for Uncorrected B-Spline"]):
#     place = place[0]
#     axis[place].sharex(axis["histosmoothed"])
#     axis[place].sharey(axis["histofinite"])
#     axis[place].set_title(title)
#     axis[place].grid(axis="y", linestyle='--', color="lightgrey")

#     if place == "histosmoothed":
#         axis[place].set_xlabel(r"$|\Delta T|$ (K)")
    
#     axis[place].set_ylabel(r"$n$")


# axis["inverse"].legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), facecolor="whitesmoke")

# fig.savefig("algorithms-constant.png", dpi=300)
print(describe(finite_tot))
print(describe(inverse_tot))
print(describe(smoothed_tot))

plt.show()