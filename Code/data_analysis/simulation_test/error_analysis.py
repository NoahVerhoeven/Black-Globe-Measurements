import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from scipy.stats import linregress, pearsonr, spearmanrho

import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=10)

path = "Data\simulation_test\semi_realistic"

df = pd.read_csv(f"{path}\Semi_realistic_{0}.CSV")

MRT = df["MRT"]

smooth_error = np.array([])
inverse_error = np.array([])
finite_error = np.array([])

for i in range(100):
    df = pd.read_csv(f"{path}\Semi_realistic_{i}.CSV")

    inverse_smoothed_MRT = df["InverseSmoothedMRT"]
    finite_diff_MRT = df["FiniteDiffMRT"]

    smoothed_estimated_MRT = df["SmoothedEstimatedMRT"]
    estimated_MRT = df["EstimatedMRT"]

    smooth_error = np.concatenate((smooth_error, estimated_MRT - smoothed_estimated_MRT), axis=None)
    inverse_error = np.concatenate((inverse_error, MRT - inverse_smoothed_MRT), axis=None)
    finite_error = np.concatenate((finite_error, MRT - finite_diff_MRT), axis=None)

inverse_res = linregress(smooth_error, inverse_error)
finit_res = linregress(smooth_error, finite_error)

inverse_m = np.array([inverse_error.min(), inverse_error.max()])
finit_m = np.array([finite_error.min(), finite_error.max()])


fig, axis = plt.subplots(1,2, figsize=(10,5), layout="constrained")

b = spearmanrho(smooth_error, inverse_error)
g = spearmanrho(smooth_error, finite_error)
print(f"for inverse smoothing: rho={b.statistic}, p-value that data has no correlation is {b.pvalue}")
print(inverse_res)
print("\n")
print(f"for finite difference: rho={g.statistic}, p-value that data has no correlation is {g.pvalue}")
print(finit_res)

h = axis[0].hist2d(smooth_error, inverse_error, bins=100, norm=LogNorm(), cmap="autumn")
axis[0].plot(inverse_m, inverse_res.intercept + inverse_res.slope * inverse_m, color="black", lw=2, label="Linear Regression Lines")
axis[0].legend(facecolor="whitesmoke", loc="lower left")
axis[0].set_ylabel("Recovery Error (K)")
axis[0].set_xlabel(r"B-Spline Error (K)")
axis[0].grid(linestyle='--', color="lightgrey")
axis[0].set_title('(a) Inverse Smoothing Operation')
fig.colorbar(h[3], ax=axis[0])


h = axis[1].hist2d(smooth_error, finite_error, bins=100, norm=LogNorm(), cmap="summer")
axis[1].plot(finit_m, finit_res.intercept + finit_res.slope * finit_m, color="black", lw=2, label="Linear Regression Lines")
axis[1].grid(linestyle="--", color="lightgrey")
axis[1].set_ylabel("Recovery Error (K)")
axis[1].set_xlabel(r"B-Spline Error (K)")
axis[1].set_title('(b) Finite Difference Approximation')
fig.colorbar(h[3], ax=axis[1])

# fig.savefig("correlation.png", dpi=300)

print(np.mean(smooth_error), np.mean(finite_error), np.mean(inverse_error))


plt.show()