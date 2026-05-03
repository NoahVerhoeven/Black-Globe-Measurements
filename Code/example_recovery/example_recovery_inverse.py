import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import make_smoothing_spline, interp1d
from mrt_tools import (
    grey_body_MRT_estimate,
    inverse_exponential_smoothing
)
import matplotlib.pyplot as plt

# READ DATA
df = pd.read_csv("Code\example_recovery\example_data.CSV")

n = len(df["TotalSeconds"])
w = np.array([1 / n] * n) # weight for spline smoothing

# CONSTANTS: In this example we use the B2 globes, we account for gaps with linear interpolation
sigma = 5.67037 * 10 ** -8
thickness = 2 * 10 ** -3 # Thickness of the shell
epsilon = 0.94  # Emissivity
rho = 1240  # Density of the globe
c = 1800 # Specific heat capacity of the globe

V_a = interp1d(df["TotalSeconds"], df["NoisyWindspeed"])
T_a = interp1d(df["TotalSeconds"], df["NoisyAirtemp"] + 273.15)
T_g = interp1d(df["TotalSeconds"], df["NoisyGlobetemp"] + 273.15)

D = 40 * 10 ** -3  # Diameter of the shell
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe
h = lambda t: (6.3 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convection
constant = c * rho * V

args = np.array([h,  T_a, epsilon, constant, A])

# UNCORRECTED MRT ESTIMATES: This is the MRT if we just apply the formula to the data
uncorrected_MRT = np.array([grey_body_MRT_estimate(T_g(t), h(t), T_a(t), epsilon) for t in df["TotalSeconds"]])

# INVERSE SMOOTHING: First, we spline smooth uncorrect MRT and h. Second, we apply the inverse smoothing
w = np.array([1 / n] * n)
smooth_uncorrected_MRT_func = make_smoothing_spline(df["TotalSeconds"], uncorrected_MRT, w)
smooth_h_func = make_smoothing_spline(df["TotalSeconds"], [h(t) for t in df["TotalSeconds"]], w)

smooth_uncorrected_MRT = np.array([smooth_uncorrected_MRT_func(t) for t in df["TotalSeconds"]])

tau = lambda k: constant / (A * (4 * epsilon * sigma * np.mean(uncorrected_MRT) ** 3 + smooth_h_func(df["TotalSeconds"][k])))
alpha = lambda k: 1 - np.exp(-((df["TotalSeconds"][1] - df["TotalSeconds"][0]) / tau(k)))

inverse_smoothed_MRT = inverse_exponential_smoothing(smooth_uncorrected_MRT, alpha, df["TotalSeconds"])

# FINITE DIFFERENCE: First, we spline smooth globe temps and h. Second, we apply the finite difference
w = np.array([1 / n] * n)
smooth_uncorrected_MRT_func = make_smoothing_spline(df["TotalSeconds"], uncorrected_MRT, w)
smooth_h_func = make_smoothing_spline(df["TotalSeconds"], [h(t) for t in df["TotalSeconds"]], w)

smooth_uncorrected_MRT = np.array([smooth_uncorrected_MRT_func(t) for t in df["TotalSeconds"]])

tau = lambda k: constant / (A * (4 * epsilon * sigma * np.mean(uncorrected_MRT) ** 3 + smooth_h_func(df["TotalSeconds"][k])))
alpha = lambda k: 1 - np.exp(-((df["TotalSeconds"][1] - df["TotalSeconds"][0]) / tau(k)))

inverse_smoothed_MRT = inverse_exponential_smoothing(smooth_uncorrected_MRT, alpha, df["TotalSeconds"])

# PLOT: We quickly plot to see the results
plt.scatter(df["TotalSeconds"], uncorrected_MRT, label="Uncorrected MRT")
plt.plot(df["TotalSeconds"], inverse_smoothed_MRT, label="Finite Difference MRT", color="red")
plt.legend()

plt.show()