import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.differentiate import derivative
from scipy.interpolate import make_smoothing_spline, interp1d
from mrt_tools import (
    grey_body_MRT_estimate,
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

V_a_func = interp1d(df["TotalSeconds"], df["NoisyWindspeed"])
T_a_func = interp1d(df["TotalSeconds"], df["NoisyAirtemp"] + 273.15)
T_g_func = interp1d(df["TotalSeconds"], df["NoisyGlobetemp"] + 273.15)

D = 40 * 10 ** -3  # Diameter of the shell
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe
h_func = lambda t: (6.3 * V_a_func(t) ** 0.6) / (D ** 0.4) # Forced convection
C = c * rho * V

args = np.array([h_func,  T_a_func, epsilon, C, A])

# UNCORRECTED MRT ESTIMATES: This is the MRT if we just apply the formula to the data
uncorrected_MRT = np.array([grey_body_MRT_estimate(T_g_func(t), h_func(t), T_a_func(t), epsilon) for t in df["TotalSeconds"]])

# FINITE DIFFERENCE: First, we spline smooth globe temps and h. Second, we apply the finite difference
w = np.array([1 / n] * n)

smooth_T_g_func = make_smoothing_spline(df["TotalSeconds"], [T_g_func(t) for t in df["TotalSeconds"]], w)
smooth_T_a_func = make_smoothing_spline(df["TotalSeconds"], [T_a_func(t) for t in df["TotalSeconds"]], w)
smooth_h_func = make_smoothing_spline(df["TotalSeconds"], [h_func(t) for t in df["TotalSeconds"]], w)

derivatives = derivative(smooth_T_g_func, df["TotalSeconds"])
finite_diff_MRT = []

for dT_g, t in zip(derivatives.df, df["TotalSeconds"]):
    T_g = smooth_T_g_func(t)
    T_a = smooth_T_a_func(t)
    h = smooth_h_func(t)

    point = np.float_power(((dT_g * C / A) + epsilon * sigma * T_g ** 4 + h * (T_g - T_a)) / (epsilon * sigma), 1/4)
    finite_diff_MRT.append(point)

# PLOT: We quickly plot to see the results
plt.scatter(df["TotalSeconds"], uncorrected_MRT, label="Uncorrected MRT")
plt.plot(df["TotalSeconds"], finite_diff_MRT, label="Finite Difference MRT", color="green")
plt.legend()

plt.show()