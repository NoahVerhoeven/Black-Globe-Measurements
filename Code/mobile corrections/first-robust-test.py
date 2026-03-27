# Mobile correction through moving average reconstruction. Currently, linear growth seems to be the best solution

from matplotlib import cm
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, UnivariateSpline
from mrt_tools import (
    dTdt,
    grey_body_MRT_estimate,
    moving_average_matrix,
    recovery_error,
    recover_mrt,
    optimize_recovery
)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

minutes = 25
t_eval = np.linspace(0, minutes*60, 1000)

def V_a(t): # needs to be smooth to, so real data has to be fit spline smoothed out as well
    return (t ** 1.3) / 3000


T_a = 295
sigma = 5.67 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 0.4 * 10 ** -3 # Thickness of the globe shell [m]

epsilon = 0.95  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 384 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 150 * 10 ** -3  # Diameter of the shell
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A_shell = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = lambda t: (6.7 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
constant = c * rho * V # [J/K]

rho_i = 1.204 # Density of the inner part (air) [kg/m3]
M_i = 0.02897 # Molar mass of the inner part (air) [kg/mol]
c_i = 20.850 # Specific heat capacity of the inner part (air) [J/mol*K]
D_i = D - thickness
V_i = quad(lambda r: 4 * np.pi * r ** 2, 0, D_i/2)[0] # Volume of the inner part [m3]
A_i = 4 * np.pi * (D_i / 2) ** 2 # Inner surface area of the globe [m2]
h_i = 2 # (3 * V_a ** 0.6) / (D_i ** 0.4)
n_i = rho_i * V_i / M_i # Number of moles of the inner part [mol]
constant_i = c_i * n_i # [J/K]

args = np.array([h,  T_a, epsilon, constant, A_shell, A_i, h_i, constant_i])


def MRT(t):
    t_1 = lambda t: 300.2 + np.cos(t/10)
    t_2 = lambda t: 40 * np.sin((t - 400) / 200) + t_1(400)
    t_3 = lambda t: np.exp(-t/850) + t_2(850) - np.exp(-801/850)

    if t <= 400:
        return t_1(t)
    elif t <= 850:
        return t_2(t)
    else:
        return t_3(t)


sol = solve_ivp(dTdt, [t_eval[0], t_eval[-1]], [295, 295], args=(MRT, args), method="Radau", t_eval=t_eval) # Implicite method to acount for stiffness

smoothing_window = 100
data_spread = 0.5

inner_temp = sol.y[0] + np.random.normal(0, data_spread, sol.t.shape)
# shell_temp = sol.y[1]

true_mrt = np.array([MRT(t) for t in sol.t])
estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a, epsilon) for T, t in zip(inner_temp, sol.t)])

smooth_inner_temp = UnivariateSpline(t_eval, inner_temp, s=1000 * data_spread ** 2)
smooth_inner_temp = [smooth_inner_temp(t) for t in t_eval]

smooth_estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a, epsilon) for T, t in zip(smooth_inner_temp, sol.t)])

# _, _, best_window_size, best_error, error_array, window_guesses = optimize_recovery(
#     empirical_mrt=estimated_mrt,
#     smooth_empirical_mrt=smooth_estimated_mrt,
#     args=args,
#     t_eval=t_eval,
#     optimizing_range=(80, 140),
#     method="brute force",
#     smoothing_window=smoothing_window,
#     error_method="absolute"
# )

# # print(f"Estimation smoothing window: {smoothing_window}")
# print(f"Best window size: {best_window_size} with error: {best_error}\n")

# In the real world, we don't know these args exactly. Therefore we pertubate these args to see if we can still
# accurately recover the true MRT even if we don't know the actual args
# print(f"Perfect args: {args[1:]}")
# args[1] += 5
# args[2] -= 0.1
# args[3] += 4
# args[-1] -= 0.12
# print(f"Pertubation of args: {args[1:]}")

# This approach can accurately deal with noisy data and inaccurate args (like density, air temperature, ...)

best_recovered_estimated_mrt, best_recovered_true_mrt, best_window_size, best_error, error_array, window_guesses = optimize_recovery(
    empirical_mrt=estimated_mrt,
    smooth_empirical_mrt=smooth_estimated_mrt,
    args=args,
    t_eval=t_eval,
    optimizing_range=(70, 90),
    method="brute force",
    smoothing_window=smoothing_window,
    error_method="absolute"
)
print(f"Best window size: {best_window_size} with error: {best_error}\n")

fig, axis  = plt.subplot_mosaic(
    [["Wind", "MRT"],
     ["Globe", "MRT"]],
    layout="constrained",
    figsize=(14,12),
    width_ratios=[1.25, 2]
)
fig.suptitle("Recovered True MRT from Noisy Estimations", fontsize=16, fontweight="bold")

axis["MRT"].plot(sol.t / 60, true_mrt, label="True MRT", color="red", alpha=0.8, lw=2.5)
axis["MRT"].scatter(sol.t / 60, estimated_mrt, label="Empirical MRT", color="tomato", alpha=0.8, s=3.5)
axis["MRT"].plot(sol.t / 60, smooth_estimated_mrt, label="Spline Smooth Empirical MRT", color="grey", lw=2.5, alpha=0.8)
axis["MRT"].plot(sol.t / 60, best_recovered_true_mrt, label="Recovered True MRT", color="green", alpha=0.8, lw=2.5)
# axis["MRT"].plot(sol.t / 60, best_recovered_estimated_mrt, label="Recovered Estimated MRT", color="lawngreen", alpha=0.75, linestyle="--", lw=2.5)
axis["MRT"].set_title("Recovered True & Estimated MRT")
axis["MRT"].set_ylabel("Temperature (K)")
axis["MRT"].legend()
axis["MRT"].grid()

# axis[1].plot(sol.t / 60, shell_temp, label="Globe Temperature", color="blue")
axis["Globe"].scatter(sol.t / 60, inner_temp, label="Thermocouple Measurements", color="royalblue", s=3.5)
axis["Globe"].plot(sol.t / 60, smooth_inner_temp, label="Spline Smooth Thermocouple", color="grey", lw=2.5)
axis["Globe"].set_title("Noisy Thermocouple Measurements")
axis["Globe"].set_xlabel("Time (min)")
axis["Globe"].set_ylabel("Temperature (K)")
axis["Globe"].legend()
axis["Globe"].grid()



axis["Wind"].plot(t_eval / 60, [V_a(t) for t in t_eval], label="Wind Speed", color="black")
axis["Wind"].set_title("Wind Speed over Time")
axis["Wind"].set_xlabel("Time")
axis["Wind"].set_ylabel("Wind Speed (m/s)")
axis["Wind"].legend()
axis["Wind"].grid()

plt.show()