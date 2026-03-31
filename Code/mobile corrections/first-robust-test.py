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
t_eval = np.linspace(0, minutes*60, 4000)
s = len(t_eval)


# We simulate the thermocouple measurements based on perfectly known constants and t-dependent variables (wind speed, air temperature and true MRT)
# To test our recovery algorithm, after this simulation we add noise to the thermocouple measurements and t-depedent variables (wind speed and air temperature)
# Our recovery algorithm should recover the true MRT, we can therefore quantify how good the alogirthm is (since we know the true MRT)

def V_a(t):
    if t <= 350:
        return 3.5
    else:
        return - np.e ** (-(t - 350)) / 2 + 4


def T_a(t):
    return 295


def MRT(t):
    # return 350
    t_1 = lambda t: 300.2 + np.cos(t/10)
    t_2 = lambda t: 40 * np.sin((t - 400) / 200) + t_1(400)
    t_3 = lambda t: np.exp(-t/850) + t_2(850) - np.exp(-801/850)

    if t <= 400:
        return t_1(t)
    elif t <= 850:
        return t_2(t)
    else:
        return t_3(t)
    

sigma = 5.67037 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 1 * 10 ** -3 # Thickness of the globe shell [m]

epsilon = 0.95  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 384 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 40 * 10 ** -3  # Diameter of the shell [m]
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

sol = solve_ivp(dTdt, [t_eval[0], t_eval[-1]], [295, 295], args=(MRT, args), method="Radau", t_eval=t_eval) # Implicite method to acount for stiffness

# Adding noise to the simulated thermocouple measurements and t-depedent variables (wind speed and air temperature)
temp_spread = 0.5
V_a_spread = 0.1

noisy_thermocouple_temp = sol.y[0] + np.random.normal(0, temp_spread, sol.t.shape)
noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, V_a_spread, sol.t.shape)
noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, temp_spread, sol.t.shape)
# shell_temp = sol.y[1]

true_mrt = np.array([MRT(t) for t in sol.t])
estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a(t), epsilon) for T, t in zip(noisy_thermocouple_temp, sol.t)])

smooth_thermocouple_temp = UnivariateSpline(t_eval, noisy_thermocouple_temp, s=10 + s * temp_spread ** 2)
smooth_V_a = UnivariateSpline(t_eval, noisy_V_a, s=5 + s * V_a_spread ** 2)
smooth_T_a = UnivariateSpline(t_eval, noisy_T_a, s=10 + s * temp_spread ** 2)

smooth_h = lambda t: (6.7 * smooth_V_a(t) ** 0.6) / (D ** 0.4)

tau = lambda index: constant / (A_shell * (4 * epsilon * sigma * 325 ** 3 + smooth_h(t_eval[index])))
alpha = lambda index: ((t_eval[1] - t_eval[0]) / tau(index))

print(tau(0))

C = lambda index: 1 / tau(index) # 1 * (4 * epsilon * sigma * 325 ** 3 + smooth_h(t_eval[index])) / constant
# base = lambda index: np.e ** ((t_eval[1] - t_eval[0]) / C(index))

smooth_estimated_mrt = np.array(
    [grey_body_MRT_estimate(smooth_thermocouple_temp(t), smooth_h(t), smooth_T_a(t), epsilon) for t in sol.t]
)

smooth_args = np.array([smooth_h,  smooth_T_a, epsilon, constant, A_shell, A_i, h_i, constant_i])

# In the real world, we don't know these args exactly. Therefore we pertubate these args to see if we can still
# accurately recover the true MRT even if we don't know the actual args
# print(f"Perfect args: {args[1:]}")
# args[1] += 5
# args[2] -= 0.1
# args[3] += 4
# args[-1] -= 0.12
# print(f"Pertubation of args: {args[1:]}")

best_recovered_estimated_mrt, best_recovered_true_mrt, best_window_size, best_error, error_array, window_guesses = optimize_recovery(
    empirical_mrt=estimated_mrt,
    smooth_empirical_mrt=smooth_estimated_mrt,
    smooth_args=smooth_args,
    t_eval=t_eval,
    optimizing_range=(s, s+1),
    method="brute force",
    moving_average="exponential smoothing",
    error_method="square",
    base_func=alpha
)
print(f"Best window size: {best_window_size} with error: {best_error}\n")

fig, axis  = plt.subplot_mosaic(
    [["Wind", "MRT"],
     ["Air", "MRT"],
     ["Globe", "MRT"]],
    layout="constrained",
    figsize=(14,12),
    width_ratios=[1.25, 2]
)
fig.suptitle("Recovered True MRT from Noisy Estimations", fontsize=16, fontweight="bold")

axis["MRT"].plot(sol.t / 60, true_mrt, label="True MRT", color="red", lw=2.5)
axis["MRT"].scatter(sol.t / 60, estimated_mrt, label="Empirical MRT", color="tomato", s=4)
axis["MRT"].plot(sol.t / 60, smooth_estimated_mrt, label="Spline Smooth Empirical MRT", color="grey", lw=2.5)
axis["MRT"].plot(sol.t / 60, best_recovered_true_mrt, label="Recovered True MRT", color="green", lw=2.5)
axis["MRT"].set_title("Recovered True & Estimated MRT")
axis["MRT"].set_xlabel("Time (min)")
axis["MRT"].set_ylabel("Temperature (K)")
axis["MRT"].legend()
axis["MRT"].grid()

axis["Wind"].scatter(t_eval / 60, noisy_V_a, label="Wind Speed", color="mediumorchid", s=4)
axis["Wind"].plot(t_eval / 60, [smooth_V_a(t) for t in t_eval], label="Spline Smooth Wind Speed", color="grey", lw=2.5)
axis["Wind"].set_title("Noisy Wind Speed Measurements")
axis["Wind"].set_xlabel("Time (min)")
axis["Wind"].set_ylabel("Wind Speed (m/s)")
axis["Wind"].legend()
axis["Wind"].grid()

axis["Air"].scatter(t_eval / 60, noisy_T_a, label="Air Temperature", color="mediumpurple", s=4)
axis["Air"].plot(t_eval / 60, [smooth_T_a(t) for t in t_eval], label="Spline Smooth Air Temperature", color="grey", lw=2.5)
axis["Air"].set_title("Noisy Air Temperature Measurements")
axis["Air"].set_xlabel("Time (min)")
axis["Air"].set_ylabel("Temperature (K)")
axis["Air"].legend()
axis["Air"].grid()

axis["Globe"].scatter(sol.t / 60, noisy_thermocouple_temp, label="Thermocouple Measurements", color="royalblue", s=4)
axis["Globe"].plot(sol.t / 60, [smooth_thermocouple_temp(t) for t in t_eval], label="Spline Smooth Thermocouple", color="grey", lw=2.5)
axis["Globe"].set_title("Noisy Thermocouple Measurements")
axis["Globe"].set_xlabel("Time (min)")
axis["Globe"].set_ylabel("Temperature (K)")
axis["Globe"].legend()
axis["Globe"].grid()

plt.show()