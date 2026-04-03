# Mobile correction through moving average reconstruction. Currently, linear growth seems to be the best solution

from matplotlib import cm
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, UnivariateSpline, make_splrep
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
t_eval = np.linspace(0, minutes*60, 3000)
s = len(t_eval)


# We simulate the thermocouple measurements based on perfectly known constants and t-dependent variables (wind speed, air temperature and true MRT)
# To test our recovery algorithm, after this simulation we add noise to the thermocouple measurements and t-depedent variables (wind speed and air temperature)
# Our recovery algorithm should recover the true MRT, we can therefore quantify how good the alogirthm is (since we know the true MRT)

def V_a(t):
    V_2 = lambda t: (t - 350) / 20 + 3.5
    if t <= 350:
        return 3.5
    elif t <= 365:
        return V_2(t)
    else:
        return V_2(365)
        # return - np.e ** (-(t - 350)) / 2 + 4


def T_a(t):
    return 295


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
    

sigma = 5.67037 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 2 * 10 ** -3 # Thickness of the globe shell [m]

epsilon = 0.95  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 384 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 40 * 10 ** -3  # Diameter of the shell [m]
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A_shell = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = lambda t: (6.3 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
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

sol = solve_ivp(dTdt, [t_eval[0], t_eval[-1]], [295, 295], args=(MRT, args), method="Radau", t_eval=t_eval) # Implicit method to account for stiffness, sol.y[0] is the Thermocouple temp, sol.y[1] is shell temp

# Adding noise to the simulated thermocouple measurements and t-depedent variables (wind speed and air temperature)
temp_spread = 0.25
V_a_spread = 0.1

noisy_thermocouple_temp = sol.y[0] + np.random.normal(0, temp_spread, sol.t.shape)
noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, V_a_spread, sol.t.shape)
noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, temp_spread, sol.t.shape)

noisy_thermocouple_temp_cont = make_splrep(t_eval, noisy_thermocouple_temp)
noisy_V_a_cont = make_splrep(t_eval, noisy_V_a)
noisy_T_a_cont = make_splrep(t_eval, noisy_T_a)

noisy_h_cont = lambda t: (6.3 * noisy_V_a_cont(t) ** 0.6) / (D ** 0.4)
noisy_h_average = np.mean([noisy_h_cont(t) for t in t_eval])

true_mrt = np.array([MRT(t) for t in sol.t])
estimated_mrt = np.array([grey_body_MRT_estimate(noisy_thermocouple_temp_cont(t), noisy_h_cont(t), noisy_T_a_cont(t), epsilon) for t in sol.t]) # estimate is based on noisy real data

tau = lambda index: constant / (A_shell * (4 * epsilon * sigma * 325 ** 3 + noisy_h_average))
alpha = lambda index: 1 - np.exp(-((t_eval[1] - t_eval[0]) / tau(index)))

noisy_args = np.array([noisy_h_cont, noisy_T_a_cont, epsilon, constant, A_shell, A_i, h_i, constant_i])

old_error = float("inf")

for guess in range(24350, 24550, 5):
    print(f"Testing guess: {guess}")

    smooth_estimated_mrt = make_splrep(t_eval, estimated_mrt, s=guess)
    smooth_estimated_mrt = [smooth_estimated_mrt(t) for t in t_eval]

    best_recovered_estimated_mrt, best_recovered_true_mrt, _, best_error, _, _ = optimize_recovery(
        empirical_mrt=estimated_mrt,
        smooth_empirical_mrt=smooth_estimated_mrt,
        smooth_args=noisy_args,
        t_eval=t_eval,
        optimizing_range=(s, s+1),
        method="brute force",
        moving_average="exponential smoothing",
        error_method="absolute",
        base_func=alpha
    )
    print(f"Error multiple spline smooths: {best_error}\n")

    if best_error < old_error:
        old_error = best_error
        best_guess = guess
        lol = best_recovered_true_mrt

print(f"Best guess: {best_guess} with error: {old_error}")

# In the real world, we don't know these args exactly. Therefore we pertubate these args to see if we can still
# accurately recover the true MRT even if we don't know the actual args
# print(f"Perfect args: {args[1:]}")
# # args[1] += 5
# args[2] -= 0.1
# args[3] += 4
# args[-1] -= 0.12
# print(f"Pertubation of args: {args[1:]}\n")

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
axis["MRT"].scatter(sol.t / 60, estimated_mrt, label="Noisy Empirical MRT", color="tomato", s=4)
axis["MRT"].plot(sol.t / 60, smooth_estimated_mrt, label="Spline Smooth Empirical MRT", color="grey", lw=2.5)
# axis["MRT"].plot(sol.t / 60, [grey_body_MRT_estimate(T, h(t), T_a(t), epsilon) for T, t in zip(sol.y[0],sol.t)], label="Simulated Empirical MRT", color="black", lw=2.5)
axis["MRT"].plot(sol.t / 60, best_recovered_true_mrt, label="Recovered True MRT", color="green", lw=2.5,alpha=0.75)
axis["MRT"].set_title("Recovered True & Estimated MRT")
axis["MRT"].set_xlabel("Time (min)")
axis["MRT"].set_ylabel("Temperature (K)")
axis["MRT"].legend()
axis["MRT"].grid()

axis["Wind"].scatter(t_eval / 60, noisy_V_a, label="Wind Speed", color="mediumorchid", s=4)
axis["Wind"].set_title("Noisy Wind Speed Measurements")
axis["Wind"].set_xlabel("Time (min)")
axis["Wind"].set_ylabel("Wind Speed (m/s)")
axis["Wind"].legend()
axis["Wind"].grid()

axis["Air"].scatter(t_eval / 60, noisy_T_a, label="Air Temperature", color="mediumpurple", s=4)
axis["Air"].set_title("Noisy Air Temperature Measurements")
axis["Air"].set_xlabel("Time (min)")
axis["Air"].set_ylabel("Temperature (K)")
axis["Air"].legend()
axis["Air"].grid()

axis["Globe"].scatter(sol.t / 60, noisy_thermocouple_temp, label="Thermocouple Measurements", color="royalblue", s=4)
# axis["Globe"].plot(sol.t / 60, sol.y[0], label="Simulated Thermocouple", color="green", lw=2.5)
# axis["Globe"].plot(sol.t / 60, sol.y[1], label="Simulated Shell", color="darkblue", lw=2.5)
axis["Globe"].set_title("Noisy Thermocouple Measurements")
axis["Globe"].set_xlabel("Time (min)")
axis["Globe"].set_ylabel("Temperature (K)")
axis["Globe"].legend()
axis["Globe"].grid()

plt.show()
