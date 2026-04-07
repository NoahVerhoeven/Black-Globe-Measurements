# Mobile correction through moving average reconstruction. Currently, linear growth seems to be the best solution

from matplotlib import cm
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from mrt_tools import dTdt, grey_body_MRT_estimate, moving_average_matrix
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

minutes = 20
t_eval = np.linspace(0, minutes*60, 1000)


def V_a(t):
    if t <= 10 * 60:
        return 0.5
    else:
        return 0.5


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

args = (h,  T_a, epsilon, constant, A_shell, A_i, h_i, constant_i)


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


sol = solve_ivp(dTdt, [0, minutes*60], [295, 295], args=(MRT, args), method="Radau", t_eval=t_eval) # Implicite method to acount for stiffness

mode = "linear decay"
smoothing_window = 100
smoothing_matrix = moving_average_matrix(sol.t, window_size=smoothing_window, mode=mode) # If estimate isn't smooth, this method will never work

inner_temp = sol.y[0]
shell_temp = sol.y[1]

true_mrt = np.array([MRT(t) for t in sol.t])
estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a, epsilon) for T, t in zip(inner_temp, sol.t)]) + np.random.normal(0, 1, sol.t.shape)

smooth_estimated_mrt = smoothing_matrix@estimated_mrt
    
old_error = float('inf')

print(t_eval)


for window_size_guess in range(195, 210):
    A = moving_average_matrix(true_mrt, window_size_guess, mode=mode, base_func=1.0185)
    A_inv = inv(A)

    recovered_true_mrt = A_inv@smooth_estimated_mrt

    new_error = np.sum(np.square(recovered_true_mrt - true_mrt))
    
    if old_error > new_error: # if old error is bigger, the new window size is better, so we update the old error
        old_error = new_error
        best_window_size = window_size_guess
        best_recovered_true_mrt = recovered_true_mrt
        best_A = A

best_recovered_true_mrt_func = interp1d(t_eval, best_recovered_true_mrt, kind='linear', fill_value="extrapolate")
recovered_sol = solve_ivp(dTdt, [0, minutes*60], [295, 295], args=(best_recovered_true_mrt_func, args), method="Radau", t_eval=t_eval) # Implicite method to acount for stiffness

recovered_inner_temp = sol.y[0]
recovered_shell_temp = sol.y[1]

recovered_estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a, epsilon) for T, t in zip(inner_temp, sol.t)])

print(f"Estimation smoothing window: {smoothing_window}")
print(f"Best window size: {best_window_size} with error: {old_error}")

fig, axis  = plt.subplots(2, 1, figsize=(10,9), sharex=True)
fig.suptitle("Recovered True MRT from Noisy Estimations", fontsize=16, fontweight="bold")

axis[0].plot(sol.t / 60, true_mrt, label="True MRT", color="red", lw=2.5)
axis[0].plot(sol.t / 60, estimated_mrt, label="Empirical Data", color="tomato")
axis[0].plot(sol.t / 60, best_recovered_true_mrt, label="Recovered True MRT", color="green", alpha=0.80, linestyle="--")
axis[0].plot(sol.t / 60, recovered_estimated_mrt, label="Recovered Estimated MRT", color="lawngreen", alpha=0.80, linestyle="--", lw=2.5)
axis[0].set_title("Recovered True & Estimated MRT")
axis[0].set_ylabel("Temperature (K)")
axis[0].legend()
axis[0].grid()

axis[1].plot(sol.t / 60, shell_temp, label="Globe Temperature", color="blue")
axis[1].plot(sol.t / 60, inner_temp, label="Inner Temperature", color="royalblue", linestyle="--")
axis[1].set_title("Recovered Globe & Inner Temperature")
axis[1].set_xlabel("Time (min)")
axis[1].set_ylabel("Temperature (K)")
axis[1].legend()
axis[1].grid()

plt.show()