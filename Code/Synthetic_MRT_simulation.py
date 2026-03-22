# Simulation with constant wind speed

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from mrt_tools import dTdt, grey_body_MRT_estimate
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

minutes = 20 

V_a = lambda t: 0.5
T_a = 295
sigma = 5.67 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 0.4 * 10 ** -3 # Thickness of the globe shell [m]

epsilon = 0.7  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 384 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 150 * 10 ** -3  # Diameter of the shell
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = lambda t: (6.7 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
constant = c * rho * V # [J/K]

rho_i = 1.204 # Density of the inner part (air) [kg/m3]
M_i = 0.02897 # Molar mass of the inner part (air) [kg/mol]
c_i = 20.850 # Specific heat capacity of the inner part (air) [J/mol*K]
D_i = D - thickness
V_i = quad(lambda r: 4 * np.pi * r ** 2, 0, D_i/2)[0] # Volume of the inner part [m3]
A_i = 4 * np.pi * (D_i / 2) ** 2 # Inner surface area of the globe [m2]
h_i = 7 # (3 * V_a ** 0.6) / (D_i ** 0.4)
n_i = rho_i * V_i / M_i # Number of moles of the inner part [mol]
constant_i = c_i * n_i # [J/K]


def MRT(t):
    if t <= 400:
        return 300.2 + np.cos(t/10)
    elif t <= 800:
        return 40 * np.sin((t - 400) / 200) + 300 + np.cos(t/15)/3
    else:
        return 325 + np.cos(t/40)/4


sol = solve_ivp(dTdt, [0, minutes*60], [295, 295], args=(MRT, h, T_a, epsilon, constant, A, A_i, h_i, constant_i), method="Radau", t_eval=np.linspace(0, minutes*60, 1000)) # Implicite method to acount for stiffness

inner_temp = sol.y[0]
shell_temp = sol.y[1]

fig, axis  = plt.subplots(2, 1, figsize=(8,7), sharex=True)
fig.suptitle("Simulated MRT Estimates from Globe Temperature", fontsize=16, fontweight="bold")

axis[0].plot(sol.t / 60, np.array([MRT(t) for t in sol.t]), label="True MRT", color="red")
axis[0].plot(sol.t / 60, np.array([grey_body_MRT_estimate(T, h, T_a, epsilon) for T in inner_temp]), label="Estimated MRT", color="tomato", linestyle="--")
axis[0].set_title("MRT Estimates & Synthetic True MRT")
axis[0].set_ylabel("Temperature (K)")
axis[0].legend()
axis[0].grid()


axis[1].plot(sol.t / 60, shell_temp, label="Globe Temperature", color="blue")
axis[1].plot(sol.t / 60, inner_temp, label="Inner Temperature", color="royalblue", linestyle="--")
axis[1].set_title("Globe & Inner Temperature")
axis[1].set_xlabel("Time (min)")
axis[1].set_ylabel("Temperature (K)")
axis[1].legend()
axis[1].grid()

plt.show()