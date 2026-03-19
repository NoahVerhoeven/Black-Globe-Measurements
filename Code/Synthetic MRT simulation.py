from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

V_a = 10
T_a = 288
sigma = 5.67 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 0.0005 # Thickness of the globe shell [m]

epsilon = 0.95  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 385 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 0.15  # Diameter of the shell
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = (6.3 * V_a ** 0.6) / (D ** 0.4)
constant = c * rho * V # [J/K]

rho_i = 1.204 # Density of the inner part (air) [kg/m3]
M_i = 0.02897 # Molar mass of the inner part (air) [kg/mol]
c_i = 20.850 # Specific heat capacity of the inner part (air) [J/mol*K]
D_i = D - thickness
V_i = quad(lambda r: 4 * np.pi * r ** 2, 0, D_i/2)[0] # Volume of the inner part [m3]
A_i = 4 * np.pi * (D_i / 2) ** 2 # Inner surface area of the globe [m2]
h_i = 7 #(3 * V_a ** 0.6) / (D_i ** 0.4)
n_i = rho_i * V_i / M_i # Number of moles of the inner part [mol]
constant_i = c_i * n_i # [J/K]


def MRT(t):
    # if  0 <= t < 10:
    #     return 300
    # elif t < 50:
    #     return 350
    # elif t < 100:
    #     return 300
    # else:
    #     return 350

    if t <= 40:
        return 300.2 + np.cos(t/4)
    elif t <= 80:
        return 40 * np.sin((t - 40) / 20) + 300 + + np.cos(t/5)
    else:
        return 325
    

def MRT_estimate(T, h):
    return (T ** 4 + h * (T - T_a) / (sigma * epsilon)) ** 0.25


def dTdt(t, T, MRT_func, h):
    T_i, T_g = T[0], T[1]

    dT_i = ((A_i * h_i) / constant_i) * (T_g - T_i)
    dT_g = (A * (sigma * epsilon * MRT_func(t) ** 4 - sigma * epsilon * T_g ** 4 - h * (T_g - T_a))  - A_i * h_i * (T_g - T_i)) / constant

    return [dT_i, dT_g]


sol = solve_ivp(dTdt, [0, 200], [288, 288], args=(MRT, h), method="Radau", t_eval=np.linspace(0, 200, 20000)) # Implicite method to acount for stiffness

shell_temp = sol.y[0]
inner_temp = sol.y[1]

fig, axis  = plt.subplots(2, 1, figsize=(8,7), sharex=True)
#fig.tight_layout(pad=3)
fig.suptitle("Simulated MRT Estimates from Globe Temperature", fontsize=16, fontweight="bold")

axis[0].plot(sol.t, np.array([MRT(t) for t in sol.t]), label="True MRT", color="red")
axis[0].plot(sol.t, np.array([MRT_estimate(T, h) for T in sol.y[1]]), label="Estimated MRT", color="tomato", linestyle="--")
axis[0].set_title("MRT Estimates & Synthetic True MRT")
#axis[0].set_xlabel("Time (min)")
axis[0].set_ylabel("Temperature (K)")
axis[0].legend()
axis[0].grid()


axis[1].plot(sol.t, sol.y[0], label="Globe Temperature", color="blue")
axis[1].plot(sol.t, sol.y[1], label="Inner Temperature", color="royalblue", linestyle="--")
axis[1].set_title("Globe & Inner Temperature")
axis[1].set_xlabel("Time (min)")
axis[1].set_ylabel("Temperature (K)")
axis[1].legend()
axis[1].grid()

plt.show()