from matplotlib import cm
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, UnivariateSpline, make_splrep, make_smoothing_spline
from mrt_tools import (
    dTdt_shell_only,
    grey_body_MRT_estimate,
    moving_average_matrix,
    recovery_error,
    recover_mrt,
    optimize_recovery,
    spline_bootstrapping_residuals
)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

minutes = 25
n=1500
t_eval = np.linspace(0, minutes*60, n)

# UNDERLYING FUNCTIONS: These are the 'functions' which we'll measure in the field
def V_a(t):
    V_2 = lambda t: (t - 500) / 20 + 3.5
    if t <= 500:
        return 3.5
    elif t <= 515:
        return V_2(t)
    else:
        return V_2(515)
        # return - np.e ** (-(t - 350)) / 2 + 4


def T_a(t):
    return 295


def MRT(t):
    # return 350 - t ** 1.5 / 1000
    t_1 = lambda t: 300.2 + np.cos(t/10)
    t_2 = lambda t: 40 * np.sin((t - 400) / 200) + t_1(400)
    t_3 = lambda t: np.exp(-t/850) + t_2(850) - np.exp(-801/850)

    if t <= 400:
        return t_1(t)
    elif t <= 850:
        return t_2(t)
    else:
        return t_3(t)


# CONSTANTS: We'll work with the shell-only simulation
sigma = 5.67037 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 0.4 * 10 ** -3 # Thickness of the globe shell [m]
epsilon = 0.95  # Emissivity of black paint
rho = 8960  # Density of the globe (copper) [kg/m3]
c = 384 # Specific heat capacity of the globe (copper) [J/kg*K]
D = 150 * 10 ** -3  # Diameter of the shell [m]
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = lambda t: (6.3 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
constant = c * rho * V # [J/K]

args = np.array([h,  T_a, epsilon, constant, A])

# TRUE MRT: This is the function we want to recover
sol = solve_ivp(
    dTdt_shell_only,
    [t_eval[0], t_eval[-1]],
    [295],
    args=(MRT, args),
    method="Radau",
    t_eval=t_eval
) # Implicit method to account for stiffness
true_mrt = np.array([MRT(t) for t in sol.t])

# SIMULATE REAL EMPIRICAL DATA: We add noise (zero-mean Gaussian) to simulate statistical error in our measurements
noisy_T_g = sol.y[0] + np.random.normal(0, 0.25, sol.t.shape)
noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, 0.1, sol.t.shape)
noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, 0.25, sol.t.shape)
noisy_h = lambda i: (6.3 * noisy_V_a[i] ** 0.6) / (D ** 0.4)

# ESTIMATE MRT: We use the grey body estimate to guess the MRT from our 'empirical' data
empirical_data = np.array([grey_body_MRT_estimate(noisy_T_g[i], noisy_h(i), noisy_T_a[i], epsilon) for i, t in enumerate(sol.t)])
estimated_mrt = np.array([grey_body_MRT_estimate(sol.y[0][i], h(t), T_a(t), epsilon) for i, t in enumerate(sol.t)])

# SMOOTHING: We apply spline smoothing based on the GCV criterion
smooth_func = make_smoothing_spline(sol.t, empirical_data)
smooth_estimated_mrt = smooth_func(sol.t)


# CONFIDENCE INTERVALS: We'll bootstrap residual to find the upper and lower bands were 95% of the true function lays
lower_estimate, upper_estimate = spline_bootstrapping_residuals(sol.t, empirical_data, 50)
outside_estimate = 0
inside_estimate = 0

for l, u, e in zip(lower_estimate, upper_estimate, estimated_mrt):
    if e < l or e > u:
        outside_estimate += 1
        print(f"Estimated MRT {e:.2f} is outside the confidence interval [{l:.2f}, {u:.2f}], outside: {outside_estimate}")
    else:
        inside_estimate += 1
        print(f"Estimated MRT {e:.2f} is inside the confidence interval [{l:.2f}, {u:.2f}], inside: {inside_estimate}")

print(inside_estimate / n)

# PLOT RESULTS
fig, axis = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True)

axis[0].scatter(sol.t, empirical_data, alpha=0.7, s=3.5, label="Empirical Data")
axis[0].fill_between(sol.t, lower_estimate, upper_estimate, color="lightblue", label="95% Confidence Interval")
axis[0].plot(sol.t, smooth_estimated_mrt, color="blue", label="Smoothing Spline")
axis[0].plot(sol.t, estimated_mrt, color="black", label="Target Function")

axis[0].set_xlabel('Time (min)')
axis[0].set_ylabel('Temperature (K)')

axis[0].grid()
axis[0].legend()


plt.show()