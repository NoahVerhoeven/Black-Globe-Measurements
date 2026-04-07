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
w = np.array([1/n] * n)

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
    return 350
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
thickness = 2 * 10 ** -3 # Thickness of the globe shell [m]
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
noisy_T_g = sol.y[0] + np.random.normal(0, 0.5, sol.t.shape)
noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, 0.2, sol.t.shape)
noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, 0.5, sol.t.shape)
noisy_h = lambda i: (6.3 * noisy_V_a[i] ** 0.6) / (D ** 0.4)

# ESTIMATE MRT: We use the grey body estimate to guess the MRT from our 'empirical' data
empirical_data = np.array([grey_body_MRT_estimate(noisy_T_g[i], noisy_h(i), noisy_T_a[i], epsilon) for i, t in enumerate(sol.t)]) # Noisy, real data
estimated_mrt = np.array([grey_body_MRT_estimate(sol.y[0][i], h(t), T_a(t), epsilon) for i, t in enumerate(sol.t)]) #

# SMOOTHING: We apply spline smoothing based on the GCV criterion
h_spline = make_smoothing_spline(t_eval, [noisy_h(i) for i in range(len(t_eval))], w)
tau = lambda index: (constant - 0) / ((A + 0.0) * (4 * (epsilon - 0.0) * sigma * 345 ** 3 + h_spline(t_eval[index])))
alpha = lambda index: 1 - np.exp(-((t_eval[1] - t_eval[0]) / tau(index)))

smooth_func = make_smoothing_spline(sol.t, empirical_data, w)
smooth_estimated_mrt = smooth_func(sol.t)

# CONFIDENCE INTERVAL (estimate): We'll bootstrap residual to find the upper and lower bands were 95% of the true function lays
lower_estimate, upper_estimate = spline_bootstrapping_residuals(sol.t, empirical_data, 250)
outside_estimate = 0
inside_estimate = 0

for l, u, e in zip(lower_estimate, upper_estimate, estimated_mrt):
    if e < l or e > u:
        outside_estimate += 1
        # print(f"Estimated MRT {e:.2f} is outside the confidence interval [{l:.2f}, {u:.2f}], outside: {outside_estimate}")
    else:
        inside_estimate += 1
        # print(f"Estimated MRT {e:.2f} is inside the confidence interval [{l:.2f}, {u:.2f}], inside: {inside_estimate}")

print(inside_estimate / n)

# RECOVERY: We recover the true mrt by inversing the exponential smoothing on the empirical data, and spline smoothing the result
M = moving_average_matrix(
        input=empirical_data,
        window_size=len(empirical_data),
        mode="exponential smoothing",
        base_func=alpha
    )
M_inv = inv(M)

recovered_mrt = M_inv@empirical_data
smooth_recovered_mrt = M_inv@smooth_estimated_mrt

# CONFIDENCE INTERVAL (recovery): Doesn't work on recovered data, since noise is too amplified for bootstrapping residuals
# lower_recovered, upper_recovered = spline_bootstrapping_residuals(sol.t, recovered_mrt, 200)
# outside_recovered = 0
# inside_recovered = 0

# for l, u, e in zip(lower_recovered, upper_recovered, true_mrt):
#     if e < l or e > u:
#         outside_recovered += 1
#         # print(f"Estimated MRT {e:.2f} is outside the confidence interval [{l:.2f}, {u:.2f}], outside: {outside_estimate}")
#     else:
#         inside_recovered += 1
#         # print(f"Estimated MRT {e:.2f} is inside the confidence interval [{l:.2f}, {u:.2f}], inside: {inside_estimate}")

# print(inside_recovered / n)

# PLOT RESULTS
left = [
    ["Wind"],
    ["Air"],
    ["T_g"]
]
right = [
    ["Sim"],
    ["Tar"]
]
fig, axis = plt.subplot_mosaic(
    [[left, right]], figsize=(13, 9),
    layout="constrained",
    width_ratios=[1.25, 2],
    sharex=True
)
fig.suptitle("Inverse Exponential Smoothing Algorithm\nfor Recovering True MRT from Mobile Measurements", fontsize=18, fontweight="bold")
fig.tight_layout(pad=2.5)

# axis[0].set_ylim(292, 344)
axis["Sim"].scatter(sol.t / 60, empirical_data, alpha=0.7, s=3.5, label="Empirical Data", lw=2)
axis["Sim"].fill_between(sol.t / 60, lower_estimate, upper_estimate, color="lightblue", label="95% Confidence Interval", lw=2.5)
axis["Sim"].plot(sol.t / 60, smooth_estimated_mrt, color="blue", label="Smoothing Spline", lw=2.5)
# axis[0].plot(sol.t, estimated_mrt, color="black", label="Target Function (Estimate)")
axis["Sim"].plot(t_eval / 60, smooth_recovered_mrt, label="Recovered MRT", color="red", lw=2.5)
axis["Sim"].set_ylabel('Temperature (K)')
axis["Sim"].set_title('Recovered MRT from Simulated Empirical Data')
axis["Sim"].grid()
axis["Sim"].legend()

axis["Tar"].plot(t_eval / 60, true_mrt, color="black", label="Target Function (True MRT)", lw=2.5)
axis["Tar"].plot(t_eval / 60, estimated_mrt, color="grey", label="Target Function (Estimate)", lw=2.5)
# axis[1].fill_between(sol.t / 60, lower_recovered, upper_recovered, color="lightcoral", label="95% Confidence Interval")
axis["Tar"].fill_between(sol.t / 60, lower_estimate, upper_estimate, color="lightblue", label="95% Confidence Interval")
axis["Tar"].plot(sol.t / 60, smooth_estimated_mrt, color="blue", label="Smoothing Spline", lw=2.5, linestyle="dashed")
axis["Tar"].plot(t_eval / 60, smooth_recovered_mrt, label="Recovered MRT", color="red", lw=2.5, linestyle="dashed")
axis["Tar"].grid()
axis["Tar"].legend()
axis["Tar"].set_xlabel("Time (min)")
axis["Tar"].set_ylabel('Temperature (K)')
axis["Tar"].set_title('Target Functions')

axis["Wind"].set_title("Simulated Wind Speed Measurements")
axis["Wind"].scatter(sol.t / 60, noisy_V_a, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="mediumorchid")
axis["Wind"].grid()
axis["Wind"].set_ylabel("Wind Speed (m/s)")

axis["Air"].set_title("Simulated Air Temperature Measurements")
axis["Air"].scatter(sol.t / 60, noisy_T_a, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="mediumpurple")
axis["Air"].grid()
axis["Air"].set_ylabel("Temperature (K)")

axis["T_g"].set_title("Simulated Globe Temperature Measurements")
axis["T_g"].scatter(sol.t / 60, noisy_T_g, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="royalblue")
axis["T_g"].grid()
axis["T_g"].set_ylabel("Temperature (K)")
axis["T_g"].set_xlabel("Time (min)")



# plt.savefig("Inverse-Exponential-Smoothing-Algorithm.png", dpi=300)
fig.savefig("Inverse-Exponential-Smoothing-Algorithm.png", dpi=600)
plt.show()