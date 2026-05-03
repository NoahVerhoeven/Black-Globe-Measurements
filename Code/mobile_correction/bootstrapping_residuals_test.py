from matplotlib import cm
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.differentiate import derivative
from scipy.interpolate import interp1d, UnivariateSpline, make_splrep, make_smoothing_spline
from mrt_tools import (
    dTdt_shell_only,
    grey_body_MRT_estimate,
    spline_bootstrapping_residuals,
    inverse_exponential_smoothing
)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=10)

minutes = 25
n=minutes * 9
t_eval = np.linspace(0, minutes*60, n)
w = np.array([1/n] * n)

# LOGISTIC FUNCTION
def L(x, y):
    try:
        return 1 / (1 + np.pow(np.e, -10 * (y-x)))
    except OverflowError:
        return 0

# UNDERLYING FUNCTIONS: These are the 'functions' which we'll measure in the field
def V_a(t):
    return np.sin(t/80) + 3.5


def T_a(t):
    return 310 - t/1000


def MRT(t):
    # return 350
    f_1 = lambda t: (2 * np.sin(t / 11) + 305) * L(t, 6 * 60)
    f_2 = lambda t: L(-t, -6 * 60) * (40 * np.sin((t - 450) / 250) + 317) * L(t, 17 * 60)
    f_3 = lambda t: L(-t, -17 * 60) * 325

    return f_1(t) + f_2(t) + f_3(t)

# CONSTANTS: We'll work with the shell-only simulation
sigma = 5.67037 * 10 ** -8 # [J/s*m^2*K^4]
thickness = 2 * 10 ** -3 # Thickness of the globe shell [m]
epsilon = 0.94  # Emissivity of black paint
rho = 1240  # Density of the globe (PLA) [kg/m3]
c = 1800 # Specific heat capacity of the globe (PLA) [J/kg*K]

D = 40 * 10 ** -3  # Diameter of the shell [m]
V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
h = lambda t: (6.3 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
constant = c * rho * V # [J/K]

args = np.array([h,  T_a, epsilon, constant, A])

# TRUE MRT: This is the function we want to recover
sol = solve_ivp(
    dTdt_shell_only,
    [t_eval[0], t_eval[-1]],
    [310], # T_g(0)
    args=(MRT, args),
    method="Radau",
    t_eval=t_eval
) # Implicit method to account for stiffness
true_mrt = np.array([MRT(t) for t in sol.t])

# SIMULATE REAL EMPIRICAL DATA: We add noise (zero-mean Gaussian) to simulate statistical error in our measurements
noisy_T_g = sol.y[0] + np.random.normal(0, 0.25, sol.t.shape)
noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, 0.5, sol.t.shape)
noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, 0.25, sol.t.shape)
noisy_h = lambda i: (6.3 * noisy_V_a[i] ** 0.6) / (D ** 0.4)

# ESTIMATE MRT: We use the grey body estimate to guess the MRT from our 'empirical' data
empirical_data = np.array([grey_body_MRT_estimate(noisy_T_g[i], noisy_h(i), noisy_T_a[i], epsilon) for i, t in enumerate(sol.t)]) # Noisy, real data
estimated_mrt = np.array([grey_body_MRT_estimate(sol.y[0][i], h(t), T_a(t), epsilon) for i, t in enumerate(sol.t)]) #

# SMOOTHING: We apply spline smoothing based on the GCV criterion
h_spline = make_smoothing_spline(t_eval, [noisy_h(i) for i in range(len(t_eval))], w)
tau = lambda index: (constant - 0) / ((A + 0.0) * (4 * (epsilon - 0.0) * sigma * np.mean(estimated_mrt) ** 3 + h_spline(t_eval[index])))
alpha = lambda index: 1 - np.exp(-((t_eval[1] - t_eval[0]) / tau(index)))

smooth_func = make_smoothing_spline(sol.t, empirical_data, w, lam=150)
smooth_estimated_mrt = smooth_func(sol.t)

# CONFIDENCE INTERVAL (estimate): We'll bootstrap residual to find the upper and lower bands were 95% of the true function lays
lower_estimate, upper_estimate = spline_bootstrapping_residuals(sol.t, empirical_data, 600)
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

# INVERSE EXPONENTIAL SMOOTHING: We recover the true mrt by inversing the exponential smoothing on the empirical data, and spline smoothing the result
inverse_smoothed_mrt = inverse_exponential_smoothing(smooth_estimated_mrt, alpha, t_eval)

# FINITE DIFFERENCE APPROXIMATION: If we numerically estimate the derivative at each point we can find the T_mrt
smooth_T_g = make_smoothing_spline(t_eval, noisy_T_g, w)
smooth_T_a = make_smoothing_spline(t_eval, noisy_T_a, w)
derivatives = derivative(smooth_T_g, t_eval)
finite_diff_mrt = []

for dT_g, T_g, T_a, h in zip(derivatives.df, smooth_T_g(t_eval), smooth_T_a(t_eval), h_spline(t_eval)):
    point = np.float_power(((dT_g * constant / A) + epsilon * sigma * T_g ** 4 + h * (T_g - T_a)) / (epsilon * sigma), 1/4)
    finite_diff_mrt.append(point)

finite_diff_mrt = np.array(finite_diff_mrt)

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
    [[left, right]], figsize=(13, 7),
    layout="constrained",
    width_ratios=[1.25, 2],
    sharex=True
)


# fig.suptitle("Exemplary Semi-Realistic MRT Simulation From Dataset", fontsize=18, fontweight="bold")
# fig.tight_layout(pad=2.5)

# axis[0].set_ylim(292, 344)
axis["Sim"].scatter(sol.t / 60, empirical_data, alpha=0.7, s=3.5, label="Uncorrected MRT Estimates", lw=2)
axis["Sim"].fill_between(sol.t / 60, lower_estimate, upper_estimate, color="lightblue", label="95% Confidence Interval", lw=2.5)
axis["Sim"].plot(sol.t / 60, smooth_estimated_mrt, color="blue", label="Uncorrected B-Spline", lw=2.5)
axis["Sim"].plot(t_eval[1:] / 60, inverse_smoothed_mrt[1:], label="Inverse Smoothing Operation", color="red", lw=2.5)
axis["Sim"].plot(t_eval / 60, finite_diff_mrt, label="Finite Difference Approximation", color="green", lw=2.5)
axis["Sim"].set_ylabel('Temperature (K)')
axis["Sim"].set_title('(a) Recovered MRT From Simulated Empirical Data')
axis["Sim"].grid(linestyle='--', color="lightgrey")
axis["Sim"].legend(facecolor="whitesmoke")

axis["Tar"].plot(t_eval / 60, true_mrt, color="black", label="Target Function (MRT)", lw=2.5)
axis["Tar"].plot(t_eval / 60, estimated_mrt, color="grey", label="Target Function (Estimate)", lw=2.5)
# axis[1].fill_between(sol.t / 60, lower_recovered, upper_recovered, color="lightcoral", label="95% Confidence Interval")
axis["Tar"].fill_between(sol.t / 60, lower_estimate, upper_estimate, color="lightblue")
axis["Tar"].plot(t_eval[1:] / 60, inverse_smoothed_mrt[1:], color="red", lw=2.5, linestyle="dashed")
axis["Tar"].plot(sol.t / 60, smooth_estimated_mrt, color="blue", lw=2.5, linestyle="dashed")
axis["Tar"].plot(t_eval / 60, finite_diff_mrt, color="green", lw=2.5, linestyle="dashed")
axis["Tar"].grid(linestyle='--', color="lightgrey")
axis["Tar"].legend(facecolor="whitesmoke")
axis["Tar"].set_xlabel("Time (min)")
axis["Tar"].set_ylabel('Temperature (K)')
axis["Tar"].set_title('(b) Target Functions')

axis["Wind"].set_title("(c) Simulated Wind Speed Measurements")
axis["Wind"].scatter(sol.t / 60, noisy_V_a, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="mediumorchid", marker="v")
axis["Wind"].grid(linestyle='--', color="lightgrey")
axis["Wind"].set_ylabel("Wind Speed (m/s)")

axis["Air"].set_title("(d) Simulated Air Temperature Measurements")
axis["Air"].scatter(sol.t / 60, noisy_T_a, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="mediumpurple", marker=">")
axis["Air"].grid(linestyle='--', color="lightgrey")
axis["Air"].set_ylabel("Temperature (K)")

axis["T_g"].set_title("(e) Simulated Globe Temperature Measurements")
axis["T_g"].scatter(sol.t / 60, noisy_T_g, alpha=0.7, s=3.5, label="Empirical Data", lw=2, color="royalblue", marker="^")
axis["T_g"].grid(linestyle='--', color="lightgrey")
axis["T_g"].set_ylabel("Temperature (K)")
axis["T_g"].set_xlabel("Time (min)")

fig.savefig("example2.png", dpi=300)
plt.show()