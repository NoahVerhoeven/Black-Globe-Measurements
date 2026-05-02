import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, quad
from scipy.differentiate import derivative
from scipy.interpolate import make_smoothing_spline
from mrt_tools import (
    dTdt_shell_only,
    grey_body_MRT_estimate,
    inverse_exponential_smoothing
)

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

for i in range(100):
    # SIMULATE REAL EMPIRICAL DATA: We add noise (zero-mean Gaussian) to simulate statistical errors in our measurements
    noisy_T_g = sol.y[0] + np.random.normal(0, 0.25, sol.t.shape)
    noisy_V_a = np.array([V_a(t) for t in sol.t]) + np.random.normal(0, 0.5, sol.t.shape)
    noisy_T_a = np.array([T_a(t) for t in sol.t]) + np.random.normal(0, 0.25, sol.t.shape)
    noisy_h = lambda i: (6.3 * noisy_V_a[i] ** 0.6) / (D ** 0.4)

    # ESTIMATE MRT: We use the grey body estimate to guess the MRT from our 'empirical' data
    empirical_data = np.array([grey_body_MRT_estimate(noisy_T_g[i], noisy_h(i), noisy_T_a[i], epsilon) for i, t in enumerate(sol.t)])
    estimated_mrt = np.array([grey_body_MRT_estimate(sol.y[0][i], h(t), T_a(t), epsilon) for i, t in enumerate(sol.t)])

    # SMOOTHING: We apply spline smoothing based on the GCV criterion
    h_spline = make_smoothing_spline(t_eval, [noisy_h(i) for i in range(len(t_eval))], w)
    tau = lambda k: constant / (A * (4 * epsilon * sigma * np.mean(empirical_data) ** 3 + h_spline(t_eval[k])))
    alpha = lambda k: 1 - np.exp(-((t_eval[1] - t_eval[0]) / tau(k)))

    smooth_func = make_smoothing_spline(sol.t, empirical_data, w)
    smooth_estimated_mrt = smooth_func(sol.t)

    # INVERSE EXPONENTIAL SMOOTHING: We recover the true mrt by inversing the exponential smoothing on the empirical data, and spline smoothing the result
    inverse_smoothed_mrt = inverse_exponential_smoothing(smooth_estimated_mrt, alpha, t_eval)

    # FINITE DIFFERENCE APPROXIMATION: If we numerically estimate the derivative at each point we can find the T_mrt
    smooth_T_g = make_smoothing_spline(t_eval, noisy_T_g, w)
    smooth_T_a = make_smoothing_spline(t_eval, noisy_T_a, w)
    derivatives = derivative(smooth_T_g, t_eval)
    finite_diff_mrt = []

    for dT_g, T_gg, T_aa, hh in zip(derivatives.df, smooth_T_g(t_eval), smooth_T_a(t_eval), h_spline(t_eval)):
        point = np.float_power(((dT_g * constant / A) + epsilon * sigma * T_gg ** 4 + hh * (T_gg - T_aa)) / (epsilon * sigma), 1/4)
        finite_diff_mrt.append(point)

    finite_diff_mrt = np.array(finite_diff_mrt)

    e1 = np.mean(np.absolute(true_mrt - inverse_smoothed_mrt))
    e2 = np.mean(np.absolute(true_mrt - finite_diff_mrt))

    if e1 < e2:
        print(f"Error without correction {np.mean(np.absolute(true_mrt - smooth_estimated_mrt))}")
        print(f"Error inverse smoothing (T_0 around MRT): {e1} [best]")
        print(f"Error finite difference: {e2}\n")
    else:
        print(f"Error without correction {np.mean(np.absolute(true_mrt - smooth_estimated_mrt))}")
        print(f"Error inverse smoothing (T_0 around MRT): {e1}")
        print(f"Error finite difference: {e2} [best]\n")

    df = pd.DataFrame({
        "MRT": true_mrt,
        "Globetemp": sol.y[0],
        "Airtemp": [T_a(t) for t in sol.t],
        "Windspeed": [V_a(t) for t in sol.t],
        "EstimatedMRT": estimated_mrt,
        "NoisyGlobetemp": noisy_T_g,
        "NoisyAirtemp": noisy_T_a,
        "NoisyWindspeed": noisy_V_a,
        "NoisyEstimatedMRT": empirical_data,
        "SmoothedEstimatedMRT": smooth_estimated_mrt,
        "FiniteDiffMRT": finite_diff_mrt,
        "InverseSmoothedMRT": inverse_smoothed_mrt
    })

    df.to_csv(f"Constant_{i}.CSV", index=False)