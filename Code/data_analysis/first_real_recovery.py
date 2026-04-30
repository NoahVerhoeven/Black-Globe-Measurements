from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.differentiate import derivative
from scipy.interpolate import interp1d, make_smoothing_spline
from mrt_tools import (
    grey_body_MRT_estimate,
    spline_bootstrapping_residuals,
    dTdt_shell_only,
    inverse_exponential_smoothing
)
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.family'] = 'Times New Roman'
plt.rc('legend',fontsize=11.5)

data_dict2 = {
    "B2": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\B2.CSV"),
        "epsilon": 0.77,
        "rho": 1250,
        "heat capacity": 1800,
        "diameter": 40 * 10 ** -3,
        "thickness": 2 * 10 ** -3
    },
    "BP": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\BP.CSV"),
        "epsilon": 0.84,
        "rho": 1100,
        "heat capacity": 1506,
        "diameter": 40 * 10 ** -3,
        "thickness": 0.4 * 10 ** -3
    },
    "G1": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\G1.CSV"),
        "epsilon": 0.74,
        "rho": 1250,
        "heat capacity": 1800,
        "diameter": 40 * 10 ** -3,
        "thickness": 1 * 10 ** -3
    },
    "G2": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\G2.CSV"),
        "epsilon": 0.77,
        "rho": 1250,
        "heat capacity": 1800,
        "diameter": 40 * 10 ** -3,
        "thickness": 2 * 10 ** -3
    },
    "GP": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\GP.CSV"),
        "epsilon": 0.62,
        "rho": 1100,
        "heat capacity": 1506,
        "diameter": 40 * 10 ** -3,
        "thickness": 0.4 * 10 ** -3
    },
    "S1": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\S1.CSV"),
        "epsilon": 0.69,
        "rho": 1250,
        "heat capacity": 1800,
        "diameter": 40 * 10 ** -3,
        "thickness": 1 * 10 ** -3
    },
    "S2": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\S2.CSV"),
        "epsilon": 0.66,
        "rho": 1250,
        "heat capacity": 1800,
        "diameter": 40 * 10 ** -3,
        "thickness": 2 * 10 ** -3
    },
    "Reference": {
        "CSV": pd.read_csv("Data\\globe_temperature_test\\Reference.CSV"),
        "epsilon": 0.94,
        "rho": 8960,
        "heat capacity": 384,
        "diameter": 150 * 10 ** -3,
        "thickness": 0.4 * 10 ** -3
    },
    "Vlinder": pd.read_csv("Data\\globe_temperature_test\\Vlinder_S9.CSV", sep=";")
}

start_time = pd.Timestamp(2026, 4, 8, 14, 15, 0)
end_time = pd.Timestamp(2026, 4, 8, 15, 45, 0)

# VLINDER DATA: These are reliable wind speed and air temperature measurements
vlinder = data_dict2["Vlinder"].dropna(ignore_index=True)
vlinder.insert(0, "datetime", pd.to_datetime(vlinder['Datum'] + ' ' + vlinder['Tijd (UTC)'], format='%Y-%m-%d %H:%M:%S'))
backup = vlinder.iloc[np.where((vlinder["datetime"] >= start_time) & (vlinder["datetime"] <= end_time))]
vlinder_filtered = backup.copy()

for device in data_dict2.keys():
    if device == "Vlinder":
        break

    print(f"\n---Finding optimal epsilon for {device}---")

    vlinder_filtered = backup.copy()

    # GLOBE DATA: These are the data we collected with our globes
    df = data_dict2[device]["CSV"].dropna(ignore_index=True)
    df.insert(0, "datetime", pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]]))
    df_filtered = df.iloc[np.where((df["datetime"] >= start_time) & (df["datetime"] <= end_time))]

    # MAKING SECONDS COLUMN
    if vlinder_filtered['datetime'].iloc[0] < df_filtered['datetime'].iloc[0]:
        vlinder_filtered["diff"] = vlinder_filtered['datetime'] - vlinder_filtered['datetime'].iloc[0]
        df_filtered["diff"] = df_filtered['datetime'] - vlinder_filtered['datetime'].iloc[0]
    else:
        vlinder_filtered["diff"] = vlinder_filtered['datetime'] - df_filtered['datetime'].iloc[0]
        df_filtered["diff"] = df_filtered['datetime'] - df_filtered['datetime'].iloc[0]

    vlinder_filtered.insert(1, "TotalSeconds", vlinder_filtered["diff"].dt.total_seconds())
    df_filtered.insert(1, "TotalSeconds", df_filtered["diff"].dt.total_seconds())

    df_filtered = df_filtered.drop('diff', axis=1)

    dt = df_filtered["TotalSeconds"][1] - df_filtered["TotalSeconds"][0]
    t_eval = df_filtered["TotalSeconds"]
    T_g = df_filtered["BlackGlobetemp"] + 273.15

    # USE VLINDER DATA FOR WIND SPEED AND AIR TEMP
    w = np.array([1/len(t_eval)] * len(t_eval))

    V_a = interp1d(vlinder_filtered["TotalSeconds"], vlinder_filtered["Windsnelheid"])
    T_a = interp1d(vlinder_filtered["TotalSeconds"], vlinder_filtered["Temperatuur"] + 273.15)

    # CONSTANTS: We'll work with the shell-only simulation [https://matmake.com/properties/density-of-polymers-and-plastics.html]
    sigma = 5.67037 * 10 ** -8 # [J/s*m^2*K^4]
    thickness = data_dict2[device]["thickness"] # Thickness of the globe shell [m]
    rho = data_dict2[device]["rho"]  # Density of the globe [kg/m3]
    c = data_dict2[device]["heat capacity"] # Specific heat capacity of the globe [J/kg*K]
    D = data_dict2[device]["diameter"]  # Diameter of the shell [m]
    V = quad(lambda r: 4 * np.pi * r ** 2, (D - thickness)/2, D/2)[0] # Volume of the globe [m3]
    A = 4 * np.pi * (D/2) ** 2 # Surface area of the globe [m2]
    C = c * rho * V # [J/K]

    h = lambda t: (6.3 * V_a(t) ** 0.6) / (D ** 0.4) # Forced convective heat transfer coefficient (McAdams) [J/s*m^2*K]
    h_spline = make_smoothing_spline(t_eval, [h(t) for t in t_eval], w)
    
    old_error = float("inf")

    if data_dict2[device]["epsilon"] == None:
       for epsilon in range(60, 85):
            epsilon /= 100
            print(f"Testing {epsilon}")
            # ESTIMATE MRT: We use the grey body estimate to guess the MRT from our empirical data
            estimated_mrt = np.array([grey_body_MRT_estimate(T_g[i], h(t), T_a(t), epsilon) for i, t in enumerate(t_eval)])

            # SMOOTHING: We apply spline smoothing based on the GCV criterion
            tau = lambda k: C / (A * (4 * epsilon * sigma * np.mean(T_g) ** 3 + h_spline(t_eval[k])))
            alpha = lambda k: 1 - np.exp(-(dt / tau(k)))

            smooth_estimated_mrt = make_smoothing_spline(t_eval, estimated_mrt, w)(t_eval)

            # FINITE DIFFERENCE APPROXIMATION: If we numerically estimate the derivative at each point we can find the T_mrt
            smooth_T_g = make_smoothing_spline(t_eval, [T_g[i] for i, _ in enumerate(t_eval)], w)
            smooth_T_a = make_smoothing_spline(t_eval, [T_a(t) for t in t_eval], w)
            derivatives = derivative(smooth_T_g, t_eval)
            finite_diff_mrt = []

            for dT_g, T_gg, T_aa, hh in zip(derivatives.df, smooth_T_g(t_eval), smooth_T_a(t_eval), h_spline(t_eval)):
                point = np.float_power(((dT_g * C / A) + epsilon * sigma * T_gg ** 4 + hh * (T_gg - T_aa)) / (epsilon * sigma), 1/4)
                finite_diff_mrt.append(point)

            # SIMULATION: Simulate what the globe temperature should be
            args = [h,  T_a, epsilon, C, A]
            sol = solve_ivp(
                dTdt_shell_only,
                y0 = [T_g[0]],
                t_span=[t_eval.iloc[0], t_eval.iloc[-1]],
                t_eval=t_eval,
                args=(interp1d(t_eval, finite_diff_mrt), args),
                method="Radau"
            )

            new_error = np.mean(np.absolute(T_g - sol.y[0]))

            if old_error > new_error:
                data_dict2[device]["epsilon"] = epsilon
                old_error = new_error

    epsilon = data_dict2[device]["epsilon"]  # Emissivity
    print(f"Optimal epsilon for {device} is {epsilon}\n")

    # ESTIMATE MRT: We use the grey body estimate to guess the MRT from our empirical data
    estimated_mrt = np.array([grey_body_MRT_estimate(T_g[i], h(t), T_a(t), epsilon) for i, t in enumerate(t_eval)])

    # SMOOTHING: We apply spline smoothing based on the GCV criterion
    tau = lambda k: C / (A * (4 * epsilon * sigma * np.mean(T_g) ** 3 + h_spline(t_eval[k])))
    alpha = lambda k: 1 - np.exp(-(dt / tau(k)))

    smooth_estimated_mrt = make_smoothing_spline(t_eval, estimated_mrt, w)(t_eval)

    # INVERSE SMOOTHING: We recover the true mrt by inversing the exponential smoothing on the empirical data, and spline smoothing the result
    inverse_smoothed_mrt = inverse_exponential_smoothing(smooth_estimated_mrt, alpha, t_eval)

    # FINITE DIFFERENCE APPROXIMATION: If we numerically estimate the derivative at each point we can find the T_mrt
    smooth_T_g = make_smoothing_spline(t_eval, [T_g[i] for i, _ in enumerate(t_eval)], w)
    smooth_T_a = make_smoothing_spline(t_eval, [T_a(t) for t in t_eval], w)
    derivatives = derivative(smooth_T_g, t_eval)
    finite_diff_mrt = []

    for dT_g, T_gg, T_aa, hh in zip(derivatives.df, smooth_T_g(t_eval), smooth_T_a(t_eval), h_spline(t_eval)):
        point = np.float_power(((dT_g * C / A) + epsilon * sigma * T_gg ** 4 + hh * (T_gg - T_aa)) / (epsilon * sigma), 1/4)
        finite_diff_mrt.append(point)

    # SIMULATION: Simulate what the globe temperature should be
    args = [h,  T_a, epsilon, C, A]
    sol = solve_ivp(
        dTdt_shell_only,
        y0 = [T_g[0]],
        t_span=[t_eval.iloc[0], t_eval.iloc[-1]],
        t_eval=t_eval,
        args=(interp1d(t_eval, finite_diff_mrt), args),
        method="Radau"
    )

    # CONFIDENCE INTERVAL (estimate): We'll bootstrap residual to find the upper and lower bands were 95% of the true function lays
    lower_estimate, upper_estimate = spline_bootstrapping_residuals(t_eval, estimated_mrt, 1)

    df_filtered = df_filtered.drop(["year", "month", "day", "hour", "minute", "second"], axis=1)

    df_filtered["SimulatedGlobetemp"] = sol.y[0] - 273.15
    df_filtered["MRT"] = estimated_mrt - 273.15
    df_filtered["CorrectedMRT"] = np.array(finite_diff_mrt) - 273.15

    df_filtered.to_csv(f'Globe-Test-Recovery-{device}.CSV', index=False)

    # PLOT RESULTS
    left = [
        ["Wind"],
        ["Air"],
        ["T_g"]
    ]
    right = [
        ["Sim"],
        ["Sim"]
    ]
    fig, axis = plt.subplot_mosaic(
        [[left, right]], figsize=(16, 9),
        layout="constrained",
        width_ratios=[1.25, 2],
        sharex=True
    )
    fig.suptitle(f"Inverse Exponential Smoothing Algorithm\nfor Recovering True MRT from {device} Measurements "+r"($\epsilon=$"+f"{epsilon})", fontsize=18, fontweight="bold")
    fig.tight_layout(pad=2.5)

    axis["Sim"].scatter(df["datetime"], estimated_mrt, alpha=0.7, s=3.5, label="Empirical Data", lw=2)
    axis["Sim"].fill_between(df["datetime"], lower_estimate, upper_estimate, color="lightblue", label="95% Confidence Interval", lw=2.5)
    axis["Sim"].plot(df["datetime"], smooth_estimated_mrt, color="blue", label="Smoothing Spline", lw=2.5)
    # axis["Sim"].plot(df["datetime"][1:], smooth_recovered_mrt[1:], label="Recovered MRT", color="red", lw=2.5)
    axis["Sim"].plot(df["datetime"], finite_diff_mrt, label="Recovered MRT", color="green", lw=2.5)
    axis["Sim"].plot(df["datetime"][1:], inverse_smoothed_mrt[1:], label="Recovered MRT", color="red", lw=2.5)
    axis["Sim"].set_ylabel('Temperature (K)')
    axis["Sim"].set_title('Recovered MRT from Empirical Data')
    axis["Sim"].grid()
    axis["Sim"].legend()

    axis["Wind"].set_title("Wind Speed")
    axis["Wind"].plot(df["datetime"], [V_a(t) for t in t_eval], label=r"Empirical $V_a$", lw=2, color="mediumorchid")
    axis["Wind"].legend()
    axis["Wind"].grid()
    axis["Wind"].set_ylabel("Wind Speed (m/s)")

    axis["Air"].set_title("Air Temperature")
    axis["Air"].plot(df["datetime"], [T_a(t) for t in t_eval], label=r"Empirical $T_a$", lw=2, color="mediumpurple")
    axis["Air"].grid()
    axis["Air"].legend()
    axis["Air"].set_ylabel("Temperature (K)")

    axis["T_g"].set_title("Globe Temperature")
    axis["T_g"].plot(df["datetime"], T_g, label=r"Empirical $T_g$", lw=2, color="royalblue")
    axis["T_g"].plot(df["datetime"], sol.y[0], color="black", lw=2, linestyle="dashed", label=r"Simulated $\hat{T}_g$")
    axis["T_g"].grid()
    axis["T_g"].legend()
    axis["T_g"].set_ylabel("Temperature (K)")
    axis["T_g"].set_xlabel("Time (min)")

    fig.savefig(f"Globe-Test-Recovery-{device}.png", dpi=600)
plt.show()