import numpy as np
from scipy.interpolate import make_smoothing_spline

def get_mrt_error(
        T_mrt: list,
        T_g: list,
        T_a: list,
        V_a: list,
        t_eval: list,
        sigma_g: float,
        sigma_a: float,
        sigma_V: float,
):
    sigma = np.power(np.array([sigma_g, sigma_a, sigma_V]), 2)
    M = np.diag(sigma)

    dt = 0.001

    errors = []
    
    for i, t in enumerate(t_eval[:-1]):
        dTdT_g = (T_mrt[i] - T_mrt[i + 1]) / (T_g[i] - T_g[i+1])
        dTdT_a = (T_mrt[i] - T_mrt[i + 1]) / (T_a[i] - T_a[i+1])
        dTdV_a = (T_mrt[i] - T_mrt[i + 1]) / (V_a[i] - V_a[i+1])

        # dTdT_g = (T_mrt(t) - T_mrt(t+dt)) / (T_g(t) - T_g(t+dt))
        # dTdT_a = (T_mrt(t) - T_mrt(t+dt)) / (T_a(t) - T_a(t+dt))
        # dTdV_a = (T_mrt(t) - T_mrt(t+dt)) / (V_a(t) - V_a(t+dt))

        J = np.array([dTdT_g, dTdT_a, dTdV_a])

        # Calculate the error in MRT using the formula: σ_MRT^2 = J * M * J^T
        sigma_mrt_squared = J.T @ M @ J
        sigma_mrt = np.sqrt(sigma_mrt_squared)

        errors.append(sigma_mrt)

    errors.append(errors[-1])  # Append the last error to maintain the same length as T_mrt

    return errors


def spline_bootstrapping_residuals(t_eval, y, n_boot=400):
    spline = make_smoothing_spline(t_eval, y)
    y_hat = spline(t_eval)
    residuals = y - y_hat
    
    fits = []
    
    for _ in range(n_boot):
        resampled = np.random.choice(residuals, size=len(y), replace=True)
        y_bootstrap = y_hat + resampled
        
        spline_bootstrap = make_smoothing_spline(t_eval, y_bootstrap)
        fits.append(spline_bootstrap(t_eval))
    
    fits = np.array(fits)
    
    lower_band = np.percentile(fits, 2.5, axis=0)
    upper_band = np.percentile(fits, 97.5, axis=0)

    return lower_band, upper_band


if __name__ == "__main__":
    T_mrt = [1, 2, 3]
    T_g = [1, 2, 3]
    T_a = [1, 2, 3]
    V_a = [1, 2, 3]

    sigma_g = 0.5
    sigma_a = 0.25
    sigma_V = 0.5

    print(get_mrt_error(T_mrt, T_g, T_a, V_a, sigma_g, sigma_a, sigma_V))

    # jackknife_confidence_intervals(x, y, f)