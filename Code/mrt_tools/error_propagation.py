import numpy as np


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


if __name__ == "__main__":
    T_mrt = [1, 2, 3]
    T_g = [1, 2, 3]
    T_a = [1, 2, 3]
    V_a = [1, 2, 3]

    sigma_g = 0.5
    sigma_a = 0.25
    sigma_V = 0.5

    print(get_mrt_error(T_mrt, T_g, T_a, V_a, sigma_g, sigma_a, sigma_V))