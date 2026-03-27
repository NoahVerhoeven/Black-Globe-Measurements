import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline
from .mrt_simulation import dTdt, grey_body_MRT_estimate


def moving_average_matrix(input, window_size, mode="linear decay", base=1.02):
    """
    Parameters

    input: Array we want to apply the moving average to, gives the necessary length.
    window_size: Window size of the moving average.
    mode: Decides the moving average matrix and its weights.
    base: Only for exponential methods, it is the base for the exponential (base^{-x}).

    Higher base give smoother results, but can start deviating from the true MRT
    """
    length = len(input)

    A = np.tri(length, length, 0) / window_size
    A[np.tril_indices(length, -window_size)] = 0

    A[0, :1] = 1

    if mode == "constant": # weights stay constant over window
        for i in range(1, window_size):
            A[i, :i+1] = 1/(i+1)
        return A
    
    elif mode == "linear growth": # weights grow linearly as we go away from current point (most right)
        for i in range(2, window_size + 2):
            linear_row = np.flip((2 / (i * (i - 1))) * np.arange(1, i))
            A[i-2, 0:i-1] = linear_row

        for i in range(window_size + 2, len(A) + 2):
            A[i-2, i-window_size-1:i-1] = linear_row
        return A
    
    elif mode == "linear decay": # weights decrease linearly as we go away from current point (most right)
        for i in range(2, window_size + 2):
            linear_row = (2 / (i * (i - 1))) * np.arange(1, i)
            A[i-2, 0:i-1] = linear_row

        for i in range(window_size + 2, len(A) + 2):
            A[i-2, i-window_size-1:i-1] = linear_row
        return A
    
    elif mode == "exponential growth": # weights decrease exponentially as we go away from current point (most right)
        for i in range(2, window_size + 2):
            exponential_row = base * ((1 - 1 / base) / (1 - base ** (-i + 1))) * np.power(base, (-np.arange(1, i)))
            A[i-2, 0:i-1] = exponential_row

        for i in range(window_size + 2, len(A) + 2):
            A[i-2, i-window_size-1:i-1] = exponential_row
        return A
    
    elif mode == "exponential decay": # weights decrease exponentially as we go away from current point (most right)
        for i in range(2, window_size + 2):
            exponential_row = base * ((1 - 1 / base) / (1 - base ** (-i + 1))) * np.power(base, (-np.flip(np.arange(1, i))))
            A[i-2, 0:i-1] = exponential_row

        for i in range(window_size + 2, len(A) + 2):
            A[i-2, i-window_size-1:i-1] = exponential_row
        return A
    
    elif mode == "gaussian":
        for i in range(window_size):
            # Define window bounds
            left = max(0, i - window_size // 2)
            right = min(window_size, i + window_size // 2 + 1)

            # Positions in the window
            positions = np.arange(left, right)

            # Gaussian weights centered at i
            weights = np.exp(-((positions - i) ** 2) / (2 * ((window_size - 1) // 6) ** 2))

            # Normalize so row sums to 1
            weights /= weights.sum()

            # Assign to matrix
            A[i, left:right] = weights
        return A


def recovery_error(recovered_data, empirical_data, mode="square"):
    '''
    Parameters

    recovered_data: Data we recovered based on empirical data, for mrt this is
    often the recovered true MRT or the recovered estimated MRT.
    empirical_data: Empirical data which we calculate the error from, our
    recovered data should match this as closely as possible.
    mode: Mode to calculate error.

    For MRT the empirical data are the estimations we obtain through measurement
    of the globe temperature, and then calculating the MRT. We obtain the recovered
    data by first recovering the true MRT and simulating how our estimate using dTdt.

    In simulations we can impose a synthetic true MRT, this now acts as the empirical
    data. If use the recovery algorithm we can check our recovered true MRT against
    the synthetic true MRT. A good recovery algorithm reproduces the synthetic true MRT
    nicely.
    '''
    if mode == "square":
        return np.sum(np.square(empirical_data - recovered_data))
    elif mode == "absolute":
        return np.sum(np.absolute(empirical_data - recovered_data))
    

def recover_mrt(empirical_data, window_size, args, t_eval, mode="linear decay", base=1.02):
    h,  T_a, epsilon, constant, A_shell, A_i, h_i, constant_i = args

    A = moving_average_matrix(
        input=empirical_data,
        window_size=window_size,
        mode=mode,
        base=base
    )
    A_inv = inv(A)

    recovered_true_mrt = A_inv@empirical_data
    recovered_true_mrt_func = interp1d(t_eval, recovered_true_mrt, kind='linear', fill_value="extrapolate")

    current_dTdt = lambda t, T: dTdt(t, T, recovered_true_mrt_func, args)

    sol = solve_ivp(current_dTdt, [t_eval[0], t_eval[-1]], [295, 295], method="Radau", t_eval=t_eval) # Implicite method to acount for stiffness

    recovered_inner_temp = sol.y[0]
    recovered_shell_temp = sol.y[1]

    recovered_estimated_mrt = np.array([grey_body_MRT_estimate(T, h(t), T_a, epsilon) for T, t in zip(recovered_inner_temp, sol.t)])

    return recovered_estimated_mrt, recovered_true_mrt, recovered_inner_temp, recovered_shell_temp


def optimize_recovery(empirical_mrt, smooth_empirical_mrt, args, t_eval, optimizing_range, method="bisection", smoothing_window=90, error_method="absolute"):
    '''
    Optimize true MRT recovery by numerically finding the optimal moving average window.
    '''
    best_error = float("inf")

    window_guesses = []
    error_array = []

    # smoothing_matrix = moving_average_matrix(
    #     input=empirical_mrt,
    #     window_size=smoothing_window,
    #     mode="linear decay"
    # ) # If estimate isn't smooth, this method will never work
    # smooth_empirical_mrt = smoothing_matrix@empirical_mrt

    if method == "brute force":
        for window_size_guess in range(optimizing_range[0], optimizing_range[1]):
            recovered_estimated_mrt, recovered_true_mrt, _, _ = recover_mrt(
                empirical_data=smooth_empirical_mrt,
                window_size=window_size_guess,
                args=args,
                mode="linear decay",
                t_eval=t_eval
            )

            new_error = recovery_error(
                recovered_data=recovered_estimated_mrt, # error calculated against empirical data!!! not smoothed out data
                empirical_data=empirical_mrt,
                mode=error_method # set way to calculate error
            )

            window_guesses.append(window_size_guess)
            error_array.append(new_error)

            print(new_error, window_size_guess)
            
            if best_error > new_error: # if old error is bigger, the new window size is better, so we update the old error
                best_error = new_error
                best_window_size = window_size_guess
                best_recovered_estimated_mrt = recovered_estimated_mrt
                best_recovered_true_mrt = recovered_true_mrt

        return best_recovered_estimated_mrt, best_recovered_true_mrt, best_window_size, best_error, error_array, window_guesses
    
    if method == "golden section":
        lo, hi = optimizing_range
        gr = (np.sqrt(5) + 1) / 2  # golden ratio ≈ 1.618

        m1 = int(hi - (hi - lo) / gr)
        m2 = int(lo + (hi - lo) / gr)

        m1_recovered, m1_true, _, _ = recover_mrt(
            empirical_data=smooth_empirical_mrt,
            window_size=m1,
            args=args,
            t_eval=t_eval
        )
        m1_error = recovery_error(recovered_data=m1_recovered, empirical_data=empirical_mrt)

        m2_recovered, m2_true, _, _ = recover_mrt(
            empirical_data=smooth_empirical_mrt,
            window_size=m2,
            args=args,
            t_eval=t_eval
        )
        m2_error = recovery_error(
            recovered_data=m2_recovered,
            empirical_data=empirical_mrt,
            mode=error_method
        )

        for _ in range(50):
            if hi - lo < 2:
                break

            print(f"lo={lo}, m1={m1} (err={m1_error:.4f}), m2={m2} (err={m2_error:.4f}), hi={hi}")

            if m1_error > m2_error:
                lo = m1
                m1, m1_error = m2, m2_error          # reuse m2 — no API call needed
                m2 = int(lo + (hi - lo) / gr)
                m2_recovered, m2_true, _, _ = recover_mrt(
                    empirical_data=smooth_empirical_mrt,
                    window_size=m2,
                    args=args,
                    t_eval=t_eval
                )
                m2_error = recovery_error(
                    recovered_data=m2_recovered,
                    empirical_data=empirical_mrt,
                    mode=error_method
                )

            else:
                hi = m2
                m2, m2_error = m1, m1_error          # reuse m1 — no API call needed
                m1 = int(hi - (hi - lo) / gr)
                m1_recovered, m1_true, _, _ = recover_mrt(
                    empirical_data=smooth_empirical_mrt,
                    window_size=m1,
                    args=args,
                    t_eval=t_eval
                )
                m1_error = recovery_error(
                    recovered_data=m1_recovered,
                    empirical_data=empirical_mrt,
                    mode=error_method
                )

        # Final sweep over the small remaining range
        best_window_size = None
        best_error = float("inf")

        for w in range(lo, hi + 1):
            recovered, true_mrt, _, _ = recover_mrt(
                empirical_data=smooth_empirical_mrt,
                window_size=w,
                args=args,
                t_eval=t_eval
            )
            err = recovery_error(
                recovered_data=recovered,
                empirical_data=empirical_mrt,
                mode=error_method
            )
            if err < best_error:
                best_error = err
                best_window_size = w
                best_recovered_estimated_mrt = recovered
                best_recovered_true_mrt = true_mrt

        return best_recovered_estimated_mrt, best_recovered_true_mrt, best_window_size, best_error, error_array, window_guesses

