import numpy as np


def grey_body_MRT_estimate(T, h, T_a, epsilon):
    sigma = 5.67 * 10 ** -8

    return (T ** 4 + h * (T - T_a) / (sigma * epsilon)) ** 0.25


def dTdt(t, T, MRT_func, h,  T_a, epsilon, constant, A, A_i, h_i, constant_i):
    sigma = 5.67 * 10 ** -8

    T_i, T_g = T[0], T[1]

    dT_i = ((A_i * h_i) / constant_i) * (T_g - T_i)
    dT_g = (A * (sigma * epsilon * MRT_func(t) ** 4 - sigma * epsilon * T_g ** 4 - h(t) * (T_g - T_a))  - A_i * h_i * (T_g - T_i)) / constant

    return [dT_i, dT_g]


def moving_average_matrix(input_array, window_size, mode="constant", base=1.02):
    """
    parameters

    input_array: array we want to apply the moving average to, gives the necessary length
    window_size: window size of the moving average
    mode: decides the moving average matrix and its weights
    base: only for exponential methods, it is the base for the exponential (base^{-x})

    higher base give smoother results, but can start deviating from the true MRT
    """
    length = len(input_array)

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
