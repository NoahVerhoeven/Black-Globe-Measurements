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


def moving_average_matrix(input_array, window_size, mode="constant"):
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
    
    elif mode == "exponential decay": # weights decrease linearly as we go away from current point (most right)
        for i in range(1, window_size + 2):
            linear_row = 2 * np.linspace(0, 1, i)[1:] / i
            A[i-2, 0:i-1] = linear_row

        for i in range(window_size + 2, len(A) + 2):
            A[i-2, i-window_size-1:i-1] = linear_row
        return A
